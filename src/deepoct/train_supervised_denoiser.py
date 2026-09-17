"""train_supervised_denoiser.py — SUPERVISED OCT denoiser on Sparsity_SDOCT_2012's
clean/noisy pairs, built to BEAT self-supervised N2N and the classical baselines.

WHY. N2N is self-supervised and NEVER uses the clean averaged references — it ranks
mid-pack (~25.3 dB), barely above a Gaussian blur and below BM3D (~26.9) / NLM
(~27.8). Sparsity HAS clean references (each folder: noisy ``Raw`` frame + high-SNR
``Averaged``). A denoiser trained WITH the clean target as supervision should beat
all of them. This script builds and evaluates that model.

Does NOT modify any existing file. Does NOT overwrite ``n2n_denoiser.pth`` — the
supervised weights save to ``supervised_denoiser.pth``.

DATA / LEAK CONTROL (the crux — only ~11 aligned subjects):
  * pairs = Sparsity folders, registered with the SAME phase-correlation alignment
    used by benchmark_denoisers (``register_pair``); misaligned folders excluded.
  * each folder is a distinct SUBJECT, so the split is PATIENT-LEVEL by construction.
  * With n≈11, a single held-out val set is noisy, so the honest full-set number is
    produced by PATIENT-LEVEL K-FOLD: every pair is predicted by a model that never
    trained on it (out-of-fold / OOF). Those OOF predictions are scored on ALL
    aligned pairs with the EXACT benchmark_denoisers metrics -> directly comparable
    to the ranked table. Overfitting is reported as (in-fold train PSNR - OOF val
    PSNR).

ARCH. A residual denoising U-Net (predict the noise, add it back) with a ZERO-INIT
final conv so it is the identity at init — the anti-darkening lesson from N2V. Wider
/ deeper than the N2N net (configurable ``--base`` / ``--levels``), but deliberately
NOT a giant Restormer: with n≈11 a huge model just memorizes. Capacity is a flag.

SPECKLE / DOMAIN. OCT speckle is multiplicative in the linear domain (additive in
log). The script trains in BOTH domains (``--domain linear|log|both``) and reports
which wins. Log training maps x∈[0,1] to a normalized log space, denoises, and maps
back; the residual+zero-init identity (hence brightness preservation) holds in both.

LOSS. Charbonnier (robust L1, better than L2 for denoising) by default, with an
optional gradient edge-preservation term (``--edge-weight``) — off by default because
the benchmark showed no PSNR/EPI trade-off here; enable only if OOF EPI lags.

EVAL. Reuses benchmark_denoisers' metric functions verbatim (PSNR/SSIM/EPI/CNR/ENL,
same ROIs, same registration) and scores supervised vs N2N/BM3D/NLM/classical on the
same aligned pairs, prints the comparable ranked table + train/val gap + panels.

MODES:
  * ``--selftest`` (no data): synthetic clean/noisy folders; confirms training
    reduces loss, supervised OOF beats the noisy floor, both domains run, and the
    checkpoint saves (to a temp path, never the real n2n file).
  * real: ``--data-root <Sparsity_SDOCT_DATASET_2012>`` (+ optional
    ``--n2n-checkpoint n2n_denoiser.pth`` to include N2N in the table).

EXAMPLE:
  python train_supervised_denoiser.py --selftest
  python train_supervised_denoiser.py \
      --data-root /path/to/Sparsity_SDOCT_DATASET_2012/ \
      --n2n-checkpoint n2n_denoiser.pth --domain both --epochs 400 \
      --out-model supervised_denoiser.pth --out supervised_bench.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

# PORTED: these two sibling modules -- validate_noise_estimator (registration /
# discovery / IO) and benchmark_denoisers (the EXACT benchmark metrics + baseline
# denoisers, reused so the numbers stay comparable) -- were imported at module
# scope in the original. They are NOT part of this repository, and nothing at
# module scope here uses them: every reference is inside a training or
# benchmarking function, while the `DenoiseUNet` class below has no dependency on
# them at all.
#
# They are therefore resolved lazily by `_reuse()`, so importing this module for
# `DenoiseUNet` alone (which is all joint_taskaware_model.py needs) works without
# them, and a run that actually needs the benchmark path fails with a clear
# message naming what is missing rather than an opaque ImportError at import time.
_REUSED_FROM = {
    "validate_noise_estimator": (
        "_summ", "discover_sparsity", "load_gray", "register_pair", "resize_to",
        "segment_regions",
    ),
    "benchmark_denoisers": (
        "METRIC_KEYS", "all_metrics", "build_method_list", "est_mad_laplacian",
        "find_rois", "load_n2n", "save_panel", "_HAS_BM3D", "_HAS_SKIMAGE",
        "d_bm3d", "d_nlm", "d_median", "d_gaussian", "d_n2n",
    ),
}


def _reuse(name: str):
    """Resolve one symbol from the un-ported sibling benchmark modules.

    Raises ImportError naming the missing module and the symbol, so the failure
    is legible instead of surfacing as a NameError deep inside a training run.
    """
    import importlib

    for module_name, symbols in _REUSED_FROM.items():
        if name not in symbols:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise ImportError(
                f"{name!r} comes from {module_name!r}, which is not part of this "
                f"repository. The benchmark/training entry points in "
                f"train_supervised_denoiser.py need it; DenoiseUNet does not."
            ) from exc
        return getattr(module, name)
    raise AttributeError(name)


def __getattr__(name: str):
    """PEP 562 module-level fallback: resolve the deferred symbols on first use.

    This keeps every original call site unchanged -- `all_metrics(...)`,
    `_HAS_BM3D`, etc. still read exactly as they did -- while the underlying
    import happens only when the name is actually touched.
    """
    if any(name in symbols for symbols in _REUSED_FROM.values()):
        return _reuse(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_MODEL_SIZE = 512  # square training/apply resolution (matches N2N's apply path)


# =========================================================================== #
# Domain transforms (linear vs log). x in [0,1] <-> net working space in ~[0,1].
# =========================================================================== #
def domain_fwd(x01: torch.Tensor, domain: str, eps: float) -> torch.Tensor:
    """[0,1] linear -> net working space. 'linear' is identity; 'log' maps to a
    NORMALIZED log space so speckle (multiplicative) becomes ~additive while the net
    still operates in ~[0,1] (keeps residual/zero-init identity meaningful)."""
    if domain == "linear":
        return x01
    lo = math.log(eps)                          # log(eps)  (negative)
    hi = math.log(1.0 + eps)                     # ~0
    return (torch.log(x01.clamp(0, 1) + eps) - lo) / (hi - lo)


def domain_inv(work: torch.Tensor, domain: str, eps: float) -> torch.Tensor:
    """Net working space -> [0,1] linear (inverse of domain_fwd), clamped."""
    if domain == "linear":
        return work.clamp(0, 1)
    lo = math.log(eps)
    hi = math.log(1.0 + eps)
    x = torch.exp(work * (hi - lo) + lo) - eps
    return x.clamp(0, 1)


# =========================================================================== #
# Architecture — residual denoising U-Net (predict noise, add back; zero-init tail)
# =========================================================================== #
class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


class _Down(nn.Module):
    def __init__(self, i, o, blocks):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True))
        self.res = nn.Sequential(*[ResBlock(o) for _ in range(blocks)])

    def forward(self, x):
        return self.res(self.head(x))


class _Up(nn.Module):
    """Bilinear upsample -> align to skip -> concat -> fuse (no checkerboard, robust
    to odd sizes)."""

    def __init__(self, in_ch, skip_ch, out_ch, blocks):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(blocks)])

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.res(self.fuse(torch.cat([x, skip], dim=1)))


class DenoiseUNet(nn.Module):
    """Residual U-Net: ``out = x + tail(features)``, ``tail`` zero-initialized so the
    net is the identity at init (brightness preserved, learns a correction = the
    negative noise). ``levels`` pooling stages, ``blocks`` residual units per stage."""

    def __init__(self, base: int = 64, in_ch: int = 1, levels: int = 4,
                 blocks: int = 2) -> None:
        super().__init__()
        self.levels = levels
        chs = [base * (2 ** i) for i in range(levels)]        # per-level channels
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.downs.append(_Down(prev, c, blocks)); prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _Down(chs[-1], chs[-1] * 2, blocks)
        self.ups = nn.ModuleList()
        prev = chs[-1] * 2
        for c in reversed(chs):
            self.ups.append(_Up(prev, c, c, blocks)); prev = c
        self.tail = nn.Conv2d(chs[0], in_ch, 1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        skips = []
        h = x
        for d in self.downs:
            h = d(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)
        return x + self.tail(h)   # residual; identity at init


# =========================================================================== #
# Losses
# =========================================================================== #
def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps * eps))


def grad_loss(pred, target):
    """L1 on spatial gradients — an edge-preservation term (optional)."""
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


# =========================================================================== #
# Data: registered (noisy512, clean512) pairs; random-crop + dihedral augmentation
# =========================================================================== #
def build_pairs(records, align_tol: int) -> List[dict]:
    """Register every folder; keep aligned ones as {folder, raw512, clean512,
    raw_reg, clean_reg}. raw/clean are resized to 512 for uniform training; the
    native-res registered crops are kept for the benchmark (scored at native res)."""
    pairs = []
    misaligned = []
    for rec in records:
        raw = load_gray(rec["raw"]); clean = load_gray(rec["averaged"])
        raw_reg, clean_reg, dy, dx, aligned = register_pair(raw, clean, align_tol)
        if not aligned:
            misaligned.append(rec["folder"]); continue
        pairs.append({
            "folder": rec["folder"],
            "raw_reg": raw_reg, "clean_reg": clean_reg,
            "raw512": resize_to(raw_reg, (_MODEL_SIZE, _MODEL_SIZE)),
            "clean512": resize_to(clean_reg, (_MODEL_SIZE, _MODEL_SIZE)),
        })
    return pairs, misaligned


def sample_batch(train_pairs, crop, batch, domain, eps, device, rng):
    """Random augmented crops -> (noisy_work, clean_work) batch in the net domain."""
    xs, ys = [], []
    for _ in range(batch):
        p = train_pairs[rng.integers(0, len(train_pairs))]
        n = p["raw512"]; c = p["clean512"]
        H, W = n.shape
        y = int(rng.integers(0, H - crop + 1)); x = int(rng.integers(0, W - crop + 1))
        ncp = n[y:y + crop, x:x + crop].copy(); ccp = c[y:y + crop, x:x + crop].copy()
        if rng.random() < 0.5:
            ncp = ncp[:, ::-1]; ccp = ccp[:, ::-1]
        if rng.random() < 0.5:
            ncp = ncp[::-1, :]; ccp = ccp[::-1, :]
        k = int(rng.integers(0, 4))
        if k:
            ncp = np.rot90(ncp, k); ccp = np.rot90(ccp, k)
        xs.append(np.ascontiguousarray(ncp)); ys.append(np.ascontiguousarray(ccp))
    xn = torch.from_numpy(np.stack(xs)[:, None].astype(np.float32)).to(device)
    yn = torch.from_numpy(np.stack(ys)[:, None].astype(np.float32)).to(device)
    return domain_fwd(xn, domain, eps), domain_fwd(yn, domain, eps)


# =========================================================================== #
# Apply the trained net to one native image (resize512 -> denoise -> back)
# =========================================================================== #
def apply_model(net, noisy_native, domain, eps, device) -> np.ndarray:
    r512 = resize_to(noisy_native, (_MODEL_SIZE, _MODEL_SIZE))
    t = torch.from_numpy(r512.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        out = domain_inv(net(domain_fwd(t, domain, eps)), domain, eps)
    out = out[0, 0].cpu().numpy()
    return resize_to(out.astype(np.float32), noisy_native.shape)


def _psnr(x, ref):
    mse = float(np.mean((x.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else float(10.0 * math.log10(1.0 / mse))


def val_psnr(net, val_pairs, domain, eps, device) -> float:
    vals = [_psnr(apply_model(net, p["raw_reg"], domain, eps, device), p["clean_reg"])
            for p in val_pairs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# =========================================================================== #
# Train one model (early stopping on val PSNR); returns (best_net, history)
# =========================================================================== #
def train_one(train_pairs, val_pairs, domain, args, device, log_prefix=""):
    rng = np.random.default_rng(args.seed)
    net = DenoiseUNet(base=args.base, levels=args.levels, blocks=args.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = max(1, args.steps_per_epoch)
    best_val = -1e9; best_state = None; best_epoch = -1; patience = 0
    hist = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        run = 0.0
        for _ in range(steps):
            xb, yb = sample_batch(train_pairs, args.crop, args.batch, domain,
                                  args.log_eps, device, rng)
            pred = net(xb)
            loss = charbonnier(pred, yb)
            if args.edge_weight > 0:
                loss = loss + args.edge_weight * grad_loss(pred, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += float(loss.detach())
        vp = val_psnr(net, val_pairs, domain, args.log_eps, device) if val_pairs \
            else float("nan")
        tp = val_psnr(net, train_pairs, domain, args.log_eps, device)  # in-sample
        hist.append({"epoch": epoch, "loss": run / steps, "val_psnr": vp,
                     "train_psnr": tp})
        improved = np.isfinite(vp) and vp > best_val + 1e-4
        if improved or best_state is None:
            best_val = vp if np.isfinite(vp) else best_val
            best_state = copy.deepcopy(net.state_dict()); best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            print(f"  {log_prefix}ep {epoch:>3}: loss={run/steps:.5f} "
                  f"train_psnr={tp:5.2f} val_psnr={vp:5.2f} "
                  f"best={best_val:5.2f}@{best_epoch}")
        if val_pairs and patience >= args.patience:
            print(f"  {log_prefix}early stop @ {epoch} (no val gain for "
                  f"{args.patience} epochs)")
            break
    net.load_state_dict(best_state)
    return net, hist, best_epoch


# =========================================================================== #
# Patient-level K-fold -> out-of-fold predictions for EVERY pair (leak-free)
# =========================================================================== #
def kfold_indices(n: int, k: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    k = min(k, n)
    return [list(idx[i::k]) for i in range(k)]   # round-robin folds (patient-level)


def train_oof(pairs, domain, args, device):
    """K-fold: train on train folds, predict the held-out fold. Returns
    (oof_pred[folder]->native array, gap_info, folds)."""
    n = len(pairs)
    folds = kfold_indices(n, args.kfold, args.seed)
    oof: Dict[str, np.ndarray] = {}
    train_psnrs, val_psnrs = [], []
    for fi, val_idx in enumerate(folds):
        if not val_idx:
            continue
        val_pairs = [pairs[i] for i in val_idx]
        train_pairs = [pairs[i] for i in range(n) if i not in set(val_idx)]
        if not train_pairs:
            continue
        print(f"\n [{domain}] fold {fi+1}/{len(folds)}: "
              f"train={[p['folder'] for p in train_pairs]} "
              f"val={[p['folder'] for p in val_pairs]}")
        net, hist, _be = train_one(train_pairs, val_pairs, domain, args, device,
                                   log_prefix=f"f{fi+1} ")
        for p in val_pairs:
            oof[p["folder"]] = apply_model(net, p["raw_reg"], domain, args.log_eps,
                                           device)
        val_psnrs.append(val_psnr(net, val_pairs, domain, args.log_eps, device))
        train_psnrs.append(val_psnr(net, train_pairs, domain, args.log_eps, device))
    gap = {"train_psnr_mean": float(np.nanmean(train_psnrs)) if train_psnrs else float("nan"),
           "oof_val_psnr_mean": float(np.nanmean(val_psnrs)) if val_psnrs else float("nan")}
    gap["overfit_gap"] = gap["train_psnr_mean"] - gap["oof_val_psnr_mean"]
    return oof, gap, folds


# =========================================================================== #
# Evaluation with the EXACT benchmark metrics (supervised OOF vs baselines)
# =========================================================================== #
def evaluate(pairs, oof_pred, n2n_net, device, check_dir, n_visual, domain_label):
    """Score supervised OOF preds + baselines on ALL aligned pairs, identical metrics
    to benchmark_denoisers. Returns summary dict + saves panels."""
    # method registry: supervised first, then whatever baselines are available.
    methods: List[Tuple[str, str, Callable]] = [("noisy", "Noisy (floor)", lambda n, c: n)]
    methods.append(("supervised", f"SUPERVISED ({domain_label})",
                    lambda n, c: oof_pred[c["folder"]]))
    if n2n_net is not None:
        methods.append(("n2n", "N2N (self-sup)", d_n2n))
    if _HAS_BM3D:
        methods.append(("bm3d", "BM3D", d_bm3d))
    if _HAS_SKIMAGE:
        methods.append(("nlm", "Non-local means", d_nlm))
    methods.append(("median", "Median 3x3", d_median))
    methods.append(("gaussian", "Gaussian s=1", d_gaussian))

    agg = {m[0]: {k: [] for k in METRIC_KEYS} for m in methods}
    os.makedirs(check_dir, exist_ok=True); visual = 0
    rows = []
    for p in pairs:
        clean = p["clean_reg"]; raw = p["raw_reg"]
        masks = segment_regions(clean)
        sig_yx, bg_yx, patch = find_rois(clean, masks)
        sigma = float(est_mad_laplacian(raw))
        ctx = {"sigma": sigma, "device": device, "n2n_net": n2n_net,
               "folder": p["folder"]}
        outs = {}
        row = {"folder": p["folder"], "metrics": {}}
        for key, _label, fn in methods:
            den = fn(raw, ctx)   # noisy/supervised closures + baselines share (raw, ctx)
            outs[key] = den
            m = all_metrics(den, clean, sig_yx, bg_yx, patch)
            row["metrics"][key] = m
            for k in METRIC_KEYS:
                agg[key][k].append(m[k])
        rows.append(row)
        if visual < n_visual:
            cols = [("noisy", raw), ("SUPERVISED", outs["supervised"])]
            if "bm3d" in outs:
                cols.append(("BM3D", outs["bm3d"]))
            elif "n2n" in outs:
                cols.append(("N2N", outs["n2n"]))
            cols.append(("clean", clean))
            save_panel(os.path.join(check_dir, f"supervised_panel_{visual+1}.png"), cols)
            visual += 1

    summ = {m[0]: {k: _summ(agg[m[0]][k]) for k in METRIC_KEYS} for m in methods}
    return methods, summ, rows, visual


def print_table(methods, summ):
    print("\n  --- RANKED SUMMARY (mean +/- std over aligned pairs; EXACT benchmark "
          "metrics) ---")
    hdr = (f"  {'method':>20} | {'PSNR(dB)':>13} | {'SSIM':>12} | {'EPI':>12} | "
           f"{'CNR':>10} | {'ENL':>12}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    order = sorted(methods, key=lambda m: -(summ[m[0]]["psnr"]["mean"]
                   if np.isfinite(summ[m[0]]["psnr"]["mean"]) else -1e9))
    for key, label, _fn in order:
        s = summ[key]
        print(f"  {label:>20} | {s['psnr']['mean']:6.2f}+-{s['psnr']['std']:4.2f} | "
              f"{s['ssim']['mean']:.3f}+-{s['ssim']['std']:.3f} | "
              f"{s['epi']['mean']:.3f}+-{s['epi']['std']:.3f} | "
              f"{s['cnr']['mean']:5.2f}+-{s['cnr']['std']:4.2f} | "
              f"{s['enl']['mean']:6.1f}+-{s['enl']['std']:5.1f}")


# =========================================================================== #
# Selftest — synthetic clean/noisy folders (no data / no deps beyond torch/scipy)
# =========================================================================== #
def selftest() -> None:
    import shutil
    import tempfile
    from PIL import Image
    from benchmark_denoisers import _synth_pair
    print("=" * 96)
    print("SELFTEST — supervised training on synthetic clean/noisy pairs (logic + "
          "OOF eval + both domains)")
    print("=" * 96)
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="supervised_selftest_")
    root = os.path.join(tmp, "data")
    n_folders = 6
    for i in range(1, n_folders + 1):
        noisy, clean = _synth_pair(rng, sigma=float(0.04 + 0.01 * i))
        d = os.path.join(root, str(i)); os.makedirs(d, exist_ok=True)
        Image.fromarray((clean * 255).astype("uint8")).save(
            os.path.join(d, f"{i}_Averaged Image.tif"))
        Image.fromarray((noisy * 255).astype("uint8")).save(
            os.path.join(d, f"{i}_Raw Image.tif"))

    records = discover_sparsity(root)
    pairs, misaligned = build_pairs(records, align_tol=2)
    print(f"  aligned pairs: {len(pairs)} (misaligned excluded: {misaligned})")
    assert len(pairs) >= 4, "need >=4 aligned synthetic pairs"

    class A:  # tiny/fast hyperparams for the logic check
        base = 8; levels = 3; blocks = 1; crop = 96; batch = 4
        epochs = 8; steps_per_epoch = 15; patience = 6; lr = 1e-3
        edge_weight = 0.0; log_eps = 1e-3; seed = 0; kfold = 3
        log_every = 4
    args = A()
    device = torch.device("cpu")

    noisy_floor = float(np.mean([_psnr(p["raw_reg"], p["clean_reg"]) for p in pairs]))
    print(f"  noisy-floor PSNR = {noisy_floor:.2f} dB")

    results = {}
    for domain in ("linear", "log"):
        oof, gap, _folds = train_oof(pairs, domain, args, device)
        sup_psnr = float(np.mean([_psnr(oof[p["folder"]], p["clean_reg"])
                                  for p in pairs]))
        results[domain] = sup_psnr
        print(f"  [{domain}] OOF supervised PSNR = {sup_psnr:.2f} dB  "
              f"(train {gap['train_psnr_mean']:.2f} / oof-val "
              f"{gap['oof_val_psnr_mean']:.2f}, gap {gap['overfit_gap']:+.2f})")
        assert sup_psnr > noisy_floor, \
            f"{domain}: supervised OOF ({sup_psnr:.2f}) must beat noisy floor "\
            f"({noisy_floor:.2f})"

    # checkpoint save path check — MUST be supervised_denoiser.pth, never the n2n file
    best_domain = max(results, key=results.get)
    ckpt_path = os.path.join(tmp, "supervised_denoiser.pth")
    net = DenoiseUNet(base=args.base, levels=args.levels, blocks=args.blocks)
    torch.save({"model_state_dict": net.state_dict(), "base": args.base,
                "levels": args.levels, "blocks": args.blocks,
                "domain": best_domain, "log_eps": args.log_eps}, ckpt_path)
    assert os.path.basename(ckpt_path) == "supervised_denoiser.pth"
    assert "n2n" not in os.path.basename(ckpt_path)

    # end-to-end eval harness runs and ranks supervised above noisy
    methods, summ, _rows, _v = evaluate(
        pairs, oof, n2n_net=None, device=device,
        check_dir=os.path.join(tmp, "panels"), n_visual=2,
        domain_label=best_domain)
    print_table(methods, summ)
    assert summ["supervised"]["psnr"]["mean"] > summ["noisy"]["psnr"]["mean"]

    shutil.rmtree(tmp)
    print(f"\n  best domain: {best_domain}  (linear {results['linear']:.2f} / "
          f"log {results['log']:.2f} dB)")
    print("  SELFTEST PASS — supervised beats noisy floor OOF; both domains train; "
          "eval harness ranks it; checkpoint saved as supervised_denoiser.pth.")


# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n2n-checkpoint", default="n2n_denoiser.pth",
                    help="include N2N in the table if present (read-only)")
    ap.add_argument("--domain", choices=["linear", "log", "both"], default="both")
    ap.add_argument("--align-tol", type=int, default=2)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--steps-per-epoch", type=int, default=50)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--edge-weight", type=float, default=0.0,
                    help="gradient edge-preservation loss weight (enable if EPI lags)")
    ap.add_argument("--log-eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--check-dir", default="supervised_panels")
    ap.add_argument("--out-model", default="supervised_denoiser.pth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.selftest or not args.data_root:
        selftest(); return

    # HARD GUARD: never clobber the N2N checkpoint.
    if os.path.abspath(args.out_model) == os.path.abspath(args.n2n_checkpoint) \
            or os.path.basename(args.out_model) == "n2n_denoiser.pth":
        raise SystemExit("refusing to write over n2n_denoiser.pth; pick another "
                         "--out-model (default supervised_denoiser.pth).")

    device = torch.device("cuda" if (args.device != "cpu"
                          and torch.cuda.is_available()) else "cpu")
    records = discover_sparsity(args.data_root)
    if not records:
        raise SystemExit(f"no Raw/Averaged pairs under {args.data_root}")
    pairs, misaligned = build_pairs(records, args.align_tol)
    print(f"aligned pairs: {len(pairs)}  (misaligned excluded: {misaligned})")
    if len(pairs) < 3:
        raise SystemExit("need >=3 aligned pairs to train/evaluate.")
    noisy_floor = float(np.mean([_psnr(p["raw_reg"], p["clean_reg"]) for p in pairs]))
    print(f"noisy-floor PSNR = {noisy_floor:.2f} dB over {len(pairs)} pairs")

    n2n_net = None
    if os.path.isfile(args.n2n_checkpoint):
        n2n_net = load_n2n(args.n2n_checkpoint, device)
        print(f"loaded N2N (for the table): {args.n2n_checkpoint}")

    domains = ["linear", "log"] if args.domain == "both" else [args.domain]
    report = {"n_aligned": len(pairs), "misaligned": misaligned,
              "noisy_floor_psnr": noisy_floor, "domains": {}}
    domain_oof = {}
    for domain in domains:
        print("\n" + "=" * 96)
        print(f"TRAIN + K-FOLD OOF — domain = {domain}")
        print("=" * 96)
        oof, gap, folds = train_oof(pairs, domain, args, device)
        domain_oof[domain] = oof
        sup_psnr = float(np.mean([_psnr(oof[p["folder"]], p["clean_reg"])
                                  for p in pairs]))
        print(f"\n[{domain}] OOF supervised PSNR (all pairs) = {sup_psnr:.2f} dB")
        print(f"[{domain}] OVERFIT CHECK: in-fold train {gap['train_psnr_mean']:.2f} "
              f"vs OOF val {gap['oof_val_psnr_mean']:.2f} dB "
              f"=> gap {gap['overfit_gap']:+.2f} dB")
        report["domains"][domain] = {"oof_psnr": sup_psnr, "gap": gap}

    best_domain = max(domain_oof, key=lambda dm: report["domains"][dm]["oof_psnr"])
    print("\n" + "=" * 96)
    if len(domains) > 1:
        print(f"LOG-vs-LINEAR: "
              + "  ".join(f"{dm}={report['domains'][dm]['oof_psnr']:.2f}dB"
                          for dm in domains)
              + f"  -> WINNER: {best_domain}")

    # Final model: retrain on ALL pairs in the best domain, save (never the n2n file).
    print(f"\nFinal model: retrain on ALL {len(pairs)} pairs, domain={best_domain}")
    final_net, _hist, _be = train_one(pairs, [], best_domain, args, device,
                                      log_prefix="final ")
    torch.save({"model_state_dict": final_net.state_dict(), "base": args.base,
                "levels": args.levels, "blocks": args.blocks, "domain": best_domain,
                "log_eps": args.log_eps}, args.out_model)
    print(f"saved {args.out_model} (domain={best_domain})")

    # Comparable ranked table on the OOF preds of the best domain.
    methods, summ, rows, visual = evaluate(
        pairs, domain_oof[best_domain], n2n_net, device, args.check_dir,
        n_visual=4, domain_label=best_domain)
    print_table(methods, summ)

    sup = summ["supervised"]["psnr"]["mean"]
    print("\n  --- VERDICT ---")
    for key, label, _fn in methods:
        if key in ("noisy", "supervised"):
            continue
        d = sup - summ[key]["psnr"]["mean"]
        print(f"    supervised {'BEATS' if d > 0 else 'loses to'} {label} by "
              f"{d:+.2f} dB PSNR")
    print(f"  panels: {visual} -> {args.check_dir}/supervised_panel_*.png "
          f"(noisy | SUPERVISED | BM3D | clean)")

    report["best_domain"] = best_domain
    report["summary"] = summ
    report["per_image"] = rows
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
