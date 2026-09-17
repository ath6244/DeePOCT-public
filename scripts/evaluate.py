"""evaluate.py — standalone, READ-ONLY scoring of a trained DeepOCT-Ultra checkpoint.

Scores BOTH tasks on the VAL set: segmentation (Dice/IoU/HD95/ASD) and denoising
image-quality (PSNR where a clean reference exists, ENL, CNR), per-class for SDOCT
(normal/abnormal) and single-class for OIMHS. It does NOT train and does NOT modify
model.py / loss.py / train.py / any dataset file — it imports and reuses them.

Model is rebuilt EXACTLY as train.py builds it (same constructor flags, taken from
the checkpoint's saved args), weights loaded, ``eval()`` + ``no_grad`` + fp32.

Domains (see model.py / oimhs_dataset.py):
  * model input      : LOG domain, 3-channel 2.5D stack; center channel = index 1.
  * denoised output  : model ``final_pred``, LINEAR [0, ~1.5] (clamped).
  * target           : LINEAR clean center frame, [0,1].
  So the "noisy input" for image-quality is exp(input_center) (undo the log), the
  "denoised" is final_pred, and PSNR's reference is the LINEAR target. data_range=1.

Run (demo):
  python scripts/evaluate.py --checkpoint checkpoints/best_model_phase1.pth \
      --dataset sdoct --ablation_phase 1 --data_dir /path/to/dataset \
      --image-size 512 --out eval_phase1.json
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- #
# src-layout bootstrap: make ``deepoct`` importable when this script is run
# directly from a checkout (``python scripts/train.py ...``) without the package
# having been pip-installed. A real install takes precedence -- this only
# appends, so an installed copy already on sys.path still wins.
# --------------------------------------------------------------------------- #
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.append(_SRC)

from deepoct.model import DeepOCTUltra
from deepoct.losses import (boundary_distribution_to_depths, rasterize_boundary_mask,
                   derive_column_boundaries, mode_decode_column_boundary,
                   viterbi_decode_column_boundary, viterbi_decode_boundary_pair,
                   count_boundary_crossings, boundary_jump_stats)
import torch.nn.functional as F
import train  # reuse build_real_dataset / require_manifests (NOT modified)

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from scipy.stats import pearsonr, spearmanr
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dice_hard(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hard Dice at threshold 0.5. Both-empty -> 1.0 (perfect agreement)."""
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou_hard(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def soft_dice(probs: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """Replicates loss.py DiceLoss (soft dice on sigmoid probs) so the eval Dice
    can be checked against training's reported val dice_score."""
    inter = (probs * gt).sum()
    return float((2.0 * inter + eps) / (probs.sum() + gt.sum() + eps))


def surface_distances(pred: np.ndarray, gt: np.ndarray
                      ) -> Tuple[float, float]:
    """(ASD, HD95) in PIXELS between the two mask boundaries. NaN if either mask
    (or its boundary) is empty — those samples are excluded from the boundary
    aggregates. Pixel->physical scaling is UNKNOWN for this data, so values are
    reported in PIXELS."""
    if not _HAVE_SCIPY or pred.sum() == 0 or gt.sum() == 0:
        return float("nan"), float("nan")
    pb = pred ^ binary_erosion(pred)
    gb = gt ^ binary_erosion(gt)
    if pb.sum() == 0 or gb.sum() == 0:
        return float("nan"), float("nan")
    dt_to_gt = distance_transform_edt(~gb)
    dt_to_pred = distance_transform_edt(~pb)
    d = np.concatenate([dt_to_gt[pb], dt_to_pred[gb]])
    return float(d.mean()), float(np.percentile(d, 95))


def psnr(x: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((x - ref) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(data_range ** 2 / mse))


def enl(img: np.ndarray, region: np.ndarray) -> float:
    """Equivalent Number of Looks = mean^2 / variance on a homogeneous region.

    A variance floor (1e-6 on [0,1] images == std 1e-3, i.e. <0.1% intensity
    variation) marks a region as effectively CONSTANT / quantization-limited and
    returns NaN — otherwise ENL explodes to ~1e13 on a noise-free clean frame and
    pollutes the aggregate. This is why ENL is only meaningful with actual noise in
    the input (use --val_speckle deterministic, or real-noise SDOCT)."""
    vals = img[region]
    if vals.size < 20:
        return float("nan")
    var = float(vals.var())
    if var <= 1e-6:
        return float("nan")
    return float(vals.mean() ** 2 / var)


def cnr(img: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> float:
    """|mean_fg - mean_bg| / sqrt(0.5*(var_fg + var_bg))."""
    if fg.sum() < 20 or bg.sum() < 20:
        return float("nan")
    vf, vb = img[fg], img[bg]
    denom = math.sqrt(0.5 * (vf.var() + vb.var()))
    if denom <= 0:
        return float("nan")
    return float(abs(vf.mean() - vb.mean()) / denom)


# --------------------------------------------------------------------------- #
# Region location (honest, anatomy-based — NOT a cherry-picked flat patch)
# --------------------------------------------------------------------------- #
def locate_regions(clean_2d: np.ndarray, gt_mask: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (vitreous_bg, choroid_fg, description).

    fg = the GT choroid mask (principled: the annotated tissue region).
    bg (homogeneous, for ENL + CNR) = the VITREOUS: the signal-free dark band at
    the TOP of the B-scan. Located ANATOMICALLY, from the CLEAN target (so it is
    noise-independent and applied identically to noisy & denoised): pixels in the
    top 30% of rows that are darker than the image's 25th percentile AND outside
    the choroid mask. This is the standard signal-free ENL region, deliberately
    NOT the flattest patch (which biases ENL high)."""
    H = clean_2d.shape[0]
    top = np.zeros_like(gt_mask, dtype=bool)
    top[: int(0.30 * H), :] = True
    thr = float(np.percentile(clean_2d, 25))
    vitreous = top & (clean_2d < thr) & (~gt_mask)
    # Fallback: if the strict top-band is too small, relax to top 40% of rows.
    if vitreous.sum() < 50:
        top2 = np.zeros_like(gt_mask, dtype=bool)
        top2[: int(0.40 * H), :] = True
        vitreous = top2 & (clean_2d < thr) & (~gt_mask)
    fg = gt_mask.astype(bool)
    ys, xs = np.where(vitreous)
    if ys.size:
        desc = (f"vitreous bg: {int(vitreous.sum())} px, rows[{ys.min()}-{ys.max()}] "
                f"cols[{xs.min()}-{xs.max()}], mean={clean_2d[vitreous].mean():.3f} "
                f"(thr<{thr:.3f}); choroid fg: {int(fg.sum())} px, "
                f"mean={clean_2d[fg].mean():.3f}" if fg.sum() else "no fg")
    else:
        desc = "vitreous bg: EMPTY (region location failed)"
    return vitreous, fg, desc


# --------------------------------------------------------------------------- #
# Aggregation helper
# --------------------------------------------------------------------------- #
def summarize(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def uncertainty_error_correlation(std_values: List[float],
                                  err_values: List[float]) -> Dict[str, float]:
    """Pearson + Spearman correlation between per-column PREDICTED uncertainty
    (std) and per-column ACTUAL boundary-placement error (|E[z]-z_gt|), pooled
    across the whole scored val set. THE key diagnostic for whether the
    boundary-distribution head's uncertainty is MEANINGFUL: a positive
    correlation means columns the model marks as uncertain really are the ones
    it gets wrong -- i.e. the model 'knows where it's wrong', not just emitting
    a fixed-width distribution everywhere. NaN fields when scipy is unavailable,
    there are too few (<3) pooled samples, or either series is constant
    (undefined correlation)."""
    n = len(std_values)
    nan_result = {"pearson_r": float("nan"), "pearson_p": float("nan"),
                  "spearman_r": float("nan"), "spearman_p": float("nan"), "n": n}
    if not _HAVE_SCIPY or n < 3:
        return nan_result
    s = np.asarray(std_values, dtype=np.float64)
    e = np.asarray(err_values, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(e)
    s, e = s[finite], e[finite]
    if s.size < 3 or s.std() == 0 or e.std() == 0:
        nan_result["n"] = int(s.size)
        return nan_result
    pr, pp = pearsonr(s, e)
    sr, sp = spearmanr(s, e)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": int(s.size)}


# --------------------------------------------------------------------------- #
# Model building (mirrors train.py exactly, flags from the checkpoint args)
# --------------------------------------------------------------------------- #
def build_model(ckpt_args: dict, ablation_phase: int, device) -> DeepOCTUltra:
    model = DeepOCTUltra(
        in_channels=3,
        base_filters=int(ckpt_args.get("base_filters", 32)),
        use_unbiasing=bool(ckpt_args.get("use_unbiasing", False)),
        use_wavelets=bool(ckpt_args.get("use_wavelets", False)),
        use_cross_gating=bool(ckpt_args.get("use_cross_gating", False)),
        ablation_phase=ablation_phase,
        film_enabled=bool(ckpt_args.get("film_enabled", True)),
        # Absent in checkpoints saved before this flag existed -> False, an
        # architecture IDENTICAL to before (no boundary_head submodule built),
        # so old checkpoints keep loading exactly as they did previously.
        boundary_distribution_head=bool(
            ckpt_args.get("boundary_distribution_head", False)),
        # Absent in checkpoints saved before this flag existed -> 0 (the ORIGINAL
        # single-softmax boundary head), so old checkpoints rebuild byte-identically.
        boundary_mixture=int(ckpt_args.get("boundary_mixture", 0)),
    ).to(device)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Read-only eval of a DeepOCT-Ultra checkpoint.")
    p.add_argument("--checkpoint", required=True)
    # PORTED: default changed "oimhs" -> "sdoct" to match scripts/train.py, so an
    # evaluation does not silently build a different loader than the run it scores.
    p.add_argument("--dataset", choices=["sdoct", "oimhs"], default="sdoct",
                   help="Dataset loader for the manifests (must match training).")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ablation_phase", type=int, required=True,
                   help="MUST match how the checkpoint was trained (builds the model "
                        "identically; load_state_dict fails loudly on a mismatch).")
    p.add_argument("--val-manifest", default="val_manifest.csv")
    p.add_argument("--train-manifest", default="train_manifest.csv")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--val_speckle", choices=["none", "deterministic"], default="none",
                   help="Match training. 'deterministic' injects whole-image gamma "
                        "(v=0.08) so PSNR/ENL/CNR measure denoising of a KNOWN noise.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="Max val images to score (0=all). Use a small value for a "
                        "fast CPU demo; the Dice-vs-best_val sanity needs the FULL set.")
    p.add_argument("--out", default=None, help="Write the metrics report as JSON.")
    p.add_argument("--dump_scans", action="store_true",
                   help="Also write <out>_per_scan.csv: one row per B-scan with eye, "
                        "image_id, class and every segmentation metric. REQUIRED for "
                        "any paired / per-patient test (the JSON stores per-class "
                        "means only). Default OFF.")
    # PORTED: the default was the POSIX-only literal "/tmp"; it now resolves via
    # tempfile so the same command line works on Windows.
    p.add_argument("--check-dir", default=tempfile.gettempdir(),
                   help="Where to save eval_check_*.png "
                        "(default: the platform temp directory).")
    p.add_argument("--n-visual", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--boundary_decode", choices=["expectation", "mode", "viterbi"],
                   default="expectation",
                   help="Only used when the checkpoint has --boundary_distribution_head "
                        "active. How to turn the per-column depth DISTRIBUTION into a "
                        "point boundary estimate for the headline Dice/HD95/ASD/MAE "
                        "numbers below (see loss.py). 'expectation' (DEFAULT, byte-"
                        "identical to the pre-existing behavior) = naive per-column "
                        "E[z] -- good Dice but HD95 is outlier-driven by bimodal/jumpy "
                        "columns. 'mode' = per-column argmax (cheap fix for bimodality, "
                        "still no smoothness constraint). 'viterbi' = the DP smooth-"
                        "path decode (loss.viterbi_decode_boundary_pair), which ALSO "
                        "enforces zero boundary crossings by construction. Regardless of "
                        "this choice, ALL THREE modes are still computed and printed in "
                        "the '--- BOUNDARY DECODE COMPARISON ---' table for a direct "
                        "apples-to-apples comparison in one pass.")
    p.add_argument("--boundary_decode_max_jump", type=int, default=5,
                   help="Viterbi's smoothness constraint: |z(x+1)-z(x)| <= this many "
                        "pixels. Only used when --boundary_decode viterbi (or always, "
                        "for the comparison table, when the checkpoint has the boundary-"
                        "distribution head). Default 5px is a STARTING PRIOR (suggested "
                        "range 3-10px per CLAUDE.md-style resolution-dependent constants "
                        "-- re-tune per resolution/dataset). The run also prints the "
                        "GT boundary's ACTUAL column-to-column jump distribution "
                        "(median/p95/p99/max) from the SAME val set being scored, so you "
                        "can set this from real data instead of a guess.")
    return p.parse_args()


def to_uint8(img2d: np.ndarray) -> np.ndarray:
    return np.clip(img2d * 255.0, 0, 255).astype(np.uint8)


def save_panel(path, noisy, denoised, gt_mask, pred_mask):
    if Image is None:
        return
    g = to_uint8(noisy)
    d = to_uint8(denoised) if denoised is not None else np.zeros_like(g)
    base = np.stack([d] * 3, axis=2).astype(np.float32)
    if _HAVE_SCIPY:
        gtb = gt_mask ^ binary_erosion(gt_mask)
        prb = pred_mask ^ binary_erosion(pred_mask)
    else:
        gtb, prb = gt_mask, pred_mask
    gt_panel = np.stack([g] * 3, axis=2).astype(np.float32); gt_panel[gtb] = [0, 255, 0]
    pr_panel = base.copy(); pr_panel[prb] = [255, 0, 0]
    strip = np.concatenate([
        np.stack([g] * 3, axis=2).astype(np.float32),   # noisy input
        base,                                            # denoised
        gt_panel,                                        # GT boundary (green) on noisy
        pr_panel,                                        # pred boundary (red) on denoised
    ], axis=1)
    Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8)).save(path)


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu"))

    # ---- Load checkpoint + build model exactly as train.py did ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_phase = ckpt_args.get("ablation_phase")
    if ckpt_phase is not None and int(ckpt_phase) != args.ablation_phase:
        print(f"[WARN] --ablation_phase {args.ablation_phase} != checkpoint's "
              f"ablation_phase {ckpt_phase}; using the checkpoint's phase for the build.")
    phase = int(ckpt_phase) if ckpt_phase is not None else args.ablation_phase
    model = build_model(ckpt_args, phase, device)
    model.load_state_dict(ckpt["model_state_dict"])  # strict: fails loudly on mismatch
    model.eval()

    best_metric = ckpt_args.get("best_metric", "dice")
    best_val = ckpt.get("best_val")
    seg_only = phase == 1

    # ---- Build the VAL dataset the SAME way train.py does ----
    val_manifest = os.path.join(args.data_dir, args.val_manifest)
    if not os.path.isfile(val_manifest):
        raise FileNotFoundError(f"val manifest not found: {val_manifest}")
    val_ds = train.build_real_dataset(
        val_manifest, args.image_size, augment=False, ablation_phase=phase,
        deterministic_speckle=(args.val_speckle == "deterministic"),
        dataset_kind=args.dataset)
    class_labels = getattr(val_ds, "class_labels", None)  # SDOCT only
    scan_records: List[dict] = []                          # for --dump_scans
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("=" * 92)
    print("DeepOCT-Ultra EVALUATION (read-only)")
    print("=" * 92)
    print(f"checkpoint : {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"best_metric={best_metric}, best_val={best_val})")
    print(f"dataset    : {args.dataset}  phase={phase}  seg_only={seg_only}  "
          f"val_speckle={args.val_speckle}  image_size={args.image_size}  device={device.type}")
    print(f"val images : {len(val_ds)}  (scoring {'ALL' if args.limit==0 else args.limit})")
    print("=" * 92)

    # metric buckets keyed by class ('all' always; SDOCT adds normal/abnormal)
    seg = defaultdict(lambda: defaultdict(list))   # class -> metric -> [values]
    den = defaultdict(lambda: defaultdict(list))
    sample_descs: List[str] = []
    range_note = {"denoised": [1e9, -1e9], "target": [1e9, -1e9], "noisy": [1e9, -1e9]}
    visual_saved = 0
    os.makedirs(args.check_dir, exist_ok=True)

    # ---- Boundary-DISTRIBUTION head (read-only; auto-detected from the rebuilt
    # model's own flag, which build_model() sets from the checkpoint's saved
    # args). `bd_by_mode[decode_name]` mirrors `seg`'s per-class bucket shape
    # for the RASTERIZED Dice/IoU/HD95/ASD (apples-to-apples with the mask
    # model above), computed for ALL THREE decode modes every batch (cheap
    # enough) so the comparison table below needs only ONE pass over val.
    # `bd_corr_by_mode` accumulates POOLED per-column (predicted std, actual
    # |z_hat-z_gt|) pairs per decode mode -- the KEY DIAGNOSTIC (does
    # uncertainty correlate with error) needs many columns, not per-sample
    # means, to be meaningful; predicted std itself is DECODE-INDEPENDENT
    # (unchanged by which point estimate is read off the distribution), only
    # the paired error changes per mode. `bd_movement` accumulates per-column
    # |viterbi - expectation| (the "does viterbi actually move things"
    # diagnostic). `bd_jump_values` accumulates the GT's raw column-to-column
    # jump magnitudes (both boundaries pooled) for the max_jump justification.
    bd_active = getattr(model, "boundary_distribution_head", False)
    _BD_MODES = ("expectation", "mode", "viterbi")
    bd_by_mode = {m: defaultdict(lambda: defaultdict(list)) for m in _BD_MODES}
    bd_corr_by_mode = {m: {"std_upper": [], "err_upper": [], "std_lower": [], "err_lower": []}
                       for m in _BD_MODES}
    bd_movement = {"upper": [], "lower": []}
    bd_jump_values: List[float] = []

    scored = 0
    with torch.no_grad():
        for inp, target, mask, sigma in loader:
            inp = inp.float().to(device)
            target = target.float().to(device)
            mask = mask.float().to(device)
            out = model(inp)
            final_pred, seg_logits = out[0], out[1]
            probs = torch.sigmoid(seg_logits)

            # Boundary-distribution outputs, batched (see model.py: the head's
            # logits are stashed on the instance, not returned from forward()).
            if bd_active and model.last_boundary_logits is not None:
                bd_logits = model.last_boundary_logits.float()
                bd_log_probs = F.log_softmax(bd_logits, dim=2)          # [B,2,H,W]
                bd_H = mask.shape[2]

                # (1) expectation -- the pre-existing/default decode.
                bd_e_upper, bd_std_upper, bd_e_lower, bd_std_lower = (
                    boundary_distribution_to_depths(bd_logits))
                # (2) mode -- per-column argmax (fixes bimodal-mean-in-valley,
                # still no smoothness constraint).
                bd_m_upper = mode_decode_column_boundary(bd_log_probs[:, 0]).float()
                bd_m_lower = mode_decode_column_boundary(bd_log_probs[:, 1]).float()
                # (3) viterbi -- DP smooth-path decode, ordering enforced by
                # construction (zero crossings guaranteed for THIS mode only).
                bd_v_upper_i, bd_v_lower_i = viterbi_decode_boundary_pair(
                    bd_logits, args.boundary_decode_max_jump)
                bd_v_upper, bd_v_lower = bd_v_upper_i.float(), bd_v_lower_i.float()

                bd_decodes = {
                    "expectation": (bd_e_upper, bd_e_lower),
                    "mode": (bd_m_upper, bd_m_lower),
                    "viterbi": (bd_v_upper, bd_v_lower),
                }
                bd_rasters = {name: rasterize_boundary_mask(u, l, bd_H)
                             for name, (u, l) in bd_decodes.items()}
                bd_crossings = {name: count_boundary_crossings(u, l)
                                for name, (u, l) in bd_decodes.items()}
                bd_moved_upper = (bd_v_upper - bd_e_upper).abs()          # [B,W]
                bd_moved_lower = (bd_v_lower - bd_e_lower).abs()

                bd_z_upper, bd_z_lower, bd_valid = derive_column_boundaries(mask)
                d_upper = (bd_z_upper[:, 1:] - bd_z_upper[:, :-1]).abs()
                d_lower = (bd_z_lower[:, 1:] - bd_z_lower[:, :-1]).abs()
                pair_valid = bd_valid[:, 1:] & bd_valid[:, :-1]
                bd_jump_values.extend(d_upper[pair_valid].cpu().tolist())
                bd_jump_values.extend(d_lower[pair_valid].cpu().tolist())

            B = inp.shape[0]
            for b in range(B):
                if args.limit and scored >= args.limit:
                    break
                idx = scored
                cls = "all"
                classes = ["all"]
                if class_labels is not None:
                    c = class_labels[idx] if idx < len(class_labels) else None
                    if c:
                        classes.append(c)

                gt = (mask[b, 0].cpu().numpy() > 0.5)
                pr_prob = probs[b, 0].cpu().numpy()
                pr = pr_prob > 0.5
                d_h = dice_hard(pr, gt); i_h = iou_hard(pr, gt)
                d_s = soft_dice(pr_prob, gt.astype(np.float32))
                asd, hd95 = surface_distances(pr, gt)
                for c in classes:
                    seg[c]["dice_hard"].append(d_h)
                    seg[c]["dice_soft"].append(d_s)
                    seg[c]["iou"].append(i_h)
                    seg[c]["asd_px"].append(asd)
                    seg[c]["hd95_px"].append(hd95)
                # PER-SCAN record WITH identity. summarize() below collapses these
                # to a per-class mean/std, which is why no paired or per-patient test
                # was ever reconstructable from an evaluate.py report.
                scan_records.append({
                    "eye": (val_ds.eye_ids[idx] if hasattr(val_ds, "eye_ids")
                            and idx < len(val_ds.eye_ids) else None),
                    "image_id": (val_ds.image_ids[idx] if hasattr(val_ds, "image_ids")
                                 and idx < len(val_ds.image_ids) else None),
                    "class": (c if (c := (class_labels[idx]
                                          if class_labels is not None
                                          and idx < len(class_labels) else None))
                              else "all"),
                    "dice_hard": d_h, "dice_soft": d_s, "iou": i_h,
                    "asd_px": asd, "hd95_px": hd95,
                })

                # ---- Boundary-DISTRIBUTION head, ALL THREE decode modes: per-
                # boundary MAE/std + rasterized-mask Dice/IoU/HD95/ASD (compared
                # against the SAME `gt` the mask model above is scored against)
                # + the pooled per-column (std, |err|) pairs for the uncertainty-
                # vs-error correlation, PER MODE (std is decode-independent; the
                # paired error is not) + the viterbi-vs-expectation per-column
                # movement diagnostic. ----
                # Per-scan values for the SELECTED (--boundary_decode) mode, stashed
                # for --dump_scans below. NaN defaults so every row has these columns
                # whenever bd_active, even on the (never-expected) edge case where
                # logits are absent for a batch. These are the SAME variables the
                # aggregate bd_by_mode[...] buckets below are built from -- captured
                # at the point they are computed, not recomputed by a second path,
                # so the CSV cannot disagree with the printed aggregate.
                sel_bd_dice = sel_bd_iou = sel_bd_asd = sel_bd_hd95 = float("nan")
                sel_bd_crossings = float("nan")
                sel_mae_upper = sel_mae_lower = float("nan")
                if bd_active and model.last_boundary_logits is not None:
                    v = bd_valid[b].cpu().numpy()
                    if v.any():
                        su = bd_std_upper[b].cpu().numpy()[v]
                        sl = bd_std_lower[b].cpu().numpy()[v]
                        for name, (u, l) in bd_decodes.items():
                            err_u = (u[b] - bd_z_upper[b]).abs().cpu().numpy()[v]
                            err_l = (l[b] - bd_z_lower[b]).abs().cpu().numpy()[v]
                            if name == args.boundary_decode:
                                sel_mae_upper = float(err_u.mean())
                                sel_mae_lower = float(err_l.mean())
                            for c in classes:
                                bd_by_mode[name][c]["mae_upper_px"].append(float(err_u.mean()))
                                bd_by_mode[name][c]["mae_lower_px"].append(float(err_l.mean()))
                                bd_by_mode[name][c]["std_upper"].append(float(su.mean()))
                                bd_by_mode[name][c]["std_lower"].append(float(sl.mean()))
                            bd_corr_by_mode[name]["std_upper"].extend(su.tolist())
                            bd_corr_by_mode[name]["err_upper"].extend(err_u.tolist())
                            bd_corr_by_mode[name]["std_lower"].extend(sl.tolist())
                            bd_corr_by_mode[name]["err_lower"].extend(err_l.tolist())
                        bd_movement["upper"].extend(bd_moved_upper[b].cpu().numpy()[v].tolist())
                        bd_movement["lower"].extend(bd_moved_lower[b].cpu().numpy()[v].tolist())
                    for name in bd_decodes:
                        pr_raster = (bd_rasters[name][b, 0].cpu().numpy() > 0.5)
                        bd_d_h = dice_hard(pr_raster, gt); bd_i_h = iou_hard(pr_raster, gt)
                        bd_asd, bd_hd95 = surface_distances(pr_raster, gt)
                        if name == args.boundary_decode:
                            sel_bd_dice, sel_bd_iou = bd_d_h, bd_i_h
                            sel_bd_asd, sel_bd_hd95 = bd_asd, bd_hd95
                            sel_bd_crossings = float(bd_crossings[name][b].item())
                        for c in classes:
                            bd_by_mode[name][c]["dice_hard"].append(bd_d_h)
                            bd_by_mode[name][c]["iou"].append(bd_i_h)
                            bd_by_mode[name][c]["asd_px"].append(bd_asd)
                            bd_by_mode[name][c]["hd95_px"].append(bd_hd95)
                            bd_by_mode[name][c]["crossings"].append(
                                float(bd_crossings[name][b].item()))
                if bd_active:
                    scan_records[-1].update({
                        "bd_dice_hard": sel_bd_dice, "bd_iou": sel_bd_iou,
                        "bd_hd95_px": sel_bd_hd95, "bd_asd_px": sel_bd_asd,
                        "bd_mae_upper_px": sel_mae_upper,
                        "bd_mae_lower_px": sel_mae_lower,
                        "bd_crossings": sel_bd_crossings,
                    })

                # ---- Denoising metrics (skip for phase 1 / seg-only) ----
                if not seg_only and final_pred is not None:
                    clean = target[b, 0].cpu().numpy()                 # LINEAR clean ref
                    noisy = np.exp(inp[b, 1].cpu().numpy())            # undo log -> LINEAR
                    denoised = final_pred[b, 0].cpu().numpy()          # LINEAR output
                    for k, arr in (("denoised", denoised), ("target", clean), ("noisy", noisy)):
                        range_note[k][0] = min(range_note[k][0], float(arr.min()))
                        range_note[k][1] = max(range_note[k][1], float(arr.max()))
                    vit, fg, desc = locate_regions(clean, gt)
                    # PSNR only where a clean reference is meaningful (OIMHS).
                    if args.dataset == "oimhs":
                        p = psnr(denoised, clean, data_range=1.0)
                        for c in classes:
                            den[c]["psnr_db"].append(p)
                    enl_n, enl_d = enl(noisy, vit), enl(denoised, vit)
                    cnr_n, cnr_d = cnr(noisy, fg, vit), cnr(denoised, fg, vit)
                    for c in classes:
                        den[c]["enl_noisy"].append(enl_n)
                        den[c]["enl_denoised"].append(enl_d)
                        den[c]["enl_delta"].append(enl_d - enl_n)
                        den[c]["cnr_noisy"].append(cnr_n)
                        den[c]["cnr_denoised"].append(cnr_d)
                        den[c]["cnr_delta"].append(cnr_d - cnr_n)
                    if len(sample_descs) < 3:
                        sample_descs.append(f"  [sample idx {idx}] {desc}")
                    if visual_saved < args.n_visual:
                        save_panel(os.path.join(args.check_dir, f"eval_check_{visual_saved+1}.png"),
                                   noisy, denoised, gt, pr)
                        visual_saved += 1
                elif visual_saved < args.n_visual:
                    # seg-only visual: noisy input (=exp center) | (no denoise) | GT | pred
                    noisy = np.exp(inp[b, 1].cpu().numpy())
                    save_panel(os.path.join(args.check_dir, f"eval_check_{visual_saved+1}.png"),
                               noisy, None, gt, pr)
                    visual_saved += 1

                scored += 1
            if args.limit and scored >= args.limit:
                break

    # ---- Report ----
    def fmt(s):
        return f"{s['mean']:.4f} +/- {s['std']:.4f} (n={s['n']})"

    classes_present = ["all"] + [c for c in seg if c != "all"]
    print("\n--- SEGMENTATION (threshold 0.5) ---")
    for c in classes_present:
        print(f"  [{c}]  Dice(hard)={fmt(summarize(seg[c]['dice_hard']))}  "
              f"IoU={fmt(summarize(seg[c]['iou']))}")
        print(f"        Dice(soft, matches train)={fmt(summarize(seg[c]['dice_soft']))}  "
              f"HD95={fmt(summarize(seg[c]['hd95_px']))} px  "
              f"ASD={fmt(summarize(seg[c]['asd_px']))} px")

    if bd_active:
        # ---- GT boundary smoothness (the max_jump justification) ----
        jarr = np.asarray(bd_jump_values, dtype=np.float64)
        jarr = jarr[np.isfinite(jarr)]
        if jarr.size:
            j_med, j_p95, j_p99, j_max = (float(np.median(jarr)), float(np.percentile(jarr, 95)),
                                          float(np.percentile(jarr, 99)), float(jarr.max()))
        else:
            j_med = j_p95 = j_p99 = j_max = float("nan")
        print("\n--- GT boundary smoothness (justifies --boundary_decode_max_jump) ---")
        print(f"  column-to-column |z(x+1)-z(x)|, pooled upper+lower, n={jarr.size}: "
              f"median={j_med:.2f}px  p95={j_p95:.2f}px  p99={j_p99:.2f}px  max={j_max:.2f}px")
        print(f"  used max_jump={args.boundary_decode_max_jump}px for viterbi "
              f"({'covers p99' if np.isfinite(j_p99) and args.boundary_decode_max_jump >= j_p99 else 'BELOW p99 -- consider raising it'})")

        # ---- Per-decode-mode comparison table ----
        print("\n--- BOUNDARY DECODE COMPARISON (expectation vs mode vs viterbi) ---")
        for c in classes_present:
            print(f"  [{c}]")
            for name in _BD_MODES:
                bkt = bd_by_mode[name][c]
                mu = summarize(bkt["mae_upper_px"]); ml = summarize(bkt["mae_lower_px"])
                d = summarize(bkt["dice_hard"]); h = summarize(bkt["hd95_px"])
                a = summarize(bkt["asd_px"]); io = summarize(bkt["iou"])
                cr = summarize(bkt["crossings"])
                marker = " <-- selected (--boundary_decode)" if name == args.boundary_decode else ""
                print(f"    {name:>11}: Dice={fmt(d)}  IoU={fmt(io)}  "
                      f"HD95={fmt(h)}px  ASD={fmt(a)}px{marker}")
                print(f"    {'':>11}  MAE upper={fmt(mu)}px lower={fmt(ml)}px  "
                      f"crossings/sample={fmt(cr)}")
        print("  (compare Dice/HD95/ASD directly against the SEGMENTATION section "
              "above -- same samples, same metrics, mask-head vs. rasterized "
              "boundary-head. 'crossings/sample' = mean # columns where "
              "z_upper>=z_lower; viterbi should read ~0.0 by construction.)")

        # ---- Viterbi-vs-expectation movement diagnostic ----
        mv_u = np.asarray(bd_movement["upper"], dtype=np.float64)
        mv_l = np.asarray(bd_movement["lower"], dtype=np.float64)
        for label, mv in (("upper", mv_u), ("lower", mv_l)):
            if mv.size:
                moved = int((mv > 5.0).sum())
                print(f"  viterbi vs expectation [{label}]: {moved}/{mv.size} columns "
                      f"({100.0*moved/mv.size:.2f}%) moved >5px  "
                      f"(mean move={mv.mean():.3f}px, max move={mv.max():.1f}px)")

        # ---- [KEY DIAGNOSTIC] uncertainty-vs-error correlation, per decode mode ----
        print("\n*** [KEY DIAGNOSTIC] uncertainty-vs-error correlation "
              "(does the model know where it's wrong?) ***")
        print("  predicted std is DECODE-INDEPENDENT; only the paired error changes "
              "with the decode mode -- recomputed below for each.")
        for name in _BD_MODES:
            bc = bd_corr_by_mode[name]
            corr_all = uncertainty_error_correlation(
                bc["std_upper"] + bc["std_lower"], bc["err_upper"] + bc["err_lower"])
            corr_upper = uncertainty_error_correlation(bc["std_upper"], bc["err_upper"])
            corr_lower = uncertainty_error_correlation(bc["std_lower"], bc["err_lower"])
            print(f"  [{name}] pooled (n={corr_all['n']}): "
                  f"Pearson r={corr_all['pearson_r']:+.4f} (p={corr_all['pearson_p']:.2e})  "
                  f"Spearman rho={corr_all['spearman_r']:+.4f} (p={corr_all['spearman_p']:.2e})  "
                  f"| upper r={corr_upper['pearson_r']:+.4f}  lower r={corr_lower['pearson_r']:+.4f}")
        print("  POSITIVE correlation = predicted uncertainty tracks ACTUAL "
              "placement error -> the model's uncertainty is INFORMATIVE, not "
              "degenerate. (NaN above means scipy is unavailable or too few "
              "valid columns were scored.)")

    if seg_only:
        print("\n--- DENOISING / IMAGE QUALITY ---")
        print("  N/A (seg-only phase 1: model has no denoise branch).")
    else:
        print("\n--- DENOISING / IMAGE QUALITY (noisy input vs denoised output) ---")
        for c in classes_present:
            if args.dataset == "oimhs":
                ps = summarize(den[c]["psnr_db"])
                psnr_str = f"PSNR={fmt(ps)} dB (data_range=1.0)"
            else:
                psnr_str = "PSNR=N/A (no clean reference for SDOCT)"
            print(f"  [{c}]  {psnr_str}")
            print(f"        ENL  noisy={fmt(summarize(den[c]['enl_noisy']))}  "
                  f"denoised={fmt(summarize(den[c]['enl_denoised']))}  "
                  f"delta={fmt(summarize(den[c]['enl_delta']))}")
            print(f"        CNR  noisy={fmt(summarize(den[c]['cnr_noisy']))}  "
                  f"denoised={fmt(summarize(den[c]['cnr_denoised']))}  "
                  f"delta={fmt(summarize(den[c]['cnr_delta']))}")

    # ---- KEY CROSS-TASK ONE-LINER ----
    dice_all = summarize(seg["all"]["dice_hard"])["mean"]
    if seg_only:
        print(f"\n[SUMMARY] phase {phase} | Dice={dice_all:.4f} | denoising N/A (seg-only)")
    else:
        enl_d = summarize(den["all"]["enl_delta"])["mean"]
        cnr_d = summarize(den["all"]["cnr_delta"])["mean"]
        pstr = (f" | dPSNR n/a" if args.dataset == "sdoct"
                else f" | PSNR={summarize(den['all']['psnr_db'])['mean']:.2f}dB")
        print(f"\n[SUMMARY] phase {phase} | Dice={dice_all:.4f} | "
              f"dENL={enl_d:+.3f} | dCNR={cnr_d:+.3f}{pstr}")

    # ---- SANITY ----
    print("\n--- SANITY ---")
    train_dice = float(best_val) if (best_metric == "dice" and best_val is not None) else None
    eval_soft = summarize(seg["all"]["dice_soft"])["mean"]
    if train_dice is not None and 0.0 <= train_dice <= 1.0:
        diff = abs(eval_soft - train_dice)
        flag = "OK" if diff <= 0.005 else "MISMATCH >0.005 — eval pipeline differs from training!"
        full = "" if args.limit == 0 else " [SUBSET run: not directly comparable to full-val best_val]"
        print(f"  eval Dice(soft)={eval_soft:.4f} vs training best_val={train_dice:.4f} "
              f"-> |diff|={diff:.4f} [{flag}]{full}")
    elif train_dice is not None:
        print(f"  eval Dice(soft)={eval_soft:.4f}; stored best_val={train_dice:.4f} is "
              f"OUTSIDE [0,1] despite best_metric='dice' -> it is a LOSS value, not a "
              f"Dice (stale / loss-selected checkpoint). Skipping the strict Dice-match "
              f"sanity; re-run against a checkpoint whose best_val is a Dice to use it.")
    else:
        print(f"  eval Dice(soft)={eval_soft:.4f} (checkpoint best_metric={best_metric}; "
              f"no comparable stored dice).")
    if not seg_only:
        print(f"  data ranges (linear): denoised[{range_note['denoised'][0]:.3f},"
              f"{range_note['denoised'][1]:.3f}] target[{range_note['target'][0]:.3f},"
              f"{range_note['target'][1]:.3f}] noisy[{range_note['noisy'][0]:.3f},"
              f"{range_note['noisy'][1]:.3f}]; PSNR data_range=1.0")
        print("  ENL/CNR region locations (3 samples):")
        for d in sample_descs:
            print(d)
    print(f"  visual panels saved: {visual_saved} -> {args.check_dir}/eval_check_*.png "
          f"(noisy | denoised | GT boundary | pred boundary)")

    # ---- JSON out ----
    if args.out:
        report = {
            "checkpoint": args.checkpoint, "dataset": args.dataset, "phase": phase,
            "val_speckle": args.val_speckle, "image_size": args.image_size,
            "n_scored": scored, "best_val_train": best_val, "best_metric": best_metric,
            "segmentation": {c: {m: summarize(v) for m, v in seg[c].items()}
                             for c in classes_present},
            "denoising": ({} if seg_only else
                          {c: {m: summarize(v) for m, v in den[c].items()}
                           for c in classes_present}),
        }
        if bd_active:
            jarr = np.asarray(bd_jump_values, dtype=np.float64)
            jarr = jarr[np.isfinite(jarr)]
            gt_jump_stats = ({"median": float(np.median(jarr)),
                             "p95": float(np.percentile(jarr, 95)),
                             "p99": float(np.percentile(jarr, 99)),
                             "max": float(jarr.max()), "n": int(jarr.size)}
                            if jarr.size else
                            {"median": None, "p95": None, "p99": None, "max": None, "n": 0})
            report["boundary_distribution"] = {
                "boundary_decode_max_jump": args.boundary_decode_max_jump,
                "selected_decode_mode": args.boundary_decode,
                "gt_boundary_jump_stats_px": gt_jump_stats,
                "by_decode_mode": {
                    name: {
                        "segmentation_rasterized": {
                            c: {m: summarize(v) for m, v in bd_by_mode[name][c].items()}
                            for c in classes_present},
                        "uncertainty_error_correlation": {
                            "pooled": uncertainty_error_correlation(
                                bd_corr_by_mode[name]["std_upper"] + bd_corr_by_mode[name]["std_lower"],
                                bd_corr_by_mode[name]["err_upper"] + bd_corr_by_mode[name]["err_lower"]),
                            "upper": uncertainty_error_correlation(
                                bd_corr_by_mode[name]["std_upper"], bd_corr_by_mode[name]["err_upper"]),
                            "lower": uncertainty_error_correlation(
                                bd_corr_by_mode[name]["std_lower"], bd_corr_by_mode[name]["err_lower"]),
                        },
                    }
                    for name in _BD_MODES
                },
                "viterbi_vs_expectation_movement_px": {
                    "upper": summarize(bd_movement["upper"]),
                    "lower": summarize(bd_movement["lower"]),
                },
            }
        from deepoct import provenance as _prov
        _prov.stamp(report, checkpoint=args.checkpoint,
                    protocol=f"evaluate/{args.dataset}/phase{phase}",
                    patients=sorted({str(r["eye"]).split("_")[0]
                                     for r in scan_records if r.get("eye")}) or None)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  wrote {args.out}")
        if args.dump_scans and scan_records:
            import pandas as _pd
            _prov.dump_frame(_pd.DataFrame(scan_records),
                             os.path.splitext(args.out)[0] + "_per_scan.csv",
                             label="scans", checkpoint=args.checkpoint,
                             protocol=f"evaluate/{args.dataset}/phase{phase}")


if __name__ == "__main__":
    main()
