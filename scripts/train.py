"""Production training entry point for DeepOCT-Ultra — Phase 3 (server/cloud).

This is the long-running, resumable training driver for full-scale 512x512 runs
on a CUDA server. It is fully compatible with the existing ``model.py`` (the
dual-decoder multi-task U-Net, all flags) and ``loss.py`` (the composite
ZNCC + GAT-variance + FFL + Seg + bias engine).

Production features:
  * **Argparse** — batch size, epochs, learning rate, base filters, the three
    architecture flags (unbiasing / wavelets / cross-gating), and data /
    checkpoint directories.
  * **Checkpointing** — at the end of every epoch ``checkpoint.pth`` (model +
    optimizer + AMP scaler + epoch) is written atomically. On startup the
    checkpoint dir is probed and, if found, training resumes from the next epoch.
  * **Device-agnostic** — ``cuda`` when available, else ``cpu``; the model, the
    loss module and every batch are moved to that device automatically.
  * **AMP** — the forward/loss pass is wrapped in ``torch.amp.autocast`` and a
    ``torch.amp.GradScaler`` guards the backward pass against fp16 underflow.
    Both are no-ops on CPU so the same code path runs locally and on the server.
  * **Logging** — all progress goes through the ``logging`` module to both the
    console and ``<checkpoint_dir>/training.log``, so detached server sessions
    can be monitored after the fact without a live terminal.

Data:
  * Real data: ``--data_dir`` holds the Phase 1 manifest CSVs (default
    ``train.csv`` / ``val.csv``, overridable). Each row carries ``image_path``,
    ``Eye ID`` and ``Image ID``; the mask is embedded in the right half of each
    OIMHS scan, so no separate mask paths are needed.
  * ``--mock`` (or an unresolved data dir) uses a tiny deterministic synthetic
    dataset so the full pipeline — including checkpoint/resume and AMP — can be
    smoke-tested locally without the dataset present.
"""

from __future__ import annotations

import argparse
import sys
import csv
import logging
import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
from deepoct.losses import (DeepOCTLoss, BoundaryDistributionLoss,
                   boundary_distribution_to_depths, rasterize_boundary_mask)
from deepoct.noise_estimation import estimate_sigma_normalized
from deepoct.sam import SAM


logger = logging.getLogger("deepoct.train")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer for REPRODUCIBLE augmentation.

    Each worker is a separate process; PyTorch seeds each worker's *torch* RNG
    deterministically (from the main generator, which --seed sets), but numpy and
    the stdlib ``random`` are NOT seeded automatically. The dataset's augmentation
    (flips, whole-image speckle, shields) uses GLOBAL ``np.random``, so without
    this each worker would draw time-based noise and runs would not reproduce. We
    derive the worker seed from ``torch.initial_seed()`` (the per-worker torch seed,
    which traces back to --seed) so augmentation is reproducible across runs.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


CHECKPOINT_NAME = "checkpoint.pth"
BEST_MODEL_NAME = "best_model.pth"
LOG_NAME = "training.log"
METRICS_CSV_NAME = "training_metrics.csv"
# ONE fixed, positional CSV schema = the UNION of every metric across phases
# 1-4. The header is written ONCE and never mutated mid-file (CSV is positional;
# growing headers on a phase change would corrupt the table). Metrics that a
# given phase does not compute are written as an empty cell — e.g. a Phase-1 row
# leaves every Phase-2+ column blank, a Phase-2 row leaves Val_Bias_MSE blank.
# NOTE: Phase-4 adds two diagnostics (predicted_bias_mean / injected_sigma_mean).
# They are APPENDED at the END of the union header — existing column order/positions
# are never mutated, so phases 1-3 simply leave the two trailing cells blank.
# The Phase-4 bias target is now the dataloader's INJECTED speckle sigma = sqrt(v)
# (retired the dead Yang-Tai estimate), hence the column is Val_Injected_Sigma_Mean.
_CSV_HEADER = ("Epoch", "Phase", "Train_Total_Loss", "Val_Total_Loss",
               "Val_BCE", "Val_Dice_Loss", "Val_Dice_Score",
               "Val_ZNCC", "Val_GAT", "Val_FFL", "Val_DC_Offset", "Val_Bias_MSE",
               "Val_Predicted_Bias_Mean", "Val_Injected_Sigma_Mean",
               # Dual val-Dice columns (filled ONLY when --val_speckle deterministic;
               # blank otherwise). Appended at the END so existing column positions
               # are never mutated. Val_Dice_Clean == Val_Dice_Score (clean pass);
               # Val_Dice_Noisy is the deterministic-speckle val pass (FiLM active).
               "Val_Dice_Clean", "Val_Dice_Noisy",
               # TRAIN segmentation metrics (appended at the END so existing column
               # positions are never mutated) so train-vs-val Dice learning curves
               # can be built straight from the CSV. Populated every epoch, all phases.
               "Train_BCE", "Train_Dice_Loss", "Train_Dice_Score",
               # Kervadec boundary-loss values (appended at the END; blank unless
               # --lambda_boundary > 0). RAW (unweighted) boundary term per pass.
               "Train_Boundary", "Val_Boundary",
               # Boundary-DISTRIBUTION head diagnostics (appended at the END;
               # blank unless --boundary_distribution_head). NLL is logged for
               # BOTH train+val (the training signal); MAE/std/rasterized-Dice
               # are VAL-only (the scientific readout -- does E[z] track the
               # boundary, and does uncertainty separate the easy/hard boundary).
               "Train_BoundaryDist_NLL", "Val_BoundaryDist_NLL",
               "Val_BoundaryDist_MAE_Upper_px", "Val_BoundaryDist_MAE_Lower_px",
               "Val_BoundaryDist_Std_Upper", "Val_BoundaryDist_Std_Lower",
               "Val_BoundaryDist_Dice")
# Maps each Phase-2+ CSV column to the val-metric key it reads (blank if absent).
# Val_Bias_MSE / the two _Mean columns are FILLED only for Phase 4.
_CSV_OPTIONAL = {
    "Val_ZNCC": "zncc", "Val_GAT": "gat", "Val_FFL": "ffl",
    "Val_DC_Offset": "dc_offset", "Val_Bias_MSE": "bias_mse",
    "Val_Predicted_Bias_Mean": "predicted_bias_mean",
    "Val_Injected_Sigma_Mean": "injected_sigma_mean",
}


def phase_artifact(name: str, phase: int) -> str:
    """Suffix an artifact filename with its ablation phase.

    Each phase trains a DIFFERENT architecture (Phase 1 has no DenoiseDecoder,
    later phases add unbiasing, etc.), so their checkpoints are mutually
    incompatible. Giving every phase its own ``*_phaseN`` files keeps each
    ablation run independently resumable and prevents one phase from clobbering
    — or crashing on — another phase's checkpoint.
    """
    stem, ext = os.path.splitext(name)
    return f"{stem}_phase{phase}{ext}"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(checkpoint_dir: str) -> None:
    """Send logs to both the console and ``<checkpoint_dir>/training.log``.

    File handler uses append mode so a resumed run keeps the full history.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, LOG_NAME)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # idempotent if main() is ever called twice
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logger.propagate = False
    logger.info("Logging to %s", log_path)


def append_epoch_metrics(checkpoint_dir: str, phase: int, epoch: int,
                         train_metrics: dict, val_metrics: dict,
                         val_dice_clean: float = None,
                         val_dice_noisy: float = None) -> None:
    """Append one epoch's metrics to the single shared ``training_metrics.csv``.

    Uses the ONE fixed union schema ``_CSV_HEADER`` (header written once, then
    appended to on resume / across phases). Metrics not active in the current
    phase are written as empty cells — Phase-1 rows leave all Phase-2+ columns
    blank; Phase-2 rows leave Val_Bias_MSE blank — so the header is never
    mutated and the table stays positionally consistent.

    ``val_dice_clean`` / ``val_dice_noisy`` populate the two trailing dual-metric
    columns and are filled ONLY under ``--val_speckle deterministic`` (both None
    otherwise -> blank cells, so the default path is unchanged except for two
    trailing empties).
    """
    csv_path = os.path.join(checkpoint_dir, METRICS_CSV_NAME)
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0

    def cell(metrics: dict, key: str) -> str:
        v = metrics.get(key)
        return "" if v is None else f"{v:.6f}"

    def scalar_cell(v) -> str:
        return "" if v is None else f"{v:.6f}"

    row = [epoch, phase, cell(train_metrics, "total_loss"),
           cell(val_metrics, "total_loss"), cell(val_metrics, "bce"),
           cell(val_metrics, "dice"), cell(val_metrics, "dice_score")]
    # Phase-2+ optional columns, in header order; blank when the phase omits them.
    for col in ("Val_ZNCC", "Val_GAT", "Val_FFL", "Val_DC_Offset", "Val_Bias_MSE",
                "Val_Predicted_Bias_Mean", "Val_Injected_Sigma_Mean"):
        row.append(cell(val_metrics, _CSV_OPTIONAL[col]))
    # Trailing dual-metric columns (deterministic val-speckle only).
    row.append(scalar_cell(val_dice_clean))
    row.append(scalar_cell(val_dice_noisy))
    # Trailing TRAIN segmentation columns (every epoch, all phases) so train-vs-val
    # learning curves can be built from the CSV alone.
    row.append(cell(train_metrics, "bce"))
    row.append(cell(train_metrics, "dice"))
    row.append(cell(train_metrics, "dice_score"))
    # Trailing Kervadec boundary columns (blank when the term is off -> absent key).
    row.append(cell(train_metrics, "boundary"))
    row.append(cell(val_metrics, "boundary"))
    # Trailing boundary-DISTRIBUTION head columns (blank unless
    # --boundary_distribution_head -> the keys are simply absent from the dicts).
    row.append(cell(train_metrics, "boundary_dist_nll"))
    row.append(cell(val_metrics, "boundary_dist_nll"))
    row.append(cell(val_metrics, "boundary_dist_mae_upper"))
    row.append(cell(val_metrics, "boundary_dist_mae_lower"))
    row.append(cell(val_metrics, "boundary_dist_std_upper"))
    row.append(cell(val_metrics, "boundary_dist_std_lower"))
    row.append(cell(val_metrics, "boundary_dist_dice"))

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_CSV_HEADER)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Mock dataset: fixed deterministic triplets, used to sanity-check the loop.
# Each sample is a constant (input, target, mask) so the model can overfit and
# the training loss is expected to fall epoch-over-epoch.
# ---------------------------------------------------------------------------
class MockOCTDataset(Dataset):
    # Constant injected-speckle level for the mock set: sqrt(0.05), matching the
    # real dataset's fixed-variance default. Returned as the 4th tuple element so
    # the mock matches the real OIMHSDataset contract (Phase 4 reads it as the bias
    # supervision target; phases 1-3 ignore it). The mock injects no actual noise,
    # so this is a fixed nominal level — enough to keep the Phase-4 loop wired.
    _MOCK_NOISE_SIGMA = math.sqrt(0.05)

    def __init__(self, num_samples: int = 3, image_size: int = 64, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        for _ in range(num_samples):
            inp = torch.randn(3, image_size, image_size, generator=g)
            target = torch.rand(1, image_size, image_size, generator=g)
            mask = (torch.rand(1, image_size, image_size, generator=g) > 0.5).float()
            self.inputs.append(inp)
            self.targets.append(target)
            self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Clone so downstream ops never mutate the cached tensors.
        return (self.inputs[index].clone(),
                self.targets[index].clone(),
                self.masks[index].clone(),
                torch.tensor([self._MOCK_NOISE_SIGMA], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Real dataset builder (cloud). Each OIMHS scan file already contains both the
# OCT image (left half) and the RGB choroid mask (right half), so the manifest's
# image_path is all we need — OIMHSDataset splits the file internally.
# ---------------------------------------------------------------------------
def build_real_dataset(manifest_path: str, image_size: int, augment: bool,
                       ablation_phase: int = 4,
                       deterministic_speckle: bool = False,
                       dataset_kind: str = "oimhs",
                       isotropic_resize: bool = False):
    import pandas as pd  # local import: only needed for the real path

    df = pd.read_csv(manifest_path)

    if dataset_kind == "sdoct":
        # SECOND dataset: separate image+overlay TIFFs, filled-contour mask. The
        # SDOCT manifest adds overlay_path + class columns to the OIMHS schema.
        # SDOCTDataset SUBCLASSES OIMHSDataset, so the augment/speckle/ablation and
        # the (input, target, mask, noise_sigma) 4-tuple contract are identical.
        from deepoct.datasets.sdoct_dataset import SDOCTDataset

        # AUTO-DETECT the clean .roi mask source: build_sdoct_manifests.py
        # --mask-source roi emits a 'mask_path' column of pre-rasterised {0,255}
        # PNGs. When present, pass them through so SDOCTDataset reads them directly
        # instead of filling the yellow overlay. When ABSENT (overlay default), this
        # is None and the construction is byte-identical to before.
        mask_paths = (df["mask_path"].tolist()
                      if "mask_path" in df.columns else None)

        # ISOTROPIC RESIZE (--isotropic_resize, default OFF): when set,
        # SDOCTDataset derives target_size itself from isotropic_target_size()
        # and REJECTS an explicit target_size -- so we must not pass one here.
        # image_size (--image-size) is IGNORED in that case; the geometry is
        # fully determined by the native SDOCT resolution + the fixed axial
        # (H) output, not by the CLI's square size.
        dataset_kwargs = dict(
            image_paths=df["image_path"].tolist(),
            overlay_paths=df["overlay_path"].tolist(),
            mask_paths=mask_paths,
            eye_ids=df["Eye ID"].tolist(),
            image_ids=df["Image ID"].tolist(),
            class_labels=df["class"].tolist(),
            augment=augment,
            ablation_phase=ablation_phase,
            deterministic_speckle=deterministic_speckle,
            isotropic_resize=isotropic_resize,
        )
        if not isotropic_resize:
            dataset_kwargs["target_size"] = (image_size, image_size)
        return SDOCTDataset(**dataset_kwargs)

    # DEFAULT: original OIMHS path — UNCHANGED.
    if isotropic_resize:
        raise ValueError(
            "--isotropic_resize is only implemented for --dataset sdoct (the "
            "flag exists to fix SDOCT's non-square native TIFF resolution; "
            "OIMHS's native geometry is a different, unaudited aspect ratio). "
            "Re-run without --isotropic_resize, or with --dataset sdoct.")
    from deepoct.datasets.oimhs_dataset import OIMHSDataset

    return OIMHSDataset(
        df["image_path"].tolist(),
        eye_ids=df["Eye ID"].tolist(),
        image_ids=df["Image ID"].tolist(),
        target_size=(image_size, image_size),
        augment=augment,
        # Pass the run's ablation phase straight through so ONLY Phase 4 randomizes
        # the per-sample speckle variance; phases 1-3 keep the fixed _SPECKLE_VARIANCE.
        ablation_phase=ablation_phase,
        # OPT-IN deterministic val speckle (only meaningful on the val build, where
        # augment=False); default False keeps the existing clean-val behavior.
        deterministic_speckle=deterministic_speckle,
    )


def require_manifests(data_dir: str, train_name: str,
                      val_name: str) -> Tuple[str, str]:
    """Resolve and STRICTLY validate the manifest CSVs inside ``data_dir``.

    When the user explicitly points the run at a dataset, a missing manifest is a
    deployment error, not a reason to silently fall back to synthetic data — so
    we fail fast with the exact paths we expected to find.
    """
    train_path = os.path.join(data_dir, train_name)
    val_path = os.path.join(data_dir, val_name)
    missing = [p for p in (train_path, val_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "--data_dir was provided but the required manifest file(s) are "
            "missing:\n  " + "\n  ".join(missing) +
            f"\nExpected '{train_name}' and '{val_name}' inside {data_dir!r}.")
    return train_path, val_path


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def is_finite_scalar(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _str2bool(value: str) -> bool:
    """Parse a truthy/falsy CLI string into a bool.

    Used by ``--film_enabled`` so the flag can be set BOTH ways explicitly
    (``--film_enabled True`` / ``--film_enabled False``), supporting the Phase-5
    two-stage warm-start schedule (Stage 1: False; Stage 2: True).
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if value.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {value!r}")


def grads_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def weights_finite(model: nn.Module) -> bool:
    """True iff EVERY model parameter is finite. Guards checkpoint writes so a
    NaN/Inf-corrupted model can never be persisted — that corrupted checkpoint is
    exactly what let a later run resume from NaN state."""
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: "torch.amp.GradScaler", args: argparse.Namespace,
                    best_val: float, criterion: nn.Module = None) -> None:
    """Write the full training state atomically (tmp file + os.replace).

    The atomic replace guarantees that a crash mid-write can never leave a
    truncated ``checkpoint.pth`` behind — the previous good checkpoint survives.

    ``criterion`` is persisted ONLY when it carries learned loss-weighting
    log-variances (Kendall & Gal); otherwise the payload is unchanged (no new key),
    so fixed-lambda checkpoints stay byte-identical to before.
    """
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "args": vars(args),
    }
    if criterion is not None and getattr(criterion, "learned_loss_weighting", False):
        payload["criterion_state_dict"] = criterion.state_dict()
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: "torch.amp.GradScaler",
                    device: torch.device,
                    criterion: nn.Module = None,
                    cur_isotropic_resize: bool = False) -> Tuple[int, float]:
    """Restore model/optimizer/scaler state. Returns (start_epoch, best_val).

    Defensively verifies the checkpoint was trained under the SAME ablation
    phase as the current model. Phases have incompatible architectures, so
    loading across them would raise an opaque state_dict error; we fail fast
    with a clear message instead.

    Also verifies the checkpoint was trained under the SAME --isotropic_resize
    setting. The model itself is fully convolutional (no shape mismatch either
    way), so this would NOT crash on load -- but resuming a 512x512 checkpoint
    into a 512x768 isotropic run (or vice versa) silently mixes two different
    physical samplings inside one training curve, which is a correctness bug,
    not a compatibility one. Fail fast rather than let it happen quietly.
    """
    ckpt = torch.load(path, map_location=device)
    ckpt_phase = ckpt.get("args", {}).get("ablation_phase")
    cur_phase = getattr(model, "ablation_phase", None)
    if ckpt_phase is not None and cur_phase is not None and ckpt_phase != cur_phase:
        raise RuntimeError(
            f"Checkpoint {path!r} was trained under ablation_phase={ckpt_phase}, "
            f"but this run is ablation_phase={cur_phase}. These architectures are "
            f"incompatible. Use a clean --checkpoint_dir or --no-resume.")
    ckpt_isotropic = bool(ckpt.get("args", {}).get("isotropic_resize", False))
    if ckpt_isotropic != bool(cur_isotropic_resize):
        raise RuntimeError(
            f"Checkpoint {path!r} was trained with isotropic_resize="
            f"{ckpt_isotropic}, but this run has isotropic_resize="
            f"{bool(cur_isotropic_resize)}. These use different input "
            f"geometries (512x512 vs 512x768) and are not resumable into each "
            f"other. Use a clean --checkpoint_dir or --no-resume.")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Restore learned loss-weighting log-variances (Kendall & Gal) when both this
    # run and the checkpoint carry them. strict=False so a criterion whose only
    # learnable state is the log_vars tolerates any parameter-free submodule keys.
    # A missing key (fixed-lambda checkpoint resumed into a learned run) just keeps
    # the fresh log-variances — never crashes the resume.
    if criterion is not None and getattr(criterion, "learned_loss_weighting", False):
        crit_sd = ckpt.get("criterion_state_dict")
        if crit_sd:
            criterion.load_state_dict(crit_sd, strict=False)
            logger.info("Restored learned loss-weighting log-variances from %s.", path)
        else:
            logger.info("Checkpoint %s carries no learned loss-weighting state; "
                        "keeping freshly-initialized log-variances.", path)
    # Scaler (AMP) state is OPTIONAL and must NEVER crash a resume. A run with AMP
    # OFF saves an EMPTY GradScaler state ({}); feeding that to an ENABLED scaler
    # (or vice-versa, when --amp is toggled between runs) raises
    # "The source state dict is empty ... saved from a disabled instance of
    # GradScaler". So load the scaler ONLY when the checkpoint carries NON-EMPTY
    # scaler state AND this run has AMP enabled; otherwise skip and continue with a
    # freshly-initialized scaler. Model/optimizer/epoch/best_val above are
    # unaffected — only the scaler load is guarded. We WARN only on a genuine AMP
    # mismatch (the toggled-between-runs case); the benign default (AMP off both
    # times -> nothing to restore) stays quiet at INFO.
    saved_scaler = ckpt.get("scaler_state_dict")
    if saved_scaler and scaler.is_enabled():
        try:
            scaler.load_state_dict(saved_scaler)
        except Exception as exc:
            logger.warning("Could not load checkpoint scaler state (%s); "
                           "reinitializing scaler and continuing.", exc)
    elif bool(saved_scaler) != scaler.is_enabled():
        # Genuine AMP mismatch: --amp was toggled between the saved run and this one
        # (saved non-empty but AMP now off, or saved empty but AMP now on).
        logger.warning("Checkpoint scaler state mismatched with current AMP setting "
                       "(saved %s, current AMP %s); reinitializing scaler.",
                       "present" if saved_scaler else "absent",
                       "on" if scaler.is_enabled() else "off")
    else:
        # Benign default: AMP off both times -> empty state, nothing to restore.
        logger.info("No scaler state to restore (AMP off); using a fresh scaler.")
    start_epoch = int(ckpt["epoch"]) + 1
    best_val = float(ckpt.get("best_val", math.inf))
    logger.info("Resumed from %s — last completed epoch %d, best_val=%.6f",
                path, ckpt["epoch"], best_val)
    return start_epoch, best_val


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer, scaler,
              amp_enabled: bool, grad_clip: float,
              nan_patience: int = 25,
              nan_fraction_threshold: float = 0.2,
              sam=None,
              boundary_criterion=None,
              boundary_dist_mode: str = "joint",
              lambda_boundary_dist: float = 1.0) -> Tuple[dict, bool]:
    """Run one epoch with the composite DeepOCTLoss engine.

    If ``optimizer`` is given -> train (AMP autocast + GradScaler + grad clip);
    else -> eval (autocast forward only, no grad). Returns (avg_components,
    diverged).

    ``diverged`` distinguishes BENIGN transient skips from REAL divergence. A few
    non-finite batches per epoch are NORMAL under AMP: their optimizer step is
    SKIPPED (weights untouched), so they do no harm. ``diverged`` is True ONLY for
    a TRAINING epoch that shows REAL corruption, defined as ANY of:
      (a) MORE than ``nan_fraction_threshold`` of the epoch's batches were
          non-finite (a large fraction = the run is actually diverging, not a
          couple of transient underflows), OR
      (b) the model WEIGHTS themselves are non-finite after the epoch (real,
          persisted corruption — the one thing skipping steps is meant to prevent),
      (c) [subsumed by (b)] a non-finite epoch-aggregate AND non-finite weights;
          since (b) already fires on non-finite weights alone, (c) adds nothing.
    A sustained run of ``nan_patience`` CONSECUTIVE non-finite-loss batches also
    trips divergence mid-epoch (fast-exit for the fully-corrupted case).

    A few SCATTERED non-finite batches with FINITE weights are logged as benign
    and training CONTINUES.

    Robustness:
      * Optimizer steps on NON-FINITE gradients are SKIPPED, so NaN/Inf can never
        land in the weights — this holds WITH and WITHOUT AMP (a disabled
        GradScaler does not skip on its own; the old code let clip_grad_norm_ turn
        a NaN grad into a NaN update, permanently corrupting the weights).
      * The epoch-mean is computed over FINITE batches ONLY, so a handful of
        skipped batches never poison the reported average to NaN.
      * Non-finite batches are counted and logged ONCE per epoch (no per-batch,
        per-term warning flood).

    ``boundary_criterion`` (default None -> byte-identical to before): when a
    ``loss.BoundaryDistributionLoss`` instance is supplied, its output is combined
    into ``loss``/``log`` every batch (see ``_combine_boundary_dist`` below) using
    ``boundary_dist_mode`` ('joint' ADDS it to the existing total; 'replace'
    substitutes it for the mask BCE+Dice[+Kervadec] contribution). Requires
    ``model.boundary_distribution_head`` to be True (so ``model.last_boundary_logits``
    is populated by the forward pass just completed).
    """
    is_train = optimizer is not None
    model.train(is_train)

    sums: dict = {}
    n_batches = 0
    finite_loss_batches = 0  # batches whose total loss was finite (mean denominator)
    nan_loss_batches = 0     # batches whose total loss was non-finite
    nan_grad_batches = 0     # batches whose grads were non-finite (step skipped)
    nan_streak = 0           # current run of consecutive non-finite-loss batches
    diverged = False
    streak_diverged = False  # tripped by a sustained CONSECUTIVE non-finite run

    # NOISE-ADAPTIVE CONDITIONING diagnostics (the scientific readout): accumulate
    # the per-image predicted noise and applied gate scale, and split each batch by
    # its mean noise into high/low bins so we can report whether the model applies a
    # LARGER coupling scale to noisier images. All no-ops when the flag is off.
    cond_active = getattr(model, "noise_adaptive_conditioning", False)
    # 'intrinsic' mode needs a REAL per-image noise scalar from the CLEAN target,
    # computed via the validated Yang-Tai estimator and fed to the model each batch.
    cond_intrinsic = cond_active and getattr(model, "noise_signal", "predicted") == "intrinsic"
    cond_noise_sum = cond_scale_sum = 0.0
    cond_n = 0
    cond_hi_scale_sum = cond_lo_scale_sum = 0.0
    cond_hi_noise_sum = cond_lo_noise_sum = 0.0
    cond_hi_n = cond_lo_n = 0

    def _combine_boundary_dist(loss_t: torch.Tensor, log_t: dict,
                               mask_t: torch.Tensor):
        """Fold BoundaryDistributionLoss into (loss_t, log_t); no-op when
        ``boundary_criterion`` is None (the flag-off default -> returns the
        inputs UNCHANGED, so every existing call site is byte-identical).

        'joint' (default): ``new_loss = loss_t + lambda_boundary_dist * b_total``
        -- purely ADDITIVE, so it is safe regardless of what ``criterion`` did
        internally (fixed lambdas OR learned/normalized Kendall-Gal weighting).

        'replace': the boundary-distribution loss SUBSTITUTES for the mask
        BCE+Dice[+Kervadec] contribution already baked into ``loss_t`` by
        ``criterion``. That contribution is reconstructed from ``log_t``'s
        RAW (unnormalized) ``seg_total``/``boundary`` entries as
        ``lambda_seg*seg_total [+ boundary_weight*boundary]`` and subtracted
        back out. This exact reconstruction is ONLY valid when ``criterion``
        added that SAME raw-weighted quantity to the total -- true for the
        fixed-lambda path always, and for phase 1 (seg-only) even with learned
        weighting (which is FORCE-DISABLED for seg-only in DeepOCTLoss). It is
        NOT valid for phases 2-5 WITH learned+normalized weighting (which adds
        ``lambda_seg*seg_total/ema_scale`` instead) -- main() fails fast on
        that combination before training starts, so it can never reach here.
        """
        if boundary_criterion is None:
            return loss_t, log_t
        with torch.autocast(device_type=device.type, enabled=False):
            b_out = boundary_criterion(model.last_boundary_logits.float(),
                                       mask_t.float())
        if boundary_dist_mode == "replace":
            seg_contribution = criterion.lambda_seg * log_t["seg_total"]
            if "boundary" in log_t:
                seg_contribution = (seg_contribution
                                    + criterion.boundary_weight * log_t["boundary"])
            new_loss = (loss_t - seg_contribution
                       + lambda_boundary_dist * b_out["total_loss"])
        else:  # 'joint' (default, recommended)
            new_loss = loss_t + lambda_boundary_dist * b_out["total_loss"]
        new_log = dict(log_t)
        new_log["total_loss"] = new_loss
        new_log["boundary_dist_total"] = b_out["total_loss"]
        new_log["boundary_dist_nll"] = b_out["nll"]
        new_log["boundary_dist_mae_upper"] = b_out["mae_upper"]
        new_log["boundary_dist_mae_lower"] = b_out["mae_lower"]
        new_log["boundary_dist_std_upper"] = b_out["mean_std_upper"]
        new_log["boundary_dist_std_lower"] = b_out["mean_std_lower"]
        # Raw Spatial Variance Penalty -- logged EVERY step regardless of
        # --lambda_spread (even at 0.0/OFF) so interval tightness is visible
        # alongside the NLL, per the Conformal-Aware Training tracking ask.
        new_log["boundary_dist_variance_penalty"] = b_out["variance_penalty"]
        with torch.no_grad():
            e_upper, _, e_lower, _ = boundary_distribution_to_depths(
                model.last_boundary_logits.float())
            raster = rasterize_boundary_mask(e_upper, e_lower, mask_t.shape[2])
            m = mask_t.float()
            inter = (raster * m).sum(dim=(1, 2, 3))
            denom = raster.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3))
            new_log["boundary_dist_dice"] = (
                (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
        return new_loss, new_log

    if boundary_criterion is not None and not getattr(
            model, "boundary_distribution_head", False):
        raise ValueError(
            "boundary_criterion was supplied but model.boundary_distribution_head "
            "is False -- model.last_boundary_logits will never be populated. "
            "Build the model with boundary_distribution_head=True.")

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        # Dataset now yields a 4-tuple: the 4th element is the per-sample injected
        # speckle level (sqrt(v)); Phase 4 uses it as the bias supervision target,
        # phases 1-3 ignore it (passed through for a uniform, branch-free loop).
        for inp, target, mask, noise_sigma in loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            noise_sigma = noise_sigma.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # NOISE-ADAPTIVE CONDITIONING ('intrinsic' mode): measure the REAL
            # per-image intrinsic noise from the CLEAN target (Yang-Tai, [0,1] domain,
            # non-differentiable) and feed it to the model as the conditioning signal.
            # None in 'predicted' mode / flag off -> model ignores it (byte-identical).
            intrinsic = None
            if cond_intrinsic:
                with torch.no_grad():
                    intrinsic = estimate_sigma_normalized(target)   # [B] in [0,1]

            # AMP: the forward + loss run under autocast on CUDA (no-op on CPU).
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(inp, intrinsic_noise=intrinsic)
                # Step 4: model returns (final, seg_logits, bias) with unbiasing.
                if len(out) == 3:
                    denoised, seg_logits, bias = out
                else:
                    denoised, seg_logits = out
                    bias = None
                loss, log = criterion((denoised, seg_logits, bias),
                                      (target, mask, noise_sigma))
                loss, log = _combine_boundary_dist(loss, log, mask)

            # The loss returns LIVE tensors in its log dict; reduce each to a
            # python float for the numeric guard, epoch averaging and CSV. Keys
            # vary by phase, so accumulate dynamically over whatever is present.
            flog = {k: (v.item() if torch.is_tensor(v) else float(v))
                    for k, v in log.items()}

            # Noise-adaptive conditioning readout: read the per-image noise/scale the
            # model just stashed (from THIS first forward, before any SAM 2nd pass)
            # and bin by the batch's mean noise. Only when the flag is on and the
            # model actually produced a scale this pass.
            if cond_active and getattr(model, "_last_gate_scale", None) is not None:
                np_vals = model._last_noise_pred.reshape(-1)
                sc_vals = model._last_gate_scale.reshape(-1)
                thr = float(np_vals.mean())
                for nv, sv in zip(np_vals.tolist(), sc_vals.tolist()):
                    cond_noise_sum += nv
                    cond_scale_sum += sv
                    cond_n += 1
                    if nv >= thr:
                        cond_hi_scale_sum += sv
                        cond_hi_noise_sum += nv
                        cond_hi_n += 1
                    else:
                        cond_lo_scale_sum += sv
                        cond_lo_noise_sum += nv
                        cond_lo_n += 1

            total_finite = is_finite_scalar(flog.get("total_loss", float("nan")))
            if total_finite:
                nan_streak = 0
                # Accumulate ONLY finite batches into the running mean so a few
                # skipped/non-finite batches never poison the reported average to
                # NaN. Guard each component too (a stray non-finite term must not
                # leak into an otherwise-finite epoch mean).
                for k, v in flog.items():
                    if math.isfinite(v):
                        sums[k] = sums.get(k, 0.0) + v
                finite_loss_batches += 1
            else:
                nan_loss_batches += 1
                nan_streak += 1

            if is_train and sam is None:
                # ---- Standard (non-SAM) single-pass step — UNCHANGED ----
                # Scale the loss before backward to keep small fp16 grads alive.
                scaler.scale(loss).backward()
                # Unscale before clipping so max_norm is applied in true scale.
                scaler.unscale_(optimizer)
                if grads_finite(model):
                    # Grad clipping is ACTIVE: max gradient L2 norm = grad_clip,
                    # applied on TRUE-scale grads (post-unscale) BEFORE the step.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                else:
                    # SKIP the step on non-finite grads so NaN/Inf never corrupts
                    # the weights (works even when AMP/scaler is disabled).
                    nan_grad_batches += 1
                scaler.update()
            elif is_train:
                # ---- Sharpness-Aware Minimization: TWO passes on the SAME batch ----
                # SAM finds FLAT minima (better generalization on our small / few-
                # patient dataset). Pass-1 grad -> ascend to the sharpest nearby
                # point w_adv; pass-2 grad at w_adv drives the AdamW update after w
                # is restored. AMP GradScaler is bypassed here (SAM requires --amp
                # off; enforced in main), so backward()/step() are called directly.
                #
                # Pass 1: gradient at the current weights w (loss computed above).
                loss.backward()
                if grads_finite(model):
                    sam.first_step(zero_grad=True)   # climb to w_adv = w + rho*g/||g||
                    # Pass 2: recompute loss + grad at w_adv on the SAME batch. The
                    # model uses GroupNorm (no BatchNorm running stats), so the two
                    # passes over one batch cannot corrupt any normalization state.
                    with torch.amp.autocast(device_type=device.type,
                                            enabled=amp_enabled):
                        out2 = model(inp, intrinsic_noise=intrinsic)
                        if len(out2) == 3:
                            denoised2, seg_logits2, bias2 = out2
                        else:
                            denoised2, seg_logits2 = out2
                            bias2 = None
                        loss2, log2 = criterion((denoised2, seg_logits2, bias2),
                                                (target, mask, noise_sigma))
                        loss2, _ = _combine_boundary_dist(loss2, log2, mask)
                    loss2.backward()
                    if grads_finite(model):
                        # Clip the w_adv gradient, then restore w and AdamW-step.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        sam.second_step(zero_grad=True)   # restore w, then base step
                    else:
                        # Non-finite w_adv grad: RESTORE w (undo the ascent) but SKIP
                        # the update — weights are never left stranded at w_adv.
                        sam.second_step(zero_grad=True, apply_step=False)
                        nan_grad_batches += 1
                else:
                    # Non-finite pass-1 grad: no ascent, no step; weights untouched
                    # (same skip semantics as the non-SAM path above).
                    nan_grad_batches += 1

            n_batches += 1

            if is_train and nan_streak >= nan_patience:
                # A sustained CONSECUTIVE run of non-finite losses is real
                # divergence (distinct from a few scattered benign skips) — stop
                # this epoch immediately.
                diverged = True
                streak_diverged = True
                break

    # Epoch mean over FINITE batches ONLY (skipped/non-finite batches excluded),
    # so the CSV/log shows a real number instead of NaN when a few batches were
    # skipped.
    n_finite = max(finite_loss_batches, 1)
    avg = {k: s / n_finite for k, s in sums.items()}

    # Attach the noise-adaptive conditioning readout (own denominators, so kept out
    # of the loss-mean `sums`). These extra keys are ignored by the fixed CSV schema
    # and read only by the training-loop diagnostic log line.
    if cond_active and cond_n > 0:
        avg["cond_noise_mean"] = cond_noise_sum / cond_n
        avg["cond_scale_mean"] = cond_scale_sum / cond_n
        if cond_hi_n > 0:
            avg["cond_scale_hi"] = cond_hi_scale_sum / cond_hi_n
            avg["cond_noise_hi"] = cond_hi_noise_sum / cond_hi_n
        if cond_lo_n > 0:
            avg["cond_scale_lo"] = cond_lo_scale_sum / cond_lo_n
            avg["cond_noise_lo"] = cond_lo_noise_sum / cond_lo_n

    # DIVERGENCE (training only) — distinguish BENIGN transient skips from REAL
    # divergence. A few scattered batches whose optimizer step was skipped are
    # harmless (weights untouched); halting on them would throw away a perfectly
    # good run. ``skipped_batches`` (non-finite GRAD -> step skipped) is the
    # complete set of batches that produced no weight update; because a non-finite
    # LOSS always yields a non-finite grad, it is a superset of nan_loss_batches
    # AND also catches finite-loss/non-finite-grad backward blowups. We halt ONLY
    # on real corruption:
    skipped_batches = nan_grad_batches
    nan_fraction = skipped_batches / max(n_batches, 1)
    weights_ok = weights_finite(model) if is_train else True
    fraction_diverged = False
    if is_train:
        # (a) a LARGE fraction of the epoch's steps were skipped (actually diverging).
        if nan_fraction > nan_fraction_threshold:
            diverged = True
            fraction_diverged = True
        # (b) the weights themselves are non-finite (real, persisted corruption).
        #     (This also covers spec case (c): "aggregate non-finite AND weights
        #     non-finite" is a strict subset of "weights non-finite".)
        if not weights_ok:
            diverged = True

    # ONE summary line per epoch instead of a per-batch, per-term warning flood.
    if is_train and diverged:
        reasons = []
        if fraction_diverged:
            reasons.append("skipped fraction %.3f > threshold %.3f"
                           % (nan_fraction, nan_fraction_threshold))
        if not weights_ok:
            reasons.append("model WEIGHTS non-finite")
        if streak_diverged:
            reasons.append("%d consecutive non-finite-loss batches (nan_patience=%d)"
                           % (nan_streak, nan_patience))
        logger.critical("[NUMERIC] FATAL divergence at epoch: %d/%d batches skipped. "
                        "Reason(s): %s. This is real corruption, not a benign "
                        "transient skip — halting.",
                        skipped_batches, n_batches, "; ".join(reasons))
    elif is_train and (skipped_batches or nan_loss_batches):
        logger.info("[NUMERIC] Epoch: %d/%d batches skipped (benign, weights "
                    "finite) — continuing. (%d non-finite loss, %d non-finite "
                    "grad; skipped fraction=%.3f <= %.3f). Epoch mean computed "
                    "over %d finite batches.",
                    skipped_batches, n_batches, nan_loss_batches, nan_grad_batches,
                    nan_fraction, nan_fraction_threshold, finite_loss_batches)
    elif nan_loss_batches or nan_grad_batches:
        # Eval pass: no divergence policy, just report.
        logger.info("[NUMERIC] %d/%d val batches non-finite (loss=%d, grad=%d); "
                    "mean over %d finite batches.",
                    max(nan_loss_batches, nan_grad_batches), n_batches,
                    nan_loss_batches, nan_grad_batches, finite_loss_batches)

    return avg, diverged


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepOCT-Ultra production training (Phase 3, server/cloud).")

    # Core hyperparameters.
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient L2 norm (applied after AMP unscale).")
    parser.add_argument("--nan_patience", type=int, default=25,
                        help="Halt training if the loss stays NON-FINITE for this "
                             "many CONSECUTIVE batches. Guards against wasting epochs "
                             "on a sustained diverged run; the last good checkpoint "
                             "is preserved.")
    parser.add_argument("--nan_fraction_threshold", type=float, default=0.2,
                        help="Halt training only if MORE than this FRACTION of an "
                             "epoch's batches are non-finite (default 0.2 = 20%%). A "
                             "handful of AMP-transient skipped batches (grads "
                             "non-finite -> step skipped -> weights untouched) is "
                             "benign and does NOT halt as long as the model weights "
                             "stay finite.")

    # Ablation Study phase selector.
    #
    # PORTED: the original accepted choices [1, 2, 3, 4, 5] and DEFAULTED to 4
    # (dynamic unbiasing). Only Phase 1 -- segmentation-only -- is carried into
    # this repository, so the flag is retained (it still tags the checkpoint and
    # the metrics CSV rows, and guards resume-compatibility) but is pinned to 1.
    parser.add_argument("--ablation_phase", type=int, choices=[1], default=1,
                        help="Ablation phase. 1 = segmentation only (the only "
                             "phase available here). Retained so checkpoint and "
                             "metrics filenames stay phase-tagged.")

    # Architecture flags (compatible with DeepOCTUltra).
    # PORTED: --use-unbiasing, --use-cross-gating, --film_enabled and
    # --noise_adaptive_conditioning were removed here; DeepOCTUltra force-disables
    # all of them on the segmentation-only path. See the model construction below
    # for the literal values they are pinned to.
    #
    # --use-wavelets IS meaningful in Phase 1: DTCWTBottleneck is built from
    # `use_wavelets` alone and is NOT gated by `seg_only` (see
    # DeepOCTUltra.__init__), so the flag is kept.
    parser.add_argument("--use-wavelets", action="store_true",
                        help="Step 5: DT-CWT + Horizontal Axial Attention bottleneck.")
    parser.add_argument("--boundary_distribution_head", action="store_true",
                        help="Build AND TRAIN the ADDITIVE probabilistic boundary-"
                             "distribution head (model.BoundaryDistributionHead) "
                             "alongside the existing mask head. Predicts a per-"
                             "column depth DISTRIBUTION for the choroid's upper/"
                             "lower boundary instead of committing to a hard mask; "
                             "trained via loss.BoundaryDistributionLoss (soft-NLL "
                             "+ expectation regression), combined into the total "
                             "per --boundary_dist_mode. Default OFF = byte-"
                             "identical (module not built; forward()'s return "
                             "signature is unchanged either way -- its logits are "
                             "stashed on model.last_boundary_logits, not returned).")
    parser.add_argument("--boundary_dist_mode", choices=["replace", "joint"],
                        default="joint",
                        help="How the boundary-distribution loss combines with the "
                             "existing mask BCE+Dice seg loss (only used with "
                             "--boundary_distribution_head). 'joint' (DEFAULT, "
                             "RECOMMENDED): the two losses are ADDED -- the shared "
                             "decoder gets both a hard-mask signal AND a "
                             "distribution signal every step, and mask-head Dice "
                             "vs. boundary-head (rasterized) Dice can be compared "
                             "head-to-head under IDENTICAL shared features -- the "
                             "most informative ablation, and the safer default "
                             "since it never alters the existing loss composition, "
                             "only adds to it. 'replace': the boundary-distribution "
                             "loss REPLACES lambda_seg*seg_total (+ the optional "
                             "Kervadec boundary term) in the total; mask BCE/Dice "
                             "are still COMPUTED and logged (for comparison) but "
                             "receive NO gradient -- a strict ablation of 'does "
                             "distribution-ONLY supervision train a usable mask'. "
                             "NOT supported together with --learned_loss_weighting "
                             "(fails fast at startup; see the code comment).")
    parser.add_argument("--lambda_boundary_dist", type=float, default=1.0,
                        help="Weight on the boundary-distribution loss's total "
                             "(NLL + lambda_boundary_dist_expectation*expectation_"
                             "L1) when combined into the training total. Only used "
                             "with --boundary_distribution_head.")
    parser.add_argument("--boundary_dist_sigma", type=float, default=2.0,
                        help="Gaussian soft-target sigma (PIXELS) for the boundary-"
                             "distribution NLL -- the assumed label-placement "
                             "uncertainty width. Resolution-dependent (like the "
                             "Kervadec SDF weight); this is a starting prior for "
                             "512px, re-tune per resolution. Only used with "
                             "--boundary_distribution_head.")
    parser.add_argument("--lambda_boundary_dist_expectation", type=float, default=0.1,
                        help="Weight on the |E[z]-z_gt| expectation-regression term "
                             "inside BoundaryDistributionLoss (pins the predicted "
                             "MEAN depth; the NLL term alone shapes the distribution "
                             "but only pulls the mean indirectly through the "
                             "Gaussian soft target). Only used with "
                             "--boundary_distribution_head.")
    # OPTION A (sharper intervals via MULTI-MODAL head). Default 0 = OFF = the
    # ORIGINAL single-softmax BoundaryDistributionHead (byte-identical). > 0 builds
    # the K-component BoundaryMixtureHead instead: a per-column mixture of K Gaussians
    # over depth that can be SHARP-and-bimodal rather than broad-and-unimodal, so the
    # derived std / prediction interval is not over-widened on ambiguous columns. The
    # output interface is identical ([B,2,H,W] log-pmf), so loss/eval/conformal are
    # unchanged. Requires --boundary_distribution_head.
    parser.add_argument("--boundary_mixture", type=int, default=0,
                        help="K components for the multi-modal BoundaryMixtureHead "
                             "(0=OFF, default; single-softmax head. Try 2-3). "
                             "Requires --boundary_distribution_head.")
    # OPTION C (sharpness-aware training). Default 0 = OFF (byte-identical loss). > 0
    # adds an AUXILIARY penalty on the predicted per-column boundary WIDTH (mean std)
    # to BoundaryDistributionLoss, pushing the distribution to be SHARP -- but ONLY
    # as an auxiliary to the NLL, which (plus the downstream conformal recalibration)
    # keeps it from collapsing to overconfidence. Keep SMALL; report the coverage vs.
    # width trade-off as it varies. Requires --boundary_distribution_head.
    parser.add_argument("--boundary_sharpness_weight", type=float, default=0.0,
                        help="Auxiliary width-penalty weight for the boundary "
                             "distribution (0=OFF, default = byte-identical). Small "
                             "values (e.g. 0.01-0.1) push sharper intervals; conformal "
                             "recalibration keeps coverage honest. Requires "
                             "--boundary_distribution_head.")
    # Spatial Variance Penalty (Conformal-Aware Training). Default 0.0 = OFF =
    # byte-identical total loss (matches every other lambda_* default-off
    # convention in this file). Same INTENT as --boundary_sharpness_weight
    # (discourage a wide predicted depth distribution) but computed via the
    # literal second central moment sum(p*(d-mu)^2) inside BoundaryDistributionLoss
    # rather than the E[X^2]-E[X]^2 std shortcut sharpness_weight uses; the raw
    # variance_penalty is logged every step regardless of this weight (see the
    # training-loop logging block) so interval tightness can be tracked even with
    # the penalty off. Requires --boundary_distribution_head.
    parser.add_argument("--lambda_spread", "--lambda-spread", type=float, default=0.0,
                        dest="lambda_spread",
                        help="Weight on the Spatial Variance Penalty for the "
                             "boundary distribution (0=OFF, default = byte-"
                             "identical). Small values (e.g. 0.01-0.1) push "
                             "narrower per-column depth distributions. Requires "
                             "--boundary_distribution_head.")

    # Composite-loss per-term lambda weights (each term has its OWN weight; the
    # terms live on different scales, so an unweighted raw sum would let one
    # dominate). seg/zncc anchor at 1.0; gat/ffl default ABOVE 1.0 (they are inert
    # at 1.0) — these are STARTING PRIORS, re-measure/tune per run (see loss.py).
    parser.add_argument("--lambda_seg", type=float, default=1.0,
                        help="Weight on the segmentation (BCE+Dice) loss.")
    parser.add_argument("--lambda_zncc", type=float, default=1.0,
                        help="Weight on the ZNCC reconstruction loss.")
    parser.add_argument("--lambda_gat", type=float, default=5.0,
                        help="Weight on the GAT-variance (tissue grain) loss. "
                             "Default 5.0 (inert at 1.0); a starting prior, re-tune.")
    parser.add_argument("--lambda_ffl", type=float, default=4.0,
                        help="Weight on the Focal Frequency Loss. "
                             "Default 4.0 (inert at 1.0); a starting prior, re-tune.")
    parser.add_argument("--lambda_bias", type=float, default=0.1,
                        help="Weight on the predicted-bias MSE penalty (Phase 4 "
                             "only; supervised toward the injected speckle sqrt(v)).")
    # OPTIONAL learned multi-task loss weighting (Kendall & Gal, CVPR 2018). Default
    # OFF = byte-identical fixed-lambda behavior. When ON, the lambda_zncc/gat/ffl/
    # bias constants above are REPLACED by LEARNED per-task weights via homoscedastic
    # uncertainty: each term i is combined as exp(-s_i)*L_i + s_i where s_i=log(sig^2)
    # is an nn.Parameter (added to the optimizer, init 0 -> all weights start at 1).
    # SEG stays the fixed reference (weight = lambda_seg). See loss.py DeepOCTLoss.
    parser.add_argument("--learned_loss_weighting", action="store_true",
                        help="Learn per-task loss weights via homoscedastic "
                             "uncertainty (Kendall & Gal 2018) INSTEAD of the fixed "
                             "lambda_zncc/gat/ffl/bias. Seg is the fixed reference "
                             "(weight=lambda_seg); the denoise-term weights are "
                             "learned nn.Parameters (init weight 1) trained by the "
                             "optimizer and logged each epoch. Default OFF = existing "
                             "fixed-lambda behavior (byte-identical). No effect in "
                             "Phase 1 (single-task).")
    # Loss-scale NORMALIZATION for the learned weighting (interpretability). Only
    # consulted when --learned_loss_weighting is set. Default True: each task loss is
    # divided by a running-EMA estimate of its own magnitude so the learned weights
    # reflect task IMPORTANCE (weight>1 = up-weight vs seg, <1 = down-weight), not raw
    # loss scale, and the total stays positive/bounded. Pass False to recover the RAW
    # Kendall-Gal (for a normalized-vs-raw comparison).
    parser.add_argument("--learned_weight_normalize", type=_str2bool, nargs="?",
                        const=True, default=True,
                        help="Normalize each task loss by a running-EMA of its "
                             "magnitude BEFORE Kendall-Gal weighting so the learned "
                             "weights are interpretable (task importance, not loss "
                             "scale). Default True. 'False' = raw Kendall-Gal. Only "
                             "used with --learned_loss_weighting.")
    parser.add_argument("--learned_weight_warmup_steps", type=int, default=100,
                        help="Warmup steps for learned+normalized weighting: use "
                             "fixed unit weights on the normalized losses for the "
                             "first N training steps while the EMA scales seed and "
                             "stabilize, then switch on the learned log-variances. "
                             "Default 100. Only used with --learned_loss_weighting "
                             "+ --learned_weight_normalize.")
    parser.add_argument("--learned_weight_ema_momentum", type=float, default=0.99,
                        help="EMA momentum for the per-task loss-scale estimate used "
                             "by --learned_weight_normalize (default 0.99). Higher = "
                             "slower-moving scale.")
    # Kervadec boundary/surface loss — a NEW, OPTIONAL ablation dimension (seg-head
    # only, orthogonal to the denoising phases). Default 0.0 = OFF = byte-identical
    # to the existing seg behavior. > 0 adds lambda_boundary*BoundaryLoss to sharpen
    # the fuzzy choroid-sclera interface (HD95/ASD). The weight is LINEARLY RAMPED
    # from 0 -> lambda_boundary over --boundary_ramp_epochs (Kervadec rebalancing).
    parser.add_argument("--lambda_boundary", type=float, default=0.0,
                        help="MAX weight on the Kervadec boundary/surface loss "
                             "(seg-head boundary localization, all phases). Default "
                             "0.0 = OFF (loss byte-identical to current). The SDF is "
                             "in PIXELS so this is resolution-dependent and must be "
                             "SMALL vs Dice/BCE; try 0.01 at 512px, then tune "
                             "(0.01-0.1). Ramped from 0 over --boundary_ramp_epochs.")
    parser.add_argument("--boundary_ramp_epochs", type=int, default=30,
                        help="Epochs over which lambda_boundary ramps LINEARLY from "
                             "0 to its full value (Kervadec 'rebalancing': region "
                             "loss dominates early, boundary loss grows later, so it "
                             "never destabilizes early training). Only used when "
                             "--lambda_boundary > 0. <=1 disables the ramp (full "
                             "weight from epoch 1).")

    # Validation speckle mode (capability; default conservative).
    parser.add_argument("--val_speckle", choices=["none", "deterministic"],
                        default="none",
                        help="Validation speckle mode. 'none' (DEFAULT) = clean val, "
                             "exactly as before. 'deterministic' = additionally run a "
                             "second, reproducible NOISY val pass (whole-image gamma "
                             "at fixed v=0.08, per-sample seeded) so FiLM is exercised "
                             "on val; logs Val_Dice_Clean and Val_Dice_Noisy. Best-"
                             "model selection still uses clean val loss. Do NOT flip "
                             "the default without sign-off.")

    # Paths.
    parser.add_argument("--data_dir", default=None,
                        help="Directory holding the train/val manifest CSVs.")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Where checkpoint.pth, best_model.pth and "
                             "training.log are written.")
    parser.add_argument("--train-manifest", default="train_manifest.csv",
                        help="Train manifest filename inside --data_dir.")
    parser.add_argument("--val-manifest", default="val_manifest.csv",
                        help="Val manifest filename inside --data_dir.")
    # PORTED: the default was "oimhs", which silently routed an SDOCT manifest
    # into the side-by-side OIMHS loader (that loader reads ONE file and splits it
    # in half), failing confusingly downstream instead of at the manifest. The
    # default is now "sdoct"; the OIMHS loader stays reachable via an explicit
    # --dataset oimhs.
    parser.add_argument("--dataset", choices=["sdoct", "oimhs"], default="sdoct",
                        help="Which real dataset to build from the manifests. "
                             "'sdoct' (DEFAULT) = separate image+overlay TIFFs with "
                             "a filled-contour mask, via SDOCTDataset; its manifests "
                             "carry the extra overlay_path/class columns. "
                             "'oimhs' = the original side-by-side loader, where one "
                             "file holds the OCT image (left half) and the RGB mask "
                             "(right half).")

    # Data / runtime.
    parser.add_argument("--image-size", type=int, default=512,
                        help="Square resize; server runs use 512, local smoke 64. "
                             "IGNORED when --isotropic_resize is set (SDOCT only).")
    parser.add_argument("--isotropic_resize", action="store_true",
                        help="SDOCT ONLY (--dataset sdoct), default OFF. Native "
                             "Bioptigen SDOCT TIFFs are 938(W)x625(H); the default "
                             "pipeline resizes to a SQUARE --image-size, which "
                             "applies a DIFFERENT scale factor per axis (today: "
                             "1.221 raw px/model px axial, 1.832 lateral) -- a "
                             "silent geometric warp that HD95/ASD cannot be "
                             "converted back to microns through. This flag instead "
                             "holds the axial (H) resolution FIXED at 512 (axial "
                             "sampling UNCHANGED from today) and derives a "
                             "non-square W (768) from the SAME single scale factor "
                             "-- no padding/letterboxing. See "
                             "sdoct_dataset.isotropic_target_size(). OFF is "
                             "byte-identical to current behavior (see "
                             "test_isotropic_resize_smoke.py). Invalidates any "
                             "checkpoint trained without it, and vice versa -- "
                             "resuming across a mismatch is refused (see "
                             "load_checkpoint).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best_metric", choices=["dice", "total_loss"],
                        default="dice",
                        help="Metric driving best_model_phaseN.pth selection. "
                             "'dice' (DEFAULT) = HIGHEST val choroid Dice (the "
                             "segmentation paper's headline; for phases 2-5 the "
                             "total loss mixes denoise/bias terms and can diverge "
                             "from best Dice). 'total_loss' = LOWEST val total loss "
                             "(legacy behavior).")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (autocast + GradScaler). "
                             "OFF by default and ONLY active on CUDA. Do the FIRST "
                             "GPU run with AMP off; enable --amp after a 1-epoch "
                             "AMP-on smoke test confirms no NaN.")

    # Sharpness-Aware Minimization (SAM) — OPTIONAL generalization boost for the
    # small / few-patient dataset. OFF by default: with --use_sam absent the
    # training step is EXACTLY the existing single-pass AdamW step (byte-identical
    # loss). When ON, each step does TWO forward-backward passes (~2x compute).
    parser.add_argument("--use_sam", action="store_true",
                        help="Wrap AdamW in Sharpness-Aware Minimization (Foret et "
                             "al. 2021). Seeks FLAT minima that generalize better on "
                             "few/unseen patients. Doubles per-step compute (two "
                             "forward-backward passes). Requires --amp OFF. Default "
                             "OFF = existing single-pass behavior, byte-identical.")
    parser.add_argument("--sam_rho", type=float, default=0.05,
                        help="SAM neighborhood radius rho (default 0.05, the standard "
                             "value). Only used with --use_sam.")
    parser.add_argument("--sam_adaptive", action="store_true",
                        help="Use ASAM (adaptive SAM, Kwon et al. 2021): per-parameter "
                             "|w|-scaled neighborhood, scale-invariant. Default OFF "
                             "= standard SAM. Only used with --use_sam.")

    # Local smoke-testing.
    parser.add_argument("--mock", action="store_true",
                        help="Force the tiny synthetic dataset (local smoke).")
    parser.add_argument("--mock-samples", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing checkpoint and start fresh.")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    setup_logging(args.checkpoint_dir)
    # Full reproducibility seeding: torch (CPU + CUDA), numpy AND stdlib random.
    # The dataset's augmentation uses GLOBAL np.random, so np.random.seed covers it
    # for num_workers=0; for worker processes, seed_worker (DataLoader worker_init_fn)
    # re-seeds numpy/random per worker. cudnn determinism is intentionally NOT forced
    # (avoids the speed cost; bit-exact GPU kernels are not the target).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Device-agnostic setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP is OPT-IN via --amp (default OFF) and only ever active on CUDA. The first
    # run on a fresh/unknown GPU should be AMP-OFF (the safe, validated-locally
    # path); enable --amp only after a 1-epoch AMP-ON smoke test shows no NaN.
    # On CPU AMP is always disabled regardless of the flag.
    amp_enabled = bool(args.amp) and device.type == "cuda"
    if device.type == "cuda":
        logger.info("Device: cuda (%s) | AMP=%s (--amp=%s)",
                    torch.cuda.get_device_name(0), amp_enabled, args.amp)
    else:
        logger.info("Device: cpu (AMP forced off; --amp=%s ignored on CPU)", args.amp)

    # ---- Data ----
    # Selection rule:
    #   * --data_dir given -> REAL data; manifests MUST exist (else FileNotFoundError).
    #   * no --data_dir     -> MOCK fallback (local smoke). --mock also forces MOCK.
    if args.data_dir and not args.mock:
        if args.isotropic_resize and args.dataset != "sdoct":
            raise ValueError(
                "--isotropic_resize requires --dataset sdoct (got "
                f"--dataset {args.dataset!r}).")
        train_manifest, val_manifest = require_manifests(
            args.data_dir, args.train_manifest, args.val_manifest)
        if args.isotropic_resize:
            from deepoct.datasets.sdoct_dataset import isotropic_target_size
            resolved_h, resolved_w = isotropic_target_size()
            logger.info("Dataset: REAL — train=%s val=%s @ %dx%d (ISOTROPIC "
                        "resize; --image-size=%d ignored)",
                        train_manifest, val_manifest, resolved_h, resolved_w,
                        args.image_size)
        else:
            logger.info("Dataset: REAL — train=%s val=%s @ %dx%d",
                        train_manifest, val_manifest, args.image_size, args.image_size)
        train_ds = build_real_dataset(train_manifest, args.image_size, augment=True,
                                      ablation_phase=args.ablation_phase,
                                      dataset_kind=args.dataset,
                                      isotropic_resize=args.isotropic_resize)
        val_ds = build_real_dataset(val_manifest, args.image_size, augment=False,
                                    ablation_phase=args.ablation_phase,
                                    dataset_kind=args.dataset,
                                    isotropic_resize=args.isotropic_resize)
        # Optional SECOND val dataset with DETERMINISTIC whole-image speckle, so the
        # noisy val pass exercises FiLM (Phase 5). Built only when explicitly enabled;
        # 'none' leaves the val path exactly as before (no second dataset/pass).
        val_noisy_ds = None
        if args.val_speckle == "deterministic":
            val_noisy_ds = build_real_dataset(
                val_manifest, args.image_size, augment=False,
                ablation_phase=args.ablation_phase, deterministic_speckle=True,
                dataset_kind=args.dataset, isotropic_resize=args.isotropic_resize)
        num_workers = args.num_workers
    else:
        reason = "forced via --mock" if args.mock else "no --data_dir provided"
        logger.info("Dataset: MOCK (%s) — %d samples @ %dx%d",
                    reason, args.mock_samples, args.image_size, args.image_size)
        train_ds = MockOCTDataset(args.mock_samples, args.image_size, seed=args.seed)
        val_ds = MockOCTDataset(args.mock_samples, args.image_size, seed=args.seed + 1)
        val_noisy_ds = None  # deterministic val speckle is a REAL-data capability
        if args.val_speckle == "deterministic":
            logger.info("--val_speckle deterministic ignored for the MOCK dataset "
                        "(it is a real-data capability).")
        if args.isotropic_resize:
            logger.info("--isotropic_resize ignored for the MOCK dataset (it is an "
                        "SDOCT real-data capability).")
        num_workers = 0  # tiny in-memory set; workers add only overhead

    pin = device.type == "cuda"
    # Reproducible loaders: a --seed-seeded generator fixes the train shuffle order,
    # and seed_worker re-seeds numpy/random inside each worker (so augmentation is
    # reproducible regardless of --num-workers).
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              drop_last=False, worker_init_fn=seed_worker,
                              generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin,
                            worker_init_fn=seed_worker)
    # Noisy (deterministic-speckle) val loader — only when enabled on real data.
    val_noisy_loader = None
    if val_noisy_ds is not None:
        val_noisy_loader = DataLoader(val_noisy_ds, batch_size=args.batch_size,
                                      shuffle=False, num_workers=num_workers,
                                      pin_memory=pin, worker_init_fn=seed_worker)
        logger.info("Validation speckle: DETERMINISTIC — second NOISY val pass ON "
                    "(whole-image gamma v=0.08, per-sample seeded). Logs "
                    "Val_Dice_Clean + Val_Dice_Noisy; best-model selection uses "
                    "CLEAN val loss. (Two eval passes per epoch.)")
    else:
        logger.info("Validation speckle: none — single CLEAN val pass (existing "
                    "behavior).")

    # ---- Model / optimizer / loss / AMP scaler ----
    if args.ablation_phase == 1:
        logger.info("=== Running Phase 1: Segmentation Only ===")
        logger.info("Denoiser, unbiasing and cross-gating are DISABLED; "
                    "loss = SegLoss (BCE + Dice) only.")
    # PORTED: the `elif args.ablation_phase == 2/3/4/5` logging blocks that
    # followed (announcing the multi-task, static-unbiasing, dynamic-unbiasing
    # and whole-image-speckle/FiLM configurations) were removed with the rest
    # of the phase 2-5 scaffolding. See the original train.py for their text.

    # PORTED: the five flags removed from the CLI above are passed here as the
    # literal values their argparse defaults produced, so the model constructed
    # for a Phase-1 run is identical to the original. DeepOCTUltra force-disables
    # each of them on this path anyway: `seg_only` -> use_cross_gating False and
    # the denoise decoder is never built; noise_adaptive_conditioning requires
    # cross-gating; film_enabled is False outside Phase 5; use_unbiasing is
    # hard-wired False for every phase inside the model.
    model = DeepOCTUltra(in_channels=3, base_filters=args.base_filters,
                         use_unbiasing=False,           # was args.use_unbiasing (default False)
                         use_wavelets=args.use_wavelets,
                         use_cross_gating=False,        # was args.use_cross_gating (default False)
                         ablation_phase=args.ablation_phase,
                         film_enabled=True,             # was args.film_enabled (default True)
                         noise_adaptive_conditioning=False,   # was args.noise_adaptive_conditioning
                         noise_signal="predicted",      # was args.noise_signal (default)
                         boundary_distribution_head=args.boundary_distribution_head,
                         boundary_mixture=args.boundary_mixture
                         ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    # OPTIONAL Sharpness-Aware Minimization. When --use_sam is set we WRAP the
    # AdamW optimizer above in SAM; the training step then runs two passes. When
    # OFF, `sam` stays None and run_epoch takes the existing single-pass path
    # (byte-identical). SAM is INCOMPATIBLE with the AMP GradScaler two-pass state
    # machine, so we fail fast if both are requested (do the first SAM run AMP-off,
    # per the same guidance as --amp itself).
    sam = None
    if args.use_sam:
        if amp_enabled:
            raise ValueError(
                "--use_sam is incompatible with --amp (the SAM two-pass step "
                "bypasses the GradScaler). Re-run with --amp off; SAM computes in "
                "full precision.")
        sam = SAM(optimizer, rho=args.sam_rho, adaptive=args.sam_adaptive)
        # From here on `optimizer` refers to the SAM wrapper so zero_grad /
        # state_dict / checkpoint-resume all route through it (SAM delegates those
        # to the wrapped AdamW, so checkpoints stay AdamW-compatible).
        optimizer = sam
        logger.warning("SAM ENABLED (rho=%g, %s): each training step runs TWO "
                       "forward-backward passes on the same batch -> expect ~2x "
                       "training time. Seeks flat minima for better generalization "
                       "on the small / few-patient dataset.",
                       args.sam_rho, "ASAM/adaptive" if args.sam_adaptive
                       else "standard SAM")
    criterion = DeepOCTLoss(
        lambda_seg=args.lambda_seg,
        lambda_zncc=args.lambda_zncc,
        lambda_gat=args.lambda_gat,
        lambda_ffl=args.lambda_ffl,
        lambda_bias=args.lambda_bias,
        lambda_boundary=args.lambda_boundary,
        ablation_phase=args.ablation_phase,
        learned_loss_weighting=args.learned_loss_weighting,
        learned_weight_normalize=args.learned_weight_normalize,
        learned_weight_warmup_steps=args.learned_weight_warmup_steps,
        ema_momentum=args.learned_weight_ema_momentum,
    ).to(device)

    # OPTIONAL Kendall & Gal learned loss weighting: the criterion holds LEARNABLE
    # log-variance nn.Parameters that must be trained by the optimizer. AdamW was
    # built from model.parameters() only, so add the criterion's log-variances as a
    # SEPARATE param group (weight_decay=0 — decaying a log-variance toward 0 would
    # bias every weight toward 1 and fight the Kendall-Gal objective). Added BEFORE
    # any checkpoint resume so the optimizer's param-group layout matches the saved
    # state on load. With SAM active, `optimizer` is the SAM wrapper which SHARES the
    # base optimizer's param_groups list by reference, so adding the group to the
    # base optimizer is visible through SAM too. No-op (byte-identical) when OFF.
    if criterion.learned_loss_weighting:
        base_opt = optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer
        base_opt.add_param_group({
            "params": list(criterion.log_vars.parameters()),
            "weight_decay": 0.0,
        })
        logger.info("Learned loss weighting (Kendall & Gal 2018) ACTIVE: %d "
                    "log-variance params [%s] added to the optimizer "
                    "(weight_decay=0, lr=%g). Fixed lambdas for these tasks are "
                    "IGNORED; seg is the fixed reference (weight=lambda_seg=%g). "
                    "log-variances init 0 -> all weights start at 1.0.",
                    len(criterion.log_vars), ", ".join(criterion.log_vars.keys()),
                    args.learning_rate, args.lambda_seg)
        if criterion.learned_weight_normalize:
            logger.info("  Loss-scale NORMALIZATION ON (interpretable weights): each "
                        "task loss divided by a running-EMA (momentum=%g) of its "
                        "magnitude before weighting; %d-step fixed-weight warmup while "
                        "EMA scales stabilize. Weights then read as task IMPORTANCE "
                        "(>1 up-weight vs seg, <1 down-weight).",
                        args.learned_weight_ema_momentum,
                        args.learned_weight_warmup_steps)
        else:
            logger.info("  Loss-scale normalization OFF (--learned_weight_normalize "
                        "False): RAW Kendall-Gal on unnormalized losses — weights "
                        "reflect loss SCALE, not importance (comparison mode).")
    elif args.learned_loss_weighting and args.ablation_phase == 1:
        logger.info("--learned_loss_weighting was set but Phase 1 is single-task "
                    "(segmentation only); the scheme is meaningless there and is "
                    "IGNORED (fixed-lambda behavior, byte-identical).")

    # OPTIONAL boundary-DISTRIBUTION head training (default OFF -> byte-identical:
    # boundary_criterion stays None, run_epoch's _combine_boundary_dist is then a
    # verified no-op at every call site). BoundaryDistributionLoss has NO learnable
    # parameters of its own (sigma/lambda_expectation are fixed floats); the head's
    # weights (model.boundary_head) are already part of model.parameters(), so the
    # existing AdamW construction above already covers them -- no optimizer surgery
    # needed here (unlike the Kendall & Gal log-variances above).
    # OPTION A/C guard: both are variants of the boundary-distribution head, so they
    # require it. Fail fast (rather than silently no-op) if requested without it.
    if (args.boundary_mixture > 0 or args.boundary_sharpness_weight > 0
            or args.lambda_spread > 0) and not args.boundary_distribution_head:
        raise ValueError(
            "--boundary_mixture / --boundary_sharpness_weight / --lambda_spread "
            "require --boundary_distribution_head (all are variants of that head).")

    boundary_criterion = None
    if args.boundary_distribution_head:
        if args.boundary_dist_mode == "replace" and criterion.learned_loss_weighting:
            raise ValueError(
                "--boundary_dist_mode replace is not supported together with "
                "--learned_loss_weighting: 'replace' reconstructs the mask seg "
                "contribution as lambda_seg*seg_total (the RAW fixed-lambda "
                "formula), but learned+normalized weighting adds "
                "lambda_seg*seg_total/ema_scale instead -- the subtraction would "
                "silently leave a residual. Use --boundary_dist_mode joint "
                "instead (purely additive, safe under any weighting scheme), or "
                "drop --learned_loss_weighting.")
        boundary_criterion = BoundaryDistributionLoss(
            sigma=args.boundary_dist_sigma,
            lambda_expectation=args.lambda_boundary_dist_expectation,
            sharpness_weight=args.boundary_sharpness_weight,
            lambda_spread=args.lambda_spread,
        ).to(device)
        logger.info(
            "Boundary-DISTRIBUTION head ACTIVE [mode=%s]: per-column depth "
            "distribution (upper/lower choroid boundary) trained via soft-NLL + "
            "%g*expectation-L1 (sigma=%gpx). %s Combined weight into the total: "
            "lambda_boundary_dist=%g.", args.boundary_dist_mode,
            args.lambda_boundary_dist_expectation, args.boundary_dist_sigma,
            ("REPLACES the mask BCE+Dice[+Kervadec] training signal (mask "
             "metrics are still computed/logged for comparison, but get no "
             "gradient)." if args.boundary_dist_mode == "replace" else
             "Runs ALONGSIDE the mask BCE+Dice loss -- both heads are trained "
             "jointly on the shared decoder for a direct head-to-head "
             "comparison."),
            args.lambda_boundary_dist)

    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Config: epochs=%d batch_size=%d lr=%g wd=%g base_filters=%d "
                "params=%.2fM", args.epochs, args.batch_size, args.learning_rate,
                args.weight_decay, args.base_filters, n_params / 1e6)
    logger.info("Ablation phase=%d | effective flags: unbiasing=%s wavelets=%s "
                "cross_gating=%s film_enabled=%s noise_adaptive_cond=%s "
                "(signal=%s) | AMP=%s",
                args.ablation_phase,
                model.use_unbiasing, model.use_wavelets, model.use_cross_gating,
                model.film_enabled, model.noise_adaptive_conditioning,
                model.noise_signal, amp_enabled)
    logger.info("Loss lambdas: seg=%g zncc=%g gat=%g ffl=%g bias=%g",
                args.lambda_seg, args.lambda_zncc, args.lambda_gat,
                args.lambda_ffl, args.lambda_bias)
    if args.lambda_boundary > 0:
        logger.info("Boundary loss (Kervadec) ACTIVE: lambda_boundary=%g "
                    "(SDF in pixels; resolution-dependent), LINEAR ramp 0 -> %g over "
                    "%d epochs. Sharpens choroid-sclera boundary (HD95/ASD); "
                    "seg-head only, all phases.",
                    args.lambda_boundary, args.lambda_boundary,
                    args.boundary_ramp_epochs)
    else:
        logger.info("Boundary loss (Kervadec): OFF (lambda_boundary=0) — "
                    "seg loss byte-identical to baseline.")
    logger.info("Best-checkpoint selection metric: %s (%s val %s = better)",
                args.best_metric,
                "higher" if args.best_metric == "dice" else "lower",
                "Dice" if args.best_metric == "dice" else "total loss")

    # ---- Resume from checkpoint if present ----
    # Phase-specific filenames so each ablation phase resumes ONLY its own
    # (architecture-compatible) checkpoint and never clobbers another phase's.
    ckpt_path = os.path.join(args.checkpoint_dir,
                             phase_artifact(CHECKPOINT_NAME, args.ablation_phase))
    best_path = os.path.join(args.checkpoint_dir,
                             phase_artifact(BEST_MODEL_NAME, args.ablation_phase))
    start_epoch = 1
    # best_val starts at the WORST value for the chosen metric so the first epoch
    # always improves on it: -inf for dice (higher is better), +inf for total loss.
    best_val = -math.inf if args.best_metric == "dice" else math.inf
    if os.path.isfile(ckpt_path) and not args.no_resume:
        start_epoch, best_val = load_checkpoint(
            ckpt_path, model, optimizer, scaler, device, criterion=criterion,
            cur_isotropic_resize=args.isotropic_resize)
    else:
        logger.info("No checkpoint to resume — starting fresh at epoch 1.")

    if start_epoch > args.epochs:
        logger.info("Checkpoint already at epoch %d >= --epochs %d. Nothing to do.",
                    start_epoch - 1, args.epochs)
        return

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        # Kervadec boundary-loss RAMP: linearly grow the effective weight from 0
        # (epoch 1) to lambda_boundary over --boundary_ramp_epochs, then hold. The
        # criterion is shared by the train + val passes, so both use this epoch's
        # weight (val total stays consistent). No-op when boundary is OFF. Set on
        # the criterion BEFORE run_epoch so this epoch's batches see the new weight.
        if args.lambda_boundary > 0:
            ramp = min(epoch / max(args.boundary_ramp_epochs, 1), 1.0)
            criterion.boundary_weight = args.lambda_boundary * ramp

        tr, tr_diverged = run_epoch(model, train_loader, device, criterion,
                                    optimizer, scaler, amp_enabled, args.grad_clip,
                                    nan_patience=args.nan_patience,
                                    nan_fraction_threshold=args.nan_fraction_threshold,
                                    sam=sam,
                                    boundary_criterion=boundary_criterion,
                                    boundary_dist_mode=args.boundary_dist_mode,
                                    lambda_boundary_dist=args.lambda_boundary_dist)

        # ABORT on training divergence BEFORE validating or saving: never waste
        # time validating a NaN model, and never overwrite the last good checkpoint
        # with NaN weights. The pre-divergence best_model_phaseN.pth is preserved.
        if tr_diverged:
            logger.critical("FATAL: training DIVERGED to NaN at epoch %d — halting. "
                            "Last good checkpoint preserved (%s); NOT saving this "
                            "epoch's NaN state. Check loss stability / lower the LR "
                            "or lambda_gat/lambda_ffl.", epoch, best_path)
            break

        va, _va_diverged = run_epoch(model, val_loader, device, criterion,
                                     optimizer=None, scaler=scaler,
                                     amp_enabled=amp_enabled, grad_clip=args.grad_clip,
                                     nan_patience=args.nan_patience,
                                     boundary_criterion=boundary_criterion,
                                     boundary_dist_mode=args.boundary_dist_mode,
                                     lambda_boundary_dist=args.lambda_boundary_dist)

        # Optional SECOND, NOISY val pass (deterministic whole-image speckle) so
        # FiLM is exercised on val. Read-only (optimizer=None); best-model selection
        # below still uses the CLEAN pass (va). None unless --val_speckle deterministic.
        va_noisy = None
        if val_noisy_loader is not None:
            va_noisy, _ = run_epoch(model, val_noisy_loader, device, criterion,
                                    optimizer=None, scaler=scaler,
                                    amp_enabled=amp_enabled, grad_clip=args.grad_clip,
                                    nan_patience=args.nan_patience,
                                    boundary_criterion=boundary_criterion,
                                    boundary_dist_mode=args.boundary_dist_mode,
                                    lambda_boundary_dist=args.lambda_boundary_dist)
        val_dice_clean = va["dice_score"] if va_noisy is not None else None
        val_dice_noisy = va_noisy["dice_score"] if va_noisy is not None else None

        if args.ablation_phase == 1:
            # Phase 1: only the segmentation metrics are meaningful.
            logger.info(
                "Epoch %d/%d | Phase 1 (Seg-only) | train seg=%.5f "
                "(bce=%.4f dice=%.4f) | val seg=%.5f "
                "(bce=%.4f dice=%.4f dice_score=%.4f)",
                epoch, args.epochs, tr["total_loss"], tr["bce"], tr["dice"],
                va["total_loss"], va["bce"], va["dice"], va["dice_score"])
        else:
            logger.info(
                "Epoch %d/%d | Phase %d | train total=%.5f (seg=%.4f zncc=%.4f "
                "gat=%.4f ffl=%.4f) | val total=%.5f (bce=%.4f dice=%.4f "
                "dice_score=%.4f dc_offset=%.4f)",
                epoch, args.epochs, args.ablation_phase, tr["total_loss"],
                tr["seg_total"], tr["zncc"], tr["gat"], tr["ffl"],
                va["total_loss"], va["bce"], va["dice"], va["dice_score"],
                va.get("dc_offset", float("nan")))

        # Extra line for the deterministic NOISY val pass: shows FiLM is exercised
        # on val. For phases 4/5 the noisy injected sigma = sqrt(0.08) ~= 0.283 (vs
        # ~0 on clean val), so predicted_bias is supervised against a real target.
        if va_noisy is not None:
            logger.info(
                "         | val(NOISY det-speckle) dice=%.4f | clean dice=%.4f"
                "%s",
                va_noisy["dice_score"], va["dice_score"],
                ("" if "predicted_bias_mean" not in va_noisy else
                 " | predicted_bias=%.4f injected_sigma=%.4f" % (
                     va_noisy["predicted_bias_mean"],
                     va_noisy["injected_sigma_mean"])))

        # Learned loss-weighting diagnostic (only when active): log the EXACT
        # end-of-epoch weights exp(-log_var) per task so their evolution is visible
        # — the whole point of the scheme (does it down-weight denoising? up-weight
        # it under noise?). Reads the criterion parameters directly (not a batch
        # mean), so it is the network's decided weight at this epoch boundary.
        learned_w = criterion.learned_weights()
        if learned_w is not None:
            logger.info("         | learned loss weights exp(-log_var) [Kendall-Gal]: "
                        "%s | seg=%.4g (fixed reference)",
                        " ".join("%s=%.4f" % (k, v) for k, v in learned_w.items()),
                        args.lambda_seg)
            # With normalization on, also log the running-EMA loss scales and the
            # warmup status so the interpretable-weight readout is fully traceable
            # (raw task losses are already in the train line above).
            ema = criterion.ema_scales_dict()
            if ema is not None:
                step = int(criterion.warmup_counter)
                logger.info("         | EMA loss scales: %s | warmup %d/%d (%s)",
                            " ".join("%s=%.3g" % (k, v) for k, v in ema.items()),
                            step, criterion.warmup_steps,
                            "warming up -> fixed unit weights"
                            if step <= criterion.warmup_steps
                            else "learned weights active")

        # Noise-adaptive conditioning diagnostic (the scientific readout, only when
        # the flag is active): log the LEARNED a,b and the noise<->scale relation.
        # The SIGN of `a` is the finding — a>0 means the model applies a LARGER
        # cross-decoder coupling scale to noisier images (denoise more where noisy);
        # the hi/lo split reports the mean applied scale for the noisier vs cleaner
        # half of each batch, and their gap should track sign(a).
        if getattr(model, "noise_adaptive_conditioning", False):
            a_val = float(model.noise_conditioner.a.detach())
            b_val = float(model.noise_conditioner.b.detach())
            sig = getattr(model, "noise_signal", "predicted")
            # Whether the noise SIGNAL itself varies (hi vs lo bin means) — the whole
            # premise of the 'intrinsic' fix: if AMD/normal (or hi/lo) genuinely
            # differ, this gap is non-zero and the conditioner has something to act on.
            noise_hi = tr.get("cond_noise_hi", float("nan"))
            noise_lo = tr.get("cond_noise_lo", float("nan"))
            logger.info(
                "         | noise-adaptive cond [signal=%s]: a=%+.4f b=%+.4f "
                "(sign(a)>0 => MORE denoise coupling on noisier images) | train "
                "scale hi(noisy)=%.4f lo(clean)=%.4f (hi-lo=%+.4f) | %s hi=%.4f "
                "lo=%.4f (gap=%+.4f) mean=%.4f",
                sig, a_val, b_val,
                tr.get("cond_scale_hi", float("nan")),
                tr.get("cond_scale_lo", float("nan")),
                tr.get("cond_scale_hi", float("nan")) - tr.get("cond_scale_lo", float("nan")),
                "intrinsic" if sig == "intrinsic" else "pred-noise",
                noise_hi, noise_lo, noise_hi - noise_lo,
                tr.get("cond_noise_mean", float("nan")))

        # Boundary-loss ramp diagnostic (only when the term is active): show the
        # current ramped weight and the raw train/val boundary values so the
        # rebalancing schedule and its effect are visible in the log.
        if args.lambda_boundary > 0:
            logger.info("         | boundary(Kervadec) weight=%.4g (ramp %d/%d, "
                        "max=%g) | train=%.4f val=%.4f",
                        criterion.boundary_weight, epoch,
                        args.boundary_ramp_epochs, args.lambda_boundary,
                        tr.get("boundary", float("nan")),
                        va.get("boundary", float("nan")))

        # Boundary-DISTRIBUTION head diagnostic (the scientific readout): per-
        # boundary placement error (px) and predicted uncertainty (std). The
        # UPPER (choroid-start) boundary is expected to be EASIER than the LOWER
        # (choroid-sclera interface, the genuinely ambiguous one) -- lower error
        # AND lower uncertainty on upper vs. lower is the first evidence the
        # uncertainty is MEANINGFUL rather than degenerate/uninformative.
        if boundary_criterion is not None:
            mae_u = va.get("boundary_dist_mae_upper", float("nan"))
            mae_l = va.get("boundary_dist_mae_lower", float("nan"))
            std_u = va.get("boundary_dist_std_upper", float("nan"))
            std_l = va.get("boundary_dist_std_lower", float("nan"))
            logger.info(
                "         | boundary-dist [mode=%s]: NLL train=%.4f val=%.4f | "
                "MAE(px) upper=%.3f lower=%.3f (lower-upper=%+.3f) | "
                "std upper=%.4f lower=%.4f (lower-upper=%+.4f) | "
                "rasterized Dice=%.4f",
                args.boundary_dist_mode,
                tr.get("boundary_dist_nll", float("nan")),
                va.get("boundary_dist_nll", float("nan")),
                mae_u, mae_l, mae_l - mae_u,
                std_u, std_l, std_l - std_u,
                va.get("boundary_dist_dice", float("nan")))
            # Spatial Variance Penalty (Conformal-Aware Training) -- logged
            # alongside NLL EVERY epoch regardless of --lambda_spread (even at
            # 0.0/OFF) so interval tightness is trackable independent of
            # whether the penalty is actively shaping the loss.
            logger.info(
                "         | boundary-dist variance_penalty (px^2): "
                "train=%.4f val=%.4f | lambda_spread=%g",
                tr.get("boundary_dist_variance_penalty", float("nan")),
                va.get("boundary_dist_variance_penalty", float("nan")),
                args.lambda_spread)

        # Publication-ready per-epoch metrics CSV (alongside console + log file).
        # (Per-epoch non-finite conditions are already logged ONCE inside run_epoch;
        # a full training divergence has already aborted above.)
        append_epoch_metrics(args.checkpoint_dir, args.ablation_phase, epoch, tr, va,
                             val_dice_clean=val_dice_clean,
                             val_dice_noisy=val_dice_noisy)

        # Best-checkpoint selection driven by --best_metric (default 'dice'):
        #   dice       -> HIGHEST val choroid Dice (the reported segmentation metric;
        #                 for phases 2-5 this can differ from lowest total loss).
        #   total_loss -> LOWEST val total loss (legacy behavior).
        # Only the SELECTED epoch changes — the loss math/numerics are untouched.
        # A NON-FINITE metric is NEVER "better" (guard with math.isfinite), so a NaN
        # epoch can never overwrite best_model_phaseN.pth with corrupted weights.
        cur_metric = (va["dice_score"] if args.best_metric == "dice"
                      else va["total_loss"])
        is_best = math.isfinite(cur_metric) and (
            cur_metric > best_val if args.best_metric == "dice"
            else cur_metric < best_val)
        if is_best:
            best_val = cur_metric

        # HARD SAVE GUARD: NEVER persist non-finite weights — that is exactly what
        # produced the corrupted checkpoint_phaseN.pth that later resumed from NaN.
        # Applies to BOTH the periodic checkpoint AND best_model. (We already abort
        # above on a non-finite training epoch; this is the belt-and-suspenders that
        # guarantees no NaN checkpoint is ever written even if that path is missed.)
        model_ok = weights_finite(model)
        if model_ok:
            # Save the resumable checkpoint at the end of EVERY (finite) epoch.
            save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, args, best_val,
                            criterion=criterion)
        else:
            logger.warning("Model weights NON-FINITE at epoch %d — NOT writing "
                           "checkpoint_phase%d.pth (last good checkpoint preserved).",
                           epoch, args.ablation_phase)

        # Track the best validation model separately — requires a FINITE improved
        # metric (guarded above via math.isfinite) AND finite weights.
        if is_best and model_ok:
            save_checkpoint(best_path, epoch, model, optimizer, scaler, args, best_val,
                            criterion=criterion)
            logger.info("New best val %s=%.5f -> saved %s",
                        args.best_metric, best_val, best_path)

    logger.info("Training complete. Best val %s=%.5f", args.best_metric, best_val)


if __name__ == "__main__":
    main()
