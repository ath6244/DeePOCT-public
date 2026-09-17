"""Standalone inference / visual-inspection script for DeepOCT-Ultra.

Loads a trained checkpoint, runs the segmentation head on a single raw OCT
B-scan, and saves a side-by-side figure (raw image | image + red mask overlay)
so segmentation progress can be eyeballed during the ablation study.

IMPORTANT — input contract (matches ``oimhs_dataset.OIMHSDataset``):
  The model is a 2.5D network: ``DeepOCTUltra`` has ``in_channels=3`` and was
  trained on a 3-channel stack ``(B_{i-1}, B_i, B_{i+1})`` of LOG-transformed
  grayscale frames. So a single image is NOT fed as ``[1, 1, H, W]`` in raw
  [0, 1] space (that would either crash on the channel mismatch or, after a
  hack, produce meaningless output from the wrong input distribution).

  Instead we reproduce the dataset's edge-replication case exactly: the single
  grayscale frame is normalized to [0, 1], ``log(x + eps)``-transformed, and
  replicated across all 3 channels -> ``[1, 3, H, W]``. This is precisely what
  the dataset yields for a 1-slice volume (prev == center == next).

Imports of PIL and matplotlib are deferred into the functions that need them so
the model/inference path stays importable and testable on machines without the
visualization stack installed.
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch

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


# Match OIMHSDataset._LOG_EPS so inference preprocessing is bit-for-bit aligned
# with how the network was trained.
_LOG_EPS = 1e-8
_INPUT_SIZE = 512
_MASK_THRESHOLD = 0.5
_OVERLAY_ALPHA = 0.45  # semi-transparent bright red


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, ablation_phase: int,
               device: torch.device) -> DeepOCTUltra:
    """Build ``DeepOCTUltra`` for ``ablation_phase`` and load checkpoint weights.

    The checkpoint written by ``train.save_checkpoint`` is a dict with a
    ``model_state_dict`` entry plus the original ``args``; we read the saved
    architecture flags (base_filters / unbiasing / wavelets / cross-gating) so
    the model is reconstructed identically before loading. A bare state_dict is
    also accepted.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path!r}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        saved_args = ckpt.get("args", {}) or {}
    else:
        # Bare state_dict (no training metadata available).
        state_dict = ckpt
        saved_args = {}

    saved_phase = saved_args.get("ablation_phase")
    if saved_phase is not None and saved_phase != ablation_phase:
        print(f"[WARN] --ablation_phase {ablation_phase} differs from the "
              f"checkpoint's phase {saved_phase}. Using {ablation_phase}; weight "
              f"loading will fail if the architectures are incompatible.")

    model = DeepOCTUltra(
        in_channels=3,
        base_filters=int(saved_args.get("base_filters", 32)),
        use_unbiasing=bool(saved_args.get("use_unbiasing", False)),
        use_wavelets=bool(saved_args.get("use_wavelets", False)),
        use_cross_gating=bool(saved_args.get("use_cross_gating", False)),
        ablation_phase=ablation_phase,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def load_grayscale(image_path: str, size: int = _INPUT_SIZE,
                   split_oimhs: bool = False) -> np.ndarray:
    """Load an OCT image as a resized grayscale array in [0, 1], shape [H, W].

    By default the whole input file is treated as the OCT B-scan. With
    ``split_oimhs=True`` the input is assumed to be a concatenated
    ``[OCT | Mask]`` OIMHS pair: it is sliced on the width midpoint and ONLY the
    left half (the raw OCT scan) is kept — matching ``OIMHSDataset._open_and_split``
    (split BEFORE the grayscale conversion / resize).
    """
    from PIL import Image  # deferred: keeps the model path import-light

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path!r}")

    with Image.open(image_path) as image:
        if split_oimhs:
            width, height = image.size
            mid = width // 2
            image = image.crop((0, 0, mid, height))  # left half = OCT scan
        gray = image.convert("L").resize((size, size), resample=Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32) / 255.0


def to_model_input(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """Grayscale [H, W] in [0, 1] -> log-domain 2.5D tensor [1, 3, H, W].

    Replicates the single frame across the 3 channels (the dataset's
    edge-replication / single-slice case) and applies the same log-transform the
    network was trained on.
    """
    log_gray = np.log(np.clip(gray, 0.0, 1.0) + _LOG_EPS).astype(np.float32)
    stack = np.stack([log_gray, log_gray, log_gray], axis=0)  # [3, H, W]
    tensor = torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0)  # [1,3,H,W]
    return tensor.to(device)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_mask(model: DeepOCTUltra, input_tensor: torch.Tensor) -> np.ndarray:
    """Run the forward pass and return a binary choroid mask [H, W] (uint8 0/1).

    ``seg_logits`` is always the 2nd element of the model output (both the
    2-tuple and 3-tuple forms), including Phase 1's ``(None, seg_logits, None)``.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
    seg_logits = outputs[1]
    probs = torch.sigmoid(seg_logits)
    mask = (probs > _MASK_THRESHOLD).squeeze().to(torch.uint8).cpu().numpy()
    return mask


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_comparison(gray: np.ndarray, mask: np.ndarray, output_path: str) -> None:
    """Save a side-by-side figure: raw grayscale | grayscale + red mask overlay.

    Uses the non-interactive 'Agg' backend so it runs head-less on a server.
    """
    import matplotlib
    matplotlib.use("Agg")  # must precede pyplot import; no GUI / display needed
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # Semi-transparent bright-red overlay, opaque (alpha) only where mask == 1.
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask.astype(bool)] = (1.0, 0.0, 0.0, _OVERLAY_ALPHA)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    ax_left.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    ax_left.set_title("Raw OCT")
    ax_left.axis("off")

    ax_right.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    ax_right.imshow(overlay)
    ax_right.set_title("Predicted Choroid Overlay")
    ax_right.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepOCT-Ultra inference: segment one OCT scan and save a "
                    "side-by-side raw|overlay figure.")
    parser.add_argument("--image_path", required=True,
                        help="Path to a raw OCT B-scan image.")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to a trained checkpoint (e.g. best_model_phase1.pth).")
    parser.add_argument("--output_path", required=True,
                        help="Where to write the comparison figure (e.g. out.png).")
    parser.add_argument("--ablation_phase", type=int, choices=[1, 2, 3, 4],
                        default=1,
                        help="Architecture phase the checkpoint was trained under.")
    parser.add_argument("--split-oimhs", action="store_true",
                        help="Treat the input as a concatenated [OCT | Mask] "
                             "OIMHS pair and keep only the left (OCT) half for "
                             "both inference and visualization.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint_path, args.ablation_phase, device)
    gray = load_grayscale(args.image_path, _INPUT_SIZE, split_oimhs=args.split_oimhs)
    input_tensor = to_model_input(gray, device)
    mask = predict_mask(model, input_tensor)

    coverage = 100.0 * float(mask.mean())
    print(f"Predicted choroid coverage: {coverage:.2f}% of pixels")

    save_comparison(gray, mask, args.output_path)
    print(f"Saved comparison figure -> {args.output_path}")


if __name__ == "__main__":
    main()
