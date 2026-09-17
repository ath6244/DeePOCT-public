"""joint_taskaware_model.py — JOINT / TASK-AWARE denoiser + segmenter.

MOTIVATION (the capstone result this exists to answer)
------------------------------------------------------
A SUPERVISED denoiser that BEATS BM3D/NLM on PSNR/SSIM/EPI *significantly DEGRADES*
choroid segmentation (all-class Dice -0.013, p=0.000, see
denoise_seg_supervised_experiment.py) — because image-quality-optimal denoising
strips the speckle TEXTURE the segmenter uses as a boundary cue. So "better
denoising" and "better segmentation" are ANTI-correlated here.

THE FIX (this file): stop optimizing the denoiser for image quality. Put a denoiser
FRONT-END in front of the existing segmenter and train the whole thing END-TO-END so
the denoiser's ONLY objective is downstream SEGMENTATION accuracy:

    x --> denoiser --> x_hat --> segmenter --> mask
    L = L_seg(segmenter(denoiser(x)), gt_mask)   [+ optional lambda_denoise * L_recon]

With ``lambda_denoise = 0`` (the MAIN experiment) the denoiser has NO image-quality
target at all — it is supervised PURELY by segmentation loss. This is the novel bit:
a denoiser learned entirely to serve segmentation, on data (SDM) that has NO clean
references. It may not learn to "denoise" at all — it may learn edge enhancement,
texture amplification, or something else that helps the segmenter. That is exactly
what the experiment script inspects (panels).

ARCHITECTURE
------------
* Front-end: ``DenoiseUNet`` (train_supervised_denoiser.py) — a residual U-Net with a
  ZERO-INIT tail, so it is the EXACT IDENTITY at initialization. Consequences:
    - at init ``denoiser(x) == x`` bit-for-bit, so the joint model's segmentation
      logits at init are IDENTICAL to the plain segmenter on the RAW input (arm A) —
      the experiment starts from the baseline and can only depart from it by learning
      (verified in the smoke test). This is the honest "does making it task-aware help
      vs. raw" starting point.
    - it is single-channel; the 2.5D input is [B,3,H,W], so the denoiser is applied
      PER-CHANNEL with SHARED weights ([B*3,1,H,W] -> denoise -> [B,3,H,W]). Each of
      the 3 adjacent B-scans is denoised by the same net, exactly like a per-frame
      denoiser would be.
* Back-end: the EXISTING segmentation network — ``DeepOCTUltra(ablation_phase=1)``,
  the same seg-only model arm A/B train. Reused unmodified.

CONTRACT: ``forward(x) -> (denoised_center [B,1,H,W], seg_logits [B,1,H,W], None)``.
The 3-tuple mirrors DeepOCTUltra's multi-task return so the same scoring code
(``out[1]`` is the seg logits) works unchanged. ``denoised_center`` is the denoised
CENTER frame (the frame the mask is defined on), exposed for panels / the optional
reconstruction regularizer.

Nothing in this file modifies model.py / train.py / loss.py — it only COMPOSES them,
so the existing arms are byte-identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import DeepOCTUltra
from .train_supervised_denoiser import DenoiseUNet


class JointTaskAwareDenoiseSeg(nn.Module):
    """Denoiser front-end + phase-1 segmenter, trainable end-to-end.

    The denoiser is optimized (by the training script) SOLELY through the
    segmentation loss when ``lambda_denoise == 0`` — it has no image-quality target.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32,
                 denoiser_base: int = 32, denoiser_levels: int = 3,
                 denoiser_blocks: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        # Residual, zero-tail denoiser -> EXACT identity at init (see module docstring).
        # Single-channel; applied per-frame with shared weights over the 2.5D stack.
        self.denoiser = DenoiseUNet(base=denoiser_base, in_ch=1,
                                    levels=denoiser_levels, blocks=denoiser_blocks)
        # The EXISTING seg-only network (arm A/B's model), reused unmodified.
        self.segmenter = DeepOCTUltra(in_channels=in_channels,
                                      base_filters=base_filters, ablation_phase=1)

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel shared-weight denoise of the 2.5D stack: [B,C,H,W] -> [B,C,H,W]."""
        b, c, h, w = x.shape
        xr = x.reshape(b * c, 1, h, w)
        den = self.denoiser(xr)
        return den.reshape(b, c, h, w)

    def forward(self, x: torch.Tensor, intrinsic_noise: torch.Tensor = None):
        # ``intrinsic_noise`` is accepted (and ignored) only so this model is a
        # drop-in for the DeepOCTUltra(x, intrinsic_noise=...) call signature.
        den = self.denoise(x)                       # [B,C,H,W] task-aware transform
        _none, seg_logits, _none2 = self.segmenter(den)   # phase-1 -> (None, logits, None)
        center = self.in_channels // 2
        denoised_center = den[:, center:center + 1]       # [B,1,H,W]
        return denoised_center, seg_logits, None
