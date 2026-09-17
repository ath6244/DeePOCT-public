"""DeepOCT-Ultra composite loss engine — Phase 2, Step 6.

Modular, physics-grounded losses for simultaneous OCT denoising + choroid
segmentation. Each component is its own ``nn.Module`` and is exercised by
``test_loss_smoke.py``; the master ``DeepOCTLoss`` wraps them with configurable
lambda weights and returns ``(total_loss, log_dict)``.

Why these losses (and NOT plain MSE):
  * The dynamic-unbiasing branch subtracts a learned scalar from the
    reconstruction, so the *global mean* of the output is deliberately shifted.
    MSE would explode on that shift; ZNCC is mean-invariant and scores only
    structural agreement.
  * GAT-variance preserves the speckle "grain" so the network does not produce
    the over-smoothed "plastic" deep-learning look.
  * Focal Frequency Loss penalises loss of high-frequency biological signatures
    in the 2D FFT domain.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZNCCLoss(nn.Module):
    """Zero-Mean Normalized Cross-Correlation, returned as ``1 - ZNCC``.

    Per image: mean-centre output and target, then divide the dot product of the
    centred tensors by the product of their L2 norms. L2 norms (== std * sqrt(N))
    are used rather than raw stds so the coefficient is properly bounded to
    [-1, 1]; the loss ``1 - ZNCC`` therefore lives in [0, 2]. Being mean-centred,
    it is invariant to the global mean shift introduced by dynamic unbiasing.

    BACKWARD STABILITY (the epsilon lives INSIDE the sqrt, not after it): a
    near-flat / constant patch has ``out ≈ 0`` so ``sum(out**2) → 0``. The old
    ``out.norm() + eps`` form added eps only AFTER the sqrt, which fixed the
    FORWARD divide but left the sqrt's own backward, ``d/ds sqrt(s) = 1/(2*sqrt(s))``,
    unbounded as ``s → 0``. Even where the norm-backward is guarded at exactly 0,
    the divide's numerator-path gradient is bounded only by ``1/eps`` (via
    Cauchy-Schwarz ``|num| ≤ norm_o*norm_t``), i.e. up to 1e8 for eps=1e-8 — which
    overflows the fp16 ceiling (65504) once it flows back out of this fp32 block
    into the AMP model, producing the observed "DivBackward0 returned nan values".
    Computing ``sqrt(sum_sq.clamp_min(eps))`` instead bounds the sqrt input (and
    hence its gradient) away from zero, capping the numerator-path gradient at
    ``1/(sqrt(eps)*norm) ≈ 1e4`` — safely inside fp16 range. The final coefficient
    is clamped to its mathematical range [-1, 1] to guard against tiny numerical
    overshoot.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = output.shape[0]
        out = output.reshape(b, -1)
        tgt = target.reshape(b, -1)

        out = out - out.mean(dim=1, keepdim=True)
        tgt = tgt - tgt.mean(dim=1, keepdim=True)

        numerator = (out * tgt).sum(dim=1)
        # eps INSIDE the sqrt (clamp the sum-of-squares BEFORE sqrt) so the sqrt
        # backward 1/(2*sqrt(x)) cannot explode on near-flat patches. For healthy
        # patches sum_sq >> eps, so this is numerically identical to the old form.
        out_norm = torch.sqrt((out * out).sum(dim=1).clamp_min(self.eps))
        tgt_norm = torch.sqrt((tgt * tgt).sum(dim=1).clamp_min(self.eps))
        zncc = numerator / (out_norm * tgt_norm)
        zncc = zncc.clamp(-1.0, 1.0)                    # [B], mathematically in [-1, 1]
        return (1.0 - zncc).mean()


class GATVarianceLoss(nn.Module):
    """Variance-stabilising "tissue grain" loss.

    Applies the Generalized Anscombe Transform ``f(x) = 2*sqrt(x + 3/8)`` to
    stabilise the (signal-dependent) speckle variance, computes a local variance
    map over a sliding window via ``E[X^2] - E[X]^2``, and returns the L1
    distance between the output's and target's local-variance maps.

    The argument to ``sqrt`` is clamped to >= 0 before adding a tiny epsilon:
    the denoised output can go negative (especially after bias subtraction), and
    an unclamped ``sqrt`` of a negative would emit NaNs in both the forward and
    backward pass.
    """

    def __init__(self, window: int = 5, eps: float = 1e-8) -> None:
        super().__init__()
        self.window = window
        self.padding = window // 2
        self.eps = eps

    def _gat(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sqrt(torch.clamp(x + 3.0 / 8.0, min=0.0) + self.eps)

    def _local_variance(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x, self.window, stride=1, padding=self.padding)
        mean_sq = F.avg_pool2d(x * x, self.window, stride=1, padding=self.padding)
        return torch.clamp(mean_sq - mean * mean, min=0.0)  # variance is >= 0

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var_out = self._local_variance(self._gat(output))
        var_tgt = self._local_variance(self._gat(target))
        return F.l1_loss(var_out, var_tgt)


class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities, returned as ``1 - Dice``."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = logits.shape[0]
        probs = torch.sigmoid(logits).reshape(b, -1)
        tgt = target.reshape(b, -1)
        intersection = (probs * tgt).sum(dim=1)
        denom = probs.sum(dim=1) + tgt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return (1.0 - dice).mean()


class SegLoss(nn.Module):
    """Segmentation loss: BCEWithLogits + soft Dice (``Total_Seg = BCE + Dice``).

    Returns a dict of the SEPARATED components (all live tensors so
    ``total_seg_loss`` stays differentiable) for publication-ready metrics:
      * ``total_seg_loss`` = BCE + Dice loss (the value to backprop),
      * ``bce_loss``       = BCEWithLogits,
      * ``dice_loss``      = soft Dice loss (``1 - Dice``),
      * ``dice_score``     = ``1 - dice_loss`` (the Dice coefficient itself).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(logits, target)
        total_seg_loss = bce_loss + dice_loss
        dice_score = 1.0 - dice_loss
        return {
            "total_seg_loss": total_seg_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "dice_score": dice_score,
        }


class BoundaryLoss(nn.Module):
    """Kervadec et al. boundary/surface loss for the choroid mask.

    (Kervadec et al., "Boundary loss for highly unbalanced segmentation",
    MIDL 2019 / MedIA 2021.) A distance-weighted regional integral that sharpens
    BOUNDARY LOCALIZATION (HD95/ASD) where a region overlap loss (Dice) is nearly
    saturated — exactly the fuzzy choroid-sclera interface where our errors
    concentrate.

    Definition (binary, single foreground channel):
        L_B = mean( sigmoid(logits) * phi_GT )
    where ``phi_GT`` is the SIGNED distance transform of the GT mask boundary:
    NEGATIVE inside the choroid region, POSITIVE outside, ~0 on the boundary.
    (The binary sigmoid probability is the two-class softmax foreground prob.)

    Effect: predicted probability mass placed FAR OUTSIDE the true region (large
    positive phi) is penalized; mass INSIDE (negative phi) is rewarded, so the
    0.5 decision level is pulled onto the true boundary. Minimizing this drives
    the predicted surface toward the GT surface — it can go NEGATIVE (the region
    loss keeps the total bounded), which is expected and matches the reference.

    The SDF is a NON-DIFFERENTIABLE constant target: it is computed on-the-fly
    from the GT mask with ``scipy.ndimage.distance_transform_edt`` (detached, no
    grad); gradient flows ONLY through the prediction. Requires scipy (already a
    project dependency; also used by evaluate.py's HD95/ASD).

    MAGNITUDE NOTE: phi is in PIXELS, so its scale grows with image resolution
    (at 512x512 the interior distance reaches the low hundreds). The composite
    weight ``lambda_boundary`` must therefore be SMALL relative to the Dice/BCE
    terms and is resolution-dependent — see ``DeepOCTLoss`` and train.py.
    """

    def __init__(self) -> None:
        super().__init__()
        # Lazy import so the boundary path is the ONLY place that requires scipy;
        # a run with the boundary term OFF never imports it (byte-identical path).
        from scipy.ndimage import distance_transform_edt
        self._edt = distance_transform_edt

    @torch.no_grad()
    def _signed_distance(self, mask: torch.Tensor) -> torch.Tensor:
        """Per-sample signed distance field of the GT boundary, [B,1,H,W].

        Convention: +dist to the boundary OUTSIDE the region, -dist INSIDE. A mask
        with NO boundary (all-background OR all-foreground) has an undefined
        surface, so its SDF is left at ZERO -> that sample contributes nothing to
        the boundary loss (safe no-op, never a NaN).
        """
        m = mask.detach().cpu().numpy() > 0.5           # [B,1,H,W] bool
        sdf = np.zeros(m.shape, dtype=np.float32)
        for b in range(m.shape[0]):
            pos = m[b, 0]
            neg = ~pos
            if pos.any() and neg.any():
                # Outside: distance INTO the background; inside: negative distance.
                sdf[b, 0] = (self._edt(neg) - self._edt(pos)).astype(np.float32)
        return torch.from_numpy(sdf)

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        phi = self._signed_distance(mask).to(device=logits.device, dtype=logits.dtype)
        probs = torch.sigmoid(logits)
        return (probs * phi).mean()


def derive_column_boundaries(mask: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-column upper/lower choroid boundary depth from a filled binary mask.

    ``mask`` is [B,1,H,W] (the existing filled-band choroid mask). For each
    column the choroid mask is a single filled band (project diagnostics: 93.7%
    of predictions are already a clean single band; topology post-processing
    does not help), so the upper boundary is simply the FIRST foreground row and
    the lower boundary the LAST foreground row. Columns with NO foreground
    (choroid absent, e.g. off the tissue edge) are flagged False in ``valid`` and
    MUST be masked out of any loss/metric that consumes the returned depths --
    their z values are meaningless placeholders (index 0), not real boundaries.

    Returns ``(z_upper[B,W], z_lower[B,W], valid[B,W] bool)``; z values are ROW
    INDICES (pixel depth) as float32, on ``mask``'s device.
    """
    b, _, h, w = mask.shape
    fg = (mask[:, 0] > 0.5)                                 # [B,H,W] bool
    valid = fg.any(dim=1)                                    # [B,W]
    fg_f = fg.float()
    # argmax on a 0/1 tensor returns the FIRST occurrence of the max value, i.e.
    # the first foreground row (an all-background column returns 0, a
    # meaningless placeholder masked out via `valid` downstream).
    z_upper = torch.argmax(fg_f, dim=1).float()               # [B,W]
    z_lower = (h - 1) - torch.argmax(fg_f.flip(dims=[1]), dim=1).float()
    return z_upper, z_lower, valid


def gaussian_soft_target(z_gt: torch.Tensor, h: int, sigma: float) -> torch.Tensor:
    """Gaussian soft-label distribution over H depth bins centered at ``z_gt``.

    ``z_gt`` is [B,W] (a per-column GT depth, e.g. from
    ``derive_column_boundaries``). Returns [B,H,W], normalized to sum to 1 over
    the H axis (dim=1) for every column -- a valid soft target for a per-column
    softmax NLL.

    WHY GAUSSIAN (not a hard one-hot): a hard one-hot target under
    cross-entropy has NO finite-loss minimum short of the predicted softmax
    collapsing to a delta (the logit gap must diverge for the loss to reach
    zero) -- exactly the OVERCONFIDENCE this head exists to avoid (the goal is a
    calibrated per-column uncertainty, not a nudged hard line). A Gaussian soft
    target instead has a well-defined FINITE-loss optimum: the KL-minimizing
    softmax is exactly the Gaussian target itself, so the network's optimal
    solution is a genuinely calibrated spread whose WIDTH is controlled by
    ``sigma`` -- not a still-overconfident one-hot merely softened by a constant
    smoothing factor. This mirrors standard heatmap-regression practice (e.g.
    Nibali et al., "Numerical Coordinate Regression with Convolutional Neural
    Networks", 2018) for precisely this reason.
    """
    device, dtype = z_gt.device, z_gt.dtype
    z_axis = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)   # [1,H,1]
    center = z_gt.unsqueeze(1)                                            # [B,1,W]
    unnorm = torch.exp(-0.5 * ((z_axis - center) / sigma) ** 2)           # [B,H,W]
    return unnorm / unnorm.sum(dim=1, keepdim=True).clamp_min(1e-8)


def boundary_distribution_to_depths(boundary_logits: torch.Tensor
                                    ) -> Tuple[torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor]:
    """(E_upper, std_upper, E_lower, std_lower), each [B,W], from raw
    ``boundary_logits`` [B,2,H,W] (model.BoundaryDistributionHead's output).

    E[z] (first moment of the per-column softmax over H) is the boundary depth
    ESTIMATE; std (sqrt of the second central moment) is the per-column
    UNCERTAINTY -- large where the network's depth distribution is spread out
    (ambiguous choroid-sclera interface), small where it is confidently peaked.
    """
    probs = F.softmax(boundary_logits, dim=2)                  # [B,2,H,W]
    h = boundary_logits.shape[2]
    z_axis = torch.arange(h, device=boundary_logits.device,
                          dtype=boundary_logits.dtype).view(1, h, 1)  # [1,H,1]
    e_upper = (probs[:, 0] * z_axis).sum(dim=1)                 # [B,W]
    e_lower = (probs[:, 1] * z_axis).sum(dim=1)                 # [B,W]
    var_upper = (probs[:, 0] * z_axis ** 2).sum(dim=1) - e_upper ** 2
    var_lower = (probs[:, 1] * z_axis ** 2).sum(dim=1) - e_lower ** 2
    std_upper = var_upper.clamp_min(0.0).sqrt()
    std_lower = var_lower.clamp_min(0.0).sqrt()
    return e_upper, std_upper, e_lower, std_lower


def rasterize_boundary_mask(e_upper: torch.Tensor, e_lower: torch.Tensor,
                            h: int) -> torch.Tensor:
    """Fill a binary mask [B,1,H,W] between the per-column predicted upper and
    lower boundary depths, so the SAME Dice/HD95/ASD metrics used for the
    existing mask model can be computed on this head's output for an
    apples-to-apples comparison. ``e_upper``/``e_lower`` are [B,W] (e.g. from
    ``boundary_distribution_to_depths``).

    A column where the predicted upper exceeds the lower (possible early in
    training, or at a genuinely ambiguous column) rasterizes to an EMPTY column
    for that A-scan -- NOT clamped/reordered, since silently forcing an order
    would hide a real miscalibration signal instead of surfacing it in Dice.
    """
    b, w = e_upper.shape
    rows = torch.arange(h, device=e_upper.device,
                        dtype=e_upper.dtype).view(1, h, 1)          # [1,H,1]
    upper = e_upper.round().unsqueeze(1)                             # [B,1,W]
    lower = e_lower.round().unsqueeze(1)                             # [B,1,W]
    filled = (rows >= upper) & (rows <= lower)                       # [B,H,W]
    return filled.unsqueeze(1).float()                                # [B,1,H,W]


# --------------------------------------------------------------------------- #
# SMOOTH-PATH (DP / Viterbi) boundary decode -- an ALTERNATIVE to the naive
# per-column E[z] point estimate (``boundary_distribution_to_depths``).
#
# DIAGNOSIS (measured on the trained boundary-distribution head): the naive
# expectation gives good Dice but a huge-variance HD95, because E[z] is a poor
# estimator whenever a column's distribution is BIMODAL (the mean falls in the
# low-probability VALLEY between two modes -- nowhere near either plausible
# boundary location) AND nothing constrains adjacent columns to agree -- the
# true choroid boundary is anatomically SMOOTH, but E[z] can jump wildly column
# to column. HD95 is outlier-driven (95th-percentile surface distance), so a
# handful of such columns dominate it even though Dice (an AREA metric, which
# barely notices a few 1px-wide spikes) looks fine.
#
# FIX: decode each boundary as the SMOOTH PATH z(x) across columns that
# MAXIMIZES sum_x log p(z(x)|x) subject to |z(x+1)-z(x)| <= max_jump -- the
# standard Chiu/Garvin-style graph/DP boundary tracer used in OCT layer
# segmentation, here restricted to a 1D BANDED Viterbi (each column's DP state
# can only transition from within +/- max_jump rows of the previous column) so
# it is both exact (a true global optimum under the banded constraint, not a
# greedy heuristic) and fast: the per-column transition is a sliding-window max
# over the H axis, vectorized via ``unfold`` (no python loop over H, only over
# the W columns, which is sequential by construction).
# --------------------------------------------------------------------------- #
def mode_decode_column_boundary(log_probs: torch.Tensor) -> torch.Tensor:
    """Per-column ARGMAX depth, [B,W], from a single boundary's [B,H,W]
    log-probability map. The cheap alternative to both E[z] and Viterbi: fixes
    the "mean falls in the valley" bimodality failure mode (argmax always lands
    ON a mode, never between two), but -- UNLIKE Viterbi -- has NO smoothness
    constraint, so it can still jump column-to-column when the two modes swap
    which one is locally highest.
    """
    return log_probs.argmax(dim=1)


def viterbi_decode_column_boundary(log_probs: torch.Tensor,
                                   max_jump: int) -> torch.Tensor:
    """Banded Viterbi smooth-path decode of ONE boundary, [B,W] LONG row
    indices, from its [B,H,W] log-probability map (e.g.
    ``F.log_softmax(boundary_logits[:, 0], dim=1)`` after moving the H axis to
    dim=1). Maximizes ``sum_x log_probs[z(x), x]`` subject to
    ``|z(x+1) - z(x)| <= max_jump`` -- see the module-level docstring above for
    the full motivation.

    Standard forward-DP / backward-backtrack Viterbi, banded to a window of
    ``2*max_jump+1`` states: the forward recursion's "max over allowed
    predecessors" is a sliding-window max over the H axis, computed for every
    depth simultaneously via ``unfold`` on a -inf-padded score vector (no
    python loop over H). The self-transition (``z(x+1)=z(x)``, offset 0) is
    ALWAYS inside the window regardless of ``max_jump``, so the window's finite
    center entry guarantees ``best_val`` is never -inf.
    """
    if max_jump < 0:
        raise ValueError(f"max_jump must be >= 0, got {max_jump}")
    B, H, W = log_probs.shape
    device = log_probs.device
    k = 2 * max_jump + 1
    dp = log_probs[:, :, 0].clone()                        # [B,H] best score ending at each depth
    backptr = torch.zeros(B, H, W, dtype=torch.long, device=device)
    # ``unfold``'s window is PER-ROW (window h covers padded[h : h+k]), so its
    # local argmax index is a displacement RELATIVE TO h, not an absolute depth
    # -- it must be recentered by "+ h" (then "- max_jump" for the padding
    # offset) to become the actual predecessor row. Omitting "+ h" silently
    # produces an absolute-index-shaped-but-wrong predecessor for every h != 0,
    # which breaks the max_jump guarantee without erroring (caught by
    # test_boundary_decode_smoke.py's max-jump assertion).
    depth_idx = torch.arange(H, device=device)                          # [H]
    for x in range(1, W):
        padded = F.pad(dp, (max_jump, max_jump), value=float("-inf"))  # [B,H+2*max_jump]
        windows = padded.unfold(1, k, 1)                                # [B,H,k]
        best_val, best_local = windows.max(dim=2)                       # [B,H]
        backptr[:, :, x] = (depth_idx.unsqueeze(0) + best_local - max_jump).clamp(0, H - 1)
        dp = log_probs[:, :, x] + best_val

    path = torch.zeros(B, W, dtype=torch.long, device=device)
    path[:, -1] = dp.argmax(dim=1)
    batch_idx = torch.arange(B, device=device)
    for x in range(W - 1, 0, -1):
        path[:, x - 1] = backptr[batch_idx, path[:, x], x]
    return path


def viterbi_decode_boundary_pair(boundary_logits: torch.Tensor, max_jump: int
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint smooth-path decode of BOTH boundaries with the anatomical
    ordering constraint ``z_upper(x) < z_lower(x)`` enforced BY CONSTRUCTION
    (zero crossings), from ``boundary_logits`` [B,2,H,W]. Returns
    ``(z_upper, z_lower)``, each [B,W] LONG row indices.

    A FULL joint DP over the pair (state = (z_upper, z_lower) together) has
    O(H^2) states per column -- 262144 states x 512 columns at H=512, an order
    of magnitude too slow for routine CPU eval. Instead: the UPPER boundary
    never depends on the lower one, so decode it FIRST via the plain per-
    boundary Viterbi; then decode the LOWER boundary via the SAME Viterbi but
    with its per-column log-probability MASKED to -inf at every depth <= the
    already-decoded ``z_upper(x)`` for that column. This keeps the exact same
    O(H*W) complexity as decoding independently, is EXACT for the lower
    boundary GIVEN the upper decode (a hard constraint, not a soft penalty),
    and guarantees ``z_lower(x) > z_upper(x)`` everywhere -- no post-hoc
    crossing repair is ever needed. ``z_upper`` is capped at ``H-2`` ONLY for
    the masking step (never in the returned value) so a column never loses
    every valid lower-boundary depth even in the degenerate case where the
    upper path is decoded all the way to the last row.
    """
    B, C, H, W = boundary_logits.shape
    assert C == 2, f"boundary_logits must have 2 channels, got {C}"
    log_probs = F.log_softmax(boundary_logits, dim=2)          # [B,2,H,W]

    z_upper = viterbi_decode_column_boundary(log_probs[:, 0], max_jump)

    lower_logp = log_probs[:, 1].clone()
    depth_idx = torch.arange(H, device=boundary_logits.device).view(1, H, 1)
    z_upper_for_mask = z_upper.clamp(max=H - 2).view(B, 1, W)
    forbidden = depth_idx <= z_upper_for_mask                   # [B,H,W]
    lower_logp = lower_logp.masked_fill(forbidden, float("-inf"))
    z_lower = viterbi_decode_column_boundary(lower_logp, max_jump)

    return z_upper, z_lower


def count_boundary_crossings(z_upper: torch.Tensor, z_lower: torch.Tensor
                             ) -> torch.Tensor:
    """Per-sample count of columns where ``z_upper(x) >= z_lower(x)`` -- an
    anatomically INVALID crossing. ``z_upper``/``z_lower`` are [B,W]. Used to
    quantify how often the naive independent decodes (expectation / mode)
    violate the ordering constraint that ``viterbi_decode_boundary_pair``
    enforces by construction (which should always report 0 crossings)."""
    return (z_upper >= z_lower).sum(dim=1)


def boundary_jump_stats(mask: torch.Tensor) -> Dict[str, float]:
    """Column-to-column ``|z(x+1)-z(x)|`` statistics of the GT boundary
    (derived from the filled mask via ``derive_column_boundaries``), pooled
    over BOTH boundaries and the whole batch. Only consecutive column pairs
    that are BOTH valid (choroid present) contribute. This is the DATA-DRIVEN
    justification for ``--boundary_decode_max_jump``: max_jump should cover the
    vast majority of the GT's actual smoothness (e.g. its p99), not be a guess.
    """
    z_upper, z_lower, valid = derive_column_boundaries(mask)
    jumps = []
    for z in (z_upper, z_lower):
        d = (z[:, 1:] - z[:, :-1]).abs()                       # [B,W-1]
        pair_valid = valid[:, 1:] & valid[:, :-1]
        jumps.append(d[pair_valid])
    all_jumps = torch.cat(jumps) if jumps else torch.zeros(0)
    if all_jumps.numel() == 0:
        return {"median": float("nan"), "p95": float("nan"), "p99": float("nan"),
                "max": float("nan"), "n": 0}
    aj = all_jumps.float()
    return {
        "median": float(aj.median()),
        "p95": float(torch.quantile(aj, 0.95)),
        "p99": float(torch.quantile(aj, 0.99)),
        "max": float(aj.max()),
        "n": int(aj.numel()),
    }


class BoundaryDistributionLoss(nn.Module):
    """Soft, distribution-aware loss for ``model.BoundaryDistributionHead``.

    ``boundary_logits`` is [B,2,H,W] raw logits (channel 0 = upper boundary,
    channel 1 = lower boundary). ``mask`` is the EXISTING filled-band choroid
    mask [B,1,H,W] the mask-based head already trains against; GT boundary
    depths are DERIVED from it via ``derive_column_boundaries`` -- no new labels
    are needed.

    Two terms, both averaged ONLY over columns where the choroid is present
    (``valid`` from ``derive_column_boundaries``; absent columns contribute 0):
      * ``nll``            -- soft cross-entropy between the predicted
        per-column softmax (over H) and a Gaussian soft target centered at the
        GT depth (see ``gaussian_soft_target`` for why NOT a hard one-hot). This
        is the PRIMARY term: it shapes the whole distribution, including its
        calibrated spread/uncertainty.
      * ``expectation_l1`` -- |E[z] - z_gt|, a direct L1 regression on the
        predicted MEAN depth. The NLL term alone calibrates the distribution's
        shape but only pulls its center indirectly (through the Gaussian
        target); this term pins the MEAN to the GT boundary directly so
        calibration (from NLL) and mean accuracy (from this term) are optimized
        somewhat independently.
    Upper and lower boundaries are combined with EQUAL weight: both are exactly
    the same kind of quantity (a depth in pixels), so there is no scale mismatch
    to correct for, unlike the heterogeneous seg/denoise composite in
    ``DeepOCTLoss``.
    """

    def __init__(self, sigma: float = 2.0, lambda_expectation: float = 0.1,
                 sharpness_weight: float = 0.0, lambda_spread: float = 0.0) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.lambda_expectation = float(lambda_expectation)
        # OPTION C (sharpness-aware training, default 0 = OFF = byte-identical): an
        # AUXILIARY penalty on the predicted per-column boundary WIDTH (mean predicted
        # std over valid columns), added to the total. It pushes the distribution to
        # be SHARP; the NLL term (which punishes a too-narrow distribution that misses
        # the truth) plus downstream conformal recalibration keep it from collapsing
        # to overconfidence. Keep SMALL. When 0, the term is SKIPPED entirely (not
        # computed-then-zeroed) so the loss is byte-identical to before.
        self.sharpness_weight = float(sharpness_weight)
        # Spatial Variance Penalty (Conformal-Aware Training), default 0 = OFF.
        # Same INTENT as sharpness_weight (discourage a wide predicted distribution)
        # but computed via the LITERAL second central moment sum(p*(d-mu)^2) rather
        # than sharpness_weight's E[X^2]-E[X]^2 std shortcut, and reported EVERY
        # step (see forward()) regardless of this weight so interval tightness can
        # be tracked even while the penalty itself is off (lambda_spread=0.0).
        self.lambda_spread = float(lambda_spread)

    def forward(self, boundary_logits: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = boundary_logits.shape
        assert c == 2, f"boundary_logits must have 2 channels, got {c}"

        z_upper, z_lower, valid = derive_column_boundaries(mask)
        z_upper = z_upper.to(boundary_logits.dtype)
        z_lower = z_lower.to(boundary_logits.dtype)
        valid_f = valid.to(boundary_logits.dtype)                  # [B,W]
        n_valid = valid_f.sum().clamp_min(1.0)

        log_probs = F.log_softmax(boundary_logits, dim=2)          # [B,2,H,W]
        probs = log_probs.exp()

        tgt_upper = gaussian_soft_target(z_upper, h, self.sigma)   # [B,H,W]
        tgt_lower = gaussian_soft_target(z_lower, h, self.sigma)

        nll_upper = -(tgt_upper * log_probs[:, 0]).sum(dim=1)      # [B,W]
        nll_lower = -(tgt_lower * log_probs[:, 1]).sum(dim=1)      # [B,W]
        nll = ((nll_upper + nll_lower) * valid_f).sum() / (2.0 * n_valid)

        z_axis = torch.arange(h, device=boundary_logits.device,
                              dtype=boundary_logits.dtype).view(1, h, 1)
        e_upper = (probs[:, 0] * z_axis).sum(dim=1)                 # [B,W]
        e_lower = (probs[:, 1] * z_axis).sum(dim=1)                 # [B,W]

        abs_err_upper = (e_upper - z_upper).abs()
        abs_err_lower = (e_lower - z_lower).abs()
        exp_l1 = ((abs_err_upper + abs_err_lower) * valid_f).sum() / (2.0 * n_valid)

        total = nll + self.lambda_expectation * exp_l1

        # Predicted per-column boundary WIDTH (std). Computed from live (grad-carrying)
        # probs so it can drive the OPTIONAL sharpness penalty below; also reported
        # (detached) as an uncertainty diagnostic.
        var_upper = (probs[:, 0] * z_axis ** 2).sum(dim=1) - e_upper ** 2
        var_lower = (probs[:, 1] * z_axis ** 2).sum(dim=1) - e_lower ** 2
        std_upper = var_upper.clamp_min(0.0).sqrt()
        std_lower = var_lower.clamp_min(0.0).sqrt()

        # OPTION C: auxiliary sharpness penalty (mean predicted std over valid columns,
        # both boundaries). SKIPPED entirely when weight==0 -> byte-identical total.
        if self.sharpness_weight > 0:
            width = ((std_upper + std_lower) * valid_f).sum() / (2.0 * n_valid)
            total = total + self.sharpness_weight * width

        # --- Spatial Variance Penalty (Conformal-Aware Training) ---
        # mu = E[d] per column is e_upper/e_lower, ALREADY computed above (the
        # first moment); sigma_sq below is the LITERAL second central moment
        # sum(p*(d-mu)^2) -- written out explicitly (not reused from var_upper/
        # var_lower's E[X^2]-E[X]^2 shortcut a few lines up) so this penalty is
        # the direct textbook variance, independent of that other formula's
        # floating-point path.
        #   probs[:, 0]            [B,H,W]
        #   z_axis                 [1,H,1]   (broadcasts over batch AND width)
        #   mu_upper = e_upper.unsqueeze(1)  [B,1,W]  (insert the depth axis back)
        #   (z_axis - mu_upper)     [1,H,1] - [B,1,W] -> broadcasts to [B,H,W]
        #   probs[:,0] * (...)**2  [B,H,W] * [B,H,W]  -> [B,H,W] (no broadcast needed)
        #   .sum(dim=1)             [B,H,W] -> [B,W]  (collapse the depth axis)
        # sigma_sq_upper/_lower are therefore [B,W], exactly like valid_f, so the
        # final masked mean below needs no further reshaping.
        mu_upper = e_upper.unsqueeze(1)                                    # [B,1,W]
        mu_lower = e_lower.unsqueeze(1)                                    # [B,1,W]
        sigma_sq_upper = (probs[:, 0] * (z_axis - mu_upper) ** 2).sum(dim=1)  # [B,W]
        sigma_sq_lower = (probs[:, 1] * (z_axis - mu_lower) ** 2).sum(dim=1)  # [B,W]
        variance_penalty = (((sigma_sq_upper + sigma_sq_lower) * valid_f).sum()
                            / (2.0 * n_valid))                              # scalar

        if self.lambda_spread > 0:
            total = total + self.lambda_spread * variance_penalty

        return {
            "total_loss": total,
            "nll": nll.detach(),
            "expectation_l1": exp_l1.detach(),
            # Raw (undetached-computation, detached-for-logging) spatial variance
            # penalty -- reported EVERY step regardless of lambda_spread so interval
            # tightness is visible even with the penalty off (lambda_spread=0.0).
            "variance_penalty": variance_penalty.detach(),
            # SEPARATE per-boundary mean absolute error (px) -- the upper
            # (choroid-start) boundary is expected to be the EASIER one; the
            # lower (choroid-sclera interface) boundary is the ambiguous one
            # this head exists to characterize (see model.py
            # BoundaryDistributionHead docstring / CLAUDE.md diagnostics).
            "mae_upper": (abs_err_upper * valid_f).sum().detach() / n_valid,
            "mae_lower": (abs_err_lower * valid_f).sum().detach() / n_valid,
            "mean_std_upper": (std_upper * valid_f).sum().detach() / n_valid,
            "mean_std_lower": (std_lower * valid_f).sum().detach() / n_valid,
            "n_valid_columns": n_valid.detach(),
        }


class DeepOCTLoss(nn.Module):
    """Master composite loss engine (ablation-phase aware).

    forward(outputs, targets) where
        outputs = (denoised, seg_logits, predicted_bias)
        targets = (clean_target, choroid_mask[, noise_sigma])
            noise_sigma (optional, Phase-4 only) = the per-sample injected speckle
            level sqrt(v) from the dataloader, used as the Bias-MSE target. A
            2-tuple (no noise_sigma) is still accepted for phases 1-3 / legacy
            callers, which never consult it.

    DOMAIN CONTRACT: ``outputs[0]`` is the model's ``final_pred`` — the LINEAR
    prediction (Phase 2: ``exp(denoised_log) - bias``, bias=0.0), NOT the raw
    log-domain decoder output. ``clean_target`` is the LINEAR clean reference
    (the dataset denoise target / mock clean image). ALL reconstruction terms
    (ZNCC, GAT-variance, FFL) therefore compare LINEAR final_pred vs LINEAR
    target — matching domains. ZNCC additionally measures only structural
    agreement (invariant to any global affine intensity shift).

    Per-phase composition:
      * Phase 1: SegLoss only (seg-only build; reconstruction losses not built).
      * Phases 2 & 3: lambda_seg*Seg + lambda_zncc*ZNCC + lambda_gat*GAT +
                 lambda_ffl*FFL. Bias is a FIXED CONSTANT (Phase 2: 0.0; Phase 3:
                 static 0.12549), so the Bias MSE term is COMPLETELY bypassed (not
                 built, not computed) — NO bias supervision in either phase.
      * Phase 4: as Phase 2/3 plus the Bias MSE penalty supervising the LEARNED
                 predicted bias toward the dataloader's KNOWN injected speckle
                 level ``noise_sigma = sqrt(v)`` (detached, [0,1]-domain, per-
                 image). The old Yang-Tai estimate (estimate_sigma_normalized,
                 which read ~0 on this ART-averaged data) is RETIRED from this
                 path.
      * Phase 5 (Whole-Image Speckle + Dual-Role FiLM): SAME composite as Phase 4
                 — Seg + ZNCC + GAT + FFL + Bias MSE. The Bias MSE still supervises
                 the GAP predictor's ``predicted_sqrt_v`` (the model's 3rd output)
                 toward the injected ``sqrt(v)``; the loss is unchanged and keys off
                 ``predicted_bias``/``noise_sigma`` presence, not a phase number, so
                 Phase 5 falls through to the bias block below just like Phase 4.
                 What CHANGED is upstream in model.py: ``predicted_sqrt_v`` now
                 DRIVES FiLM conditioning instead of being SUBTRACTED as a DC term,
                 and the dataset injects whole-image speckle. Consequently
                 ``dc_offset`` here measures the residual DC error of FiLM-
                 conditioned denoising WITH NO subtraction — it is EXPECTED to
                 differ from Phase 4 (whose subtraction nulls the DC term); that
                 difference is precisely the Phase 4 -> 5 ablation signal, not a bug.
                 Phases 4 & 5 are the phases where Bias MSE is active.

    Each term carries its OWN weight (terms live on different scales — an
    unweighted raw sum would let one dominate). Returns (total_loss, log_dict)
    where every term in log_dict is a LIVE tensor; ``dc_offset`` is a detached
    diagnostic (see forward).
    """

    def __init__(self,
                 lambda_seg: float = 1.0,
                 lambda_zncc: float = 1.0,
                 lambda_gat: float = 5.0,
                 lambda_ffl: float = 4.0,
                 lambda_bias: float = 0.1,
                 lambda_boundary: float = 0.0,
                 ffl_alpha: float = 1.0,
                 ablation_phase: int = 4,
                 learned_loss_weighting: bool = False,
                 learned_weight_normalize: bool = True,
                 learned_weight_warmup_steps: int = 100,
                 ema_momentum: float = 0.99) -> None:
        # NOTE on lambda_gat / lambda_ffl defaults (5.0 / 4.0, up from 1.0): these
        # are STARTING PRIORS. At 1.0 the GAT (tissue-grain) and FFL (frequency)
        # terms contributed only ~3% / ~13% of the composite gradient and were
        # effectively inert. The 5.0 / 4.0 figures were measured on the OLD
        # choroid-only + subtraction setup; whole-image speckle (Phase 5) will shift
        # the real contributions, so these are NOT final — re-measure and re-tune
        # per run. Do not treat them as fixed.
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_zncc = lambda_zncc
        self.lambda_gat = lambda_gat
        self.lambda_ffl = lambda_ffl
        self.lambda_bias = lambda_bias
        self.ablation_phase = ablation_phase

        # --- OPTIONAL Kervadec boundary/surface loss (NEW ablation dimension) ---
        # A flag-gated ADD-ON that sharpens choroid-sclera boundary localization
        # (HD95/ASD); orthogonal to the denoising ablation phases (applies to the
        # seg head in EVERY phase). ``lambda_boundary`` is the MAX weight; the
        # ACTUAL per-step weight is ``self.boundary_weight``, which train.py RAMPS
        # from 0 -> lambda_boundary over the run (Kervadec "rebalancing": region
        # loss dominates early, boundary loss grows later) so it never destabilizes
        # early training. When lambda_boundary == 0 (DEFAULT) the term is COMPLETELY
        # bypassed: BoundaryLoss is not built (scipy not imported), no SDF is
        # computed, nothing is added -> the loss is BYTE-IDENTICAL to before.
        self.lambda_boundary = lambda_boundary
        # Current (possibly ramped) weight; train.py overwrites this each epoch.
        # Defaults to the full max so non-ramping callers (eval/tests) still apply
        # the configured weight. Only consulted when self.boundary is not None.
        self.boundary_weight = lambda_boundary
        self.boundary = BoundaryLoss() if lambda_boundary > 0 else None

        # Phase 1 is segmentation-only: the reconstruction losses are never
        # called, so we skip building them (and the focal_frequency_loss import)
        # entirely — saving compute and avoiding an unnecessary hard dependency.
        self.seg_only = ablation_phase == 1
        self.seg = SegLoss()
        if self.seg_only:
            self.zncc = None
            self.gat_var = None
            self.ffl = None
        else:
            from focal_frequency_loss import FocalFrequencyLoss
            self.zncc = ZNCCLoss()
            self.gat_var = GATVarianceLoss()
            self.ffl = FocalFrequencyLoss(alpha=ffl_alpha)

        # --- OPTIONAL learned (homoscedastic-uncertainty) loss weighting ---------
        # Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses"
        # (CVPR 2018). Instead of the hand-set lambda_zncc/gat/ffl/bias constants,
        # each of these tasks gets a LEARNABLE log-variance ``s_i = log(sigma_i^2)``
        # and is combined as ``exp(-s_i) * L_i + s_i`` (the numerically-stable
        # standard form: weight = exp(-s_i) = 1/sigma_i^2, regularizer = s_i, which
        # prevents the collapse to sigma -> inf). SEG is the REFERENCE task and keeps
        # its fixed weight ``lambda_seg`` (anchors the overall scale to the primary
        # segmentation objective) so the learned weights read as "denoise-term weight
        # RELATIVE to segmentation". Boundary (when on) keeps its own ramped weight,
        # orthogonal to this scheme. log-variances init to 0 -> every weight starts
        # at exp(0)=1 (a neutral start). When OFF (default) NONE of this is built and
        # forward() is byte-identical to the fixed-lambda path. Phase 1 is
        # single-task (seg only), where this scheme is meaningless, so it is forced
        # OFF there regardless of the flag.
        self.learned_loss_weighting = bool(learned_loss_weighting) and not self.seg_only

        # --- Loss-scale NORMALIZATION (interpretability fix; default ON when learned
        #     weighting is on) -------------------------------------------------------
        # The raw task losses live on wildly different scales (seg~1e-1, zncc~1e-2,
        # gat~1e-3, ffl~1e-4). Feeding those straight into Kendall & Gal makes the
        # optimum weight exp(-s_i)=1/L_i INFLATE mechanically for the tiny terms and
        # drives the total negative/unbounded — the weights then read as a loss-scale
        # artifact, NOT task importance. Fix: divide each task loss by a DETACHED
        # running-EMA estimate of its OWN magnitude, so every task sits near O(1)
        # BEFORE the uncertainty weighting. Then exp(-s_i) reflects RELATIVE task
        # importance and stays bounded/self-correcting (a task that gets up-weighted
        # is optimized harder -> its loss drops -> the EMA follows -> the normalized
        # loss returns toward 1 -> the weight relaxes back). A short WARMUP uses fixed
        # unit weights while the EMA scales seed and stabilize, so the log-variances
        # never learn on distorted early scales. Set ``learned_weight_normalize=False``
        # to recover the pre-normalization RAW Kendall-Gal for a normalized-vs-raw
        # comparison. When learned weighting is OFF, NONE of this is built.
        self.learned_weight_normalize = bool(learned_weight_normalize)
        self.ema_momentum = float(ema_momentum)
        self.warmup_steps = int(learned_weight_warmup_steps)
        self.ema_eps = 1e-8
        self.log_vars = None
        self._norm_idx: Dict[str, int] = {}
        if self.learned_loss_weighting:
            tasks = ["zncc", "gat", "ffl"]
            if ablation_phase in (4, 5):
                # Bias MSE is only active in the dynamic-unbiasing phases, so its
                # log-variance is only learned there.
                tasks.append("bias")
            self.log_vars = nn.ParameterDict(
                {t: nn.Parameter(torch.zeros(())) for t in tasks})
            if self.learned_weight_normalize:
                # SEG is normalized too (so the seg-vs-denoise comparison is on the
                # SAME footing) but stays the FIXED reference (weight lambda_seg, no
                # log-variance). Per-task EMA scale + a per-task "seeded yet?" flag +
                # a global warmup step counter are BUFFERS -> checkpointed and moved
                # with .to(device), never gradient-tracked.
                norm_names = ["seg"] + tasks
                self._norm_idx = {n: i for i, n in enumerate(norm_names)}
                self.register_buffer("ema_scales", torch.ones(len(norm_names)))
                self.register_buffer("ema_init", torch.zeros(len(norm_names)))
                self.register_buffer("warmup_counter",
                                     torch.zeros((), dtype=torch.long))

    def _norm_loss(self, name: str, l_raw: torch.Tensor) -> torch.Tensor:
        """Update the task's running-EMA magnitude (TRAIN only, detached) and return
        the SCALE-NORMALIZED loss ``L / (ema + eps)``.

        The EMA is a buffer (no grad); only ``l_raw`` carries gradient, so this is a
        per-task CONSTANT rescale of the gradient — it removes the raw-magnitude
        artifact without opening a gradient path to the scale estimate. Seeded from
        the FIRST training batch (``ema_init``) so tiny-scale terms are immediately
        O(1) instead of decaying slowly from the 1.0 init. Gated on
        ``torch.is_grad_enabled()`` so validation passes (run under ``no_grad``) read
        the scales but never move them.
        """
        i = self._norm_idx[name]
        if torch.is_grad_enabled():                 # training pass only
            l_det = l_raw.detach()
            if self.ema_init[i] == 0:
                self.ema_scales[i] = l_det           # seed from the first batch
                self.ema_init[i] = 1
            else:
                self.ema_scales[i] = (self.ema_momentum * self.ema_scales[i]
                                      + (1.0 - self.ema_momentum) * l_det)
        return l_raw / (self.ema_scales[i] + self.ema_eps)

    def ema_scales_dict(self) -> Optional[Dict[str, float]]:
        """Current running-EMA loss scales per task (None unless normalization on)."""
        if not (self.learned_loss_weighting and self.learned_weight_normalize):
            return None
        return {n: float(self.ema_scales[i]) for n, i in self._norm_idx.items()}

    def _kg_term(self, name: str, l_term: torch.Tensor) -> torch.Tensor:
        """Kendall & Gal weighted term: ``exp(-s)*L + s`` with ``s = log(sigma^2)``.

        Gradient flows to the log-variance parameter from BOTH the ``exp(-s)*L``
        data term (drives s toward log L) and the ``+s`` regularizer (prevents
        s -> +inf / weight -> 0). At the optimum ``exp(-s)*L = 1`` so the learned
        weight ``exp(-s) = 1/L`` — high-loss (high-uncertainty) tasks are damped.
        """
        s = self.log_vars[name]
        return torch.exp(-s) * l_term + s

    def learned_weights(self) -> Optional[Dict[str, float]]:
        """Current learned per-task weights ``exp(-log_var)`` (None when OFF).

        A plain float dict for logging; ``seg`` is omitted (it is the fixed
        reference task at weight ``lambda_seg``).
        """
        if not self.learned_loss_weighting or self.log_vars is None:
            return None
        return {name: float(torch.exp(-s).detach())
                for name, s in self.log_vars.items()}

    def forward(self,
                outputs: Tuple[Optional[torch.Tensor], torch.Tensor,
                               Optional[torch.Tensor]],
                targets: Tuple[torch.Tensor, ...]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # outputs[0] is the LINEAR prediction (Phase 2: exp(denoised_log) - bias),
        # NOT the raw log-domain decoder output — see the class DOMAIN CONTRACT.
        final_pred, seg_logits, predicted_bias = outputs
        # targets is (clean_target, choroid_mask) or, for Phase 4, also carries the
        # per-sample injected speckle level noise_sigma=sqrt(v) as a 3rd element.
        if len(targets) == 3:
            clean_target, choroid_mask, noise_sigma = targets
        else:
            clean_target, choroid_mask = targets
            noise_sigma = None

        seg_metrics = self.seg(seg_logits, choroid_mask)
        l_seg = seg_metrics["total_seg_loss"]

        # OPTIONAL Kervadec boundary term (all phases; seg-head only). Bypassed
        # ENTIRELY when disabled (self.boundary is None) -> byte-identical path.
        # Computed in FP32 (autocast disabled) for parity with the recon terms:
        # the SDF reaches the low hundreds at 512px and the raw product could
        # overflow fp16 under AMP. A no-op when AMP is off (already fp32).
        boundary_term = None
        if self.boundary is not None:
            with torch.autocast(device_type=seg_logits.device.type, enabled=False):
                boundary_term = self.boundary(seg_logits.float(), choroid_mask.float())

        # ---- Phase 1: ONLY SegLoss (no reconstruction / bias terms exist) ----
        if self.seg_only:
            total = self.lambda_seg * l_seg
            if boundary_term is not None:
                total = total + self.boundary_weight * boundary_term
            log = {
                "total_loss": total,
                "seg_total": l_seg,
                "bce": seg_metrics["bce_loss"],
                "dice": seg_metrics["dice_loss"],
                "dice_score": seg_metrics["dice_score"],
            }
            if boundary_term is not None:
                log["boundary"] = boundary_term
            return total, log

        # ---- Phases 2-4: composite reconstruction + segmentation ----
        # All three reconstruction terms compare the LINEAR final_pred against
        # the LINEAR clean_target (matching domains). ZNCC -> structure; GAT ->
        # Anscombe local-variance "grain"; FFL operates in the LINEAR domain too
        # (its 2D FFT is taken over final_pred and clean_target), kept consistent
        # with ZNCC/GAT so no term mixes log/linear domains.
        #
        # NUMERICAL STABILITY (fp16/AMP): final_pred is exp-linearized and can grow
        # large during training (up to exp(_EXP_CLAMP_MAX)=~4.85e8). Under fp16 the
        # recon terms overflow the 65504 ceiling and go NaN — measured breakpoints:
        # ZNCC ~1e4, GAT ~1e5, and FocalFrequencyLoss does not even accept fp16
        # (its FFT rejects Half). So compute ALL THREE in FP32 with autocast
        # DISABLED: this raises the overflow ceiling to fp32's ~3.4e38 (finite for
        # the entire exp-clamp range) and NEVER changes the math when AMP is off
        # (final_pred is already fp32, so .float() and the disabled autocast are
        # no-ops). Only the numerical precision of these ops changes, not the loss.
        with torch.autocast(device_type=final_pred.device.type, enabled=False):
            fp32_pred = final_pred.float()
            fp32_target = clean_target.float()
            l_zncc = self.zncc(fp32_pred, fp32_target)
            l_gat = self.gat_var(fp32_pred, fp32_target)
            l_ffl = self.ffl(fp32_pred, fp32_target)

        # ``in_warmup`` is reused by the Phase-4 bias block below (same schedule).
        in_warmup = False
        if self.learned_loss_weighting and self.learned_weight_normalize:
            # NORMALIZED Kendall & Gal (default): divide each task loss by its
            # running-EMA magnitude so all tasks are O(1), then weight. Seg is
            # normalized too but stays the FIXED reference (weight lambda_seg). A
            # short warmup uses fixed unit weights while the EMA scales stabilize;
            # during warmup the log-variances stay OUT of the graph (no gradient ->
            # frozen at init), so they never learn on distorted early scales.
            seg_n = self._norm_loss("seg", l_seg)
            zncc_n = self._norm_loss("zncc", l_zncc)
            gat_n = self._norm_loss("gat", l_gat)
            ffl_n = self._norm_loss("ffl", l_ffl)
            if torch.is_grad_enabled():
                self.warmup_counter += 1
            in_warmup = int(self.warmup_counter) <= self.warmup_steps
            if in_warmup:
                total = (self.lambda_seg * seg_n + zncc_n + gat_n + ffl_n)
            else:
                total = (self.lambda_seg * seg_n
                         + self._kg_term("zncc", zncc_n)
                         + self._kg_term("gat", gat_n)
                         + self._kg_term("ffl", ffl_n))
        elif self.learned_loss_weighting:
            # RAW Kendall & Gal (normalization OFF): the pre-normalization behavior,
            # kept for a normalized-vs-raw comparison. Seg is the fixed reference;
            # each denoise term is exp(-s)*L + s on the RAW loss.
            total = (self.lambda_seg * l_seg
                     + self._kg_term("zncc", l_zncc)
                     + self._kg_term("gat", l_gat)
                     + self._kg_term("ffl", l_ffl))
        else:
            total = (self.lambda_seg * l_seg
                     + self.lambda_zncc * l_zncc
                     + self.lambda_gat * l_gat
                     + self.lambda_ffl * l_ffl)

        # Add the optional boundary term BEFORE the Phase-4/5 bias block so its
        # contribution is carried into the final total for every non-seg-only
        # phase (bias, if any, is added on top below).
        if boundary_term is not None:
            total = total + self.boundary_weight * boundary_term

        # Diagnostic ONLY (not backpropagated, hence .detach()): the absolute
        # difference between the mean intensity of the LINEAR final_pred and the
        # LINEAR clean target — i.e. the post-linearization residual DC error.
        # ZNCC is invariant to affine intensity, so with the bias branch OFF the
        # absolute DC level is unconstrained; this MEASURES that drift — the
        # empirical justification for re-introducing the bias branch in Phase 3.
        # PHASE 5: there is NO DC subtraction (the predictor drives FiLM instead),
        # so this dc_offset reflects FiLM-conditioned denoising WITHOUT subtraction
        # — expected to differ from Phase 4 (whose subtraction targets the DC term).
        dc_offset = (final_pred.mean() - clean_target.mean()).abs().detach()

        log: Dict[str, torch.Tensor] = {
            "total_loss": total,
            "seg_total": l_seg,
            "bce": seg_metrics["bce_loss"],
            "dice": seg_metrics["dice_loss"],
            "dice_score": seg_metrics["dice_score"],
            "zncc": l_zncc,
            "gat": l_gat,
            "ffl": l_ffl,
            "dc_offset": dc_offset,
        }
        if boundary_term is not None:
            log["boundary"] = boundary_term

        # Expose the current learned weights exp(-s) as detached diagnostics so they
        # ride along in the returned log dict (train.py logs the exact end-of-epoch
        # values via learned_weights(); these are unused by the CSV schema).
        if self.learned_loss_weighting and self.log_vars is not None:
            for name, s in self.log_vars.items():
                log[f"w_{name}"] = torch.exp(-s).detach()

        if self.ablation_phase in (2, 3):
            # Phases 2 & 3: bias is a FIXED CONSTANT (0.0 / static 0.12549), NOT
            # learned, so there is NO bias supervision — the Bias MSE term is
            # COMPLETELY bypassed (never constructed, never computed; NOT
            # computed-then-zeroed). 'bias_mse' is intentionally absent from the
            # log dict, so the CSV's Val_Bias_MSE stays blank for these phases.
            return total, log

        # ---- Phase 4 (and Phase 5, which reuses this path): Dynamic Unbiasing ----
        # Supervise the LEARNED per-image bias toward the dataloader's KNOWN
        # injected speckle level noise_sigma = sqrt(v) (multiplicative gamma var v
        # -> sqrt(v) is the log-domain additive-equivalent std; see oimhs_dataset).
        # Both are [0,1]-domain per-image scalars; sqrt([0.01,0.15]) = [0.10,0.387]
        # sits inside the sigmoid-bounded (0,1) output range of the GAP predictor,
        # so the head can reach it. The target is DETACHED (a fixed, non-
        # differentiable regression target). This REPLACES the retired Yang-Tai
        # estimate_sigma_normalized path, which is no longer imported or called
        # here (it remains in noise_estimation.py only as a baseline-comparison
        # tool). This is the ONLY phase with Bias MSE.
        if predicted_bias is not None and noise_sigma is not None:
            pred = predicted_bias.reshape(predicted_bias.shape[0])      # [B] in [0,1]
            injected_sigma = noise_sigma.reshape(-1).to(pred).detach()  # [B] = sqrt(v)
            l_bias = F.mse_loss(pred, injected_sigma)
            if self.learned_loss_weighting and self.learned_weight_normalize:
                # Same normalize-then-Kendall-Gal (or warmup unit weight) as above.
                bias_n = self._norm_loss("bias", l_bias)
                total = total + (bias_n if in_warmup
                                 else self._kg_term("bias", bias_n))
            elif self.learned_loss_weighting:
                total = total + self._kg_term("bias", l_bias)  # RAW Kendall-Gal
            else:
                total = total + self.lambda_bias * l_bias
            log["total_loss"] = total
            log["bias_mse"] = l_bias
            log["predicted_bias_mean"] = pred.mean().detach()
            log["injected_sigma_mean"] = injected_sigma.mean().detach()
        return total, log
