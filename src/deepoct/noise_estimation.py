"""noise_estimation.py

Standalone Python port of the Yang-Tai hybrid image-noise estimator, faithful to
the reference MATLAB in ``noiseest_matlab/noiseest.m`` and
``noiseest_matlab/refinednoiseest.m``.

    S.-M. Yang and S.-C. Tai, "Fast and reliable image-noise estimation using a
    hybrid approach", J. Electronic Imaging 19(3), 033007, 2010.
    MATLAB reference: Chris Schwemmer, Universitaet Erlangen-Nuernberg.

This module is INTENTIONALLY decoupled from model.py / loss.py / train.py. It is a
correctness anchor for (eventual) Phase-4 dynamic-unbiasing supervision, but bakes
in NO assumption about [0,1] vs [0,255] scaling: ``valrange`` is fully
parameterized exactly as in the MATLAB.

Two ports are provided:
  * ``noiseest_numpy`` / ``refined_noiseest_numpy``  -- faithful per-block port.
  * ``noiseest_torch`` / ``refined_noiseest_torch``  -- vectorized, device-agnostic,
    optional leading batch dim, no per-block Python loops.

----------------------------------------------------------------------------------
MATLAB semantics that were ambiguous / required a judgment call (documented here
and inline):

  (J1) Histogram bin count.  In MATLAB, the *data* passed to ``hist`` is
       ``G_blocks(blocklist>0)`` (valid blocks only), but ``grange`` (the bin
       count) is computed from ``max/min`` over the FULL ``G_blocks`` array, where
       clipped blocks were left at 0.  These two ranges differ only when at least
       one block is clipped.  Following the user's explicit recipe, we use a single
       min/max -- that of the valid-block G-values (which is also what MATLAB's
       ``hist`` uses internally to place the bin centers/binwidth).  When no block
       is clipped (the common case and all of the equivalence tests) this is
       identical to the MATLAB.

  (J2) Non-integer bin count.  MATLAB's ``hist(y, m)`` feeds a possibly
       non-integer ``m`` into ``0:m`` (effectively a floor on the number of bin
       edges).  We follow the user's recipe and use ``nbins = round(max-min+1)``.
       For integer-valued ``grange`` (the usual case here, since G is a sum of
       integer-weighted integer pixels at 8-bit scale) round and floor agree.

  (J3) Degenerate guard.  If NO homogeneous block survives (Nh == 0) the MATLAB
       would divide by zero and return NaN/Inf.  A NaN here would silently poison
       Phase-4 supervision, so we return 0.0 instead (documented, intentional
       deviation from literal MATLAB).
----------------------------------------------------------------------------------
"""

import math

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is a hard dep for the torch port only
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Filter masks (identical to MATLAB).  Sobel-like 5x5 derivative kernels and the
# 3x3 Laplacian.  Held as module constants so numpy/torch ports share one source.
# ---------------------------------------------------------------------------
_WB = 5  # block size

_S_V = np.array(
    [[1, 2, 0, -2, -1],
     [4, 8, 0, -8, -4],
     [6, 12, 0, -12, -6],
     [4, 8, 0, -8, -4],
     [1, 2, 0, -2, -1]],
    dtype=np.float64,
)

_S_H = np.array(
    [[1, 4, 6, 4, 1],
     [2, 8, 12, 8, 2],
     [0, 0, 0, 0, 0],
     [-2, -8, -12, -8, -2],
     [-1, -4, -6, -4, -1]],
    dtype=np.float64,
)

_L_A = np.array(
    [[1, -2, 1],
     [-2, 4, -2],
     [1, -2, 1]],
    dtype=np.float64,
)


# ===========================================================================
# Shared scalar helper: the automated histogram threshold (judgment calls J1/J2)
# ===========================================================================
def _auto_threshold_from_G(g_valid, k):
    """Replicate MATLAB ``hist``-based auto thresholding for a 1-D array of valid
    block homogeneity values ``g_valid`` and target cumulative count ``k``.

    Returns the scalar G_th. Implemented in pure-python/numpy floats so the numpy
    and torch ports produce bit-identical thresholds (the torch port feeds in the
    same float64 min/max via ``.item()``).
    """
    g_valid = np.asarray(g_valid, dtype=np.float64)
    gmin = float(g_valid.min())
    gmax = float(g_valid.max())

    # Guard: all valid blocks share the same homogeneity (e.g. perfectly flat
    # image -> all G == 0).  MATLAB's hist would collapse; we keep every valid
    # block by putting the threshold just above the single value.  (guard min==max)
    if gmax == gmin:
        return gmax + 1.0

    # (J2) nbins = round(max - min + 1); guard nbins < 1.
    nbins = int(round(gmax - gmin + 1.0))
    if nbins < 1:
        nbins = 1

    binwidth = (gmax - gmin) / nbins
    idx = np.arange(nbins, dtype=np.float64)
    # bin CENTERS = min + binwidth*(0.5, 1.5, ..., nbins-0.5)
    centers = gmin + binwidth * (idx + 0.5)
    # Right edge of bin i = min + binwidth*(i+1).  Last bin's true right edge is
    # +inf (MATLAB's outermost hist bin catches everything), so its cumulative
    # count is the full valid-block count.
    right_edges = gmin + binwidth * (idx + 1.0)

    sorted_g = np.sort(g_valid)
    # cum[i] = #{ G < right_edges[i] } == np.histogram cumulative with [lo, hi) bins
    cum = np.searchsorted(sorted_g, right_edges, side="left").astype(np.int64)
    cum[-1] = g_valid.size  # last bin right edge -> +inf

    # hloc = ceil(center + 0.5); G_th = first hloc whose cumulative count >= k.
    hloc = np.ceil(centers + 0.5)
    ge = cum >= k
    if not ge.any():
        # cum[-1] == n_valid >= k always (k = floor(p*n_valid) <= n_valid), so this
        # branch is unreachable for valid p; kept as a defensive fallback.
        return float(hloc[-1])
    first = int(np.argmax(ge))  # argmax returns the first True index
    return float(hloc[first])


# ===========================================================================
# FAITHFUL NUMPY PORT (the correctness anchor)
# ===========================================================================
def noiseest_numpy(img, valrange=256, p=0.1):
    """Faithful NumPy port of ``noiseest.m``.

    Parameters
    ----------
    img : 2-D array-like
    valrange : value range for clipped-pixel identification (MATLAB default 256).
    p : percentile for the automated homogeneity threshold (MATLAB default 0.1).
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("noiseest_numpy expects a 2-D image")
    H, W = img.shape

    # (Trap 2) Block size Wb=5; both dims must be divisible by 5, else ValueError.
    if (H % _WB != 0) or (W % _WB != 0):
        raise ValueError("Image size not divisible by 5")

    # (Trap 3) Clip bounds in RAW units.
    min_val = 0.0625 * valrange
    max_val = 0.91796875 * valrange

    n_by = H // _WB
    n_bx = W // _WB
    clip_threshold = math.floor(0.5 * _WB ** 2)  # floor(0.5*25) = 12

    blocklist = np.ones((n_by, n_bx), dtype=bool)
    g_blocks = np.zeros((n_by, n_bx), dtype=np.float64)

    # Pass 1: drop clipped blocks, compute homogeneity measure for the rest.
    for by in range(n_by):
        for bx in range(n_bx):
            block = img[by * _WB:by * _WB + _WB, bx * _WB:bx * _WB + _WB]

            # (Trap 3) Count NON-clipped (in-range) pixels; drop the block if the
            # count is <= 12 (use <=, not <).
            nonclipped = (block >= min_val) & (block <= max_val)
            if nonclipped.sum() <= clip_threshold:
                blocklist[by, bx] = False
                continue

            # (Trap 1) Block gradient is an ELEMENT-WISE multiply then SUM over the
            # whole 5x5 block (Frobenius inner product per block), NOT a sliding
            # convolution. G = |sum(S_v (.) block)| + |sum(S_h (.) block)|.
            g_v = np.sum(_S_V * block)
            g_h = np.sum(_S_H * block)
            g_blocks[by, bx] = abs(g_v) + abs(g_h)

    n_valid = int(blocklist.sum())
    # (Trap 9 / J3) No non-clipped blocks at all -> nothing to estimate from.
    if n_valid == 0:
        return 0.0

    # (Trap 4) Automated threshold over the currently-valid blocks.
    g_valid = g_blocks[blocklist]
    k = math.floor(p * n_valid)
    g_th = _auto_threshold_from_G(g_valid, k)

    # (Trap 5) Remove blocks with G >= G_th (use >=). Survivors are homogeneous.
    remove = blocklist & (g_blocks >= g_th)
    blocklist[remove] = False

    # (Trap 6) Normalisation factor.
    n_h = int(blocklist.sum()) * _WB ** 2
    # (Trap 9 / J3) Degenerate guard: do NOT divide by zero -> return finite 0.0.
    if n_h == 0:
        return 0.0
    fac = math.sqrt(math.pi / 2.0) / (6.0 * n_h)

    # (Trap 7) Laplacian via VALID 2-D convolution, then embedded in a zero-border
    # array the size of the image.  L_A is 180-symmetric so conv == correlation.
    tmp = _laplacian_valid_numpy(img)
    lap = np.zeros((H, W), dtype=np.float64)
    lap[1:-1, 1:-1] = tmp  # ZERO BORDER kept (rows/cols 0 and -1 stay 0)

    # (Trap 8) Sum |L| over the homogeneous 5x5 blocks.  The blocks DO include the
    # zero border rows/cols of `lap`.
    sigma = 0.0
    for by in range(n_by):
        for bx in range(n_bx):
            if not blocklist[by, bx]:
                continue
            lblock = lap[by * _WB:by * _WB + _WB, bx * _WB:bx * _WB + _WB]
            sigma += np.sum(np.abs(lblock))

    return fac * sigma


def _laplacian_valid_numpy(img):
    """VALID 2-D convolution of `img` with the 3x3 Laplacian L_A.

    L_A is symmetric under 180-deg rotation, so MATLAB's conv2(...,'valid') (which
    flips the kernel) equals correlation.  Computed as a 9-term shifted sum (a loop
    over the 3x3 kernel, NOT a per-block loop).
    """
    H, W = img.shape
    out = np.zeros((H - 2, W - 2), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            out += _L_A[a, b] * img[a:a + H - 2, b:b + W - 2]
    return out


def refined_noiseest_numpy(img, valrange=256):
    """Faithful NumPy port of ``refinednoiseest.m`` (two-round refinement)."""
    low_threshold = 5
    medium_threshold = 10
    high_threshold = 20

    # First round (p = 0.1).
    sigma1 = noiseest_numpy(img, valrange, 0.1)

    # Medium noise -> no refinement.
    if (sigma1 > low_threshold) and (sigma1 <= medium_threshold):
        return sigma1

    # Second round: pick p from the regime of sigma1.
    if sigma1 <= low_threshold:
        p = 0.03
    elif sigma1 <= high_threshold:
        p = 0.2
    else:
        p = 0.5

    sigma2 = noiseest_numpy(img, valrange, p)

    if sigma1 <= low_threshold:
        return min(sigma1, sigma2)
    return sigma2


# ===========================================================================
# VECTORIZED PYTORCH PORT (device-agnostic, optional leading batch dim)
# ===========================================================================
def _require_torch():
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for the *_torch noise estimators")


def _to_blocks_torch(x):
    """(B, H, W) -> (B, nBy, nBx, Wb, Wb) without copying block-by-block."""
    B, H, W = x.shape
    n_by, n_bx = H // _WB, W // _WB
    x = x.reshape(B, n_by, _WB, n_bx, _WB)
    return x.permute(0, 1, 3, 2, 4).contiguous()


def noiseest_torch(img, valrange=256, p=0.1):
    """Vectorized Torch port of ``noiseest``.

    Accepts a 2-D ``(H, W)`` tensor (returns a 0-dim tensor) or a 3-D
    ``(B, H, W)`` tensor (returns a ``(B,)`` tensor).  Device-agnostic; uses
    float64 internally for fidelity with the numpy anchor.  Block operations are
    fully vectorized -- no per-block Python loops (the only loop is over the batch,
    for the per-image histogram threshold).
    """
    _require_torch()
    squeeze = False
    if img.dim() == 2:
        img = img.unsqueeze(0)
        squeeze = True
    if img.dim() != 3:
        raise ValueError("noiseest_torch expects (H,W) or (B,H,W)")

    device = img.device
    img = img.to(torch.float64)
    B, H, W = img.shape

    # (Trap 2) Divisibility check.
    if (H % _WB != 0) or (W % _WB != 0):
        raise ValueError("Image size not divisible by 5")

    n_by, n_bx = H // _WB, W // _WB
    min_val = 0.0625 * valrange
    max_val = 0.91796875 * valrange
    clip_threshold = math.floor(0.5 * _WB ** 2)  # 12

    s_v = torch.as_tensor(_S_V, dtype=torch.float64, device=device)
    s_h = torch.as_tensor(_S_H, dtype=torch.float64, device=device)

    blocks = _to_blocks_torch(img)  # (B, nBy, nBx, 5, 5)

    # (Trap 3) Clipping: drop a block if #non-clipped <= 12.
    nonclipped = ((blocks >= min_val) & (blocks <= max_val)).sum(dim=(-1, -2))
    valid = nonclipped > clip_threshold  # (B, nBy, nBx) ; > 12 == not (<= 12)

    # (Trap 1) Frobenius inner product per block (vectorized tensordot over 5x5).
    g_v = (blocks * s_v).sum(dim=(-1, -2))
    g_h = (blocks * s_h).sum(dim=(-1, -2))
    g_blocks = g_v.abs() + g_h.abs()  # (B, nBy, nBx)

    # (Trap 4/5) Per-image automated threshold (loop over batch only).
    g_th = torch.empty(B, dtype=torch.float64, device=device)
    for b in range(B):
        gb = g_blocks[b][valid[b]]
        if gb.numel() == 0:
            g_th[b] = float("inf")  # nothing survives -> guarded to 0 below
            continue
        k = math.floor(p * gb.numel())
        g_th[b] = _auto_threshold_from_G(gb.detach().cpu().numpy(), k)

    homo = valid & (g_blocks < g_th.view(B, 1, 1))  # keep G < G_th  (remove G >= G_th)

    # (Trap 7) Laplacian via valid conv2d (cross-correlation; L_A is symmetric).
    kernel = torch.as_tensor(_L_A, dtype=torch.float64, device=device).view(1, 1, 3, 3)
    tmp = F.conv2d(img.unsqueeze(1), kernel).squeeze(1)  # (B, H-2, W-2)
    lap = torch.zeros((B, H, W), dtype=torch.float64, device=device)
    lap[:, 1:-1, 1:-1] = tmp  # ZERO BORDER kept

    lap_blocks = _to_blocks_torch(lap)  # (B, nBy, nBx, 5, 5)
    # (Trap 8) Sum |L| over homogeneous blocks (border zeros included).
    sigma_block = lap_blocks.abs().sum(dim=(-1, -2))  # (B, nBy, nBx)
    sigma = (sigma_block * homo).sum(dim=(-1, -2))  # (B,)

    # (Trap 6 / J3) Normalisation + degenerate guard (Nh == 0 -> 0.0, no div-by-0).
    n_h = homo.sum(dim=(-1, -2)).to(torch.float64) * (_WB ** 2)  # (B,)
    safe = n_h > 0
    fac = torch.zeros(B, dtype=torch.float64, device=device)
    fac[safe] = math.sqrt(math.pi / 2.0) / (6.0 * n_h[safe])
    result = fac * sigma  # zero where n_h == 0

    if squeeze:
        return result[0]
    return result


def refined_noiseest_torch(img, valrange=256):
    """Vectorized Torch port of ``refinednoiseest`` (per-image two-round logic)."""
    _require_torch()
    if img.dim() == 2:
        return _refined_single_torch(img, valrange)
    if img.dim() != 3:
        raise ValueError("refined_noiseest_torch expects (H,W) or (B,H,W)")
    # Branch decisions are per-image; evaluate each independently and re-stack.
    return torch.stack([_refined_single_torch(img[b], valrange) for b in range(img.shape[0])])


def _refined_single_torch(img2d, valrange):
    low_threshold = 5
    medium_threshold = 10
    high_threshold = 20

    sigma1 = noiseest_torch(img2d, valrange, 0.1)  # 0-dim tensor
    s1 = float(sigma1)

    if (s1 > low_threshold) and (s1 <= medium_threshold):
        return sigma1

    if s1 <= low_threshold:
        p = 0.03
    elif s1 <= high_threshold:
        p = 0.2
    else:
        p = 0.5

    sigma2 = noiseest_torch(img2d, valrange, p)

    if s1 <= low_threshold:
        return torch.minimum(sigma1, sigma2)
    return sigma2


# ===========================================================================
# INTEGRATION WRAPPER (Phase-4 domain bridge — Option A: rescale, no re-derived
# constants). The validated core above is NOT touched by this.
# ===========================================================================
def estimate_sigma_normalized(img_01, valrange=256):
    """Per-image Yang-Tai noise sigma for a LINEAR ``[0,1]`` image, returned in
    the same ``[0,1]`` domain.

    Domain bridge (Option A — rescale, do NOT re-derive the estimator's RAW-unit
    constants): the estimator was validated at 8-bit scale, so we lift the linear
    ``[0,1]`` image to ``[0,255]``, estimate there with the unchanged
    ``valrange=256`` constants, then divide the resulting sigma back by 255 to
    return a ``[0,1]``-domain scalar.  Concretely, per image::

        sigma_target = refined_noiseest(img_01 * 255.0, valrange=256) / 255.0

    INTEGRATION TRAP (real, not theoretical): the validated estimator raises
    ``ValueError`` unless BOTH spatial dims are multiples of 5.  Production tile
    sizes (128, 512, ...) are not, so each image is CENTER-CROPPED to the nearest
    lower multiple of 5 in both dims before estimation (e.g. 128->125, 512->510).
    The crop is symmetric so the estimate samples the image center.

    SUPERVISION SEMANTICS: this is a NON-differentiable supervision target (runs
    under ``torch.no_grad`` and the result is detached), computed on the clean
    denoise TARGET image — i.e. the real-speckle level of the reference frame, NOT
    the synthetically-augmented network input.

    Parameters
    ----------
    img_01 : torch.Tensor
        Linear ``[0,1]`` image. Accepts ``(H,W)``, ``(B,H,W)``, or ``(B,C,H,W)``
        (the single-channel center frame, ``C==1``, is squeezed; if ``C>1`` the
        first channel is used).
    valrange : passed straight through to ``refined_noiseest_torch``.

    Returns
    -------
    torch.Tensor
        Per-image ``[0,1]`` sigma. ``(B,)`` for batched input, 0-dim for a single
        image. Same device as ``img_01``.
    """
    _require_torch()
    x = img_01
    squeeze_batch = False
    if x.dim() == 2:                 # (H, W)
        x = x.unsqueeze(0)
        squeeze_batch = True
    elif x.dim() == 3:               # (B, H, W)
        pass
    elif x.dim() == 4:               # (B, C, H, W) -> use the center channel
        x = x[:, 0] if x.shape[1] == 1 else x[:, 0]
    else:
        raise ValueError(
            "estimate_sigma_normalized expects (H,W), (B,H,W) or (B,C,H,W)")

    with torch.no_grad():
        # Lift to the 8-bit scale the estimator's constants were validated at.
        x255 = x.to(torch.float64) * 255.0

        # Center-crop both dims to the nearest lower multiple of 5 (the trap).
        _, H, W = x255.shape
        h_c = (H // 5) * 5
        w_c = (W // 5) * 5
        if h_c == 0 or w_c == 0:
            raise ValueError(
                f"image too small to crop to a multiple of 5: got {H}x{W}")
        top = (H - h_c) // 2
        left = (W - w_c) // 2
        x255 = x255[:, top:top + h_c, left:left + w_c]

        sigma_raw = refined_noiseest_torch(x255, valrange=valrange)  # (B,) raw units
        sigma_01 = (sigma_raw / 255.0).to(img_01.device)

    if squeeze_batch:
        return sigma_01[0] if sigma_01.dim() > 0 else sigma_01
    return sigma_01
