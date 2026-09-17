"""DeepOCT-Ultra core architecture — Phase 2, Step 2 (skeleton).

This is the BASELINE multi-task U-Net skeleton only:
  * Shared CNN encoder (4 downsampling stages, GroupNorm + GELU, MaxPool).
  * Plain CNN bottleneck (2 conv blocks). NO wavelet transform, NO axial
    attention yet — those are swapped in at Step 5 behind config flags.
  * Two fully separate, parallel decoders (denoise + segmentation).

Design notes (Golden Rules):
  * NO BatchNorm anywhere. OCT speckle variance violates BatchNorm's batch
    statistics assumptions; we use GroupNorm throughout.
  * Everything is built incrementally — this file is intentionally a working
    end-to-end skeleton that later steps enrich one component at a time.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_GN_GROUPS = 8  # GroupNorm groups; all channel counts here are multiples of 8.

# Upper clamp on the LOG-domain denoiser output before the deterministic exp()
# linearization, so a large log value can never produce inf (exp(20) ~ 4.85e8).
_EXP_CLAMP_MAX = 20.0

# Magnitude bound on the LINEAR reconstruction AFTER exp() (the durable NaN fix).
# The (normalized) target images live in [0,1] linear intensity, so a physically
# valid reconstruction cannot exceed that range; we allow a small headroom above 1.
# ZNCC is scale-invariant, so nothing else anchors the output scale — without this
# the exp() output can drift toward exp(_EXP_CLAMP_MAX)=~4.85e8 and overflow the
# reconstruction losses (fp16 first, eventually fp32). Clamping final_pred to this
# range BOUNDS the magnitude so it can never blow up regardless of precision. It
# only bites on genuinely out-of-[0,1.5] (unphysical) values; an in-range output is
# untouched. Complementary to the fp32-loss fix (precision headroom + bounded mag).
_RECON_CLAMP_MIN = 0.0
_RECON_CLAMP_MAX = 1.5

# Phase 3 static Spectralis bias: 32/255, Heidelberg Spectralis machine bias,
# linear [0,1] domain, post-exp. OIMHS is a Heidelberg Spectralis dataset (NOT
# Cirrus), so the static DC offset is the Spectralis value of 32 in raw 8-bit
# intensity, expressed in the normalized [0,1] LINEAR domain that final_pred and
# the target live in. Subtracted AFTER exp() (linear domain), NOT in the log domain.
_SPECTRALIS_STATIC_BIAS = 32.0 / 255.0  # = 0.12549


class ConvBlock(nn.Module):
    """Conv2d(k=3, pad=1) -> GroupNorm(8) -> GELU. The repeated atomic unit."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=_GN_GROUPS, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    """Two stacked ConvBlocks (in->out, then out->out)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedEncoder(nn.Module):
    """4-stage downsampling CNN encoder.

    Returns the deepest pre-bottleneck feature map plus the skip connection from
    each stage (captured BEFORE its MaxPool), ordered shallow -> deep.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32) -> None:
        super().__init__()
        # Channel progression: 32 -> 64 -> 128 -> 256 (base * 2**stage).
        self.stage_channels = [base_filters * (2 ** s) for s in range(4)]

        self.stages = nn.ModuleList()
        prev_ch = in_channels
        for out_ch in self.stage_channels:
            self.stages.append(DoubleConv(prev_ch, out_ch))
            prev_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = self.stage_channels[-1]  # feeds the bottleneck

    def forward(self, x: torch.Tensor, film_coeffs=None
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        current = x
        for idx, stage in enumerate(self.stages):
            current = stage(current)
            # Phase-5 FiLM injection point: after this level's main DoubleConv,
            # BEFORE the skip is captured and BEFORE pooling — so BOTH the skip and
            # the downsampled path carry the noise-level conditioning. When
            # film_coeffs is None (phases 1-4, or FiLM disabled) this is skipped
            # entirely -> byte-identical to the original encoder.
            if film_coeffs is not None:
                gamma, beta = film_coeffs[idx]
                current = gamma * current + beta
            skips.append(current)          # skip captured before downsampling
            current = self.pool(current)   # downsample all 4 stages
        return current, skips


class Decoder(nn.Module):
    """One U-Net decoder branch.

    Mirrors the encoder: at each level ConvTranspose2d upsamples, the matching
    encoder skip is concatenated, and a DoubleConv fuses them. A final 1x1-free
    Conv2d projects to ``out_channels`` (no activation — raw output/logits).
    """

    def __init__(self, skip_channels: List[int], bottleneck_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        # Decode from deep -> shallow, mirroring the encoder skip list reversed.
        decode_channels = list(reversed(skip_channels))  # e.g. [256, 128, 64, 32]

        self.upsamplers = nn.ModuleList()
        self.fusers = nn.ModuleList()
        in_ch = bottleneck_channels
        for skip_ch in decode_channels:
            # Upsample halves channels to the skip's channel count.
            self.upsamplers.append(
                nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
            )
            # Concatenate skip (skip_ch) with upsampled (skip_ch) -> 2*skip_ch.
            self.fusers.append(DoubleConv(skip_ch * 2, skip_ch))
            in_ch = skip_ch

        self.head = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        # Consume skips from deepest to shallowest.
        skips_deep_to_shallow = list(reversed(skips))
        current = bottleneck
        for upsample, fuse, skip in zip(self.upsamplers, self.fusers,
                                        skips_deep_to_shallow):
            current = upsample(current)
            current = torch.cat([current, skip], dim=1)
            current = fuse(current)
        return self.head(current)

    def forward_collect(self, bottleneck: torch.Tensor,
                        skips: List[torch.Tensor]
                        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Same as ``forward`` but also returns the post-fuse feature map of each
        decoder block (deep -> shallow). Used by the SegDecoder under cross-gating
        so its intermediate features can be injected into the denoise pathway.
        """
        skips_deep_to_shallow = list(reversed(skips))
        feats: List[torch.Tensor] = []
        current = bottleneck
        for upsample, fuse, skip in zip(self.upsamplers, self.fusers,
                                        skips_deep_to_shallow):
            current = upsample(current)
            current = torch.cat([current, skip], dim=1)
            current = fuse(current)
            feats.append(current)
        return self.head(current), feats

    def forward_gated(self, bottleneck: torch.Tensor, skips: List[torch.Tensor],
                      seg_feats: List[torch.Tensor], confidence: torch.Tensor,
                      gates: "nn.ModuleList", film_coeffs=None,
                      gate_scale: torch.Tensor = None) -> torch.Tensor:
        """Decode while gating each block with the matching SegDecoder feature.

        After each block's fuse, ``gates[i]`` blends the confidence-weighted
        segmentation feature ``seg_feats[i]`` into the denoise feature. This is
        what makes the denoise loss flow backward into the SegDecoder.

        ``film_coeffs`` (Phase 5 only) optionally FiLM-modulates each level's
        feature AFTER its main DoubleConv (fuse) and BEFORE the cross-gate. When
        None (phases 1-4, or FiLM disabled) the FiLM step is skipped entirely ->
        byte-identical to the original gated decoder.

        ``gate_scale`` (noise-adaptive conditioning only) optionally scales the
        seg-feature injection at EVERY block by the same per-image scalar
        [B,1,1,1] (a monotonic function of the predicted noise level). When None
        (flag off) the gates run exactly as before -> byte-identical.
        """
        skips_deep_to_shallow = list(reversed(skips))
        current = bottleneck
        for i, (upsample, fuse, skip) in enumerate(
                zip(self.upsamplers, self.fusers, skips_deep_to_shallow)):
            current = upsample(current)
            current = torch.cat([current, skip], dim=1)
            current = fuse(current)
            if film_coeffs is not None:
                gamma, beta = film_coeffs[i]
                current = gamma * current + beta
            current = gates[i](seg_feats[i], current, confidence, scale=gate_scale)
        return self.head(current)


class DenoiseDecoder(Decoder):
    """Denoising branch: outputs the reconstructed center image (1 channel)."""

    def __init__(self, skip_channels: List[int], bottleneck_channels: int) -> None:
        super().__init__(skip_channels, bottleneck_channels, out_channels=1)


class SegDecoder(Decoder):
    """Segmentation branch: outputs raw choroid-mask logits (1 channel).

    No Sigmoid here — BCEWithLogitsLoss is applied downstream (Step 3).
    """

    def __init__(self, skip_channels: List[int], bottleneck_channels: int) -> None:
        super().__init__(skip_channels, bottleneck_channels, out_channels=1)


class Linearization(nn.Module):
    """Reverse-engineer the log-domain OCT input back to LINEAR optical space.

    The Phase-1 pipeline feeds the network ``log(image + eps)``, so the exact
    analytic inverse is ``exp``. We use that here.

    NOTE (calibration): the architecture spec calls for the *machine calibration
    curve* — a per-device, monotonic intensity LUT mapping displayed values to
    linear optical reflectivity. We do not have that LUT, and we do not invent
    one. ``exp`` is the honest, pipeline-exact placeholder; a real LUT can be
    dropped in later by overriding ``forward`` (e.g. a registered buffer +
    interpolation) without touching the rest of the model.
    """

    def __init__(self, clamp_min: float = -30.0, clamp_max: float = 30.0) -> None:
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp before exp to keep the linear-domain values finite.
        return torch.exp(torch.clamp(x, self.clamp_min, self.clamp_max))


class BiasEstimator(nn.Module):
    """Global Average Pooling branch at the bottleneck that predicts a single
    NON-NEGATIVE scalar speckle bias B_e per sample in the ``[0,1]`` domain
    (shape [B, 1, 1, 1]).

    The estimate is broadcast-subtracted from the reconstructed LINEAR image
    inside ``DeepOCTUltra.forward`` (Phase 4: ``final_pred = exp(clamp(log)) -
    predicted_bias``), and supervised (loss.py, Phase 4) toward the dataloader's
    KNOWN injected speckle level ``sqrt(v)`` — both [0,1]-domain per-image scalars.

    ACTIVATION CHOICE: a final ``sigmoid`` bounds the output to ``(0,1)`` — the
    exact domain of both the LINEAR reconstruction it is subtracted from and the
    supervision target. For v ~ Uniform(0.01, 0.15) the target sqrt(v) lies in
    ``[0.10, 0.387]``, comfortably inside the sigmoid range, so the head can reach
    it. Sigmoid is chosen over softplus precisely because the target is BOUNDED to
    [0,1]; softplus (unbounded above) would let the predictor wander out of range.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(in_channels // 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        b = bottleneck.shape[0]
        pooled = self.gap(bottleneck).flatten(1)   # [B, C]
        bias = self.mlp(pooled)                     # [B, 1] raw logit
        bias = torch.sigmoid(bias)                  # -> non-negative [0,1] scalar
        return bias.view(b, 1, 1, 1)


class HorizontalAxialAttention(nn.Module):
    """Restormer-style self-attention restricted to the HORIZONTAL axis (rows).

    Vertical vessel shadows are columns of obliterated signal; to inpaint them we
    must attend along each row, pulling intact context from the left/right of the
    shadow. So instead of global attention over ``N = H*W`` tokens, we treat each
    row independently: ``W`` becomes the sequence length and ``B*H`` the batch.
    Complexity collapses from O((H*W)^2) to O(H * W^2) -- mandatory for VRAM.

    Standard pre-norm transformer block: LayerNorm -> MHA -> residual, then
    LayerNorm -> GELU FFN -> residual. Operates entirely on real tensors.
    """

    def __init__(self, channels: int, num_heads: int = 8,
                 ffn_expansion: int = 2) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # [B, C, H, W] -> [B*H, W, C]: each row is an independent length-W sequence.
        seq = x.permute(0, 2, 3, 1).reshape(b * h, w, c)

        # Pre-norm multi-head attention along the row (W) axis, with residual.
        normed = self.norm1(seq)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        seq = seq + attn_out

        # Pre-norm feed-forward, with residual.
        seq = seq + self.ffn(self.norm2(seq))

        # [B*H, W, C] -> [B, C, H, W].
        return seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class DTCWTBottleneck(nn.Module):
    """Complex-wavelet bottleneck (Phase 2A: DT-CWT, J=1, 6 directional subbands).

    Pipeline:
      1. DoubleConv projects ``in_channels`` -> ``out_channels`` (channel mixing,
         keeps this a drop-in replacement for the plain DoubleConv bottleneck).
      2. ``DTCWTForward(J=1)`` splits the feature map into a low-frequency band
         ``Yl`` (the coarse base geometry) and high-frequency complex directional
         subbands ``Yh`` (oriented speckle / fine edges).
      3. ``HorizontalAxialAttention`` is applied ONLY to ``Yl``. Vessel shadows
         are large, low-frequency structural absences, so the base-geometry band
         is exactly where horizontal inpainting must act.
      4. ``DTCWTInverse`` recombines the attended ``Yl`` with the UNTOUCHED ``Yh``
         back to a spatial feature map.

    TRAP AVOIDANCE: PyTorch autograd is brittle on complex-tensor ops, and
    ``Yh`` is returned in a complex/extra-dim packed layout. We never touch
    ``Yh`` -- it is passed straight through to the inverse. All learnable work
    happens on the real-valued ``Yl`` and the conv projection.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int = 8) -> None:
        super().__init__()
        # Lazy import: pytorch_wavelets is only required on the wavelet path, so
        # the Step 2-4 baselines stay importable without the dependency.
        from pytorch_wavelets import DTCWTForward, DTCWTInverse

        self.proj = DoubleConv(in_channels, out_channels)
        self.dtcwt_forward = DTCWTForward(J=1)
        self.dtcwt_inverse = DTCWTInverse()
        self.axial_attention = HorizontalAxialAttention(out_channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)                       # [B, out_channels, H, W]
        yl, yh = self.dtcwt_forward(feat)         # yl real; yh complex/packed
        yl = self.axial_attention(yl)             # attend low-freq geometry only
        # yh passed through completely untouched (autograd-safe).
        return self.dtcwt_inverse((yl, yh))       # [B, out_channels, H, W]


class UncertaintyConfidence(nn.Module):
    """Epistemic confidence map from segmentation logits.

    Binary entropy of the per-pixel sigmoid probability,
    ``H = -(p*log2 p + (1-p)*log2(1-p))``, is naturally bounded to [0, 1] and
    peaks at p=0.5 (maximum uncertainty). Confidence is ``C = 1 - H``: ~1 where
    the segmentation is decisive (p->0 or p->1), ~0 where it is unsure. The gate
    uses C so the seg branch only steers the denoiser where it is confident,
    preventing it from hallucinating boundaries in obliterated tissue.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, seg_logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(seg_logits)
        entropy = -(p * torch.log2(p + self.eps)
                    + (1.0 - p) * torch.log2(1.0 - p + self.eps))
        return 1.0 - entropy


class CrossDecoderGate(nn.Module):
    """Confidence-gated injection of segmentation features into the denoise path.

    Given the seg/denoise feature maps at one decoder block (same channels and
    spatial size) and the global confidence map ``C``:
      1. resize ``C`` (bilinear) to the block's spatial size,
      2. modulate the seg features: ``gated_seg = seg_features * C``,
      3. concatenate with the denoise features and fuse with a 1x1 conv back to
         the original denoise channel count.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, seg_features: torch.Tensor, denoise_features: torch.Tensor,
                confidence: torch.Tensor,
                scale: torch.Tensor = None) -> torch.Tensor:
        c = F.interpolate(confidence, size=denoise_features.shape[-2:],
                          mode="bilinear", align_corners=False)
        gated_seg = seg_features * c                       # C broadcasts over C-dim
        if scale is not None:
            # OPTIONAL noise-adaptive per-image modulation (flag-gated). ``scale``
            # is a per-image scalar [B,1,1,1] that broadcasts over channels/space
            # and scales HOW STRONGLY the confidence-gated seg feature is injected
            # into the denoise path for THIS image — i.e. the strength of the
            # cross-decoder coupling, conditioned on the predicted noise level (see
            # NoiseAdaptiveConditioner). ``None`` (default) leaves the injection
            # untouched -> byte-identical to the un-conditioned gate.
            gated_seg = gated_seg * scale
        fused = torch.cat([gated_seg, denoise_features], dim=1)
        return self.fuse(fused)


class NoiseAdaptiveConditioner(nn.Module):
    """Per-image noise-adaptive scale for the cross-decoder coupling (flag-gated).

    Maps the model's PREDICTED per-image noise level (the GAP noise predictor's
    scalar, DETACHED by the caller) to a single per-image gate scale
    ``s = sigmoid(a * noise_pred + b)`` with LEARNABLE scalars ``a, b``. That
    scale multiplies the confidence-gated segmentation feature injected into the
    denoise path at EVERY cross-gate block (see CrossDecoderGate), so it modulates
    HOW STRONGLY the two decoders are coupled AS A FUNCTION OF PREDICTED NOISE:
        * a > 0  ->  noisier images get a LARGER scale -> MORE cross-decoder
                     coupling (the denoising objective shapes the shared/seg
                     features more) — the hypothesized "denoise more where noisy".
        * a < 0  ->  the opposite; a == 0 -> noise-independent (a global scale).
    The SIGN and magnitude of the LEARNED ``a`` are the scientific readout: does
    the model choose to couple denoising more in harder (noisier / AMD) images?
    A single GLOBAL weight cannot express this per-image, class-dependent effect;
    this can.

    WHY DETACH the noise signal (done by the caller): the predictor stays honest
    to the noise level — it is supervised ONLY by its own Bias MSE (phases 4/5),
    never pulled by the recon/seg gradient that flows back through this scale. This
    mirrors the Phase-5 FiLM design, which detaches ``predicted_sqrt_v`` for the
    same reason. ``a, b`` STILL receive gradient (through ``s``), so the modulation
    itself is learned end-to-end; only the noise ESTIMATE is shielded.

    INIT (a=0, b=0 -> s=0.5): a NEUTRAL start that assumes NEITHER a particular
    coupling strength NOR any noise dependence, and sits at the sigmoid's point of
    MAXIMUM gradient sensitivity so ``a`` can move off zero readily. ``b`` absorbs
    the overall coupling level while ``a`` captures the noise dependence — the
    quantity of interest. (Flag-OFF never builds or calls this module, so the
    default path is byte-identical regardless of this init; flag-ON is a distinct
    experimental arm and is NOT required to match flag-off.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, noise_pred: torch.Tensor) -> torch.Tensor:
        # noise_pred: [B,1,1,1] per-image scalar in [0,1] (DETACHED by the caller).
        # Returns s in (0,1), same [B,1,1,1] shape, broadcast over the gate.
        return torch.sigmoid(self.a * noise_pred + self.b)


class FiLMGenerator(nn.Module):
    """Feature-wise Linear Modulation (FiLM) coefficient generator (Phase 5).

    Maps a per-image conditioning SCALAR — the GAP predictor's estimated noise
    level ``predicted_sqrt_v`` in [0,1] — to per-channel ``(gamma, beta)`` affine
    coefficients for a SEQUENCE of feature maps (one entry per U-Net level on a
    given path). FiLM then modulates each feature map as ``out = gamma*feat+beta``
    (Perez et al., FiLM, arXiv:1709.07871; the dual-role encoder+decoder
    conditioning follows the 2008.10418 family).

    gamma is parameterized as ``1 + delta`` (a RESIDUAL around identity) so a
    freshly-initialized head starts NEAR identity (gamma≈1, beta≈0) rather than
    near zero — multiplying features by ~0 at init would destroy signal. The head
    is otherwise default-initialized (NOT zero-init): with FiLM enabled it
    modulates from step 0, which is exactly what the Phase-5 cold-start probe
    exercises. EXACT identity when FiLM is DISABLED is guaranteed UPSTREAM by
    SKIPPING FiLM entirely (the model never calls this head), not by the weights.

    One head serves a whole path: ``channels_per_level`` lists the channel count
    at each injection point (encoder: [32,64,128,256]; decoder: [256,128,64,32]).
    The MLP emits ``2 * sum(channels_per_level)`` numbers, sliced per level into
    the per-level gamma/beta tensors of shape ``[B, C_level, 1, 1]``.
    """

    def __init__(self, channels_per_level: List[int], hidden: int = 64) -> None:
        super().__init__()
        self.channels_per_level = list(channels_per_level)
        self.total = sum(self.channels_per_level)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.total),
        )

    def forward(self, cond: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # cond: [B, 1] conditioning scalar (the DETACHED predicted_sqrt_v).
        b = cond.shape[0]
        params = self.mlp(cond)                       # [B, 2*total]
        gamma_deltas = params[:, :self.total]         # [B, total]
        betas = params[:, self.total:]                # [B, total]
        coeffs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        i = 0
        for c in self.channels_per_level:
            # gamma = 1 + delta -> residual around identity (see class docstring).
            gamma = (1.0 + gamma_deltas[:, i:i + c]).view(b, c, 1, 1)
            beta = betas[:, i:i + c].view(b, c, 1, 1)
            coeffs.append((gamma, beta))
            i += c
        return coeffs


class BoundaryDistributionHead(nn.Module):
    """Probabilistic boundary-distribution segmentation head (flag-gated, ADDITIVE).

    Alternative to the binary mask head: for each A-scan COLUMN, predicts a
    DISTRIBUTION over depth (the H axis) for the choroid's upper boundary
    (choroid start) and lower boundary (choroid-sclera interface), instead of a
    per-pixel binary mask. Motivation (see loss.py ``BoundaryDistributionLoss``):
    diagnostics on this project's existing mask model show its errors are pure
    BOUNDARY DISPLACEMENT of an otherwise clean, topologically-perfect band (the
    residual error IS genuine depth ambiguity at the choroid-sclera interface),
    so modeling the boundary as a distribution lets the network express WHERE it
    is uncertain instead of being forced to commit to a single displaced line.

    A single 1x1 Conv2d maps the SegDecoder's final PRE-HEAD feature map (the
    same ``in_channels``-channel tensor the existing binary-mask 1x1 head already
    consumes) to 2 raw logit channels -- [upper, lower]. No activation here: the
    caller applies ``softmax`` over the H axis (dim=2) downstream (see loss.py),
    mirroring how ``SegDecoder`` returns raw BCEWithLogits-ready logits rather
    than a Sigmoid probability.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)   # [B, 2, H, W] raw logits (upper, lower)


class BoundaryMixtureHead(nn.Module):
    """MULTI-MODAL boundary head: a per-column MIXTURE of ``K`` Gaussians over depth,
    rendered onto the H grid so it is a BYTE-COMPATIBLE drop-in for
    ``BoundaryDistributionHead`` (flag-gated via ``--boundary_mixture K``, default OFF).

    WHY (interval SHARPNESS): the single per-column softmax
    (BoundaryDistributionHead) is effectively UNIMODAL. On an ambiguous column its
    ONLY way to express "the boundary is at depth a OR depth b" is a single BROAD
    blob spanning both -- which OVER-widens the derived std / prediction interval
    (thickness_uncertainty.py: ~+-19px on a ~78px choroid). A mixture can instead be
    SHARP-and-BIMODAL: two narrow components at a and b, so E[z]/std reflect genuine
    "here or here" ambiguity WITHOUT inflating the width where the model is actually
    confident. The head uses 1 component where unambiguous, 2+ where genuinely
    bimodal, learned from the SAME Gaussian soft-target NLL (no loss change needed).

    OUTPUT CONTRACT (identical to BoundaryDistributionHead): returns ``[B,2,H,W]``
    where channel 0/1 = upper/lower boundary and dim=2 (H) is the depth axis. The
    returned tensor is the per-column NORMALIZED LOG-PMF over depth, so the exact
    same downstream ``F.softmax(logits, dim=2)`` (boundary_distribution_to_depths)
    and ``F.log_softmax(logits, dim=2)`` (loss.BoundaryDistributionLoss) recover the
    mixture pmf / log-pmf UNCHANGED -- softmax(log_pmf) == pmf, log_softmax(log_pmf)
    == log_pmf, because the head already normalizes over depth. Everything after the
    head (E[z], std, rasterize, conformal, thickness) is therefore byte-compatible.

    PARAMETERIZATION: per boundary, per A-scan COLUMN, K components
    (weight, mean, log-scale) are predicted from the H-pooled per-column feature
    (Conv1d over the channel axis). Means are bounded to ``[0, H-1]`` via a sigmoid;
    scales are ``softplus + sigma_floor`` so a component can be SHARP (small sigma)
    but never collapse to a spike. The mixture pmf is assembled in LOG-SPACE via
    ``logsumexp`` (numerically stable; no underflow) and normalized over depth.
    """

    def __init__(self, in_channels: int, num_components: int = 2,
                 sigma_floor: float = 0.5) -> None:
        super().__init__()
        self.k = int(num_components)
        self.sigma_floor = float(sigma_floor)
        # 2 boundaries x K components x 3 params (weight logit, mean-raw, log-scale-raw),
        # predicted per COLUMN from the H-pooled feature.
        self.proj = nn.Conv1d(in_channels, 2 * self.k * 3, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        pooled = feat.mean(dim=2)                       # [B, C, W] per-column context
        params = self.proj(pooled).view(b, 2, self.k, 3, w)
        w_logit = params[:, :, :, 0, :]                 # [B,2,K,W]
        mean_raw = params[:, :, :, 1, :]
        logsig_raw = params[:, :, :, 2, :]

        log_pi = torch.log_softmax(w_logit, dim=2)      # mixture weights (over K)
        mu = (h - 1) * torch.sigmoid(mean_raw)          # means in [0, H-1]
        sigma = torch.nn.functional.softplus(logsig_raw) + self.sigma_floor

        grid = torch.arange(h, device=feat.device, dtype=feat.dtype
                            ).view(1, 1, 1, h, 1)        # depth axis
        mu_ = mu.unsqueeze(3)                            # [B,2,K,1,W]
        sig_ = sigma.unsqueeze(3)
        log_pi_ = log_pi.unsqueeze(3)
        # log N(h; mu, sigma) up to a per-column constant (the -0.5*log(2*pi) term is
        # identical across depth AND components, so it cancels in the depth
        # normalization below -- dropped for cleanliness, not an approximation).
        log_norm = -0.5 * ((grid - mu_) / sig_) ** 2 - torch.log(sig_)   # [B,2,K,H,W]
        log_mix = torch.logsumexp(log_pi_ + log_norm, dim=2)             # [B,2,H,W]
        # Normalize over depth so softmax(log_pmf)==pmf downstream (see class docstring).
        log_pmf = log_mix - torch.logsumexp(log_mix, dim=2, keepdim=True)
        return log_pmf                                   # [B,2,H,W] normalized log-pmf


class DeepOCTUltra(nn.Module):
    """Dual-decoder multi-task U-Net.

    Ablation Study (``ablation_phase``):
      * Phase 1 — SEGMENTATION ONLY. The DenoiseDecoder, unbiasing branch and
        cross-decoder gating are NOT built and NOT run (saving VRAM/compute);
        ``forward`` returns ``(None, seg_logits, None)``.
      * Phases 2-4 — the full multi-task model, configured by the architecture
        flags exactly as before.

    forward(x: [B, 3, H, W]) ->
      * ablation_phase == 1 (segmentation only):
            (None, seg_logits [B,1,H,W], None)
      * ablation_phase == 2 (multi-task, no bias):
            (final_pred [B,1,H,W] LINEAR = exp(denoised_log) - 0.0,
             seg_logits [B,1,H,W], bias=0.0 scalar tensor)
            NOTE: the returned denoised tensor is in the LINEAR domain (post-exp).
      * ablation_phase == 3 (multi-task + static Spectralis bias):
            (final_pred [B,1,H,W] LINEAR = exp(denoised_log) - 0.12549,
             seg_logits [B,1,H,W], bias=0.12549 scalar tensor)
            Identical pipeline to Phase 2 except the constant bias value.
      * ablation_phase == 4 (multi-task + LEARNED dynamic bias — final form):
            (final_pred [B,1,H,W] LINEAR = exp(denoised_log) - predicted_bias,
             seg_logits [B,1,H,W], predicted_bias [B,1,1,1] in [0,1])
            SHARES the Phase 2/3 OUTPUT-exp pipeline; the only difference is that
            the subtracted bias is a LEARNED, non-negative per-image scalar from
            the GAP predictor (supervised in loss.py toward the dataloader's KNOWN
            injected speckle level sqrt(v)) instead of the constant 0.0 / 0.12549.
      * ablation_phase == 5 (Whole-Image Speckle + Dual-Role FiLM): handled by the
            dedicated ``_forward_phase5`` (two-pass: an unconditioned predictor
            pass, then a FiLM-conditioned reconstruction). The predicted noise
            level ``predicted_sqrt_v`` is DETACHED and drives FiLM on BOTH the
            shared encoder and the denoise decoder — it is NOT subtracted as a DC
            term (the Phase-4 subtraction is REMOVED; that is the 4->5 ablation).
            Returns (final_pred = exp(clamp(denoised_log)) [no subtraction],
            seg_logits, predicted_sqrt_v [B,1,1,1]). ``film_enabled=False`` makes
            FiLM EXACT identity (Stage-1 warm-start).

    NOTE (retired path): the old ``use_unbiasing`` INPUT-exp formulation
    (linearize the input pre-encoder, subtract the bias from the log-domain
    decoder output) has been REMOVED. ``use_unbiasing`` is now an always-False
    attribute kept only so legacy flag/logging plumbing stays valid; the baseline
    2-tuple ``(denoised_output, seg_logits)`` return is unreachable for the
    in-spec phases 1-4 and remains only as a defensive fallback.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32,
                 use_unbiasing: bool = False, use_wavelets: bool = False,
                 use_cross_gating: bool = False, ablation_phase: int = 4,
                 film_enabled: bool = True,
                 noise_adaptive_conditioning: bool = False,
                 noise_signal: str = "predicted",
                 boundary_distribution_head: bool = False,
                 boundary_mixture: int = 0) -> None:
        super().__init__()
        self.ablation_phase = ablation_phase
        # Phase 1 is segmentation-only: every denoiser-dependent module is
        # force-disabled here regardless of the incoming flags, so the denoiser,
        # unbiasing branch and cross-gating are never built or run.
        self.seg_only = ablation_phase == 1
        # Phases 2 & 3 = Multi-Task with a CONSTANT bias (0.0 in Phase 2, the
        # static Spectralis 0.12549 in Phase 3). ablation_phase drives the
        # architecture: force the DenoiseDecoder + uncertainty-aware cross-gating
        # ON and the DYNAMIC bias-predictor branch OFF regardless of the incoming
        # flags. The bias-predictor module is therefore NOT constructed in Phase
        # 2/3 (VRAM bypass per CLAUDE.md); forward() subtracts a constant bias.
        static_bias_multitask = ablation_phase in (2, 3)
        # Phase 4 = Multi-Task with a LEARNED, per-image DYNAMIC bias (replaces
        # Phase 3's static 0.12549). UNLIKE Phase 2/3 the GAP bias-predictor IS
        # constructed here, and cross-gating is forced ON exactly as in Phase 2/3.
        # Phase 5 (Whole-Image Speckle) shares Phase 4's model path EXACTLY for now
        # (predictor built, same exp()-minus-predicted_bias subtraction); only the
        # dataset's speckle scope differs. FiLM / subtraction-removal is a LATER
        # round — NOT here. Hence phase 5 is grouped with phase 4 throughout.
        self.dynamic_bias_multitask = ablation_phase in (4, 5)

        # The legacy input-exp / `use_unbiasing` numeric path has been REMOVED:
        # Phase 4 now shares Phase 2/3's OUTPUT-exp formulation (exp AFTER the
        # log-domain decoder), differing only in that the subtracted bias is
        # learned. `use_unbiasing` is retained as an always-False attribute so the
        # older CLI flag / logging plumbing keeps working; it drives nothing.
        self.use_unbiasing = False
        self.use_wavelets = use_wavelets
        self.use_cross_gating = ((use_cross_gating or static_bias_multitask
                                  or self.dynamic_bias_multitask)
                                 and not self.seg_only)

        self.encoder = SharedEncoder(in_channels=in_channels, base_filters=base_filters)

        enc_out = self.encoder.out_channels           # 256
        bottleneck_channels = enc_out * 2             # 512
        # Step 5: DT-CWT + Horizontal Axial Attention bottleneck behind a flag.
        # Both variants take enc_out -> bottleneck_channels at the same spatial
        # resolution, so they are interchangeable for the decoders / bias branch.
        if self.use_wavelets:
            self.bottleneck = DTCWTBottleneck(enc_out, bottleneck_channels)
        else:
            # Plain CNN bottleneck: 2 conv blocks (Steps 2-4 baseline).
            self.bottleneck = DoubleConv(enc_out, bottleneck_channels)

        skip_channels = self.encoder.stage_channels   # [32, 64, 128, 256]
        # Phase 1: skip building the DenoiseDecoder entirely (VRAM/compute save).
        self.denoise_decoder = (
            None if self.seg_only
            else DenoiseDecoder(skip_channels, bottleneck_channels))
        self.seg_decoder = SegDecoder(skip_channels, bottleneck_channels)

        # Phase 4 ONLY: build the GAP dynamic bias-predictor. Phase 2/3 do NOT
        # build it (so ``hasattr(model, 'bias_estimator')`` stays False there —
        # asserted by verify_phase3.py); the retired input-exp path no longer
        # constructs Linearization/BiasEstimator at all.
        if self.dynamic_bias_multitask:
            self.bias_estimator = BiasEstimator(bottleneck_channels)

        # Step 7: uncertainty-aware cross-decoder gating. One gate per decoder
        # block; channels follow the decode order (deep -> shallow).
        if self.use_cross_gating:
            decode_channels = list(reversed(skip_channels))   # [256, 128, 64, 32]
            self.confidence = UncertaintyConfidence()
            self.cross_gates = nn.ModuleList(
                CrossDecoderGate(ch) for ch in decode_channels)

        # Phase 5 ONLY (Step 2): dual-role FiLM conditioning driven by the
        # predictor's estimated noise level. Built LAST and gated strictly on
        # phase 5, so phases 1-4 construct an IDENTICAL module set in an IDENTICAL
        # order — the parameter-init RNG stream is unchanged and their weights stay
        # byte-identical. Two SEPARATE heads (per the spec): one conditions the
        # shared encoder path, one conditions the denoise-decoder path.
        self.is_phase5 = ablation_phase == 5
        # film_enabled is meaningful ONLY in Phase 5; force it False elsewhere so
        # phases 1-4 never build or run FiLM regardless of the incoming flag.
        self.film_enabled = bool(film_enabled) and self.is_phase5
        if self.is_phase5:
            # encoder injection channels in forward order; decoder in decode order.
            self.film_encoder_head = FiLMGenerator(self.encoder.stage_channels)
            self.film_decoder_head = FiLMGenerator(list(reversed(skip_channels)))

        # NOISE-ADAPTIVE CONDITIONING (flag-gated, default OFF). Modulates the
        # cross-decoder coupling per image by a monotonic function of the PREDICTED
        # noise level, so the model can couple denoising MORE on noisier images and
        # LESS on clean ones — the per-image, class-dependent effect a single global
        # loss weight cannot express. Built LAST (after the FiLM heads) so that with
        # the flag OFF phases 1-5 construct an IDENTICAL module set in an IDENTICAL
        # order -> the parameter-init RNG stream is unchanged and every existing
        # weight stays byte-identical. Requires the cross-gate pathway it modulates
        # (and a denoiser), so it is force-disabled for seg-only Phase 1.
        self.noise_adaptive_conditioning = (bool(noise_adaptive_conditioning)
                                            and self.use_cross_gating
                                            and not self.seg_only)
        # Diagnostics stashed each forward (plain attrs, NOT buffers -> not in the
        # state_dict) so train.py can log the predicted-noise <-> gate-scale relation.
        self._last_noise_pred = None
        self._last_gate_scale = None
        # Which per-image signal drives the conditioner:
        #   'predicted' (DEFAULT) — the GAP noise predictor's output (the injected-
        #                speckle estimate). This is the ORIGINAL behavior; it was
        #                inconclusive because the injected speckle is ~uniform, so the
        #                predicted noise was ~flat across images (nothing to act on).
        #   'intrinsic' — a REAL per-image intrinsic-noise scalar measured from the
        #                CLEAN input B-scan (Yang-Tai, passed in via forward's
        #                ``intrinsic_noise`` arg), normalized here by a running
        #                z-score so it genuinely distinguishes noisier (AMD) from
        #                cleaner (normal) scans. This is the FIX.
        if noise_signal not in ("predicted", "intrinsic"):
            raise ValueError(
                f"noise_signal must be 'predicted' or 'intrinsic', got {noise_signal!r}")
        self.noise_signal = str(noise_signal)
        if self.noise_adaptive_conditioning:
            self.noise_conditioner = NoiseAdaptiveConditioner()
            if self.noise_signal == "predicted":
                # Reuse the existing GAP predictor when the phase already builds one
                # (phases 4/5 -> self.bias_estimator). Phases 2/3 have NO predictor,
                # so build a dedicated one ONLY here (flag-on) to supply the
                # conditioning signal; flag-off 2/3 never construct it.
                if not self.dynamic_bias_multitask:
                    self.cond_noise_predictor = BiasEstimator(bottleneck_channels)
            else:
                # 'intrinsic': normalize the passed-in per-image intrinsic-noise
                # scalar by a RUNNING z-score (train-updated, frozen at eval), so the
                # conditioner sees a well-scaled O(1) signal regardless of the raw
                # Yang-Tai magnitude. Buffers (checkpointed, moved with .to(device),
                # NOT gradient-tracked) — mirrors the loss.py EMA-normalization idea.
                # These consume NO init RNG, so base weights stay byte-identical.
                self.register_buffer("intr_ema_mean", torch.zeros(()))
                self.register_buffer("intr_ema_var", torch.ones(()))
                self.register_buffer("intr_ema_init", torch.zeros(()))

        # BOUNDARY-DISTRIBUTION HEAD (flag-gated, default OFF, fully ADDITIVE).
        # An alternative segmentation head: predicts a per-column DEPTH
        # DISTRIBUTION for the choroid's upper/lower boundary instead of a binary
        # mask (see BoundaryDistributionHead / loss.py BoundaryDistributionLoss).
        # It consumes the SAME SegDecoder pre-head feature the existing mask head
        # already computes -- no new encoder/decoder path. Built LAST (after
        # EVERY other module, including the intrinsic-noise buffers above) so
        # that with the flag OFF every phase constructs an IDENTICAL module set
        # in an IDENTICAL order -> the parameter-init RNG stream (and therefore
        # every existing weight) stays byte-identical. The output is NOT added to
        # forward()'s return tuple (that would change the tuple length/contract
        # for every existing caller, flag on or off); instead it is stashed on
        # ``self.last_boundary_logits`` each forward, mirroring the existing
        # ``_last_noise_pred`` / ``_last_gate_scale`` diagnostic-stash pattern --
        # so the forward() return signature is byte-identical REGARDLESS of this
        # flag's value.
        self.boundary_distribution_head = bool(boundary_distribution_head)
        # OPTION A (interval sharpness, flag-gated, default 0 = OFF): a MULTI-MODAL
        # mixture boundary head REPLACING the single-softmax BoundaryDistributionHead.
        # It is a byte-COMPATIBLE drop-in (same [B,2,H,W] normalized-log-pmf output,
        # stashed on the same self.last_boundary_logits), so forward()/loss/eval are
        # UNCHANGED. When boundary_mixture == 0 (default) the ORIGINAL
        # BoundaryDistributionHead is built EXACTLY as before -> byte-identical; the
        # mixture is a DISTINCT experimental arm with its own parameter count, so a
        # mixture checkpoint is a separate architecture (evaluate.build_model reads
        # boundary_mixture from ckpt_args to rebuild it identically).
        self.boundary_mixture = int(boundary_mixture)
        self.last_boundary_logits = None
        if self.boundary_distribution_head:
            if self.boundary_mixture > 0:
                self.boundary_head = BoundaryMixtureHead(
                    skip_channels[0], num_components=self.boundary_mixture)
            else:
                self.boundary_head = BoundaryDistributionHead(skip_channels[0])

    def _intrinsic_gate_signal(self, intrinsic_noise, batch_size: int,
                               device, dtype):
        """Turn the raw per-image intrinsic-noise scalar into the conditioner input.

        Returns ``(signal, raw)`` both shaped ``[B,1,1,1]`` (detached): ``signal`` is
        the running-z-scored intrinsic used by the conditioner; ``raw`` is the
        UN-normalized intrinsic kept for interpretable logging. When
        ``intrinsic_noise`` is None (e.g. an eval script that cannot supply it), the
        signal degrades to zeros -> a constant scale ``sigmoid(b)`` (safe no-op
        modulation), so ``model(x)`` still runs without the intrinsic input.
        """
        if intrinsic_noise is None:
            z = torch.zeros(batch_size, 1, 1, 1, device=device, dtype=dtype)
            return z, z
        raw = intrinsic_noise.reshape(-1).to(device=device, dtype=dtype).detach()
        m = 0.99
        if self.training and torch.is_grad_enabled():
            bmean = raw.mean()
            if float(self.intr_ema_init) == 0:
                self.intr_ema_mean.copy_(bmean)
                bvar = raw.var(unbiased=False)
                self.intr_ema_var.copy_(bvar if float(bvar) > 1e-8
                                        else torch.ones_like(bvar))
                self.intr_ema_init.fill_(1)
            else:
                self.intr_ema_mean.mul_(m).add_((1.0 - m) * bmean)
                dev2 = ((raw - self.intr_ema_mean) ** 2).mean()
                self.intr_ema_var.mul_(m).add_((1.0 - m) * dev2)
        std = torch.sqrt(self.intr_ema_var + 1e-6)
        z = ((raw - self.intr_ema_mean) / std).clamp(-5.0, 5.0)
        return z.view(-1, 1, 1, 1), raw.view(-1, 1, 1, 1)

    def _conditioning_scale(self, bottleneck, x, intrinsic_noise):
        """Compute the per-image cross-gate scale from whichever noise signal the
        run is configured for. Returns ``(gate_scale, cond_bias)`` where
        ``cond_bias`` is the (grad-on) predictor output to REUSE for the Phase-4/5
        subtraction (predicted mode only; None otherwise). Also stashes the
        interpretable noise value + scale for logging."""
        if self.noise_signal == "intrinsic":
            signal, raw = self._intrinsic_gate_signal(
                intrinsic_noise, x.shape[0], bottleneck.device, bottleneck.dtype)
            cond_bias = None
            self._last_noise_pred = raw            # RAW intrinsic (interpretable)
        else:  # predicted
            if self.dynamic_bias_multitask:
                cond_bias = self.bias_estimator(bottleneck)     # [B,1,1,1] in [0,1]
                signal = cond_bias.detach()
            else:
                cond_bias = None
                signal = self.cond_noise_predictor(bottleneck).detach()
            self._last_noise_pred = signal.detach()
        gate_scale = self.noise_conditioner(signal)             # [B,1,1,1] in (0,1)
        self._last_gate_scale = gate_scale.detach()
        return gate_scale, cond_bias

    def forward(self, x: torch.Tensor, intrinsic_noise: torch.Tensor = None):
        # Phase 1: segmentation-only path. Encoder -> bottleneck -> seg decoder.
        # No linearization, no denoiser, no gating. Denoised + bias are None.
        if self.seg_only:
            deep_features, skips = self.encoder(x)
            bottleneck = self.bottleneck(deep_features)
            if self.boundary_distribution_head:
                # forward_collect computes the IDENTICAL head(current) as plain
                # forward() (same loop, same final `current`), so seg_logits here
                # is byte-identical to the non-collecting call -- only routed
                # through forward_collect (flag ON) so its pre-head feature
                # (feats[-1]) is available for the boundary head too.
                seg_logits, seg_feats = self.seg_decoder.forward_collect(bottleneck, skips)
                self.last_boundary_logits = self.boundary_head(seg_feats[-1])
            else:
                seg_logits = self.seg_decoder(bottleneck, skips)
            return None, seg_logits, None

        # Phase 5 (Whole-Image Speckle + Dual-Role FiLM) has its OWN forward: a
        # two-pass schedule (unconditioned predictor pass, then FiLM-conditioned
        # reconstruction) with NO DC subtraction. Dispatched here so phases 2/3/4
        # below stay on their original code path, byte-for-byte unchanged.
        if self.ablation_phase == 5:
            return self._forward_phase5(x, intrinsic_noise)

        # NOTE: the legacy input-exp linearization (encoder_input = exp(x)) has
        # been REMOVED. All multi-task phases (2/3/4) encode the raw log-domain
        # input and apply the deterministic exp() linearization on the DECODER
        # OUTPUT instead (output-exp), so the encoder always sees ``x`` directly.
        deep_features, skips = self.encoder(x)
        bottleneck = self.bottleneck(deep_features)

        # NOISE-ADAPTIVE CONDITIONING (flag-gated): compute the per-image cross-gate
        # scale from the configured noise signal BEFORE decoding. 'predicted' reads
        # the bottleneck GAP predictor (the ORIGINAL, inconclusive path); 'intrinsic'
        # uses the REAL per-image intrinsic-noise scalar passed in ``intrinsic_noise``
        # (the FIX). Both feed the same learnable a,b conditioner. When OFF,
        # gate_scale/cond_bias stay None -> the cross-gate and the Phase-4 bias call
        # below run exactly as before (byte-identical).
        gate_scale = None
        cond_bias = None   # phases 4/5 (predicted mode): reuse this predictor call
        if self.noise_adaptive_conditioning:
            gate_scale, cond_bias = self._conditioning_scale(
                bottleneck, x, intrinsic_noise)

        if self.use_cross_gating:
            # Seg decoder runs FIRST and caches its per-block features, so the
            # confidence map and those features can gate the denoise decoder.
            # Nothing is detached: the denoise loss flows back into the seg branch
            # through both the gated features and the confidence map.
            seg_logits, seg_feats = self.seg_decoder.forward_collect(bottleneck, skips)
            confidence = self.confidence(seg_logits)
            denoised_output = self.denoise_decoder.forward_gated(
                bottleneck, skips, seg_feats, confidence, self.cross_gates,
                gate_scale=gate_scale)
            if self.boundary_distribution_head:
                self.last_boundary_logits = self.boundary_head(seg_feats[-1])
        else:
            denoised_output = self.denoise_decoder(bottleneck, skips)
            seg_logits = self.seg_decoder(bottleneck, skips)

        if self.ablation_phase in (2, 3):
            # Phases 2 & 3 share ONE code path; they differ ONLY by the bias
            # constant. The DenoiseDecoder operates and outputs in the LOG domain
            # (multiplicative OCT speckle is additive after log). Apply the SAME
            # DETERMINISTIC, non-learned output linearization
            # linear_pred = exp(clamp(denoised_log, max=_EXP_CLAMP_MAX)), fully
            # DECOUPLED from any bias branch and clamped to prevent inf. Then
            # subtract a CONSTANT bias in the LINEAR domain AFTER exp():
            #   Phase 2 -> 0.0                               (no correction)
            #   Phase 3 -> _SPECTRALIS_STATIC_BIAS = 0.12549 (static Spectralis DC offset)
            # The dynamic bias-predictor module is NOT built in either phase, so
            # the bias here is a fixed constant (no supervision). The RETURNED
            # denoised tensor is in the LINEAR domain (post-exp, post-bias).
            linear_pred = torch.exp(torch.clamp(denoised_output, max=_EXP_CLAMP_MAX))
            bias_value = _SPECTRALIS_STATIC_BIAS if self.ablation_phase == 3 else 0.0
            bias = torch.full((), bias_value, device=linear_pred.device,
                              dtype=linear_pred.dtype)
            final_pred = linear_pred - bias   # Phase 2: bias 0.0 -> == linear_pred
            # Bound the reconstruction to the physical [0,1] (+headroom) range so
            # the exp-linearized magnitude can never drift large enough to overflow
            # the recon losses (see _RECON_CLAMP_* above).
            final_pred = torch.clamp(final_pred, _RECON_CLAMP_MIN, _RECON_CLAMP_MAX)
            return final_pred, seg_logits, bias

        if self.ablation_phase == 4:
            # Phase 4 — Dynamic Unbiasing (final form). Phase 5 NO LONGER shares
            # this path: it is dispatched to _forward_phase5 above (FiLM + no
            # subtraction). Phase 4 KEEPS its bias subtraction, unchanged — the
            # subtraction-vs-FiLM contrast is the Phase 4 -> 5 ablation.
            # SHARES Phase 2/3's
            # OUTPUT-exp formulation exactly: the DenoiseDecoder outputs in the LOG
            # domain, then the SAME deterministic linearization
            # linear_pred = exp(clamp(denoised_log, max=_EXP_CLAMP_MAX)). The only
            # difference from Phase 3 is the bias: instead of the constant 0.12549
            # we subtract the LEARNED, non-negative per-image scalar predicted by
            # the GAP bias-predictor (broadcast over H,W), in the LINEAR domain
            # AFTER exp(). The bias is supervised in loss.py toward the dataloader's
            # KNOWN injected speckle level sqrt(v). The RETURNED denoised tensor is
            # in the LINEAR domain (post-exp, post-bias).
            #
            # CAVEAT (research question, intentionally NOT changed this round): under
            # the new supervision `predicted_bias` is an estimated LOG-DOMAIN SPECKLE
            # STD (a NOISE-LEVEL estimate), NOT a DC offset. We KEEP the existing
            # `final_pred = exp(clamp(log)) - predicted_bias` subtraction unchanged
            # for now; whether a noise-level scalar should be subtracted as a DC term
            # vs. used to MODULATE denoising is a separate question deferred to a
            # later round.
            linear_pred = torch.exp(torch.clamp(denoised_output, max=_EXP_CLAMP_MAX))
            # Reuse the single predictor call made for conditioning when the flag is
            # on (cond_bias is that exact tensor, grad-on -> Bias MSE still trains
            # the predictor). Flag off: cond_bias is None -> call it here as before,
            # so this line is byte-identical on the default path.
            predicted_bias = (cond_bias if cond_bias is not None
                              else self.bias_estimator(bottleneck))  # [B,1,1,1] in [0,1]
            final_pred = linear_pred - predicted_bias          # broadcast per-image
            # Bound the reconstruction to the physical [0,1] (+headroom) range so
            # the exp-linearized magnitude can never drift large enough to overflow
            # the recon losses (see _RECON_CLAMP_* above).
            final_pred = torch.clamp(final_pred, _RECON_CLAMP_MIN, _RECON_CLAMP_MAX)
            return final_pred, seg_logits, predicted_bias

        return denoised_output, seg_logits

    def _forward_phase5(self, x: torch.Tensor, intrinsic_noise: torch.Tensor = None):
        """Phase 5 — Whole-Image Speckle + Dual-Role FiLM (NO DC subtraction).

        TWO-PASS design (resolves the predictor/encoder ordering circularity — the
        predictor reads the bottleneck, which is downstream of the very encoder we
        want to condition):
          * Pass 1 (ALWAYS): UNCONDITIONED encoder+bottleneck -> GAP predictor ->
            ``predicted_sqrt_v``. The predictor therefore always reads CLEAN,
            unmodulated features (honest noise estimation) and is supervised ONLY
            by Bias MSE; its gradient flows back through THIS unconditioned pass.
          * Pass 2 (only when ``film_enabled``): ``predicted_sqrt_v`` is DETACHED
            and fed to the two FiLM heads, which condition a SECOND encoder pass
            and the denoise-decoder path. Detaching means NO recon/seg gradient
            reaches the predictor through FiLM — it stays honest to the noise level
            (the predictor learns ONLY from Bias MSE).

        ``film_enabled=False`` makes FiLM EXACT identity by reusing the Pass-1
        features verbatim (the FiLM heads are never called) — byte-identical to a
        no-FiLM model. This is the Stage-1 warm-start regime: the predictor trains
        on clean features exactly as in Phase 4, then FiLM is switched on (Stage 2).

        Output: ``final_pred = exp(clamp(denoised_log, max=_EXP_CLAMP_MAX))`` with
        NO bias subtraction. The noise estimate now DRIVES FiLM; it is NOT
        subtracted as a DC term (this removal is the Phase 4 -> 5 ablation; Phase 4
        KEEPS its subtraction). The 3rd return is ``predicted_sqrt_v`` (grad-on),
        which loss.py supervises via Bias MSE toward the injected ``sqrt(v)``.
        """
        b = x.shape[0]

        # --- Pass 1: UNCONDITIONED features -> noise-level predictor (grad-on) ---
        deep_u, skips_u = self.encoder(x)
        bottleneck_u = self.bottleneck(deep_u)
        predicted_sqrt_v = self.bias_estimator(bottleneck_u)   # [B,1,1,1] in [0,1]

        # NOISE-ADAPTIVE CONDITIONING (flag-gated): scale the cross-gate coupling by
        # a monotonic function of the noise signal. Composes with FiLM — FiLM
        # modulates feature CONTENT per noise, this modulates the denoise<->seg
        # COUPLING strength per noise. In 'predicted' mode the signal is the DETACHED
        # predicted_sqrt_v (unconditioned Pass-1 estimate); in 'intrinsic' mode it is
        # the real per-image intrinsic-noise scalar (``intrinsic_noise``). None (flag
        # off) -> byte-identical. NOTE: _conditioning_scale re-runs the bias_estimator
        # in predicted mode, but Pass 1 already computed predicted_sqrt_v grad-on for
        # the Bias-MSE supervision, and here we only need the DETACHED value for the
        # scale — so we call the shared helper on bottleneck_u for a consistent signal.
        gate_scale = None
        if self.noise_adaptive_conditioning:
            gate_scale, _ = self._conditioning_scale(
                bottleneck_u, x, intrinsic_noise)

        if self.film_enabled:
            # DETACH before FiLM: predictor learns ONLY from Bias MSE, never via
            # the seg/denoise losses that flow through the FiLM-conditioned passes.
            cond = predicted_sqrt_v.detach().reshape(b, 1)     # [B,1] scalar
            enc_coeffs = self.film_encoder_head(cond)          # per encoder level
            dec_coeffs = self.film_decoder_head(cond)          # per decoder level
            # --- Pass 2: CONDITIONED encoder+bottleneck for the reconstruction ---
            deep_c, skips_c = self.encoder(x, film_coeffs=enc_coeffs)
            bottleneck_c = self.bottleneck(deep_c)
        else:
            # Identity FiLM: reuse the Pass-1 features verbatim (byte-identical to
            # no-FiLM). No FiLM head is ever touched.
            dec_coeffs = None
            skips_c, bottleneck_c = skips_u, bottleneck_u

        # Decoders: cross-gating is ON for Phase 5 — the SegDecoder runs first and
        # caches its per-block features, then the FiLM-conditioned DenoiseDecoder
        # consumes them. The encoder FiLM already conditions the shared features
        # the SegDecoder reads, so segmentation is conditioned indirectly; only the
        # DenoiseDecoder receives the SEPARATE decoder-head FiLM directly.
        seg_logits, seg_feats = self.seg_decoder.forward_collect(bottleneck_c, skips_c)
        confidence = self.confidence(seg_logits)
        denoised_output = self.denoise_decoder.forward_gated(
            bottleneck_c, skips_c, seg_feats, confidence, self.cross_gates,
            film_coeffs=dec_coeffs, gate_scale=gate_scale)
        if self.boundary_distribution_head:
            self.last_boundary_logits = self.boundary_head(seg_feats[-1])

        # OUTPUT-exp, NO subtraction (Phase 5 removes the Phase-4 DC subtraction).
        final_pred = torch.exp(torch.clamp(denoised_output, max=_EXP_CLAMP_MAX))
        # Bound the reconstruction to the physical [0,1] (+headroom) range so the
        # exp-linearized magnitude can never drift large enough to overflow the
        # recon losses (see _RECON_CLAMP_* above).
        final_pred = torch.clamp(final_pred, _RECON_CLAMP_MIN, _RECON_CLAMP_MAX)
        return final_pred, seg_logits, predicted_sqrt_v
