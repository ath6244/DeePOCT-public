# DeepOCT

Choroid boundary segmentation in OCT B-scans with per-column uncertainty intervals
that are conformally calibrated, so a stated 90% interval contains the true boundary
about 90% of the time.

A choroid thickness measurement only supports a clinical decision if you know whether
an observed change is larger than the measurement's own error — which requires an
interval you can trust, not a point estimate.

## Reliability

**Alt text (upper boundary, all):** Reliability diagram plotting observed coverage
against nominal coverage for the choroid upper boundary. A dashed diagonal marks
perfect calibration. The red uncalibrated curve sits far above the diagonal across
all four nominal levels. The blue conformally calibrated curve lies close to the
diagonal.

**Alt text (thickness, abnormal):** The same reliability diagram for choroid
thickness on AMD eyes. Both the red uncalibrated curve and the blue calibrated curve
fall below the diagonal at every nominal level, with the calibrated curve the lower
of the two.

**Caption:** Reliability of the boundary-distribution head's uncertainty, before and
after split-conformal calibration, on the two held-out test patients of the
`single_split` protocol (41,477 A-scan columns over 84 B-scans). *Left:* the raw
predicted standard deviation is badly miscalibrated for the upper boundary — a
nominal 95% interval covers 99.9% of columns — and conformal calibration brings
expected calibration error from 0.3164 to 0.0399. *Right:* the same procedure fails
on thickness in AMD eyes, where ECE moves the wrong way, from 0.1284 to 0.2012. The
pooled quantile fitted across both classes does not transfer to the abnormal
subgroup. Both panels are the two-patient split; the six-patient cross-conformal
numbers in the Results section below are measured under a different protocol and
have no figure in this repository.

## Results

All figures below are in **pixels**. This repository has no recorded axial pixel
spacing for the scanner used, so no micron conversion is applied anywhere.

**Expected calibration error of 0.0022 for the upper boundary, 0.0040 for the lower
boundary, and 0.0051 for thickness**, under grouped leave-one-patient-out conformal
calibration over 6 patients and 124,804 A-scan columns.
Calibration is harder than it looks here because the quantile has to be fitted on
patients the reported number never touches; fitting and evaluating on the same eyes
would make almost any interval look calibrated.

**Observed coverage of 94.4% at a nominal 95% level for the upper boundary, with a
patient-level bootstrap CI of [91.2%, 97.2%].**
Thickness reaches 93.5% at the same level.
The CI is wide because it resamples 6 patients, not 124,804 columns — columns within
an eye are not independent evidence, and reporting a column-level CI would overstate
the precision by an order of magnitude.

**The 95% interval is 4.22 px wide at the upper boundary but 16.67 px at the lower
boundary** — a 4x asymmetry.
This is the substantive finding rather than a defect: the choroid–sclera interface is
genuinely ambiguous in these scans, and the calibrated interval widens to say so
instead of committing to a confident wrong line.

**Predicted uncertainty is discriminative, not merely calibrated: Spearman 0.4218
between predicted sigma and absolute thickness error across 124,804 columns.**
An interval can hit its nominal coverage while being uniformly wide and therefore
useless; this says the width varies with where the model is actually wrong.

## Architecture

```
3 adjacent B-scans  [B,3,H,W]          (2.5D stack: center frame carries the label)
        |
   shared encoder   4 stages, DoubleConv = Conv3x3 -> GroupNorm(8) -> GELU
                    32 -> 64 -> 128 -> 256 channels, MaxPool between stages
        |
   bottleneck       plain CNN; optional DT-CWT + horizontal axial attention
                    behind --use-wavelets
        |
   +----+----+
   |         |
 denoise    seg decoder
 decoder      |
 (not built   +--> 1x1 conv -> binary mask logits [B,1,H,W]
  in phase 1) |
              +--> 1x1 conv -> 2 channels [upper, lower], softmax over depth
                                 = per-column depth DISTRIBUTION
                                       |
                        mean -> boundary estimate
                        std  -> per-column uncertainty
                                       |
                        thickness = lower - upper, sigma propagated
                                       |
                        split conformal: halfwidth = q_hat(level) * sigma
                        q_hat fitted on held-out calibration patients
```

No BatchNorm anywhere; OCT speckle variance violates BatchNorm's batch-statistics
assumption, so the network uses GroupNorm throughout.
The boundary-distribution head is a single 1x1 convolution on the segmentation
decoder's pre-head feature map, producing two raw logit channels that are softmaxed
over the depth axis.
It exists because diagnostics showed the mask model's residual errors were pure
boundary displacement of an otherwise topologically clean band — genuine depth
ambiguity at the choroid–sclera interface — so modelling the boundary as a
distribution lets the network express where it is uncertain instead of committing to
a single displaced line.

The calibration step is deliberately separate from training: it loads an existing
checkpoint read-only and fits distribution-free conformal quantiles, because a
predicted sigma that merely *correlates* with error is not a guarantee that a stated
90% interval contains the truth 90% of the time.

## Repo structure

```
src/deepoct/
  model.py                        DeepOCTUltra: dual-decoder U-Net, DT-CWT and axial
                                  attention bottleneck, boundary-distribution and
                                  boundary-mixture heads
  losses.py                       DeepOCTLoss (ZNCC + GAT variance + FFL + BCE/Dice)
                                  and BoundaryDistributionLoss with decode helpers
  joint_taskaware_model.py        Denoiser front-end + segmenter trained end-to-end
                                  on segmentation loss alone
  noise_estimation.py             Per-image speckle sigma estimation
  train_supervised_denoiser.py    DenoiseUNet (residual, zero-init tail = identity
                                  at initialization) and its training loop
  sam.py                          Sharpness-Aware Minimization optimizer wrapper
  provenance.py                   Run provenance stamped into every result JSON
  calibration/
    thickness_uncertainty.py      Split conformal and grouped leave-one-patient-out
                                  conformal; coverage tables, ECE, reliability plots
  datasets/
    oimhs_dataset.py              OIMHS loader; 2.5D stacking, speckle, log transform
    sdoct_dataset.py              SDOCT loader; subclasses OIMHS, fills contour masks
    build_sdoct_manifests.py      Patient-level class-stratified train/val manifests
    parse_roi_masks.py            ROI mask parsing
scripts/
  train.py                        Training entry point
  evaluate.py                     Checkpoint evaluation and metrics JSON
  predict.py                      Single-image inference figure
results/                          Result JSONs and reliability diagrams
tests/                            Empty placeholder
requirements.txt                  Runtime dependencies except torch
LICENSE                           Placeholder; no license selected yet
```

## Setup

There is no `pyproject.toml`. Install torch first, matched to your CUDA driver, then
the rest — `pytorch_wavelets` and `focal-frequency-loss` both declare `torch` as a
dependency and will otherwise pull an arbitrary wheel.

```bash
nvidia-smi                                    # read the driver's CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your driver
pip install -r requirements.txt
```

Pins are the locally validated set: Python 3.14, torch 2.12, numpy 2.4.6,
pandas 3.0.3, Pillow 12.2.0, matplotlib 3.11.0, focal-frequency-loss 0.3.0,
pytorch_wavelets 1.3.0, PyWavelets 1.9.0, openpyxl 3.1.5.
torchvision is not used.

Scripts in `scripts/` add `src/` to `sys.path` themselves, so no install step is
needed to run them.
Running the calibration module directly needs `PYTHONPATH=src`.

## Usage

Train (segmentation only — `--ablation_phase` accepts only `1` in this repository):

```bash
python scripts/train.py \
  --data_dir <dir with train_manifest.csv and val_manifest.csv> \
  --dataset sdoct \
  --boundary_distribution_head \
  --image-size 512 \
  --epochs 100 --batch_size 4 --learning_rate 1e-4 \
  --checkpoint_dir checkpoints
```

Defaults: `--batch_size 4`, `--epochs 100`, `--learning_rate 0.0001`,
`--base_filters 32`, `--weight_decay 0.01`, `--grad_clip 1.0`, `--image-size 512`,
`--dataset sdoct`, `--best_metric dice`, `--num-workers 4`, `--seed 0`.
Boundary-head defaults: `--boundary_dist_mode joint`, `--lambda_boundary_dist 1.0`,
`--boundary_dist_sigma 2.0`, `--lambda_boundary_dist_expectation 0.1`,
`--boundary_mixture 0`. Optional: `--use-wavelets`, `--amp`, `--use_sam`
(`--sam_rho 0.05`), `--isotropic_resize`, `--mock` for a synthetic smoke run.

Evaluate:

```bash
python scripts/evaluate.py \
  --checkpoint <path.pth> --data_dir <dir> --ablation_phase 1 \
  --out metrics.json --dump_scans
```

`--checkpoint`, `--data_dir` and `--ablation_phase` are required.
`--boundary_decode` defaults to `expectation`; `viterbi` applies a smoothness
constraint of `--boundary_decode_max_jump 5` pixels.

Predict on one B-scan:

```bash
python scripts/predict.py \
  --image_path <scan.tif> --checkpoint_path <path.pth> --output_path out.png
```

All three arguments are required.

Calibrate:

```bash
PYTHONPATH=src python -m deepoct.calibration.thickness_uncertainty \
  --checkpoint runs/boundary_dist_sdoct/best_model_phase1.pth \
  --data_dir <dir> --out_dir thickness_uncertainty_out \
  --calibration_mode cross --conformal_diagnostics --dump_columns
```

`--calibration_mode` is `single_split` by default; `cross` runs grouped
leave-one-patient-out with a patient-level bootstrap (`--n_bootstrap 2000`).
`--calib_fraction 0.5` and `--seed 42` control the calibration/test patient split,
`--coverage_levels` defaults to `0.50,0.68,0.90,0.95`, and `--conformal_per_class`
fits separate quantiles for normal and AMD eyes. `--selftest` verifies the conformal
and bootstrap arithmetic on synthetic columns with no checkpoint or data.

## Limitations

**The data cannot be redistributed here.** Scans, derived arrays and manifests are
excluded by `.gitignore`; manifests embed absolute paths and patient identifiers and
must be regenerated locally.
Patient and eye identifiers in the published result JSONs were replaced with stable
pseudonyms before release, with no numeric result altered and the mapping kept out of
this repository.
Nothing here runs end to end without the private dataset and a trained checkpoint.

**Everything rests on very few patients.** The cross-conformal result pools 6
patients; the single-split result fits on 4 and reports coverage on 2.
The bootstrap CIs are correspondingly wide — the 90% normal-eye upper-boundary
interval spans [77.8%, 96.0%].
The column counts are large, but columns within an eye are not independent samples.

**Calibration does not transfer to AMD eyes under a pooled quantile.** On the
single-split protocol, thickness ECE for abnormal eyes rises from 0.1284
uncalibrated to 0.2012 calibrated, and lower-boundary ECE from 0.0932 to 0.1550.
The code flags this itself rather than reporting the pooled number alone.
The independence assumption behind thickness propagation also breaks down in this
subgroup: upper and lower boundary errors correlate at r = 0.545 in abnormal eyes
versus r = -0.022 in normal eyes.

**No micron conversion.** All thickness and interval figures are in pixels, because
no axial pixel spacing for this scanner is recorded anywhere in the codebase. Micron
conversion needs the acquisition protocol's axial scale, which is not in this
repository, so none of these numbers can be compared directly against published
thickness values.

**Single institution, single scanner.** The calibration set is Bioptigen SDOCT data
from one source. Nothing here tests transfer to another device or site.

**The segmentation target rests on an unconfirmed annotation assumption.** The loader
treats the yellow `(255,255,0)` contour as tracing the choroid, matching the OIMHS
colour convention and the visible band, but this is flagged in the source as not yet
confirmed with the supervisor.

**The denoiser comparison is small and partly excluded.** It covers 11 aligned image
pairs; 6 further folders were dropped as misaligned.

**`tests/` is empty** and the repository has no license selected yet.

---

Drawn from an ongoing, unpublished research project; the results here are internal
and have not been peer reviewed.
