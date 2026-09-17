"""thickness_uncertainty.py -- CLINICAL PAYOFF: choroid thickness measurement
with a CALIBRATED uncertainty interval, from the boundary-distribution head.

READ-ONLY / no training. Uses an EXISTING trained checkpoint
(default ``runs/boundary_dist_sdoct/best_model_phase1.pth``) and does NOT
modify model.py / loss.py / evaluate.py / train.py / sdoct_dataset.py /
build_sdoct_manifests.py -- it only imports and reuses them.

WHY THIS EXISTS (see loss.py BoundaryDistributionLoss / model.py
BoundaryDistributionHead): the boundary-distribution head already predicts a
per-column depth DISTRIBUTION for the choroid's upper and lower boundary, with
a std (uncertainty) per column. Diagnostics from evaluate.py showed the RAW
predicted std CORRELATES with actual error -- but correlation is NOT
calibration. A stated "90% interval" from the raw std is not guaranteed to
actually contain the truth 90% of the time; it might be a systematically
too-narrow (overconfident) or too-wide (useless) interval. This script:

  1. Propagates the per-boundary std into a THICKNESS uncertainty (with an
     explicit, TESTED independence assumption -- see ``upper_lower_correlation``
     and ``apply_corrected_sigma_t``).
  2. CALIBRATES that uncertainty on a HELD-OUT, PATIENT-LEVEL calibration split
     (zero leakage from the checkpoint's own training patients, and zero
     leakage between the calibration split and the final test split) via SPLIT
     CONFORMAL PREDICTION -- the standard, distribution-free way to convert a
     merely-correlated uncertainty score into a GUARANTEED-coverage interval
     (Vovk et al. / Lei et al., "distribution-free predictive inference").
  3. Reports whether the calibrated intervals actually achieve their stated
     coverage on genuinely held-out test patients -- INCLUDING per class
     (normal vs. AMD/"abnormal"), since the project's own diagnostics flagged
     the model as possibly overconfident on AMD lower boundaries specifically.
  4. Produces the clinical output: per-eye choroid thickness +/- a calibrated
     95% interval, and example thickness-profile figures with a shaded
     calibrated confidence band.

PIXEL <-> MICRON: this codebase has NO recorded axial (depth) pixel spacing
for the Bioptigen SDOCT scanner/settings used to acquire this dataset (grepped
the full repo -- nothing). Per the task, we do NOT invent a conversion factor.
ALL thickness/uncertainty numbers below are reported in PIXELS, with an
explicit flag on every clinical output that a micron conversion requires the
scan's actual axial scale (typically documented in the acquisition protocol /
DICOM header / instrument export settings -- NOT present anywhere in this
repository).

PATIENT-LEVEL SPLIT: the checkpoint's OWN train/val manifests were already
split patient-level (build_sdoct_manifests.py, seed 42) so val_manifest.csv
contains ONLY patients the model never trained on. This script further splits
THOSE val patients (class-stratified, patient-level, a FRESH seed) into a
calibration set (fits the conformal quantiles) and a test set (evaluates
coverage) -- so calibration is fit on patients the test-set numbers never
touch, and a leakage check against the ORIGINAL train_manifest.csv patients is
run and reported explicitly (see ``verify_no_leakage``).

Run (on the GPU box, with the real checkpoint + SDOCT data):
  python thickness_uncertainty.py \\
      --checkpoint runs/boundary_dist_sdoct/best_model_phase1.pth \\
      --data_dir <sdoct_data_dir> --out_dir thickness_uncertainty_out
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..losses import boundary_distribution_to_depths, derive_column_boundaries
from ..datasets.build_sdoct_manifests import stem_of_eye  # patient-key extraction (NOT modified)


# --------------------------------------------------------------------------- #
# PORTED: this module reuses ``train.build_real_dataset`` and
# ``evaluate.build_model`` (NOT modified -- the calibration must run against the
# exact dataset/model construction the training and evaluation entry points use).
# Those two live in ``scripts/``, outside the installed package, so the original
# top-level ``import train`` / ``import evaluate`` cannot work from inside
# ``deepoct.calibration``.
#
# The proxy below defers the import until the first attribute access, so merely
# importing this module (for its pure calibration helpers -- coverage_table,
# summarize, the conformal fitters) does NOT require the scripts directory to be
# present. Call sites are unchanged: ``train.build_real_dataset(...)`` and
# ``evaluate.build_model(...)`` still read exactly as before.
# --------------------------------------------------------------------------- #
class _LazyPipelineModule:
    """Import a top-level pipeline script on first attribute access.

    Resolution order: whatever is already importable on ``sys.path`` first, then
    the repository's ``scripts/`` directory located relative to this file.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def _resolve(self):
        if self._module is None:
            import importlib
            try:
                self._module = importlib.import_module(self._name)
            except ImportError:
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))))
                scripts_dir = os.path.join(repo_root, "scripts")
                if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
                    sys.path.append(scripts_dir)
                self._module = importlib.import_module(self._name)
        return self._module

    def __getattr__(self, attr):
        return getattr(self._resolve(), attr)


train = _LazyPipelineModule("train")        # reuse build_real_dataset (NOT modified)
evaluate = _LazyPipelineModule("evaluate")  # reuse build_model (NOT modified)

from scipy.stats import norm, pearsonr, spearmanr  # hard dependency: calibration needs it

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: write files, never open a window
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAVE_MPL = False


DEFAULT_CHECKPOINT = os.path.join("runs", "boundary_dist_sdoct", "best_model_phase1.pth")
DEFAULT_COVERAGE_LEVELS = (0.50, 0.68, 0.90, 0.95)
# Fine grid used ONLY for Expected Calibration Error (Kuleshov et al. 2018-style
# regression ECE: mean |observed - nominal| over a grid of nominal levels) -- a
# finer grid than the headline coverage table gives a more robust single number.
FINE_ECE_GRID = tuple(round(float(x), 6) for x in np.linspace(0.05, 0.95, 19))

# PIXEL<->MICRON: intentionally None. See module docstring -- do NOT invent this.
PIXEL_TO_MICRON_UM: Optional[float] = None


def _lvl(x: float) -> float:
    """Round a coverage level to a stable dict key (avoids float-union misses
    between CLI-parsed levels and the linspace-generated ECE grid)."""
    return round(float(x), 6)


def calibration_protocol_string(calibration_mode: str, conformal_per_class: bool) -> str:
    """The ONE description of which q-hat FIT protocol is in effect for this
    run -- printed loudly at startup and stamped into ``_provenance`` under
    the SAME key (``calibration_protocol``) by BOTH --calibration_mode paths.

    ``--conformal_per_class`` now means exactly one thing in both modes: it
    gates the per-class FIT (see grouped_lopo_conformal / the single_split
    qhat_by_class loop). This string is the single source of truth for what
    that gate actually did on this run, so the two can never silently drift
    apart the way they used to (cross: flag gated the fit; single_split: the
    fit was unconditional and the flag only toggled a diagnostic)."""
    return f"calibration: {calibration_mode} / {'per_class=True' if conformal_per_class else 'pooled'}"


# --------------------------------------------------------------------------- #
# 1. Patient-level split utilities (mirrors build_sdoct_manifests.py's
#    class-stratified, image-count-balanced patient split, operating on an
#    ALREADY-BUILT manifest DataFrame instead of raw scanned records).
# --------------------------------------------------------------------------- #
def patient_of(eye_id: object) -> str:
    """Patient key from an 'Eye ID' == '{patient}_{eye}' -- reuses the EXACT
    function build_sdoct_manifests.py used for the original train/val split,
    so patient identity is derived identically everywhere in this project."""
    return stem_of_eye(eye_id)


def patient_class_map(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    """Majority class per patient (a patient spanning both classes is assigned
    its majority class, flagged). Mirrors build_sdoct_manifests.py's handling."""
    patients = df["Eye ID"].map(patient_of)
    patient_class: Dict[str, str] = {}
    mixed: List[str] = []
    for p, g in df.groupby(patients):
        classes = set(g["class"])
        if len(classes) == 1:
            patient_class[p] = next(iter(classes))
        else:
            patient_class[p] = g["class"].value_counts().idxmax()
            mixed.append(p)
    return patient_class, mixed


def stratified_patient_split(df: pd.DataFrame, calib_fraction: float, seed: int
                             ) -> Tuple[set, set, Dict[str, str]]:
    """Split df's PATIENTS (not rows) into (calib_patients, test_patients),
    stratified by class and balanced by IMAGE COUNT (not just patient count,
    since patients can have very different scan counts) -- the same greedy
    image-count-target allocation build_sdoct_manifests.py uses for its
    train/val split, applied here to the val-only patient pool.

    Guards against a degenerate all-or-nothing allocation when a class has
    very few patients (keeps at least one patient on each side when >1 exist).
    """
    patient_class, mixed = patient_class_map(df)
    if mixed:
        print(f"  [WARN] {len(mixed)} patient(s) span BOTH classes within "
              f"val_manifest.csv; assigned to their majority class for the "
              f"calib/test split: {mixed[:5]}{'...' if len(mixed) > 5 else ''}")

    patient_imgs = df.groupby(df["Eye ID"].map(patient_of)).size().to_dict()
    by_class: Dict[str, List[str]] = defaultdict(list)
    for p, c in patient_class.items():
        by_class[c].append(p)

    rng = random.Random(seed)
    calib_patients: set = set()
    for c, plist in by_class.items():
        plist = sorted(plist)
        rng.shuffle(plist)
        target = round(calib_fraction * sum(patient_imgs[p] for p in plist))
        chosen: List[str] = []
        cum = 0
        for p in plist:
            if cum >= target:
                break
            chosen.append(p)
            cum += patient_imgs[p]
        if len(plist) > 1 and len(chosen) == len(plist):
            chosen = chosen[:-1]           # keep >=1 patient for test
        if not chosen and plist:
            chosen = [plist[0]]            # keep >=1 patient for calib
        calib_patients.update(chosen)

    all_patients = set(patient_class.keys())
    test_patients = all_patients - calib_patients
    return calib_patients, test_patients, patient_class


def write_subset_manifest(df: pd.DataFrame, patients: set, path: str) -> pd.DataFrame:
    subset = df[df["Eye ID"].map(patient_of).isin(patients)].reset_index(drop=True)
    subset.to_csv(path, index=False)
    return subset


def verify_no_leakage(data_dir: str, train_manifest_name: str,
                      calib_patients: set, test_patients: set) -> bool:
    """Prints (and returns) whether calib/test are mutually disjoint AND
    disjoint from the checkpoint's ORIGINAL training patients. This is the
    explicit "VERIFY: zero leakage" check -- it should trivially pass (the
    original build_sdoct_manifests.py split already made val patients disjoint
    from train), but we RE-VERIFY rather than assume."""
    ok = True
    inter = calib_patients & test_patients
    if inter:
        print(f"  [FAIL] calib/test patient overlap: {sorted(inter)[:10]}")
        ok = False
    else:
        print(f"  [OK] calib ({len(calib_patients)}) and test "
              f"({len(test_patients)}) patients are mutually disjoint.")

    train_path = os.path.join(data_dir, train_manifest_name)
    if not os.path.isfile(train_path):
        print(f"  [WARN] {train_path} not found -- cannot verify calib/test "
              f"are disjoint from the checkpoint's TRAINING patients. Assumed "
              f"true (val_manifest.csv was already patient-split away from "
              f"train_manifest.csv by build_sdoct_manifests.py), but UNVERIFIED "
              f"here.")
        return ok
    train_df = pd.read_csv(train_path)
    train_patients = set(train_df["Eye ID"].map(patient_of))
    leak_calib = train_patients & calib_patients
    leak_test = train_patients & test_patients
    if leak_calib or leak_test:
        print(f"  [FAIL] train-set leakage into calib={sorted(leak_calib)[:10]} "
              f"test={sorted(leak_test)[:10]}")
        ok = False
    else:
        print(f"  [OK] calib and test patients have ZERO overlap with the "
              f"{len(train_patients)} patients in {train_path}.")
    return ok


# --------------------------------------------------------------------------- #
# 2. Model inference -> per-column table (the core data-collection pass,
#    reused identically for the calibration split and the test split).
# --------------------------------------------------------------------------- #
def collect_columns(model: torch.nn.Module, ds, device: torch.device,
                    batch_size: int, subfoveal_frac: float, limit: int = 0
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the model over ``ds`` and return (col_df, scan_df).

    col_df: one row per VALID A-scan column (choroid present in GT) with the
    predicted expectation/std for both boundaries and the GT depths.
    scan_df: one row per SCAN (B-scan image), aggregating col_df's SUBFOVEAL
    window (the central ``subfoveal_frac`` of columns) -- the clinically
    reported "subfoveal thickness". Adjacent A-scan columns are NOT
    independent (they trace the same continuous tissue boundary), so this
    aggregate is a plain MEAN over the window, NOT a sqrt(n)-shrunk
    standard-error estimate -- shrinking by sqrt(n_columns) would assume
    independence that spatially adjacent columns do not have.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    class_labels = getattr(ds, "class_labels", None)
    col_rows: List[dict] = []
    scan_rows: List[dict] = []
    scored = 0
    model.eval()
    with torch.no_grad():
        for inp, target, mask, sigma in loader:
            inp = inp.float().to(device)
            mask = mask.float().to(device)
            model(inp)
            logits = model.last_boundary_logits.float()
            e_upper, std_upper, e_lower, std_lower = boundary_distribution_to_depths(logits)
            z_upper_gt, z_lower_gt, valid = derive_column_boundaries(mask)

            B, _, H, W = mask.shape
            lo = int(W * (0.5 - subfoveal_frac / 2.0))
            hi = int(W * (0.5 + subfoveal_frac / 2.0))

            for b in range(B):
                if limit and scored >= limit:
                    break
                idx = scored
                eye = ds.eye_ids[idx]
                image_id = ds.image_ids[idx]
                cls = class_labels[idx] if class_labels is not None else None

                v = valid[b].cpu().numpy()
                eu = e_upper[b].cpu().numpy(); el = e_lower[b].cpu().numpy()
                su = std_upper[b].cpu().numpy(); sl = std_lower[b].cpu().numpy()
                zu = z_upper_gt[b].cpu().numpy(); zl = z_lower_gt[b].cpu().numpy()

                cols = np.nonzero(v)[0]
                for c in cols:
                    col_rows.append({
                        "eye": eye, "image_id": image_id, "class": cls, "col": int(c),
                        "e_upper": float(eu[c]), "e_lower": float(el[c]),
                        "std_upper": float(su[c]), "std_lower": float(sl[c]),
                        "z_upper_gt": float(zu[c]), "z_lower_gt": float(zl[c]),
                    })

                sub_mask = v.copy()
                sub_mask[:lo] = False
                sub_mask[hi:] = False
                sub_cols = np.nonzero(sub_mask)[0]
                if sub_cols.size:
                    scan_rows.append({
                        "eye": eye, "image_id": image_id, "class": cls,
                        "n_subfoveal": int(sub_cols.size),
                        "mean_e_upper": float(eu[sub_cols].mean()),
                        "mean_e_lower": float(el[sub_cols].mean()),
                        "mean_std_upper": float(su[sub_cols].mean()),
                        "mean_std_lower": float(sl[sub_cols].mean()),
                        "mean_z_upper_gt": float(zu[sub_cols].mean()),
                        "mean_z_lower_gt": float(zl[sub_cols].mean()),
                        "mean_t_pred": float((el[sub_cols] - eu[sub_cols]).mean()),
                        "mean_t_gt": float((zl[sub_cols] - zu[sub_cols]).mean()),
                    })
                scored += 1
            if limit and scored >= limit:
                break

    col_df = pd.DataFrame(col_rows)
    scan_df = pd.DataFrame(scan_rows)
    if not col_df.empty:
        col_df["err_upper"] = col_df["e_upper"] - col_df["z_upper_gt"]
        col_df["err_lower"] = col_df["e_lower"] - col_df["z_lower_gt"]
        col_df["t_pred"] = col_df["e_lower"] - col_df["e_upper"]
        col_df["t_gt"] = col_df["z_lower_gt"] - col_df["z_upper_gt"]
        col_df["err_t"] = col_df["t_pred"] - col_df["t_gt"]
        col_df["sigma_t_naive"] = np.sqrt(col_df["std_upper"] ** 2 + col_df["std_lower"] ** 2)
    return col_df, scan_df


# --------------------------------------------------------------------------- #
# 3. Upper/lower error correlation + corrected thickness sigma (PART 1's
#    explicit independence-assumption test).
# --------------------------------------------------------------------------- #
def upper_lower_correlation(col_df: pd.DataFrame) -> Dict[str, dict]:
    """Pearson correlation between the SIGNED upper and lower boundary errors,
    pooled ('all') and per class. This directly tests PART 1's independence
    assumption behind ``sigma_t_naive = sqrt(su^2+sl^2)``: since thickness is a
    DIFFERENCE (t = z_lower - z_upper), Var(t) = su^2+sl^2 - 2*rho*su*sl.
    POSITIVE rho (errors move together, e.g. a whole-column brightness/noise
    shift moving both estimates the same way) makes the true thickness
    variance SMALLER than the naive independent-sum formula (some noise
    cancels in the difference); NEGATIVE rho makes it LARGER.
    """
    out: Dict[str, dict] = {}

    def _corr(g: pd.DataFrame) -> dict:
        n = len(g)
        if n < 3 or g["err_upper"].std() == 0 or g["err_lower"].std() == 0:
            return {"pearson_r": float("nan"), "pearson_p": float("nan"), "n": n}
        r, p = pearsonr(g["err_upper"], g["err_lower"])
        return {"pearson_r": float(r), "pearson_p": float(p), "n": n}

    out["all"] = _corr(col_df)
    for cls, g in col_df.groupby("class"):
        out[cls] = _corr(g)
    return out


def apply_corrected_sigma_t(col_df: pd.DataFrame, rho_map: Dict[str, dict],
                            min_n_for_class_rho: int = 200) -> pd.DataFrame:
    """Adds ``sigma_t_corr`` = the correlation-adjusted thickness sigma, using
    the PER-CLASS rho when that class has enough calib columns
    (>=min_n_for_class_rho) to estimate it reliably, else falling back to the
    POOLED rho. ``rho_map`` MUST come from the CALIBRATION split (never fit
    the correlation on the same data whose calibrated intervals we then
    report coverage on)."""
    pooled_rho = rho_map["all"]["pearson_r"]
    pooled_rho = pooled_rho if np.isfinite(pooled_rho) else 0.0

    def rho_for(cls):
        entry = rho_map.get(cls)
        if entry and np.isfinite(entry["pearson_r"]) and entry["n"] >= min_n_for_class_rho:
            return entry["pearson_r"]
        return pooled_rho

    rhos = col_df["class"].map(rho_for).astype(float)
    var_t = (col_df["std_upper"] ** 2 + col_df["std_lower"] ** 2
             - 2.0 * rhos * col_df["std_upper"] * col_df["std_lower"]).clip(lower=0.0)
    col_df = col_df.copy()
    col_df["rho_used"] = rhos
    col_df["sigma_t_corr"] = np.sqrt(var_t)
    return col_df


# --------------------------------------------------------------------------- #
# 4. Calibration: split conformal prediction (PART 2).
#
# WHY CONFORMAL (over a single Gaussian temperature scalar): a single scale
# factor s (error ~ N(0,(s*sigma)^2)) forces the ratio between e.g. the 90%
# and 50% interval widths to always equal the FIXED Gaussian ratio
# z_0.90/z_0.50 (~2.44), regardless of whether the model's ERROR distribution
# actually has that shape (OCT boundary errors are plausibly heavier-tailed).
# Split conformal instead fits an INDEPENDENT quantile multiplier per coverage
# level directly from the empirical distribution of standardized residuals
# |error|/sigma on a held-out calibration set, which is exactly the "monotonic
# map from predicted sigma to empirical error quantiles" the task asks for --
# and it comes with a finite-sample marginal-coverage GUARANTEE under
# exchangeability (Vovk et al.), which a parametric Gaussian assumption does
# not. ``fit_temperature`` is also provided (and reported) as the simpler
# comparison point.
# --------------------------------------------------------------------------- #
def z_for_coverage(c: float) -> float:
    """Two-sided Normal quantile multiplier for nominal coverage c (e.g.
    c=0.90 -> ~1.645). Used ONLY for the UNCALIBRATED/naive baseline."""
    return float(norm.ppf(0.5 + c / 2.0))


def conformal_qhat(err_abs: Sequence[float], sigma: Sequence[float], coverage: float) -> float:
    """Split-conformal quantile multiplier for one coverage level: the
    ceil((n+1)*coverage)-th SMALLEST (1-indexed) of n standardized calibration
    scores |error|/sigma -- the standard finite-sample-correct split-conformal
    rule (Vovk et al.; Lei et al. 2018). A NEW point's interval half-width is
    ``qhat * sigma_new``.

    IMPLEMENTED VIA DIRECT SORTED-ARRAY INDEXING, not ``np.quantile``: an
    earlier version used ``np.quantile(scores, ceil((n+1)*coverage)/n,
    method="higher")``, which looks equivalent but is NOT -- numpy's "higher"
    interpolation index is ``ceil(q*(n-1))``, which only coincides with the
    textbook ``k``-th order statistic (index ``k-1``) in special cases (e.g.
    when ``k==n``). Caught by test_thickness_uncertainty_smoke.py comparing
    against a hand-computed order statistic on n=99 synthetic scores (numpy
    quantile gave 2.5074, the correct 90th-smallest was 2.4788) -- direct
    indexing removes the ambiguity entirely and needs no numpy-version shim.
    """
    err_abs = np.asarray(err_abs, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(sigma) & (sigma > 1e-6) & np.isfinite(err_abs)
    scores = np.sort(err_abs[ok] / sigma[ok])
    n = scores.size
    if n == 0:
        return float("nan")
    k = min(int(np.ceil((n + 1) * coverage)), n)  # 1-indexed rank, clipped to n
    return float(scores[k - 1])


def min_conformal_n(level: float) -> int:
    """Minimum calibration-set size for a FINITE-SAMPLE-VALID split-conformal
    quantile at nominal coverage ``level`` (Vovk et al.): ``conformal_qhat``
    picks the ``ceil((n+1)*level)``-th smallest of n standardized scores, so
    that rank must not exceed n -- i.e. ``n >= ceil(1/(1-level)) - 1``. Below
    this, the requested quantile does not exist as a distinct order statistic
    (the guarantee degenerates to "the single largest score", regardless of
    how large n actually is short of the requirement) -- e.g. 9 at 90%, 19 at
    95%. Used to gate PER-CLASS conformal fits (grouped_lopo_conformal): a
    class group below its level's threshold falls back to the pooled q-hat
    for that (class, level) instead of emitting a degenerate one.
    """
    return max(int(np.ceil(1.0 / (1.0 - level))) - 1, 1)


def fit_conformal_table(err_abs: Sequence[float], sigma: Sequence[float],
                        levels: Sequence[float]) -> Dict[float, float]:
    return {_lvl(c): conformal_qhat(err_abs, sigma, c) for c in levels}


# --------------------------------------------------------------------------- #
# 4b. Selective prediction / deferral: distribution-free RISK control on the
# calibrated conformal half-width (Chow's reject rule for regression, applied
# to a conformal score instead of raw sigma).
#
# WHY THIS IS THE RIGHT SCORE TO THRESHOLD: sigma alone is only a RANKING of
# uncertainty; conformal_qhat is what turns it into a WIDTH with a coverage
# guarantee. Deferring on ``qhat_level * sigma`` (the SAME half-width the
# coverage table already reports) means "defer" and "the stated interval" are
# reading the same number -- a clinician sees one width, not two different
# uncertainty scales.
# --------------------------------------------------------------------------- #
def risk_control_threshold(halfwidth: Sequence[float], abs_err: Sequence[float],
                           target_risk: float) -> Tuple[float, float, int]:
    """Distribution-free selective-risk threshold: the LARGEST calibrated
    half-width ``tau`` such that, restricted to calibration points with
    half-width <= tau, the empirical mean |error| ("risk") does not exceed
    ``target_risk``.

    MECHANISM: sort the CALIBRATION set by half-width ascending, then walk the
    cumulative mean of |error| in that order. Because sorting by half-width
    puts (on average, given a positively-ordered uncertainty) the smallest
    errors first, this cumulative mean is not exactly monotone but its RUNNING
    MAXIMUM prefix satisfying the constraint is well-defined -- ``tau`` is the
    half-width of the LAST index (in sorted order) whose prefix mean is still
    <= target_risk. This is the same "count from an empirical order statistic"
    idea ``conformal_qhat`` uses for coverage, applied to risk instead: no
    parametric assumption on the error distribution is made (distribution-
    free), but exactly like split conformal, the guarantee this ``tau``
    provides is only as good as how exchangeable the set it is later APPLIED
    to is with the calibration set it was fit on here -- see
    ``conditional_coverage_audit.lopo_deferral_curve`` for the leakage-correct
    (fit-on-other-patients, apply-to-held-out-patient) use of this primitive.

    Returns ``(tau, achieved_fit_risk, n_accepted_fit)``. ``tau = -inf`` (defer
    everything -- no width, however small, satisfies the target) when even the
    single smallest-half-width calibration point already exceeds
    ``target_risk``; this is reported explicitly, never silently clipped to
    some other threshold.
    """
    hw = np.asarray(halfwidth, dtype=np.float64)
    e = np.asarray(abs_err, dtype=np.float64)
    ok = np.isfinite(hw) & np.isfinite(e)
    hw, e = hw[ok], e[ok]
    if hw.size == 0:
        return float("-inf"), float("nan"), 0
    order = np.argsort(hw, kind="stable")
    hw_s, e_s = hw[order], e[order]
    cum_mean = np.cumsum(e_s) / np.arange(1, hw_s.size + 1)
    satisfies = cum_mean <= target_risk
    if not satisfies.any():
        return float("-inf"), float(cum_mean[0]), 0
    last = int(np.max(np.nonzero(satisfies)[0]))
    return float(hw_s[last]), float(cum_mean[last]), last + 1


def fit_temperature(err_signed: Sequence[float], sigma: Sequence[float]) -> float:
    """Simple comparison baseline: the scalar s minimizing the Normal NLL of
    error ~ N(0,(s*sigma)^2), i.e. s = sqrt(mean((error/sigma)^2)) -- the
    value that makes the standardized residuals have unit variance."""
    err = np.asarray(err_signed, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(sigma) & (sigma > 1e-6) & np.isfinite(err)
    z = err[ok] / sigma[ok]
    return float(np.sqrt(np.mean(z ** 2))) if z.size else float("nan")


def coverage_table(err_abs: Sequence[float], sigma: Sequence[float],
                   levels: Sequence[float],
                   multipliers: Optional[Dict[float, float]] = None) -> List[dict]:
    """For each nominal level, the OBSERVED fraction of points whose actual
    |error| falls inside the predicted interval (multiplier * sigma).
    ``multipliers=None`` -> naive Normal z_alpha (uncalibrated baseline);
    otherwise a per-level dict (e.g. from ``fit_conformal_table``)."""
    err_abs = np.asarray(err_abs, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(sigma) & (sigma > 1e-6) & np.isfinite(err_abs)
    err_abs, sigma = err_abs[ok], sigma[ok]
    rows = []
    for c in levels:
        m = z_for_coverage(c) if multipliers is None else multipliers.get(_lvl(c), float("nan"))
        if not np.isfinite(m) or sigma.size == 0:
            rows.append({"level": c, "multiplier": m, "observed_coverage": float("nan"),
                        "mean_halfwidth": float("nan"), "n": int(sigma.size)})
            continue
        halfwidth = m * sigma
        rows.append({
            "level": c, "multiplier": m,
            "observed_coverage": float((err_abs <= halfwidth).mean()),
            "mean_halfwidth": float(halfwidth.mean()),
            "n": int(sigma.size),
        })
    return rows


def expected_calibration_error(err_abs: Sequence[float], sigma: Sequence[float],
                               qhat_table: Optional[Dict[float, float]] = None,
                               grid: Sequence[float] = FINE_ECE_GRID) -> float:
    """Regression ECE (Kuleshov et al. 2018 style): mean |observed - nominal|
    coverage over a fine grid of nominal levels. ``qhat_table=None`` scores the
    UNCALIBRATED (naive Normal) baseline; otherwise the CALIBRATED table."""
    diffs = []
    for c in grid:
        m = None if qhat_table is None else {_lvl(c): qhat_table.get(_lvl(c), float("nan"))}
        rows = coverage_table(err_abs, sigma, [c], multipliers=m)
        obs = rows[0]["observed_coverage"]
        if np.isfinite(obs):
            diffs.append(abs(obs - c))
    return float(np.mean(diffs)) if diffs else float("nan")


# --------------------------------------------------------------------------- #
# 5. Plots (reliability diagram; thickness profile with a shaded calibrated
#    band, overlaid on the B-scan). Degrades gracefully (prints + skips) if
#    matplotlib is unavailable, never crashes the numeric report.
# --------------------------------------------------------------------------- #
def plot_reliability(levels: Sequence[float], observed_before: Sequence[float],
                     observed_after: Sequence[float], out_path: str, title: str) -> None:
    if not _HAVE_MPL:
        print(f"  [WARN] matplotlib unavailable -- skipping {out_path}")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(levels, observed_before, "o-", color="tab:red", label="uncalibrated (raw std)")
    ax.plot(levels, observed_after, "o-", color="tab:blue", label="calibrated (conformal)")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("observed coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_thickness_profile(image_2d: np.ndarray, cols: np.ndarray,
                           e_upper: np.ndarray, hw_upper: np.ndarray,
                           e_lower: np.ndarray, hw_lower: np.ndarray,
                           t_pred: np.ndarray, hw_t: np.ndarray, t_gt: np.ndarray,
                           out_path: str, title: str) -> None:
    """Two-panel figure: (top) the B-scan with the calibrated boundary bands
    overlaid; (bottom) the thickness profile with a shaded calibrated
    confidence band vs. the GT thickness. Units are PIXELS (see module
    docstring -- no micron conversion factor is available)."""
    if not _HAVE_MPL:
        print(f"  [WARN] matplotlib unavailable -- skipping {out_path}")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.imshow(image_2d, cmap="gray", aspect="auto")
    ax1.fill_between(cols, e_upper - hw_upper, e_upper + hw_upper,
                     color="tab:orange", alpha=0.35, label="upper boundary (calib. 95%)")
    ax1.plot(cols, e_upper, color="tab:orange", lw=1)
    ax1.fill_between(cols, e_lower - hw_lower, e_lower + hw_lower,
                     color="tab:cyan", alpha=0.35, label="lower boundary (calib. 95%)")
    ax1.plot(cols, e_lower, color="tab:cyan", lw=1)
    ax1.set_ylabel("depth (px)"); ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=7)

    ax2.plot(cols, t_gt, color="k", lw=1, label="GT thickness")
    ax2.plot(cols, t_pred, color="tab:green", lw=1.2, label="predicted thickness")
    ax2.fill_between(cols, t_pred - hw_t, t_pred + hw_t, color="tab:green", alpha=0.25,
                     label="calibrated 95% interval")
    ax2.set_xlabel("A-scan column (px)"); ax2.set_ylabel("thickness (px)")
    ax2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Reporting helper
# --------------------------------------------------------------------------- #
def summarize(values) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


# --------------------------------------------------------------------------- #
# 6b. CROSS-CONFORMAL (PART A) + PER-CLASS CONFORMAL (PART B)
#
# The single-split path (above) estimates coverage from ONE held-out test split
# (here, the 2 test patients of val_manifest.csv). That is a 2-patient estimate.
# The functions below estimate coverage across MANY held-out patients via
# grouped LEAVE-ONE-PATIENT-OUT (LOPO) conformal, and add a per-patient bootstrap
# CI so the (necessarily large) sampling error of a few-patient coverage estimate
# is reported honestly.
#
# LEAKAGE RULE (enforced structurally): a prediction is only ever used if the
# model that produced it did NOT train on that patient. Two ways to get such
# out-of-sample (OOS) predictions:
#   * VAL-ONLY: the single checkpoint's val patients (unseen by it). LOPO over
#     those => coverage over ~6 patients. group_col='val' (one group).
#   * CROSS-FOLD: K-fold checkpoints, each with a DIFFERENT held-out patient set.
#     Each fold's val patients are OOS for THAT fold's model; pooling their OOS
#     predictions spans all patients. group_col='fold'. Within grouped-LOPO the
#     conformal FIT for a held-out patient uses only OTHER patients from the SAME
#     fold (same model) — so fit and eval are always patient-disjoint AND scored
#     by a model that trained on neither.
# --------------------------------------------------------------------------- #
def grouped_lopo_conformal(col_df: pd.DataFrame, err_col: str, sigma_col: str,
                           levels: Sequence[float], per_class: bool = False,
                           group_col: str = "fold", min_fit: int = 30
                           ) -> Tuple[pd.DataFrame, List[str]]:
    """Leave-one-PATIENT-out split conformal, aggregated per (patient, class,
    level). For each held-out patient p, the conformal quantile table is FIT on
    the OTHER patients WITHIN p's group (``group_col``) — never on p, never
    across groups — then p's own columns are scored against it. This guarantees
    fit/eval patient-disjointness; the caller guarantees every group is a set of
    patients OOS for the model that scored them.

    Returns ``(rec_df, skipped)`` where rec_df has one row per (patient, class,
    level) with ``n`` (columns), ``cov_sum`` (columns covered), ``hw_sum`` (sum of
    interval half-widths) — a compact form sufficient for pooled coverage, mean
    width, ECE and the PATIENT-level bootstrap. ``skipped`` lists patients whose
    group had < ``min_fit`` valid fit columns (cannot calibrate — reported, not
    silently dropped).

    PER-CLASS FINITE-SAMPLE GUARD: ``per_class=True`` fits a class group's own
    quantile at level ``c`` only if that group has >= ``min_conformal_n(c)``
    valid points AT THAT LEVEL (9 at 90%, 19 at 95%, ...) -- a class can pass
    the guard at 68% while still falling back to the POOLED q-hat at 95% in
    the SAME patient/group, since the threshold is per-level, not a single
    per-class yes/no. Every fallback is logged once per (group, class, level)
    (deduplicated across the LOPO patient loop, which would otherwise repeat
    it once per held-out patient) — never silent, never a degenerate quantile.
    """
    levels = [_lvl(c) for c in levels]
    patients = list(dict.fromkeys(col_df["patient"]))
    out_rows: List[tuple] = []
    skipped: List[str] = []
    _warned_fallbacks: set = set()
    for p in patients:
        pmask = (col_df["patient"] == p).to_numpy()
        p_rows = col_df[pmask]
        grp = p_rows[group_col].iloc[0]
        fit_df = col_df[(col_df[group_col] == grp) & (~pmask)]
        e_fit = fit_df[err_col].abs().to_numpy(); s_fit = fit_df[sigma_col].to_numpy()
        okf = np.isfinite(s_fit) & (s_fit > 1e-6) & np.isfinite(e_fit)
        if int(okf.sum()) < min_fit:
            skipped.append(str(p))
            continue
        qh_pool = fit_conformal_table(e_fit, s_fit, levels)
        qh_cls: Dict[str, Dict[float, float]] = {}
        if per_class:
            for c, g in fit_df.groupby("class"):
                ge = g[err_col].abs().to_numpy(); gs = g[sigma_col].to_numpy()
                okc = np.isfinite(gs) & (gs > 1e-6) & np.isfinite(ge)
                ge, gs = ge[okc], gs[okc]
                n_c = int(okc.sum())
                per_level: Dict[float, float] = {}
                for lvl in levels:
                    need = min_conformal_n(lvl)
                    if n_c >= need:
                        per_level[lvl] = conformal_qhat(ge, gs, lvl)
                    else:
                        key = (str(grp), str(c), lvl)
                        if key not in _warned_fallbacks:
                            _warned_fallbacks.add(key)
                            print(f"    [WARN] per-class conformal fallback: "
                                  f"group={grp!r} class={c!r} level={lvl:.2f} has "
                                  f"n={n_c} calib columns < required {need} for a "
                                  f"finite-sample-valid quantile at this level -- "
                                  f"using the POOLED q-hat for this (class, level) "
                                  f"instead of a degenerate per-class one.")
                if per_level:
                    qh_cls[c] = per_level
        e_ev = p_rows[err_col].abs().to_numpy(); s_ev = p_rows[sigma_col].to_numpy()
        cls_ev = p_rows["class"].astype(object).to_numpy()
        okv = np.isfinite(s_ev) & (s_ev > 1e-6) & np.isfinite(e_ev)
        e_ev, s_ev, cls_ev = e_ev[okv], s_ev[okv], cls_ev[okv]
        if e_ev.size == 0:
            continue
        for c in levels:
            if per_class:
                mult = np.array([qh_cls.get(cc, {}).get(c, qh_pool[c]) for cc in cls_ev])
            else:
                mult = np.full(e_ev.shape, qh_pool[c])
            hw = mult * s_ev
            covered = (e_ev <= hw)
            for cc in np.unique(cls_ev):
                sel = cls_ev == cc
                out_rows.append((str(p), str(cc), c, int(sel.sum()),
                                 float(covered[sel].sum()), float(hw[sel].sum())))
    rec_df = pd.DataFrame(out_rows,
                          columns=["patient", "class", "level", "n", "cov_sum", "hw_sum"])
    return rec_df, skipped


def _scope_filter(rec_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return rec_df if scope == "all" else rec_df[rec_df["class"] == scope]


def aggregate_coverage(rec_df: pd.DataFrame, levels: Sequence[float],
                       scope: str = "all") -> List[dict]:
    """Pooled (column-weighted) observed coverage + mean half-width per level over
    all held-out columns in ``scope``."""
    rows = []
    for c in levels:
        sub = _scope_filter(rec_df[rec_df["level"] == _lvl(c)], scope)
        n = float(sub["n"].sum())
        rows.append({
            "level": float(c),
            "observed_coverage": float(sub["cov_sum"].sum() / n) if n else float("nan"),
            "mean_halfwidth": float(sub["hw_sum"].sum() / n) if n else float("nan"),
            "n_columns": int(n),
            "n_patients": int(sub["patient"].nunique()),
        })
    return rows


def lopo_ece(rec_df: pd.DataFrame, scope: str = "all",
             grid: Sequence[float] = FINE_ECE_GRID) -> float:
    diffs = []
    for c in grid:
        sub = _scope_filter(rec_df[rec_df["level"] == _lvl(c)], scope)
        n = float(sub["n"].sum())
        if n:
            diffs.append(abs(sub["cov_sum"].sum() / n - c))
    return float(np.mean(diffs)) if diffs else float("nan")


# --------------------------------------------------------------------------- #
# ECE -- pooled vs within-class.
#
# lopo_ece() takes abs() AFTER pooling cov_sum/n across whatever the scope
# contains. At scope="all" that pools BOTH classes before subtracting the
# nominal level, so an over-covering class and an under-covering class cancel
# inside the mean instead of adding -- confirmed on the real cross-conformal
# run (upper_boundary): pooled ECE 0.01404 vs abnormal 0.01803 / normal
# 0.01016. The pooled estimator is NOT deleted -- it is still a legitimate
# summary of MARGINAL calibration. ece_within_class aggregates the per-class
# ABS deviations (already correctly signed-then-abs'd by lopo_ece) by
# column-count-weighted mean, so opposite-signed class errors ADD instead of
# cancelling. This is the SINGLE canonical estimator -- conditional_coverage_
# audit.py imports it rather than reimplementing, so the two scripts cannot
# drift apart on this math again.
# --------------------------------------------------------------------------- #
def _signed_class_deviation(rec_df: pd.DataFrame, cls: str,
                            grid: Sequence[float]) -> float:
    """Mean SIGNED (coverage - nominal) for one class across ``grid`` -- the
    un-abs'd twin of lopo_ece, used only to detect opposite-signed class
    deviations (the cancellation signature), never reported as an ECE itself."""
    diffs = []
    for c in grid:
        sub = _scope_filter(rec_df[rec_df["level"] == _lvl(c)], str(cls))
        n = float(sub["n"].sum())
        if n:
            diffs.append(sub["cov_sum"].sum() / n - c)
    return float(np.mean(diffs)) if diffs else float("nan")


def ece_pooled_vs_within_class(rec_df: pd.DataFrame, classes: Sequence[str],
                               grid: Sequence[float] = FINE_ECE_GRID) -> dict:
    """Both estimators, the per-class values they're built from, the signed
    per-class deviations, and whether the pooled number is safe to quote."""
    ece_pooled = lopo_ece(rec_df, scope="all", grid=grid)
    ece_by_class = {str(c): lopo_ece(rec_df, scope=str(c), grid=grid)
                    for c in classes}
    # weight by TOTAL columns for that class (level-invariant, i.e. summed
    # across every row of rec_df for that class -- not just one grid point).
    weights = {str(c): float(rec_df[rec_df["class"] == str(c)]["n"].sum())
              for c in classes}
    tot = sum(weights.values())
    ece_within_class = (float(sum(ece_by_class[c] * weights[c] for c in weights) / tot)
                       if tot > 0 else float("nan"))
    signed = {str(c): _signed_class_deviation(rec_df, c, grid) for c in classes}
    finite_signs = [np.sign(v) for v in signed.values() if np.isfinite(v) and v != 0]
    opposite_signed = len(set(finite_signs)) > 1 if len(finite_signs) >= 2 else False
    return {
        "ece_pooled": ece_pooled,
        "ece_within_class": ece_within_class,
        "ece_by_class": ece_by_class,
        "class_signed_deviation": signed,
        "class_weights_n_columns": weights,
        "ece_pooled_reliable": bool(not opposite_signed),
        "ece_pooled_reliability_note": (
            "opposite-signed per-class deviations detected -- pooling CANCELS "
            "them, so ece_pooled UNDERSTATES miscalibration and must not be "
            "quoted alone; use ece_within_class or the per-class values"
            if opposite_signed else
            "per-class deviations agree in sign -- pooling does not cancel; "
            "ece_pooled is a reasonable single-number summary here"),
    }


def patient_bootstrap_ci(rec_df: pd.DataFrame, level: float, scope: str = "all",
                         n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI on the pooled observed coverage, resampling
    PATIENTS (not columns). Columns within a patient trace the same continuous
    boundary and are highly correlated, so a column-level bootstrap would be
    fake-tight; resampling whole patients captures the real between-patient
    sampling error that a 2-6-patient coverage estimate actually has.

    Statistic per resample = column-pooled coverage over the resampled patients
    (each drawn patient contributes ALL its columns), i.e. the exact pooled
    headline number recomputed on the patient bootstrap sample."""
    sub = _scope_filter(rec_df[rec_df["level"] == _lvl(level)], scope)
    per_p = sub.groupby("patient")[["n", "cov_sum"]].sum()
    N = per_p["n"].to_numpy(dtype=np.float64)
    C = per_p["cov_sum"].to_numpy(dtype=np.float64)
    point = float(C.sum() / N.sum()) if N.sum() else float("nan")
    if len(N) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = np.arange(len(N))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        tot = N[s].sum()
        stats[b] = C[s].sum() / tot if tot else np.nan
    stats = stats[np.isfinite(stats)]
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi, point


def sigma_error_correlation(col_df: pd.DataFrame, sigma_col: str, err_col: str
                            ) -> Dict[str, dict]:
    """Pearson + Spearman correlation between predicted sigma and ACTUAL |error|,
    pooled and per class. A well-ordered uncertainty has POSITIVE correlation
    (larger predicted std where the error is actually larger). Near-zero/negative
    => the sigma is NOT discriminative (mis-ordered): per-class conformal can then
    still FIX the marginal coverage (by widening), but the interval width no
    longer tracks WHERE the error is — a calibrated-but-blunt interval."""
    out: Dict[str, dict] = {}

    def _c(g: pd.DataFrame) -> dict:
        e = g[err_col].abs().to_numpy(); s = g[sigma_col].to_numpy()
        ok = np.isfinite(e) & np.isfinite(s) & (s > 1e-6)
        if int(ok.sum()) < 10 or np.std(s[ok]) == 0 or np.std(e[ok]) == 0:
            return {"pearson_r": float("nan"), "spearman_r": float("nan"),
                    "n": int(ok.sum())}
        return {"pearson_r": float(pearsonr(s[ok], e[ok])[0]),
                "spearman_r": float(spearmanr(s[ok], e[ok])[0]),
                "n": int(ok.sum())}

    out["all"] = _c(col_df)
    for cls, g in col_df.groupby("class"):
        out[str(cls)] = _c(g)
    return out


def report_cross_target(name: str, rec_df: pd.DataFrame, skipped: List[str],
                        coverage_levels: Sequence[float], per_class: bool,
                        n_boot: int, seed: int) -> dict:
    """Print + return the pooled and per-class coverage table (with per-patient
    bootstrap CIs), mean width and ECE for one target (upper/lower/thickness)."""
    scopes = ["all"] + sorted(c for c in rec_df["class"].unique() if c and c != "all")
    print(f"\n--- {name} ---")
    if skipped:
        print(f"  [note] {len(skipped)} patient(s) skipped (group had too few fit "
              f"columns to calibrate): {skipped[:6]}{'...' if len(skipped) > 6 else ''}")
    result: Dict[str, dict] = {}
    classes = [c for c in scopes if c != "all"]
    if classes:
        ece_block = ece_pooled_vs_within_class(rec_df, classes)
        if not ece_block["ece_pooled_reliable"]:
            print(f"  [WARNING] {name}: ece_pooled is NOT reliable -- "
                 f"{ece_block['ece_pooled_reliability_note']} "
                 f"(class_signed_deviation={ece_block['class_signed_deviation']})")
        result["ece_pooled_vs_within_class"] = ece_block
    for scope in scopes:
        agg = aggregate_coverage(rec_df, coverage_levels, scope)
        ece = lopo_ece(rec_df, scope)
        n_pat = int(_scope_filter(rec_df, scope)["patient"].nunique())
        n_col = int(_scope_filter(rec_df[rec_df["level"] == _lvl(coverage_levels[0])],
                                  scope)["n"].sum())
        print(f"  [{scope}] held-out patients={n_pat}  cols={n_col}  "
              f"ECE(LOPO)={ece:.4f}")
        print(f"    {'stated':>7} {'observed':>9} {'95% CI (patient-bootstrap)':>28} "
              f"{'mean width':>11}")
        rows = []
        for r in agg:
            lo, hi, pt = patient_bootstrap_ci(rec_df, r["level"], scope, n_boot, seed)
            print(f"    {r['level']*100:6.0f}% {r['observed_coverage']*100:8.1f}% "
                  f"   [{lo*100:5.1f}%, {hi*100:5.1f}%]        "
                  f"{r['mean_halfwidth']:8.2f}px")
            rows.append({**r, "ci_lo": lo, "ci_hi": hi})
        result[scope] = {"coverage": rows, "ece": ece, "n_patients": n_pat}
    return result


# --------------------------------------------------------------------------- #
# 6c. Cross-conformal drivers: build OOS columns (val-only OR cross-fold),
#     then run grouped-LOPO conformal.
# --------------------------------------------------------------------------- #
def _patient_col(col_df: pd.DataFrame) -> pd.DataFrame:
    col_df = col_df.copy()
    col_df["patient"] = col_df["eye"].map(patient_of)
    return col_df


def _ckpt_has_boundary_head(ckpt: dict) -> bool:
    """True iff the checkpoint's state_dict actually carries boundary-distribution
    head weights (``boundary_head.*``). This is the GROUND TRUTH — the significance/
    multiseed folds were trained with the MASK head (--boundary_distribution_head
    OFF), so they have NO such weights and cannot produce per-column depth
    distributions. We never fabricate them from the mask head."""
    sd = ckpt.get("model_state_dict", {}) or {}
    return any(str(k).startswith("boundary_head.") for k in sd.keys())


# --------------------------------------------------------------------------- #
# OPTION B — DEEP ENSEMBLE of boundary-distribution models (flag-gated,
# --boundary_ensemble; default OFF -> single-checkpoint path is byte-identical).
#
# Deep ensembles (Lakshminarayanan et al. 2017) typically SHARPEN the predictive
# distribution AND improve calibration: averaging N per-column depth PMFs from
# models trained with DIFFERENT SEEDS collapses the seed-variance while keeping
# genuine ambiguity. This wrapper AVERAGES the members' per-column depth PMFs
# (softmax over depth) and re-exposes the result as a normalized log-pmf on
# ``self.last_boundary_logits`` -- the SAME interface a single model exposes -- so
# ``collect_columns`` / ``boundary_distribution_to_depths`` / conformal all consume
# it UNCHANGED. The E[z]/std are then derived from the AVERAGED distribution.
# --------------------------------------------------------------------------- #
class EnsembleBoundaryModel(torch.nn.Module):
    """Average the per-column depth PMFs of N boundary-distribution models.

    All members MUST share the SAME train split (so a given val/test patient is
    out-of-sample for EVERY member) -- typically the same experiment retrained with
    N different seeds. The ensemble presents the boundary-distribution-head interface
    (``boundary_distribution_head=True``, ``last_boundary_logits`` populated each
    forward), so it is a drop-in for a single model in ``collect_columns``.
    """

    def __init__(self, models: Sequence[torch.nn.Module]) -> None:
        super().__init__()
        self.members = torch.nn.ModuleList(models)
        self.boundary_distribution_head = True
        self.last_boundary_logits: Optional[torch.Tensor] = None

    def forward(self, inp: torch.Tensor):
        pmfs = []
        for m in self.members:
            m(inp)
            # softmax over depth (dim=2) -> a proper per-column PMF for each member,
            # regardless of whether the member is the single-softmax OR the mixture
            # head (both emit [B,2,H,W] where softmax(dim=2) is the depth PMF).
            pmfs.append(torch.softmax(m.last_boundary_logits.float(), dim=2))
        avg = torch.stack(pmfs, dim=0).mean(dim=0)          # [B,2,H,W] averaged PMF
        # Re-expose as a normalized log-pmf so downstream F.softmax(.,dim=2) recovers
        # the AVERAGED pmf exactly (softmax(log p)==p for a normalized p).
        self.last_boundary_logits = torch.log(avg.clamp_min(1e-12))
        return None                                          # collect_columns reads the stash


def _resolve_ensemble_checkpoints(ensemble_glob: str) -> List[str]:
    """Comma-separated glob(s)/paths -> ordered, de-duplicated list of existing
    checkpoint files."""
    out: List[str] = []
    seen: set = set()
    for pat in ensemble_glob.split(","):
        pat = pat.strip()
        if not pat:
            continue
        matched = sorted(glob.glob(pat, recursive=True)) or (
            [pat] if os.path.isfile(pat) else [])
        for p in matched:
            if p not in seen and os.path.isfile(p):
                seen.add(p)
                out.append(p)
    return out


def build_ensemble_model(ensemble_glob: str, args, device
                         ) -> Tuple[EnsembleBoundaryModel, List[str], int]:
    """Load every ensemble member, verify each has a boundary-distribution head,
    wrap in an EnsembleBoundaryModel. Returns (model, member_paths, phase)."""
    paths = _resolve_ensemble_checkpoints(ensemble_glob)
    if not paths:
        raise SystemExit(f"--boundary_ensemble matched no checkpoint files: "
                         f"{ensemble_glob!r}")
    models: List[torch.nn.Module] = []
    phase = None
    for cp in paths:
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        if not _ckpt_has_boundary_head(ckpt):
            raise SystemExit(
                f"ensemble member {cp!r} has NO boundary-distribution head "
                f"(no 'boundary_head.*' weights) -> cannot produce depth "
                f"distributions to average. Train members with "
                f"--boundary_distribution_head.")
        ckpt_args = ckpt.get("args", {}) or {}
        ph = args.ablation_phase or int(ckpt_args.get("ablation_phase", 1))
        phase = ph if phase is None else phase
        m = evaluate.build_model(ckpt_args, ph, device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    print(f"  ENSEMBLE: {len(models)} boundary-distribution member(s) -> averaging "
          f"per-column depth PMFs:")
    for cp in paths:
        print(f"    - {cp}")
    return EnsembleBoundaryModel(models).to(device), paths, phase


def _infer_fold_val_manifest(ckpt_path: str, val_manifest_name: str) -> Optional[str]:
    """Locate a fold checkpoint's val manifest.

    The significance harness does NOT put the manifest beside the checkpoint: the
    checkpoint lives at ``<exp_root>/ckpt/fold<K>_<COND>/best_model_phase1.pth`` while
    the per-fold manifests live in the fold's DATA dir,
    ``<exp_root>/{raw_folds|denoised_folds}/fold<K>/val_manifest.csv``. Condition
    A and C are RAW; condition B is DENOISED. We infer that mapping first, then fall
    back to the (older) beside-the-checkpoint search, then give up."""
    d = os.path.dirname(ckpt_path)                       # .../ckpt/fold<K>_<COND>
    base = os.path.basename(d)                           # fold<K>_<COND>
    candidates: List[str] = []
    m = re.match(r"fold(\d+)_([A-Za-z0-9]+)$", base)
    if m:
        k, cond = m.group(1), m.group(2).upper()
        exp_root = os.path.dirname(os.path.dirname(d))   # strip /ckpt/fold<K>_<COND>
        # A, C (and any non-B) -> raw_folds; B -> denoised_folds.
        sub = "denoised_folds" if cond == "B" else "raw_folds"
        candidates.append(os.path.join(exp_root, sub, f"fold{k}", val_manifest_name))
    # Fallbacks: beside the checkpoint (older layout).
    candidates.append(os.path.join(d, val_manifest_name))
    candidates.append(os.path.join(os.path.dirname(d), val_manifest_name))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def discover_folds(folds_glob: Optional[str], val_manifest_name: str) -> List[dict]:
    """Detect K-fold checkpoints + their val manifests. Returns [] if none found.

    Each returned dict: {name, checkpoint, val_manifest}. The val manifest is
    resolved via ``_infer_fold_val_manifest`` (raw_folds/denoised_folds mapping,
    then beside-the-checkpoint). A checkpoint whose val manifest cannot be located
    is skipped (we cannot know which patients are OOS for it). Boundary-head
    verification happens later, at scoring time (``build_oos_columns_cross_fold``),
    since it requires loading the checkpoint."""
    patterns = ([p.strip() for p in folds_glob.split(",") if p.strip()]
                if folds_glob else [
                    os.path.join("runs", "sig_experiment_roi", "ckpt", "fold*_*",
                                 "best_model_phase1.pth"),
                    os.path.join("runs", "multiseed", "seed*", "ckpt", "fold*_*",
                                 "best_model_phase1.pth"),
                    # older / simpler layouts kept as a fallback
                    os.path.join("runs", "sig_experiment_roi", "ckpt", "fold*",
                                 "best_model_phase1.pth"),
                    os.path.join("runs", "multiseed", "seed*", "**",
                                 "best_model_phase1.pth"),
                ])
    folds: List[dict] = []
    seen_ckpts = set()
    for pat in patterns:
        for ckpt in sorted(glob.glob(pat, recursive=True)):
            if ckpt in seen_ckpts:
                continue
            seen_ckpts.add(ckpt)
            vm = _infer_fold_val_manifest(ckpt, val_manifest_name)
            if vm is None:
                print(f"  [skip fold] {ckpt}: could not locate '{val_manifest_name}' "
                      f"(tried raw_folds/denoised_folds mapping + beside the "
                      f"checkpoint) -> cannot determine its OOS patients.")
                continue
            # name includes the fold-dir basename so fold<K>_<COND> stays distinct.
            folds.append({"name": os.path.basename(os.path.dirname(ckpt)),
                          "checkpoint": ckpt, "val_manifest": vm})
    return folds


def build_oos_columns_val_only(args, device, phase: int, model
                               ) -> Tuple[pd.DataFrame, dict]:
    """OOS columns = the single checkpoint's val patients (unseen by it), all in
    one LOPO group. Also runs + prints the train/val leakage check."""
    val_manifest_path = os.path.join(args.data_dir, args.val_manifest)
    val_df = pd.read_csv(val_manifest_path)
    val_patients = set(val_df["Eye ID"].map(patient_of))
    print(f"  path=VAL-ONLY LOPO: {len(val_patients)} val patients "
          f"(all OOS for the single checkpoint).")
    print("  --- leakage verification (val vs the checkpoint's train patients) ---")
    verify_no_leakage(args.data_dir, args.train_manifest, val_patients, set())
    ds = train.build_real_dataset(val_manifest_path, args.image_size, augment=False,
                                  ablation_phase=phase, dataset_kind="sdoct")
    col_df, _scan = collect_columns(model, ds, device, args.batch_size,
                                    args.subfoveal_frac, args.limit)
    if col_df.empty:
        raise SystemExit("No valid columns scored on the val set.")
    col_df = _patient_col(col_df)
    # ONE group named 'val' inside the 'fold' COLUMN (not a column called 'val'):
    # grouped_lopo_conformal groups BY the 'fold' column exactly like the cross-fold
    # path, and with a single distinct value it degenerates to plain LOPO over these
    # patients — which is correct for a single checkpoint. group_col is the COLUMN
    # NAME ('fold'), consistent with the cross-fold path (this fixes the KeyError
    # that came from previously passing the group VALUE 'val' as the column name).
    col_df["fold"] = "val"
    meta = {"path": "val_only_lopo", "group_col": "fold",
            "n_patients": int(col_df["patient"].nunique()),
            "patients": sorted(col_df["patient"].unique())}
    return col_df, meta


def build_oos_columns_cross_fold(args, device, folds: List[dict]
                                 ) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """OOS columns pooled across folds: each fold's model scores its OWN val
    patients (OOS for it). Each patient is assigned to the FIRST fold that holds
    it (so every patient appears once, scored by a model that did not train on
    it). Prints per-fold leakage checks against each fold's train manifest when
    present.

    A fold whose checkpoint has NO boundary-distribution head is SKIPPED with an
    explicit message (it was trained with the mask head and cannot produce depth
    distributions — we never fall back to the mask head or fabricate a sigma).
    Returns ``(None, None)`` if NO fold is usable (all headless / no columns), so
    the caller can fall back to the val-only path."""
    frames = []
    seen: set = set()
    fold_meta = []
    n_headless = 0
    for f in folds:
        ckpt = torch.load(f["checkpoint"], map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {}) or {}
        # VERIFY the boundary-distribution head BEFORE building/scoring — the
        # significance/multiseed folds were trained with the mask head.
        if not _ckpt_has_boundary_head(ckpt):
            argflag = bool(ckpt_args.get("boundary_distribution_head", False))
            print(f"  [skip fold {f['name']}] checkpoint has NO boundary-distribution "
                  f"head (no 'boundary_head.*' weights in the state_dict; trained with "
                  f"--boundary_distribution_head={argflag}). It CANNOT produce "
                  f"per-column depth distributions; NOT falling back to the mask head.")
            n_headless += 1
            continue
        phase = args.ablation_phase or int(ckpt_args.get("ablation_phase", 1))
        model = evaluate.build_model(ckpt_args, phase, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        vdf = pd.read_csv(f["val_manifest"])
        val_patients = set(vdf["Eye ID"].map(patient_of))
        # Leakage guard: a fold's train manifest (if beside it) must be disjoint
        # from its val patients — the fold model must NOT have trained on them.
        train_beside = os.path.join(os.path.dirname(f["val_manifest"]),
                                    args.train_manifest)
        if os.path.isfile(train_beside):
            tp = set(pd.read_csv(train_beside)["Eye ID"].map(patient_of))
            leak = tp & val_patients
            status = "OK (disjoint)" if not leak else f"LEAK {sorted(leak)[:5]}"
            print(f"  [{f['name']}] val patients={len(val_patients)} vs its train "
                  f"patients={len(tp)} -> {status}")
        else:
            print(f"  [{f['name']}] val patients={len(val_patients)} "
                  f"(no train_manifest beside it to cross-check; assumed OOS).")
        ds = train.build_real_dataset(f["val_manifest"], args.image_size, augment=False,
                                      ablation_phase=phase, dataset_kind="sdoct")
        col, _scan = collect_columns(model, ds, device, args.batch_size,
                                     args.subfoveal_frac, args.limit)
        if col.empty:
            continue
        col = _patient_col(col)
        # keep only patients not already assigned to an earlier fold
        col = col[~col["patient"].isin(seen)]
        seen |= set(col["patient"].unique())
        col["fold"] = f["name"]
        frames.append(col)
        fold_meta.append({"name": f["name"], "n_patients": int(col["patient"].nunique())})
    if not frames:
        if n_headless:
            print(f"\n  [CROSS-FOLD UNAVAILABLE] none of the {len(folds)} detected fold "
                  f"checkpoint(s) has a boundary-distribution head — they were trained "
                  f"for the significance experiment with the MASK head "
                  f"(--boundary_distribution_head OFF), so they cannot produce the "
                  f"per-column depth distributions this calibration needs. The "
                  f"cross-fold path is UNAVAILABLE with these checkpoints; falling back "
                  f"to VAL-ONLY LOPO (the honest coverage ceiling we can reach today).")
        else:
            print("\n  [CROSS-FOLD UNAVAILABLE] no columns scored from any fold "
                  "(empty val sets?). Falling back to VAL-ONLY LOPO.")
        return None, None
    col_df = pd.concat(frames, ignore_index=True)
    meta = {"path": "cross_fold", "group_col": "fold",
            "n_patients": int(col_df["patient"].nunique()),
            "n_folds": len(frames), "folds": fold_meta,
            "patients": sorted(col_df["patient"].unique())}
    print(f"  path=CROSS-FOLD: {meta['n_patients']} unique OOS patients across "
          f"{meta['n_folds']} folds (each scored by a model that did NOT train on it).")
    return col_df, meta


def run_cross_conformal(args, device, coverage_levels: Sequence[float]) -> None:
    print("=" * 92)
    print("CROSS-CONFORMAL CALIBRATION (PART A) — grouped leave-one-patient-out")
    print("=" * 92)

    # OPTION B — deep ensemble (flag-gated). All members share the SAME train split,
    # so the cross-FOLD path (which relies on per-fold DISTINCT held-out sets) does
    # not apply; the honest OOS coverage is VAL-ONLY LEAVE-ONE-PATIENT-OUT over the
    # shared val patients (out-of-sample for EVERY member). The averaged distribution
    # feeds the same grouped-LOPO conformal below, unchanged.
    if getattr(args, "boundary_ensemble", None):
        print("ENSEMBLE mode -> VAL-ONLY leave-one-patient-out (members share the "
              "train split; val patients are OOS for all). Cross-fold path skipped.")
        model, _paths, phase = build_ensemble_model(args.boundary_ensemble, args, device)
        col_df, meta = build_oos_columns_val_only(args, device, phase, model)
        # Record WHICH members produced these numbers. Its absence is precisely why an
        # ensemble run and a single-model run were indistinguishable on disk (audit A).
        meta["ensemble_members"] = list(_paths)
        _run_cross_conformal_from_columns(args, col_df, meta, coverage_levels)
        return

    folds = discover_folds(args.folds_glob, args.fold_val_manifest_name)
    col_df = meta = None
    if folds:
        print(f"detected {len(folds)} fold checkpoint(s) -> attempting CROSS-FOLD path "
              f"(honest OOS coverage spanning all folds' held-out patients); verifying "
              f"each has a boundary-distribution head...")
        col_df, meta = build_oos_columns_cross_fold(args, device, folds)
    if col_df is None:
        # No folds detected, OR none of the detected folds had a boundary head:
        # fall back to VAL-ONLY LEAVE-ONE-PATIENT-OUT over the single checkpoint's
        # val patients (the honest ceiling with the boundary-dist checkpoint we have).
        if not folds:
            print("no fold checkpoints detected -> VAL-ONLY LEAVE-ONE-PATIENT-OUT path "
                  "(coverage over the single checkpoint's val patients only).")
        else:
            print("-> using VAL-ONLY LEAVE-ONE-PATIENT-OUT path with --checkpoint.")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {}) or {}
        phase = args.ablation_phase or int(ckpt_args.get("ablation_phase", 1))
        model = evaluate.build_model(ckpt_args, phase, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if not getattr(model, "boundary_distribution_head", False):
            raise SystemExit(
                f"The val-only path needs a boundary-distribution head, but "
                f"--checkpoint {args.checkpoint!r} has none. Point --checkpoint at a "
                f"model trained with --boundary_distribution_head.")
        col_df, meta = build_oos_columns_val_only(args, device, phase, model)

    _run_cross_conformal_from_columns(args, col_df, meta, coverage_levels)


def _run_cross_conformal_from_columns(args, col_df: pd.DataFrame, meta: dict,
                                      coverage_levels: Sequence[float]) -> None:
    """Grouped-LOPO conformal + PART B report from an already-built OOS column table.
    Shared by the normal (single/cross-fold) path and the OPTION-B ensemble path."""
    # sigma_t via a pooled OOS rho (a nuisance reshaping of sigma; the conformal
    # quantile is still fit LOPO-honestly, so this does not touch the coverage
    # guarantee — noted explicitly).
    _ens_members = meta.get("ensemble_members")
    protocol_str = calibration_protocol_string(args.calibration_mode, args.conformal_per_class)
    rho_map = upper_lower_correlation(col_df)
    col_df = apply_corrected_sigma_t(col_df, rho_map, args.min_n_for_class_rho)

    print(f"\nOOS pool: {len(col_df)} columns from {meta['n_patients']} patients "
          f"(group='{meta['group_col']}'). {protocol_str}")

    # Persist the raw per-column OOS table. WITHOUT this the run's per-column
    # predictions exist only in memory and are DISCARDED at exit -- only the
    # aggregated cross_conformal_report.json survives, which has no per-patient
    # and no per-column rows, so NO post-hoc conditional-coverage audit is
    # possible after the fact. Default OFF -> byte-identical to before.
    if getattr(args, "dump_columns", False):
        from .. import provenance as _prov
        _prov.dump_frame(col_df, os.path.join(args.out_dir, "oos_columns.csv"),
                         label="A-scan columns -> conditional_coverage_audit.py",
                         checkpoint=(None if _ens_members else getattr(args, "checkpoint", None)),
                         ensemble_members=_ens_members,
                         protocol=f"cross/{meta.get('path')}",
                         patients=meta.get("patients"))

    targets = {
        "upper_boundary": ("err_upper", "std_upper"),
        "lower_boundary": ("err_lower", "std_lower"),
        "thickness": ("err_t", "sigma_t_corr"),
    }
    all_levels = sorted(set(_lvl(c) for c in coverage_levels) | set(FINE_ECE_GRID))
    report: Dict[str, object] = {"mode": "cross", **meta,
                                 "per_class": bool(args.conformal_per_class),
                                 "pixel_to_micron": PIXEL_TO_MICRON_UM}
    target_reports = {}
    for name, (err_col, sigma_col) in targets.items():
        rec_df, skipped = grouped_lopo_conformal(
            col_df, err_col, sigma_col, all_levels,
            per_class=args.conformal_per_class, group_col=meta["group_col"])
        target_reports[name] = report_cross_target(
            name, rec_df, skipped, coverage_levels, args.conformal_per_class,
            args.n_bootstrap, args.seed)
    report["targets"] = target_reports

    # PART B honesty (own flag, --conformal_diagnostics): sigma-vs-error
    # correlation (does the AMD sigma even ORDER the errors?) — quantifies
    # "calibrated but not discriminative". INDEPENDENT of --conformal_per_class
    # -- this used to run UNCONDITIONALLY here (never gated at all), which is
    # itself half of the inconsistency this change fixes: requesting/skipping a
    # diagnostic must never be entangled with the fit, in EITHER direction.
    if args.conformal_diagnostics:
        print("\n" + "=" * 92)
        print("PART B — sigma-vs-|error| correlation (is the uncertainty DISCRIMINATIVE?)")
        print("=" * 92)
        corr_report = {}
        for name, (err_col, sigma_col) in targets.items():
            sc = sigma_error_correlation(col_df, sigma_col, err_col)
            corr_report[name] = sc
            print(f"  {name}:")
            for scope, r in sc.items():
                print(f"    [{scope:>8}] pearson={r['pearson_r']:+.3f}  "
                      f"spearman={r['spearman_r']:+.3f}  (n={r['n']})")
        report["sigma_error_correlation"] = corr_report
        ab = corr_report["lower_boundary"].get("abnormal", {})
        if np.isfinite(ab.get("spearman_r", float("nan"))):
            print(f"\n  [READ THIS] Per-class conformal FIXES AMD coverage (by widening the "
                  f"intervals) but does NOT fix the underlying problem: on the AMD lower "
                  f"boundary the predicted sigma orders the error with spearman="
                  f"{ab['spearman_r']:+.3f}. Near-zero/negative => the AMD interval is "
                  f"correctly-covering but poorly-DISCRIMINATIVE (blunt): it does not know "
                  f"WHERE it is uncertain. Reported honestly, not smoothed over.")

        if args.conformal_per_class:
            print("\n  [PART B trade-off] Compare per-class 'mean width' above: AMD "
                  "('abnormal') intervals are EXPECTED to be WIDER than pooled/normal — "
                  "that is the honest cost of correct AMD coverage given the mis-ordered "
                  "sigma. A calibrated-but-blunt AMD interval is the correct outcome.")

    out_json = os.path.join(args.out_dir, "cross_conformal_report.json")
    from .. import provenance as _prov
    _prov.stamp(report,
                checkpoint=(None if getattr(args, "boundary_ensemble", None)
                            else getattr(args, "checkpoint", None)),
                ensemble_members=_ens_members,
                protocol=f"cross/{meta.get('path')}"
                         f"{'/per_class' if args.conformal_per_class else '/pooled'}",
                patients=meta.get("patients"),
                extra={"calibration_protocol": protocol_str})
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(f"\nwrote {out_json}")


# --------------------------------------------------------------------------- #
# 6d. Selftest: verify the cross-conformal + per-class MATH on synthetic OOS
#     columns with a KNOWN generative model (no checkpoint / data / model).
# --------------------------------------------------------------------------- #
def _synth_oos(seed: int = 0, n_folds: int = 3, per_fold_patients: int = 4,
               cols_per_patient: int = 220) -> pd.DataFrame:
    """Synthetic OOS column table with a KNOWN calibration story:
      * NORMAL: predicted sigma is WELL-CALIBRATED and WELL-ORDERED (error scale
        == predicted sigma) -> conformal ~ z, sigma-vs-error corr strongly +.
      * ABNORMAL: errors are ~2x too big for the predicted sigma (pooled conformal
        undercovers AMD) AND MIS-ORDERED (true error scale is INVERSELY related to
        predicted sigma) -> per-class conformal restores coverage by WIDENING, but
        sigma-vs-error corr is ~0/negative (blunt).
    Each fold holds distinct patients (half normal, half abnormal)."""
    rng = np.random.default_rng(seed)
    rows = []
    for fi in range(n_folds):
        for pj in range(per_fold_patients):
            cls = "normal" if pj < per_fold_patients // 2 else "abnormal"
            pid = f"P{fi}_{pj}"
            su = rng.uniform(1.0, 3.0, cols_per_patient)
            sl = rng.uniform(1.0, 3.0, cols_per_patient)
            eu = rng.normal(0.0, su)                      # upper: calibrated
            if cls == "normal":
                el = rng.normal(0.0, sl)                  # calibrated + ordered
            else:
                true_sl = 2.0 * (2.0 / sl)                # BIG + INVERSE to sl
                el = rng.normal(0.0, true_sl)
            for k in range(cols_per_patient):
                rows.append({"eye": f"{pid}_LE", "patient": pid, "class": cls,
                             "fold": f"fold{fi}",
                             "err_upper": eu[k], "std_upper": su[k],
                             "err_lower": el[k], "std_lower": sl[k],
                             "err_t": el[k] - eu[k],
                             "sigma_t_corr": float(np.sqrt(su[k] ** 2 + sl[k] ** 2))})
    return pd.DataFrame(rows)


def run_selftest(args) -> None:
    print("=" * 92)
    print("SELFTEST — cross-conformal (grouped LOPO) + per-class + bootstrap CI, "
          "synthetic OOS")
    print("=" * 92)
    df = _synth_oos()
    levels = [0.50, 0.68, 0.90, 0.95]
    all_levels = sorted(set(_lvl(c) for c in levels) | set(FINE_ECE_GRID))

    # --- lower boundary is where the AMD mis-ordering lives ---
    print("\n[1] POOLED conformal (per_class=False) on the LOWER boundary:")
    rec_pool, sk = grouped_lopo_conformal(df, "err_lower", "std_lower", all_levels,
                                          per_class=False, group_col="fold")
    _ = report_cross_target("lower_boundary [pooled]", rec_pool, sk, levels,
                            False, 500, 0)
    cov_norm = {r["level"]: r["observed_coverage"]
                for r in aggregate_coverage(rec_pool, levels, "normal")}
    cov_ab = {r["level"]: r["observed_coverage"]
              for r in aggregate_coverage(rec_pool, levels, "abnormal")}

    print("\n[2] PER-CLASS conformal (per_class=True) on the LOWER boundary:")
    rec_pc, sk2 = grouped_lopo_conformal(df, "err_lower", "std_lower", all_levels,
                                         per_class=True, group_col="fold")
    _ = report_cross_target("lower_boundary [per-class]", rec_pc, sk2, levels,
                            True, 500, 0)
    ab_pool_w = {r["level"]: r["mean_halfwidth"]
                 for r in aggregate_coverage(rec_pool, levels, "abnormal")}
    ab_pc_cov = {r["level"]: r["observed_coverage"]
                 for r in aggregate_coverage(rec_pc, levels, "abnormal")}
    ab_pc_w = {r["level"]: r["mean_halfwidth"]
               for r in aggregate_coverage(rec_pc, levels, "abnormal")}

    print("\n[3] sigma-vs-|error| correlation (lower boundary):")
    sc = sigma_error_correlation(df, "std_lower", "err_lower")
    for scope, r in sc.items():
        print(f"    [{scope:>8}] pearson={r['pearson_r']:+.3f} "
              f"spearman={r['spearman_r']:+.3f} (n={r['n']})")

    print("\n[4] patient-bootstrap CI sanity (abnormal, 90%, per-class):")
    lo, hi, pt = patient_bootstrap_ci(rec_pc, 0.90, "abnormal", 1000, 0)
    print(f"    observed={pt*100:.1f}%  95% CI=[{lo*100:.1f}%, {hi*100:.1f}%]")

    cov_pc_norm = {r["level"]: r["observed_coverage"]
                   for r in aggregate_coverage(rec_pc, levels, "normal")}

    # ---- assertions (the KNOWN pooled-vs-per-class story must hold) ----
    print("\n--- assertions ---")
    # Under ONE POOLED quantile, the (2-4x bigger) abnormal errors inflate the
    # shared q-hat: NORMAL is over-covered (conservative), ABNORMAL under-covered.
    assert cov_norm[0.90] > 0.92, cov_norm
    print(f"  [OK] pooled OVER-covers normal@90% = {cov_norm[0.90]*100:.1f}% (>92, "
          f"shared quantile inflated by abnormal)")
    assert cov_ab[0.90] < 0.86, cov_ab
    print(f"  [OK] pooled UNDER-covers abnormal@90% = {cov_ab[0.90]*100:.1f}% (<86)")
    # per-class RESTORES coverage for BOTH classes.
    assert abs(ab_pc_cov[0.90] - 0.90) < 0.05, ab_pc_cov
    print(f"  [OK] per-class abnormal coverage@90% = {ab_pc_cov[0.90]*100:.1f}% (~90)")
    assert abs(cov_pc_norm[0.90] - 0.90) < 0.05, cov_pc_norm
    print(f"  [OK] per-class normal coverage@90% = {cov_pc_norm[0.90]*100:.1f}% (~90)")
    # ...by WIDENING the abnormal interval
    assert ab_pc_w[0.90] > ab_pool_w[0.90], (ab_pc_w[0.90], ab_pool_w[0.90])
    print(f"  [OK] per-class WIDENS abnormal@90%: {ab_pool_w[0.90]:.2f} -> "
          f"{ab_pc_w[0.90]:.2f}px (honest cost)")
    # normal sigma ORDERS its error (+corr); abnormal sigma does NOT (mis-ordered)
    assert sc["normal"]["spearman_r"] > 0.2, sc["normal"]
    assert sc["abnormal"]["spearman_r"] < 0.1, sc["abnormal"]
    print(f"  [OK] normal sigma discriminative (spearman={sc['normal']['spearman_r']:+.3f}); "
          f"abnormal NOT (spearman={sc['abnormal']['spearman_r']:+.3f})")
    assert lo <= pt <= hi
    print(f"  [OK] bootstrap CI brackets the point estimate")
    print("\nALL CROSS-CONFORMAL SELFTEST ASSERTIONS PASSED")


# --------------------------------------------------------------------------- #
# 7. CLI + main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrated choroid thickness uncertainty from the "
                    "boundary-distribution head (read-only, no retraining).")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--data_dir", default=None,
                   help="Directory holding the SDOCT train/val manifest CSVs. "
                        "Required except for --selftest.")
    # ---- PART A: cross-conformal over patients (default keeps the single split) ----
    p.add_argument("--calibration_mode", choices=["single_split", "cross"],
                   default="single_split",
                   help="single_split (DEFAULT, unchanged/reproducible): fit conformal "
                        "on a calib patient subset of val, report coverage on the "
                        "held-out test patients. cross: patient-level cross-conformal "
                        "(grouped leave-one-patient-out) over ALL out-of-sample "
                        "patients — cross-fold if K-fold checkpoints are found, else "
                        "leave-one-patient-out over the single checkpoint's val "
                        "patients. NEVER fits/evaluates on a patient the scoring model "
                        "trained on.")
    p.add_argument("--conformal_per_class", action="store_true",
                   help="Fit SEPARATE conformal quantiles for 'normal' and "
                        "'abnormal' (AMD) instead of one pooled quantile; evaluate "
                        "each class against its own fit (widens AMD intervals -- "
                        "the honest cost). GATES THE FIT ONLY, identically in BOTH "
                        "--calibration_mode values: cross grouped-LOPO and the "
                        "single_split calib/test split. Below the finite-sample "
                        "guard (min_conformal_n: 9 pts @90%, 19 @95%) a class/level "
                        "falls back to the pooled q-hat, logged loudly -- never a "
                        "degenerate quantile. Default OFF (pooled everywhere). "
                        "NOTE: in single_split this WAS unconditional (always fit "
                        "per-class) with the flag only toggling the Part B "
                        "diagnostic below -- that entanglement is exactly the bug "
                        "this flag's semantics were fixed to remove; single_split's "
                        "DEFAULT output now differs from before (pooled, not "
                        "per-class) -- see the loud 'calibration:' startup line and "
                        "_provenance.calibration_protocol on every run.")
    p.add_argument("--conformal_diagnostics", action="store_true",
                   help="PART B (both modes): report the sigma-vs-|error| "
                        "correlation (is the uncertainty DISCRIMINATIVE, not just "
                        "calibrated?) as its own diagnostic block. INDEPENDENT of "
                        "--conformal_per_class -- requesting this diagnostic can "
                        "never alter the q-hat fit (that used to be conflated: "
                        "--conformal_per_class alone used to also decide whether "
                        "this ran). Default OFF. NOTE: in --calibration_mode cross "
                        "this WAS unconditional (always printed/reported); it is "
                        "now opt-in like everywhere else in this script.")
    p.add_argument("--folds_glob", default=None,
                   help="Comma-separated glob(s) to K-fold checkpoints for cross-fold "
                        "conformal (e.g. 'runs/sig_experiment_roi/ckpt/fold*_A/"
                        "best_model_phase1.pth'). Default: auto-probe the known "
                        "sig_experiment_roi / multiseed locations. Each fold needs its "
                        "val manifest beside the checkpoint.")
    p.add_argument("--fold_val_manifest_name", default="val_manifest.csv",
                   help="Filename of each fold's val manifest (searched beside its "
                        "checkpoint) for the cross-fold path.")
    p.add_argument("--boundary_ensemble", default=None,
                   help="OPTION B (deep ensemble): comma-separated glob(s)/paths to N "
                        "boundary-distribution checkpoints (e.g. different seeds) that "
                        "SHARE the same train split. Their per-column depth PMFs are "
                        "AVERAGED and fed to the SAME calibration/coverage machinery. "
                        "Works in the DEFAULT single_split path and in "
                        "--calibration_mode cross (forces the VAL-ONLY LOPO path, since "
                        "all members share one train split). Default OFF (single "
                        "checkpoint, byte-identical).")
    p.add_argument("--n_bootstrap", type=int, default=2000,
                   help="Patient-level bootstrap resamples for the coverage CI "
                        "(cross mode). Resamples PATIENTS, not columns.")
    p.add_argument("--dump_columns", action="store_true",
                   help="Also write the raw per-column OOS table to "
                        "<out_dir>/oos_columns.csv (cross mode). REQUIRED for any "
                        "post-hoc conditional-coverage / Mondrian audit: without it "
                        "the per-column predictions are discarded at exit and only "
                        "the aggregated report survives. Default OFF.")
    p.add_argument("--selftest", action="store_true",
                   help="Verify the cross-conformal + per-class + bootstrap MATH on "
                        "synthetic OOS columns (no checkpoint/data/model needed).")
    p.add_argument("--train-manifest", default="train_manifest.csv")
    p.add_argument("--val-manifest", default="val_manifest.csv")
    p.add_argument("--ablation_phase", type=int, default=None,
                   help="Override; default reads the checkpoint's own saved phase.")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--calib_fraction", type=float, default=0.5,
                   help="Fraction of val_manifest.csv's PATIENTS (by image "
                        "count) used to FIT calibration; the rest is the "
                        "held-out test set that reports coverage.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the calib/test patient split (independent "
                        "of the original train/val split's seed).")
    p.add_argument("--coverage_levels", default="0.50,0.68,0.90,0.95",
                   help="Comma-separated nominal coverage levels to report.")
    p.add_argument("--subfoveal_frac", type=float, default=0.10,
                   help="Width (fraction of image width) of the central "
                        "'subfoveal' column window used for per-scan/per-eye "
                        "thickness summaries. NOTE: this is a geometric-center "
                        "proxy for the fovea (no foveal-center annotation is "
                        "available in this dataset) -- flagged as an "
                        "assumption, not a clinical foveal localization.")
    p.add_argument("--min_n_for_class_rho", type=int, default=200,
                   help="Minimum calib columns required to use a CLASS-"
                        "specific upper/lower error correlation; below this, "
                        "falls back to the pooled correlation.")
    p.add_argument("--n_example_scans", type=int, default=3)
    p.add_argument("--out_dir", default="thickness_uncertainty_out")
    p.add_argument("--limit", type=int, default=0,
                   help="Max scans to score per split (0=all); use a small "
                        "value for a fast dry run.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Selftest: pure-math verification of the cross-conformal machinery. ---
    if args.selftest:
        run_selftest(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu"))
    coverage_levels = [float(x) for x in args.coverage_levels.split(",") if x.strip()]

    if args.data_dir is None:
        raise SystemExit("--data_dir is required (except for --selftest).")

    # LOUD, MODE-INDEPENDENT protocol line: the ONE place both --calibration_mode
    # paths announce what --conformal_per_class actually did, before either path's
    # own work starts. See calibration_protocol_string's docstring for why this
    # exists (the flag used to mean something different per mode).
    protocol_str = calibration_protocol_string(args.calibration_mode,
                                               args.conformal_per_class)
    print(f"[{protocol_str}]  (diagnostics={'ON' if args.conformal_diagnostics else 'OFF'})")

    # --- PART A: cross-conformal path (grouped LOPO over ALL OOS patients). The
    #     default (single_split) path below is UNCHANGED / byte-identical. ---
    if args.calibration_mode == "cross":
        run_cross_conformal(args, device, coverage_levels)
        return

    print("=" * 92)
    print("CHOROID THICKNESS -- CALIBRATED UNCERTAINTY (read-only, no retraining)")
    print("=" * 92)

    # ---- 1. Checkpoint + model (mirrors evaluate.py exactly) ----
    # OPTION B: an ensemble of N same-split checkpoints replaces the single model;
    # everything downstream consumes its averaged depth distribution unchanged.
    if args.boundary_ensemble:
        model, _paths, phase = build_ensemble_model(args.boundary_ensemble, args, device)
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {}) or {}
        phase = args.ablation_phase or int(ckpt_args.get("ablation_phase", 1))
        model = evaluate.build_model(ckpt_args, phase, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if not getattr(model, "boundary_distribution_head", False):
            raise SystemExit(
                f"Checkpoint {args.checkpoint!r} was NOT trained with "
                f"--boundary_distribution_head; this script requires it.")
        print(f"checkpoint: {args.checkpoint} (phase={phase}, epoch={ckpt.get('epoch')})")

    # ---- 2. Patient-level calib/test split of val_manifest.csv ----
    val_manifest_path = os.path.join(args.data_dir, args.val_manifest)
    val_df = pd.read_csv(val_manifest_path)
    if "class" not in val_df.columns:
        raise SystemExit(
            f"{val_manifest_path} has no 'class' column -- this script needs "
            f"the SDOCT manifest (normal/abnormal), built via "
            f"build_sdoct_manifests.py.")
    calib_patients, test_patients, patient_class = stratified_patient_split(
        val_df, args.calib_fraction, args.seed)
    print(f"\nval_manifest patients: {len(patient_class)} total -> "
         f"{len(calib_patients)} calib / {len(test_patients)} test")
    print("--- leakage verification ---")
    verify_no_leakage(args.data_dir, args.train_manifest, calib_patients, test_patients)

    calib_manifest_path = os.path.join(args.out_dir, "calib_manifest.csv")
    test_manifest_path = os.path.join(args.out_dir, "test_manifest.csv")
    write_subset_manifest(val_df, calib_patients, calib_manifest_path)
    write_subset_manifest(val_df, test_patients, test_manifest_path)

    calib_ds = train.build_real_dataset(calib_manifest_path, args.image_size, augment=False,
                                        ablation_phase=phase, dataset_kind="sdoct")
    test_ds = train.build_real_dataset(test_manifest_path, args.image_size, augment=False,
                                       ablation_phase=phase, dataset_kind="sdoct")
    print(f"calib scans: {len(calib_ds)}   test scans: {len(test_ds)}")

    # ---- 3. Inference -> per-column tables ----
    print("\n--- running inference (calib) ---")
    calib_col, calib_scan = collect_columns(model, calib_ds, device, args.batch_size,
                                            args.subfoveal_frac, args.limit)
    print(f"  {len(calib_col)} valid columns, {len(calib_scan)} scans")
    print("--- running inference (test) ---")
    test_col, test_scan = collect_columns(model, test_ds, device, args.batch_size,
                                          args.subfoveal_frac, args.limit)
    print(f"  {len(test_col)} valid columns, {len(test_scan)} scans")

    if calib_col.empty or test_col.empty:
        raise SystemExit("No valid columns scored in calib or test split -- aborting.")

    # ---- 4. PART 1: independence-assumption test + corrected sigma_t ----
    print("\n" + "=" * 92)
    print("PART 1 -- upper/lower error correlation (independence-assumption test, on CALIB)")
    print("=" * 92)
    rho_map = upper_lower_correlation(calib_col)
    for cls, r in rho_map.items():
        flag = ("naive sqrt(su^2+sl^2) OVERESTIMATES sigma_t" if r["pearson_r"] > 0.05 else
                "naive sqrt(su^2+sl^2) UNDERESTIMATES sigma_t" if r["pearson_r"] < -0.05 else
                "independence assumption approximately holds")
        print(f"  [{cls}] rho(err_upper, err_lower)={r['pearson_r']:+.4f} "
             f"(p={r['pearson_p']:.2e}, n={r['n']}) -> {flag}")
    calib_col = apply_corrected_sigma_t(calib_col, rho_map, args.min_n_for_class_rho)
    test_col = apply_corrected_sigma_t(test_col, rho_map, args.min_n_for_class_rho)
    # Same dump as the cross path, for the single_split protocol: without it the
    # per-column predictions die at exit and only the aggregated report survives.
    if getattr(args, "dump_columns", False):
        from .. import provenance as _prov
        for _name, _df in (("calib", calib_col), ("test", test_col)):
            _prov.dump_frame(_df, os.path.join(args.out_dir, f"{_name}_columns.csv"),
                             label="A-scan columns", checkpoint=args.checkpoint,
                             protocol="single_split",
                             patients=sorted(calib_patients if _name == "calib"
                                             else test_patients))
    print(f"  mean sigma_t: naive={calib_col['sigma_t_naive'].mean():.3f}px  "
         f"corrected={calib_col['sigma_t_corr'].mean():.3f}px (calib set)")

    # ---- 5. PART 2: fit conformal calibration (CALIB only) ----
    print("\n" + "=" * 92)
    print("PART 2 -- calibration (split conformal, fit on CALIB, evaluated on TEST)")
    print("=" * 92)
    all_levels = sorted(set(_lvl(c) for c in coverage_levels) | set(FINE_ECE_GRID))
    targets = {
        "upper_boundary": ("err_upper", "std_upper"),
        "lower_boundary": ("err_lower", "std_lower"),
        "thickness": ("err_t", "sigma_t_corr"),
    }
    # PER-CLASS FIT gate (--conformal_per_class, SAME meaning as --calibration_mode
    # cross now): when OFF, qhat_by_class[name] stays {} and every consumer below
    # (coverage_report, the reliability plots, PART 3 clinical output, the example
    # figures) falls back to qhat_pooled via its existing `.get(cls, qhat_pooled[..])`
    # -- so leaving this OFF makes the ENTIRE single_split run pooled-only, not just
    # the headline coverage table. Previously this loop ran UNCONDITIONALLY (the
    # bug this fix removes): the fit happened regardless of the flag, which only
    # toggled the Part B diagnostic below.
    qhat_pooled: Dict[str, Dict[float, float]] = {}
    qhat_by_class: Dict[str, Dict[str, Dict[float, float]]] = {}
    temperature_pooled: Dict[str, float] = {}
    for name, (err_col, sigma_col) in targets.items():
        qhat_pooled[name] = fit_conformal_table(
            calib_col[err_col].abs(), calib_col[sigma_col], all_levels)
        temperature_pooled[name] = fit_temperature(calib_col[err_col], calib_col[sigma_col])
        qhat_by_class[name] = {}
        if args.conformal_per_class:
            for cls, g in calib_col.groupby("class"):
                qhat_by_class[name][cls] = fit_conformal_table(
                    g[err_col].abs(), g[sigma_col], all_levels)

    report: Dict[str, dict] = {
        "checkpoint": args.checkpoint, "phase": phase,
        "calib_patients": sorted(calib_patients), "test_patients": sorted(test_patients),
        "upper_lower_error_correlation": rho_map,
        "temperature_scale_pooled": temperature_pooled,
        "pixel_to_micron": PIXEL_TO_MICRON_UM,
    }

    coverage_report: Dict[str, dict] = {}
    for name, (err_col, sigma_col) in targets.items():
        print(f"\n--- {name} ---")
        coverage_report[name] = {}
        scopes = [("all", test_col)] + list(test_col.groupby("class"))
        for scope, g in scopes:
            qh = qhat_pooled[name] if scope == "all" else qhat_by_class[name].get(scope, qhat_pooled[name])
            err_abs = g[err_col].abs(); sigma = g[sigma_col]
            cov_naive = coverage_table(err_abs, sigma, coverage_levels, multipliers=None)
            cov_calib = coverage_table(err_abs, sigma, coverage_levels, multipliers=qh)
            ece_naive = expected_calibration_error(err_abs, sigma, qhat_table=None)
            ece_calib = expected_calibration_error(err_abs, sigma, qhat_table=qh)
            coverage_report[name][scope] = {
                "n_columns": int(len(g)), "n_scans": int(g[["eye", "image_id"]].drop_duplicates().shape[0]),
                "uncalibrated": cov_naive, "calibrated": cov_calib,
                "ece_uncalibrated": ece_naive, "ece_calibrated": ece_calib,
            }
            print(f"  [{scope}] n_cols={len(g)}  ECE uncalibrated={ece_naive:.4f} "
                 f"-> calibrated={ece_calib:.4f}")
            print(f"    {'level':>6} {'obs(naive)':>11} {'w(naive)':>9} | "
                 f"{'obs(calib)':>11} {'w(calib)':>9}")
            for cn, cc in zip(cov_naive, cov_calib):
                print(f"    {cn['level']*100:5.0f}% {cn['observed_coverage']*100:10.1f}% "
                     f"{cn['mean_halfwidth']:8.2f}px | {cc['observed_coverage']*100:10.1f}% "
                     f"{cc['mean_halfwidth']:8.2f}px")
            if scope == "abnormal":
                worst = max(abs(c["observed_coverage"] - c["level"]) for c in cov_calib
                           if np.isfinite(c["observed_coverage"]))
                if worst > 0.07:
                    print(f"    [FLAG] AMD ('abnormal') calibrated coverage is still "
                         f"off by up to {worst*100:.1f} pts from nominal -- "
                         f"reporting honestly, NOT smoothing this over. Consider "
                         f"a larger AMD calibration pool or a stricter "
                         f"per-class-only conformal fit.")
        # reliability diagram (pooled + per class) for this target
        for scope, g in scopes:
            qh = qhat_pooled[name] if scope == "all" else qhat_by_class[name].get(scope, qhat_pooled[name])
            err_abs = g[err_col].abs(); sigma = g[sigma_col]
            obs_before = [coverage_table(err_abs, sigma, [c], None)[0]["observed_coverage"]
                         for c in coverage_levels]
            obs_after = [coverage_table(err_abs, sigma, [c], qh)[0]["observed_coverage"]
                        for c in coverage_levels]
            plot_reliability(coverage_levels, obs_before, obs_after,
                            os.path.join(args.out_dir, f"reliability_{name}_{scope}.png"),
                            f"{name} reliability [{scope}]")
    report["coverage"] = coverage_report

    # ---- 5b. PART B supplement (single_split): per-class discrimination check ----
    # Own flag (--conformal_diagnostics), INDEPENDENT of --conformal_per_class --
    # this diagnostic never alters the fit above, in either direction. The
    # "READ THIS" sub-message specifically describes the per-class fit's effect,
    # so it stays additionally gated on --conformal_per_class (nested): with
    # diagnostics ON but per_class OFF, the correlation numbers still print (they
    # describe the pooled fit's residuals same as ever), just without a claim
    # about per-class widening that would not apply.
    if args.conformal_diagnostics:
        print("\n" + "=" * 92)
        print("PART B (single_split) — per-class discrimination: sigma-vs-|error| corr")
        print("=" * 92)
        pb = {}
        for name, (err_col, sigma_col) in targets.items():
            sc = sigma_error_correlation(test_col, sigma_col, err_col)
            pb[name] = sc
            print(f"  {name}:")
            for scope, r in sc.items():
                print(f"    [{scope:>8}] pearson={r['pearson_r']:+.3f}  "
                      f"spearman={r['spearman_r']:+.3f}  (n={r['n']})")
        report["sigma_error_correlation"] = pb
        ab = pb["lower_boundary"].get("abnormal", {})
        if args.conformal_per_class and np.isfinite(ab.get("spearman_r", float("nan"))):
            print(f"\n  [READ THIS] Per-class conformal fixes AMD marginal coverage by "
                  f"WIDENING, but the AMD lower-boundary sigma orders the error with "
                  f"spearman={ab['spearman_r']:+.3f} — near-zero/negative means the "
                  f"AMD interval is calibrated-but-BLUNT (does not know WHERE it is "
                  f"uncertain). NOTE: single_split coverage is a 2-patient estimate; "
                  f"use --calibration_mode cross for a many-patient coverage number.")

    # ---- 6. PART 3: clinical output ----
    print("\n" + "=" * 92)
    print("PART 3 -- clinical output (PIXELS -- see PIXEL_TO_MICRON note below)")
    print("=" * 92)
    if PIXEL_TO_MICRON_UM is None:
        print("  [FLAG] No axial (depth) pixel spacing is recorded anywhere in "
             "this repository for the Bioptigen SDOCT acquisition used here. "
             "ALL thickness/uncertainty numbers below are in PIXELS. Converting "
             "to microns requires the scan's actual axial scale (instrument "
             "export settings / DICOM header) -- NOT invented here.")

    # 95% calibrated half-width per column (thickness), applied per-eye.
    qh95_pooled = {cls: qhat_by_class["thickness"].get(cls, qhat_pooled["thickness"])[_lvl(0.95)]
                  for cls in test_col["class"].dropna().unique()}
    qh95_pooled["all"] = qhat_pooled["thickness"][_lvl(0.95)]

    test_scan = test_scan.copy()
    test_scan["qhat95"] = test_scan["class"].map(lambda c: qh95_pooled.get(c, qh95_pooled["all"]))
    # per-scan sigma_t_corr for the subfoveal window: reuse the per-column
    # correlation-adjustment on the scan-level mean stds (same rho logic).
    def _scan_sigma_t(row):
        rho = rho_map.get(row["class"], rho_map["all"])["pearson_r"]
        rho = rho if np.isfinite(rho) else 0.0
        var = max(row["mean_std_upper"] ** 2 + row["mean_std_lower"] ** 2
                 - 2 * rho * row["mean_std_upper"] * row["mean_std_lower"], 0.0)
        return float(np.sqrt(var))
    test_scan["sigma_t_corr"] = test_scan.apply(_scan_sigma_t, axis=1)
    test_scan["halfwidth95"] = test_scan["qhat95"] * test_scan["sigma_t_corr"]

    print("\nper-EYE subfoveal thickness +/- calibrated 95% interval (px), test set:")
    per_eye_rows = []
    for eye, g in test_scan.groupby("eye"):
        n_scans = len(g)
        mean_t = float(g["mean_t_pred"].mean())
        # Different SCANS of the same eye are plausibly closer to independent
        # observations than adjacent COLUMNS within one scan, so the per-eye
        # mean's RAW uncertainty IS shrunk by sqrt(n_scans) here (unlike the
        # within-scan subfoveal-window aggregation above, which is NOT
        # shrunk). CAVEAT: qhat95 was fit against PER-COLUMN residuals, not
        # per-eye-mean residuals, so applying it to this shrunk SE is an
        # error-propagation APPROXIMATION, not an independently eye-level-
        # conformal-calibrated interval -- flagged here rather than
        # overclaiming a coverage guarantee this number doesn't strictly have.
        se_t = float(np.sqrt((g["sigma_t_corr"] ** 2).mean()) / np.sqrt(max(n_scans, 1)))
        qh95 = qh95_pooled.get(g["class"].iloc[0], qh95_pooled["all"])
        hw95 = qh95 * se_t
        per_eye_rows.append({"eye": eye, "class": g["class"].iloc[0], "n_scans": n_scans,
                             "mean_thickness_px": mean_t, "calibrated_95_halfwidth_px": hw95})
        print(f"  {eye:>20} [{g['class'].iloc[0]:>8}] n_scans={n_scans:3d}  "
             f"thickness={mean_t:7.2f} +/- {hw95:6.2f} px")
    report["per_eye_thickness_px"] = per_eye_rows

    # ---- 7. Example thickness-profile figures ----
    print("\n--- example thickness-profile figures ---")
    example_indices: List[Tuple[int, str]] = []
    classes_seen = list(dict.fromkeys(test_ds.class_labels))
    for cls in classes_seen:
        idxs = [i for i, c in enumerate(test_ds.class_labels) if c == cls]
        if idxs:
            example_indices.append((idxs[0], f"{cls}_example"))
    # "most uncertain": scan with the largest mean predicted std (upper+lower),
    # computed from calib_scan/test_scan already collected.
    if not test_scan.empty:
        unc = (test_scan["mean_std_upper"] + test_scan["mean_std_lower"])
        worst_i = int(unc.values.argmax())
        worst_eye, worst_img = test_scan.iloc[worst_i][["eye", "image_id"]]
        for i in range(len(test_ds)):
            if test_ds.eye_ids[i] == worst_eye and test_ds.image_ids[i] == worst_img:
                example_indices.append((i, "most_uncertain_example"))
                break

    with torch.no_grad():
        for i, tag in example_indices[: max(args.n_example_scans, len(example_indices))]:
            inp, target, mask, _ = test_ds[i]
            inp_b = inp.unsqueeze(0).float().to(device)
            mask_b = mask.unsqueeze(0).float().to(device)
            model(inp_b)
            logits = model.last_boundary_logits.float()
            e_upper, std_upper, e_lower, std_lower = boundary_distribution_to_depths(logits)
            z_upper_gt, z_lower_gt, valid = derive_column_boundaries(mask_b)
            v = valid[0].cpu().numpy()
            cols = np.nonzero(v)[0]
            if cols.size == 0:
                continue
            cls = test_ds.class_labels[i]
            qh_t = qhat_by_class["thickness"].get(cls, qhat_pooled["thickness"])
            qh_u = qhat_by_class["upper_boundary"].get(cls, qhat_pooled["upper_boundary"])
            qh_l = qhat_by_class["lower_boundary"].get(cls, qhat_pooled["lower_boundary"])
            rho = rho_map.get(cls, rho_map["all"])["pearson_r"]
            rho = rho if np.isfinite(rho) else 0.0

            eu = e_upper[0].cpu().numpy()[cols]; el = e_lower[0].cpu().numpy()[cols]
            su = std_upper[0].cpu().numpy()[cols]; sl = std_lower[0].cpu().numpy()[cols]
            zu = z_upper_gt[0].cpu().numpy()[cols]; zl = z_lower_gt[0].cpu().numpy()[cols]
            hw_u = qh_u[_lvl(0.95)] * su
            hw_l = qh_l[_lvl(0.95)] * sl
            sigma_t = np.sqrt(np.clip(su ** 2 + sl ** 2 - 2 * rho * su * sl, 0.0, None))
            hw_t = qh_t[_lvl(0.95)] * sigma_t
            t_pred = el - eu
            t_gt = zl - zu
            image_2d = inp[1].numpy()  # center frame of the 2.5D stack, log domain

            out_path = os.path.join(args.out_dir, f"thickness_profile_{tag}.png")
            plot_thickness_profile(image_2d, cols, eu, hw_u, el, hw_l, t_pred, hw_t, t_gt,
                                   out_path, f"{tag} (eye={test_ds.eye_ids[i]}, "
                                             f"image={test_ds.image_ids[i]}, class={cls})")
            print(f"  wrote {out_path}")

    # ---- 8. Write JSON report ----
    out_json = os.path.join(args.out_dir, "thickness_uncertainty_report.json")
    from .. import provenance as _prov
    _prov.stamp(report, checkpoint=args.checkpoint, protocol="single_split",
                patients=sorted(set(calib_patients) | set(test_patients)),
                extra={"calibration_protocol": protocol_str})
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
