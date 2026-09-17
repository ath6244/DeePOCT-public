"""provenance.py -- one shared ``_provenance`` block for every report and dump.

WHY THIS EXISTS
The persistence survey found that thickness_uncertainty.py's cross-conformal report
CANNOT SAY WHICH MODEL PRODUCED IT: no checkpoint field, no ensemble member list. That
is why a single-model run and a 2-member deep-ensemble run were indistinguishable on
disk, and why audit A could not be answered even after the artifacts were synced. A
second instance: two supervised-denoiser benchmarks (v1 -> supervised_bench.json,
v2 -> supervised_v2_bench.jso) differ by 5.1 dB and neither file names its own model
or script, so an audit read the wrong one.

Dumping finer-grained numbers does not fix that. Naming the run does.

UNCONDITIONAL BY DESIGN
``_provenance`` is written ALWAYS, never behind a flag. A provenance block that can be
forgotten will be forgotten exactly when it matters. This means report bytes DO change
versus the un-patched code -- the deliberate, single exception to byte-identity. The
regression test in test_persistence_provenance.py therefore compares the metrics
payload with ``_provenance`` STRIPPED, which is the invariant that actually matters:
no measured number may move.

Keep this module dependency-light (stdlib + subprocess git) so importing it can never
perturb a numeric pipeline.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

PROVENANCE_KEY = "_provenance"
_HASH_MAX_BYTES = 256 * 1024 * 1024      # don't hash absurdly large files


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                     # git absent / not a repo / timeout
        return None


def file_digest(path: Optional[str]) -> Optional[str]:
    """sha256 of a file, or None. Used for checkpoints so a report identifies the
    exact WEIGHTS, not just a path that may later be overwritten."""
    if not path or not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) > _HASH_MAX_BYTES:
            return "SKIPPED_TOO_LARGE"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def provenance_block(checkpoint: Optional[str] = None,
                     ensemble_members: Optional[Iterable[str]] = None,
                     patients: Optional[Iterable[object]] = None,
                     protocol: Optional[str] = None,
                     extra: Optional[Dict[str, object]] = None) -> dict:
    """The block to store under ``_provenance``.

    checkpoint       -- path to the weights that produced the numbers
    ensemble_members -- member checkpoint paths when an ensemble was used (the field
                        that was missing when a 2-member ensemble could not be told
                        apart from a single model)
    patients         -- patient IDs the numbers are computed over
    protocol         -- calibration/eval protocol name, e.g. "cross/val_only_lopo",
                        "cross/cross_fold", "single_split", "kfold_oof"
    """
    members: Optional[List[str]] = (
        [str(m) for m in ensemble_members] if ensemble_members is not None else None)
    block: Dict[str, object] = {
        "script": os.path.basename(sys.argv[0]) if sys.argv else None,
        "argv": list(sys.argv[1:]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_digest(checkpoint),
        "ensemble_members": members,
        "ensemble_members_sha256": ([file_digest(m) for m in members]
                                    if members else None),
        "n_ensemble_members": len(members) if members else None,
        "protocol": protocol,
        "patients": ([str(p) for p in patients] if patients is not None else None),
        "n_patients": (len(list(patients)) if patients is not None else None),
    }
    if extra:
        block.update(extra)
    return block


def stamp(report: dict, **kwargs) -> dict:
    """Attach ``_provenance`` at the TOP LEVEL of a report dict, in place."""
    report[PROVENANCE_KEY] = provenance_block(**kwargs)
    return report


def strip_provenance(obj):
    """Recursively remove every ``_provenance`` key -- the metrics payload that the
    byte-identity regression test hashes."""
    if isinstance(obj, dict):
        return {k: strip_provenance(v) for k, v in obj.items() if k != PROVENANCE_KEY}
    if isinstance(obj, list):
        return [strip_provenance(v) for v in obj]
    return obj


def payload_digest(obj) -> str:
    """Stable sha256 of a report's METRICS payload, provenance excluded. Sorted keys
    so dict ordering can never register as a change."""
    import json
    return hashlib.sha256(
        json.dumps(strip_provenance(obj), sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def dump_frame(df, path: str, label: str = "records", **prov) -> str:
    """Write a per-unit dump as CSV, with a sidecar ``<path>.provenance.json`` so the
    dump identifies its own run exactly as the report does. Returns ``path``."""
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    with open(path + ".provenance.json", "w") as f:
        json.dump({PROVENANCE_KEY: provenance_block(**prov),
                   "rows": int(len(df)), "label": label,
                   "columns": [str(c) for c in df.columns]}, f, indent=2)
    print(f"  wrote {path} ({len(df)} {label}) + .provenance.json")
    return path
