#!/usr/bin/env python
"""Stage 2 CLI: fit (or reuse) the motion-artifact noise calibration.

Each severity's noise amplitude is anchored to a quantile of real smartphone
PPG noise (BUT PPG) and solved on clean PulseDB-MIMIC reference windows -- see
trustbio/degradation/calibrate.py for why the accelerometer-based mapping was
abandoned. Degradation itself is applied in-line during feature extraction
(extract_features.py --degrade-kind / --degrade-severity), not cached as
signals, because it must hit the raw signal before each model's own
preprocessing.

    python scripts/inject_degradation.py --refit-calibration \
        --reference-cohort-cache features_cache/taxonomy --n-reference 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trustbio.data.but_ppg import DEFAULT_ROOT as BUT_PPG_ROOT
from trustbio.data.pulsedb import DEFAULT_ROOT as PULSEDB_ROOT, make_pulsedb_signal_loader
from trustbio.degradation.calibrate import fit_motion_noise_amplitude, load_cached_noise_amplitude


def reference_windows(cohort_csv: Path, pulsedb_root: Path, n_reference: int, per_subject: int):
    """Up to `per_subject` clean PulseDB-MIMIC PPG windows per subject, in
    cohort order, until `n_reference` windows are collected."""
    cohort = pd.read_csv(cohort_csv, dtype={"visit_id": str, "subject_id": str})
    load = make_pulsedb_signal_loader(pulsedb_root, "mimic")
    out, counts = [], {}
    for vid, sid in zip(cohort["visit_id"], cohort["subject_id"]):
        if counts.get(sid, 0) >= per_subject:
            continue
        ppg, fs = load(vid, "ppg")
        out.append((ppg, fs))
        counts[sid] = counts.get(sid, 0) + 1
        if len(out) >= n_reference:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--but-ppg-root", type=Path, default=BUT_PPG_ROOT)
    ap.add_argument("--pulsedb-root", type=Path, default=PULSEDB_ROOT)
    ap.add_argument("--reference-cohort-cache", type=Path, default=Path("features_cache/taxonomy"),
                    help="dir holding pulsedb_mimic_cohort.csv to draw clean reference PPG windows from")
    ap.add_argument("--n-reference", type=int, default=200)
    ap.add_argument("--per-subject", type=int, default=4)
    ap.add_argument("--refit-calibration", action="store_true")
    args = ap.parse_args()

    if not args.refit_calibration:
        try:
            amplitudes = load_cached_noise_amplitude()
            print(f"[inject_degradation] cached noise amplitudes by severity: {amplitudes}")
            return 0
        except FileNotFoundError:
            print("[inject_degradation] no cache; fitting")
    ref = reference_windows(args.reference_cohort_cache / "pulsedb_mimic_cohort.csv", args.pulsedb_root,
                            args.n_reference, args.per_subject)
    print(f"[inject_degradation] {len(ref)} clean reference PPG windows", flush=True)
    amplitudes = fit_motion_noise_amplitude(args.but_ppg_root, ref, cache=True)
    print(f"[inject_degradation] noise amplitudes by severity: {amplitudes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
