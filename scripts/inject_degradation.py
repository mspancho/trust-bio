#!/usr/bin/env python
"""Stage 2 CLI: materialize degraded segment caches for PulseDB and
MIMIC-III-Ext-PPG (BUT PPG already contains real, not synthetic, degradation
and is used as-is)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from trustbio.config import DEGRADATION_KINDS, DEGRADATION_SEVERITIES
from trustbio.degradation.calibrate import fit_motion_noise_amplitude, load_cached_noise_amplitude
from trustbio.degradation.inject import apply_degradation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--but-ppg-root", type=Path, required=True,
                     help="used to (re)fit the motion-artifact noise-amplitude calibration")
    ap.add_argument("--refit-calibration", action="store_true")
    args = ap.parse_args()

    if args.refit_calibration:
        amplitudes = fit_motion_noise_amplitude(args.but_ppg_root, cache=True)
    else:
        try:
            amplitudes = load_cached_noise_amplitude()
        except FileNotFoundError:
            amplitudes = fit_motion_noise_amplitude(args.but_ppg_root, cache=True)
    print(f"[inject_degradation] noise amplitudes by severity: {amplitudes}")
    print(
        "[inject_degradation] calibration ready. Degradation is applied "
        "in-line during feature extraction (see extract_features.py "
        "--degrade-kind / --degrade-severity), not as a separate cached "
        "signal store, since it must be applied per-model at the raw-signal "
        "stage before each model's own preprocessing."
    )


if __name__ == "__main__":
    main()
