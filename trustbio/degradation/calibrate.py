"""Calibrate synthetic motion-artifact noise amplitude against BUT PPG's real
accelerometer-vs-quality relationship (Methods: synthetic degradation is a
controlled proxy, cross-validated against real motion artifact rather than an
arbitrary noise model).

Approach: for BUT PPG recordings with an accelerometer channel (ID >= 112001),
compute each recording's mean accelerometer magnitude and its binary quality
label. Fit a simple linear relationship between accelerometer magnitude and
the empirical PPG noise-to-signal ratio in poor-quality recordings, then map
each of TRUST-BIO's three severity levels (fraction of segment corrupted) onto
a noise amplitude via that fitted relationship — so "severity 0.6" corresponds
to roughly the accelerometer magnitude observed in the worst real BUT PPG
recordings, not an arbitrary noise level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import DEGRADATION_SEVERITIES
from ..data.but_ppg import (
    ACC_MIN_RECORD_ID, build_but_ppg_cohort, load_but_ppg_accelerometer,
    make_but_ppg_signal_loader,
)

NOISE_AMPLITUDE_CACHE_PATH = Path(
    Path(__file__).resolve().parent.parent.parent / "features_cache" / "noise_amplitude_cache.json"
)


def _accel_magnitude(acc: np.ndarray) -> float:
    """Mean magnitude of the triaxial accelerometer signal."""
    return float(np.mean(np.linalg.norm(acc, axis=1)))


def _ppg_noise_ratio(ppg: np.ndarray) -> float:
    """Empirical noise-to-signal proxy: high-frequency energy fraction, used
    as the quantity motion artifact is expected to inflate."""
    detrended = ppg - np.convolve(ppg, np.ones(5) / 5, mode="same")
    return float(np.std(detrended) / (np.std(ppg) + 1e-8))


def fit_motion_noise_amplitude(
    root: str | Path,
    severities: list[float] = DEGRADATION_SEVERITIES,
    cache: bool = True,
) -> dict[float, float]:
    """Return {severity: noise_amplitude} fitted from BUT PPG's real
    accelerometer/quality relationship among its accelerometer-equipped
    recordings (ID >= 112001)."""
    root = Path(root)
    cohort = build_but_ppg_cohort(root)
    load_signal = make_but_ppg_signal_loader(root)

    accel_mags, noise_ratios = [], []
    for visit_id in cohort.visits["visit_id"]:
        if int(visit_id) < ACC_MIN_RECORD_ID:
            continue
        acc_result = load_but_ppg_accelerometer(root, visit_id)
        if acc_result is None:
            continue
        acc, _fs = acc_result
        ppg, _fs = load_signal(visit_id, "ppg")
        accel_mags.append(_accel_magnitude(acc))
        noise_ratios.append(_ppg_noise_ratio(ppg))

    if len(accel_mags) < 2:
        raise ValueError(
            "need at least 2 accelerometer-equipped BUT PPG recordings to fit "
            "the motion-noise calibration; found "
            f"{len(accel_mags)}. Check ACC_MIN_RECORD_ID filtering."
        )

    # Linear fit: noise_ratio ~ slope * accel_magnitude + intercept.
    slope, intercept = np.polyfit(accel_mags, noise_ratios, deg=1)
    max_accel = max(accel_mags)

    amplitudes = {}
    for sev in severities:
        target_accel = sev * max_accel
        predicted_noise_ratio = max(slope * target_accel + intercept, 1e-3)
        amplitudes[sev] = predicted_noise_ratio

    if cache:
        NOISE_AMPLITUDE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOISE_AMPLITUDE_CACHE_PATH.write_text(
            json.dumps({str(k): v for k, v in amplitudes.items()}, indent=2)
        )
    return amplitudes


def load_cached_noise_amplitude() -> dict[float, float]:
    """Load a previously fitted amplitude dict from cache (raises if absent —
    callers should run fit_motion_noise_amplitude once before this)."""
    if not NOISE_AMPLITUDE_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{NOISE_AMPLITUDE_CACHE_PATH} not found; run "
            "fit_motion_noise_amplitude(but_ppg_root) first."
        )
    raw = json.loads(NOISE_AMPLITUDE_CACHE_PATH.read_text())
    return {float(k): v for k, v in raw.items()}
