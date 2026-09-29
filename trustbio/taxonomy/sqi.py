"""Per-sub-window signal-quality traces for ECG and PPG.

The taxonomy's transient/persistent distinction is about WHEN and for HOW
LONG quality drops, so it needs a quality value per second, on the exact
(possibly degraded) waveform a model was fed. Two ingredients, deliberately
nothing more:

  * flat-line: a sub-window whose spread is a negligible fraction of the whole
    window's spread scores 0 -- an electrode off or zeroed span;
  * high-frequency residual ratio: std(x - moving_average(x, 40 ms)) / std(x),
    the same quantity degradation/calibrate.py uses to define "noise". The
    trace is 1 - ratio / hf_ref, clipped to [0, 1]; `hf_ref` is the ratio at
    which quality is called zero, one free scale per modality, fitted against
    MIMIC-III-Ext-PPG's native SQI (scripts/build_fault_features.py).
"""
from __future__ import annotations

import numpy as np

FLAT_REL_STD = 1e-4
SMOOTH_SEC = 0.04
DEFAULT_HF_REF = 0.5


def hf_noise_ratio(x: np.ndarray, fs: int) -> float:
    """High-frequency residual energy as a fraction of total spread; 0 for a
    constant signal."""
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd == 0.0:
        return 0.0
    x = x - x.mean()
    k = max(3, int(round(SMOOTH_SEC * fs)))
    # Edge-pad before smoothing: np.convolve(mode="same") zero-pads, so a
    # signal with a large DC offset (camera PPG sits at ~100-200 intensity
    # units) gets DC-sized residuals at the edges and ratios far above 1.
    padded = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
    return float(np.std(x - smooth) / sd)


def _sub_windows(x: np.ndarray, fs: int, window_sec: float) -> list[np.ndarray]:
    n = max(1, int(round(window_sec * fs)))
    n_win = max(1, len(x) // n)
    return [x[i * n:(i + 1) * n] for i in range(n_win)]


def mean_hf_ratio(x: np.ndarray, fs: int, window_sec: float = 1.0) -> float:
    """Mean sub-window residual ratio -- the per-segment statistic that gets
    compared against a native quality label when calibrating hf_ref."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean([hf_noise_ratio(w, fs) for w in _sub_windows(x, fs, window_sec)]))


def sqi_trace(x: np.ndarray, fs: int, hf_ref: float, window_sec: float = 1.0) -> np.ndarray:
    """Quality in [0, 1] per sub-window of `window_sec` seconds."""
    x = np.asarray(x, dtype=np.float64)
    whole_sd = float(np.std(x))
    out = []
    for w in _sub_windows(x, fs, window_sec):
        if whole_sd == 0.0 or float(np.std(w)) < FLAT_REL_STD * whole_sd:
            out.append(0.0)
        else:
            out.append(float(np.clip(1.0 - hf_noise_ratio(w, fs) / hf_ref, 0.0, 1.0)))
    return np.asarray(out, dtype=np.float64)


def combined_sqi_trace(ecg_sqi: np.ndarray, ppg_sqi: np.ndarray) -> np.ndarray:
    n = min(len(ecg_sqi), len(ppg_sqi))
    return np.minimum(np.asarray(ecg_sqi[:n]), np.asarray(ppg_sqi[:n]))


def calibrate_hf_ref(hf_good: np.ndarray, hf_poor: np.ndarray) -> float:
    """Pick hf_ref so that the trace's 0.5 threshold (hf_ratio == hf_ref / 2)
    best separates natively-good from natively-poor segments (max Youden J
    over the pooled per-segment mean ratios)."""
    good, poor = np.asarray(hf_good, float), np.asarray(hf_poor, float)
    if len(good) == 0 or len(poor) == 0:
        return DEFAULT_HF_REF
    best_t, best_j = float(np.median(np.r_[good, poor])), -np.inf
    for t in np.unique(np.r_[good, poor]):
        j = float(np.mean(poor >= t) - np.mean(good >= t))
        if j > best_j:
            best_j, best_t = j, float(t)
    return 2.0 * best_t
