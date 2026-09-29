"""Per-sub-window signal-quality traces for ECG and PPG.

The taxonomy's transient/persistent distinction is about WHEN and for HOW
LONG quality drops, so it needs a quality value per second, on the exact
(possibly degraded) waveform a model was fed. Deliberately little:

  * ECG: an electrode-off detector. A sub-window whose spread is a negligible
    fraction of the whole window's spread scores 0, everything else 1. The
    only ECG fault injected here (lead-off) and the only one the native
    MIMIC-III-Ext-PPG code -3 confirms (missing samples) is a flat line, and
    a noise-based measure gets ECG BACKWARDS -- QRS complexes are the high-
    frequency content, so a clean ECG looks "noisy" (native AUROC 0.29 in the
    first run of this analysis).
  * PPG: out-of-band noise. The window is low-passed at 8 Hz (pulse content
    and respiratory baseline lie below); per sub-window, std(residual) /
    std(signal) is the noise ratio, and quality is 1 - ratio / ref, clipped
    to [0, 1]. Frequency-based so it
    means the same thing at 30 Hz (smartphone camera) and 125 Hz (monitor) --
    a fixed-length moving-average kernel did not, and made every camera-PPG
    recording look destroyed. `ref` is the ratio at which quality is called
    zero: the 95th percentile of real BUT PPG per-second ratios, i.e. "as
    noisy as the noisiest 5% of real smartphone-PPG seconds"
    (scripts/build_fault_features.py). Flat sub-windows score 0 as well.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

FLAT_REL_STD = 1e-4
PULSE_LOWPASS_HZ = 8.0     # pulse morphology and respiratory baseline both lie below this
DEFAULT_PPG_REF = 0.5


def _sub_windows(x: np.ndarray, fs: int, window_sec: float) -> list[np.ndarray]:
    n = max(1, int(round(window_sec * fs)))
    n_win = max(1, len(x) // n)
    return [x[i * n:(i + 1) * n] for i in range(n_win)]


def _is_flat(w: np.ndarray, whole_sd: float) -> bool:
    return whole_sd == 0.0 or float(np.std(w)) < FLAT_REL_STD * whole_sd


def pulse_band_residual(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """(in-band, residual) decomposition of a demeaned PPG window: in-band is
    an 8-Hz zero-phase low-pass (pulse morphology and respiratory baseline),
    the residual is everything above it. A band-pass with a 0.5-Hz high-pass
    edge was tried first; its multi-second transient made the first and last
    second of every clean window look noisy."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    hi = min(PULSE_LOWPASS_HZ, 0.45 * fs)
    if len(x) < 3 * 3 * 2 + 1 or hi <= 0:
        return x, np.zeros_like(x)
    b, a = butter(3, hi, btype="lowpass", fs=fs)
    band = filtfilt(b, a, x, padlen=min(len(x) - 1, int(fs)))
    return band, x - band


def ppg_oob_ratio_trace(x: np.ndarray, fs: int, window_sec: float = 1.0) -> np.ndarray:
    """Per sub-window out-of-pulse-band noise ratio, clipped to [0, 2]."""
    x = np.asarray(x, dtype=np.float64)
    band, resid = pulse_band_residual(x, fs)
    out = []
    for xw, rw in zip(_sub_windows(x - x.mean(), fs, window_sec), _sub_windows(resid, fs, window_sec)):
        sd = float(np.std(xw))
        out.append(0.0 if sd == 0.0 else float(np.clip(np.std(rw) / sd, 0.0, 2.0)))
    return np.asarray(out, dtype=np.float64)


def mean_ppg_oob_ratio(x: np.ndarray, fs: int, window_sec: float = 1.0) -> float:
    """Mean per-sub-window out-of-band ratio -- the per-recording statistic
    the motion-noise calibration matches to real smartphone PPG."""
    return float(np.mean(ppg_oob_ratio_trace(x, fs, window_sec)))


def ppg_sqi_trace(x: np.ndarray, fs: int, ref: float, window_sec: float = 1.0) -> np.ndarray:
    """PPG quality in [0, 1] per sub-window: flat -> 0, else 1 - ratio / ref."""
    x = np.asarray(x, dtype=np.float64)
    whole_sd = float(np.std(x))
    ratios = ppg_oob_ratio_trace(x, fs, window_sec)
    out = []
    for w, r in zip(_sub_windows(x, fs, window_sec), ratios):
        out.append(0.0 if _is_flat(w, whole_sd) else float(np.clip(1.0 - r / ref, 0.0, 1.0)))
    return np.asarray(out, dtype=np.float64)


def ecg_sqi_trace(x: np.ndarray, fs: int, window_sec: float = 1.0) -> np.ndarray:
    """ECG quality per sub-window: 0 for a flat (electrode-off / dropout)
    sub-window, 1 otherwise."""
    x = np.asarray(x, dtype=np.float64)
    whole_sd = float(np.std(x))
    return np.asarray([0.0 if _is_flat(w, whole_sd) else 1.0 for w in _sub_windows(x, fs, window_sec)])


def combined_sqi_trace(ecg_sqi: np.ndarray, ppg_sqi: np.ndarray) -> np.ndarray:
    n = min(len(ecg_sqi), len(ppg_sqi))
    return np.minimum(np.asarray(ecg_sqi[:n]), np.asarray(ppg_sqi[:n]))


def calibrate_ppg_ref(per_second_ratios: np.ndarray, quantile: float = 0.95) -> float:
    """Quality-zero reference: the `quantile` of real per-second out-of-band
    ratios (default: the noisiest 5% of real smartphone-PPG seconds)."""
    r = np.asarray(per_second_ratios, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return DEFAULT_PPG_REF
    return float(max(np.quantile(r, quantile), 1e-3))
