"""Degradation injection: motion artifact, lead-off (electrode disconnect),
and missing-PPG-channel conditions, at three severity levels (fraction of the
segment affected). Motion-artifact noise amplitude is calibrated against real
BUT PPG accelerometer/quality data (degradation/calibrate.py); lead-off and
missing-PPG have no free parameters to calibrate (a disconnected electrode
reads zero; a missing channel is simply absent).
"""
from __future__ import annotations

import numpy as np

from .calibrate import load_cached_noise_amplitude


def inject_motion_artifact(
    sig: np.ndarray,
    fs: int,
    severity: float,
    rng: np.random.Generator,
    noise_amplitudes: dict[float, float] | None = None,
) -> np.ndarray:
    """Add colored (smoothed white) noise to a randomly placed contiguous span
    covering `severity` fraction of the signal. Noise amplitude is looked up
    from `noise_amplitudes` (or the BUT-PPG-calibrated cache if not given)."""
    amplitudes = noise_amplitudes if noise_amplitudes is not None else load_cached_noise_amplitude()
    amplitude = amplitudes[severity]

    out = sig.copy()
    burst_len = max(1, int(severity * len(sig)))
    start = int(rng.integers(0, max(1, len(sig) - burst_len + 1)))
    raw_noise = rng.normal(0, amplitude * np.std(sig), burst_len)
    # Smooth (colored) noise better approximates real motion artifact than
    # white noise, matching the "colored noise" framing in the study design.
    kernel = np.ones(5) / 5
    colored_noise = np.convolve(raw_noise, kernel, mode="same")
    out[start:start + burst_len] += colored_noise.astype(out.dtype)
    return out


def inject_lead_off(
    sig: np.ndarray, severity: float, rng: np.random.Generator,
) -> np.ndarray:
    """Zero out (flat-line) a randomly placed contiguous span covering
    `severity` fraction of the signal, simulating an electrode disconnect."""
    out = sig.copy()
    span = max(1, int(round(severity * len(sig))))
    start = int(rng.integers(0, max(1, len(sig) - span + 1)))
    out[start:start + span] = 0.0
    return out


def inject_missing_ppg(ppg_sig: np.ndarray | None) -> None:
    """Drop the PPG channel entirely for this segment."""
    return None


def apply_degradation(
    ecg: np.ndarray,
    ppg: np.ndarray | None,
    fs: int,
    kind: str,
    severity: float,
    rng: np.random.Generator,
    noise_amplitudes: dict[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dispatch to the appropriate injection function by `kind`.

    - "motion_artifact": corrupts PPG (the modality most susceptible to
      motion artifact in practice); ECG is passed through unchanged.
    - "lead_off": corrupts ECG (electrode disconnect is an ECG-specific
      failure mode); PPG is passed through unchanged.
    - "missing_ppg": drops PPG entirely; ECG is passed through unchanged.

    Raises KeyError for any `kind` not in config.DEGRADATION_KINDS.
    """
    dispatch = {"motion_artifact", "lead_off", "missing_ppg"}
    if kind not in dispatch:
        raise KeyError(f"unknown degradation kind {kind!r}; expected one of {dispatch}")

    if kind == "motion_artifact":
        ppg_out = (
            inject_motion_artifact(ppg, fs, severity, rng, noise_amplitudes)
            if ppg is not None else None
        )
        return ecg, ppg_out
    if kind == "lead_off":
        return inject_lead_off(ecg, severity, rng), ppg
    # kind == "missing_ppg"
    return ecg, inject_missing_ppg(ppg)
