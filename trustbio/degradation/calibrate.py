"""Calibrate synthetic motion-artifact noise amplitude against REAL smartphone
PPG noise levels (BUT PPG).

Design note (2026-09-28). The original approach fitted noise ~ accelerometer
magnitude across BUT PPG recordings and mapped severity to a fraction of the
maximum accelerometer magnitude. On the real data that relationship does not
exist: accelerometer dynamics separate quality-1 from quality-0 recordings at
AUROC 0.57, PPG noise measures at 0.50-0.57, and noise vs accelerometer
Spearman correlation is -0.11 (n = 3,795 accelerometer-era recordings); the
fit collapsed to its floor. What BUT PPG does provide is a large sample of
real camera-PPG noise levels, so each severity is anchored to a QUANTILE of
that distribution: the corrupted span of an injected window is made as noisy
(high-frequency residual ratio, trustbio.taxonomy.sqi) as the 50th / 75th /
95th percentile real BUT PPG recording for severity 0.1 / 0.3 / 0.6. The
amplitude achieving each target is solved on clean reference PPG windows
(PulseDB) by bisection. The accelerometer cross-check is a negative result
and is reported as such.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import DEGRADATION_SEVERITIES
from ..data.but_ppg import build_but_ppg_cohort, make_but_ppg_signal_loader
from ..taxonomy.sqi import mean_hf_ratio

NOISE_AMPLITUDE_CACHE_PATH = Path(
    Path(__file__).resolve().parent.parent.parent / "features_cache" / "noise_amplitude_cache.json"
)
CALIBRATION_DETAILS_NAME = "noise_amplitude_calibration.json"
SEVERITY_QUANTILES = {0.1: 0.50, 0.3: 0.75, 0.6: 0.95}
SMOOTH_KERNEL = 5


def colored_noise(rng: np.random.Generator, n: int, std: float) -> np.ndarray:
    """Smoothed (5-sample moving average) white noise -- the motion-artifact
    noise model shared by injection and calibration."""
    raw = rng.normal(0.0, std, n)
    return np.convolve(raw, np.ones(SMOOTH_KERNEL) / SMOOTH_KERNEL, mode="same")


def real_noise_quantiles(root: str | Path, quantiles: dict[float, float] = SEVERITY_QUANTILES,
                         min_std: float = 1e-9) -> tuple[dict[float, float], int]:
    """{severity: HF-ratio target} from every readable, non-flat BUT PPG
    recording, plus the number of recordings used."""
    root = Path(root)
    cohort = build_but_ppg_cohort(root, annotate_reconstruction=False)
    load = make_but_ppg_signal_loader(root)
    ratios = []
    for vid in cohort.visits["visit_id"]:
        try:
            ppg, fs = load(vid, "ppg")
        except Exception:  # noqa: BLE001 -- an unreadable record is simply not a sample
            continue
        ppg = np.asarray(ppg, dtype=float)
        if np.std(ppg) <= min_std or np.mean(np.abs(np.diff(ppg)) == 0) > 0.5:
            continue
        ratios.append(mean_hf_ratio(ppg, fs))
    if len(ratios) < 10:
        raise ValueError(f"only {len(ratios)} usable BUT PPG recordings under {root}; need >= 10")
    r = np.asarray(ratios)
    return {sev: float(np.quantile(r, q)) for sev, q in quantiles.items()}, int(len(r))


def _achieved_ratio(reference_ppg, amplitude: float, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for sig, fs in reference_ppg:
        sig = np.asarray(sig, dtype=float)
        noisy = sig + colored_noise(rng, len(sig), amplitude * float(np.std(sig)))
        vals.append(mean_hf_ratio(noisy, fs))
    return float(np.mean(vals))


def solve_amplitude(reference_ppg, target: float, lo: float = 1e-3, hi: float = 20.0,
                    iters: int = 40, seed: int = 0) -> float:
    """Bisection (in log space) on the amplitude -- noise std as a multiple of
    the window's std -- so that the mean HF ratio over the reference windows
    hits `target`. The ratio is monotone in the amplitude."""
    if _achieved_ratio(reference_ppg, lo, seed) >= target:
        return lo
    if _achieved_ratio(reference_ppg, hi, seed) <= target:
        return hi
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        if _achieved_ratio(reference_ppg, mid, seed) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def fit_motion_noise_amplitude(
    root: str | Path,
    reference_ppg,
    severities: list[float] = DEGRADATION_SEVERITIES,
    cache: bool = True,
    seed: int = 0,
) -> dict[float, float]:
    """Return {severity: noise_amplitude}. `reference_ppg` is a list of
    (clean_ppg_window, fs) pairs the amplitudes are solved on."""
    if len(reference_ppg) == 0:
        raise ValueError("need at least one clean reference PPG window")
    targets, n_real = real_noise_quantiles(root, {s: SEVERITY_QUANTILES[s] for s in severities})
    amplitudes = {sev: solve_amplitude(reference_ppg, targets[sev], seed=seed) for sev in severities}
    if cache:
        NOISE_AMPLITUDE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOISE_AMPLITUDE_CACHE_PATH.write_text(
            json.dumps({str(k): v for k, v in amplitudes.items()}, indent=2)
        )
        details = dict(
            method="severity -> quantile of real BUT PPG HF-noise ratio; amplitude solved on clean reference PPG",
            quantiles={str(s): SEVERITY_QUANTILES[s] for s in severities},
            targets={str(k): v for k, v in targets.items()},
            achieved={str(s): _achieved_ratio(reference_ppg, amplitudes[s], seed) for s in severities},
            n_real_recordings=n_real, n_reference_windows=len(reference_ppg), seed=seed,
        )
        (NOISE_AMPLITUDE_CACHE_PATH.parent / CALIBRATION_DETAILS_NAME).write_text(json.dumps(details, indent=2))
    return amplitudes


def load_cached_noise_amplitude() -> dict[float, float]:
    """Load a previously fitted amplitude dict from cache (raises if absent --
    callers should run fit_motion_noise_amplitude once before this)."""
    if not NOISE_AMPLITUDE_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{NOISE_AMPLITUDE_CACHE_PATH} not found; run "
            "scripts/inject_degradation.py --refit-calibration first."
        )
    raw = json.loads(NOISE_AMPLITUDE_CACHE_PATH.read_text())
    return {float(k): v for k, v in raw.items()}
