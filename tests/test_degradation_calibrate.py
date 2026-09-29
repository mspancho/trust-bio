import json

import numpy as np
import pandas as pd
import pytest
import wfdb

from trustbio.config import DEGRADATION_SEVERITIES
from trustbio.degradation.calibrate import fit_motion_noise_amplitude


@pytest.fixture(autouse=True)
def _isolate_noise_cache(tmp_path, monkeypatch):
    """fit_motion_noise_amplitude(cache=True) writes NOISE_AMPLITUDE_CACHE_PATH.
    Left pointing at the repo, the test suite overwrote the real
    features_cache/noise_amplitude_cache.json with fixture-derived numbers
    (observed 2026-09-25). Every test in this module gets a throwaway path."""
    monkeypatch.setattr(
        "trustbio.degradation.calibrate.NOISE_AMPLITUDE_CACHE_PATH",
        tmp_path / "noise_amplitude_cache.json",
    )


@pytest.fixture
def reference_ppg():
    """Six clean synthetic PPG windows at 125 Hz the amplitudes are solved on."""
    out = []
    for i in range(6):
        t = np.arange(10 * 125) / 125.0
        rng = np.random.default_rng(100 + i)
        sig = np.sin(2 * np.pi * (1.0 + 0.1 * i) * t) + 0.3 * np.sin(2 * np.pi * (2.0 + 0.2 * i) * t)
        out.append(((sig + 0.01 * rng.standard_normal(len(t))).astype(np.float32), 125))
    return out


@pytest.fixture
def fake_but_ppg_with_accel(tmp_path):
    root = tmp_path / "but-ppg"
    root.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    # Recordings with progressively higher accel magnitude and lower quality,
    # so the fit has real signal to recover (higher motion -> more noise needed
    # to reproduce the observed quality degradation).
    for i, rec_id in enumerate(str(112001 + k) for k in range(12)):
        accel_scale = 1.0 + i * 0.7
        quality = 1 if i < 6 else 0   # first half "good", second half "poor"
        # PPG mixes a smooth low-frequency baseline with high-frequency noise,
        # with the noise fraction tracking accel_scale, so higher real motion
        # actually produces a higher high-frequency noise-to-signal ratio in
        # the PPG channel (matching this fixture's intent below) rather than
        # leaving the PPG draw independent of accel_scale/quality, which would
        # make the fitted slope's sign a coin flip across seeds.
        n = 30 * 10
        smooth = np.sin(2 * np.pi * np.arange(n) / n)
        hf_noise = rng.standard_normal(n)
        w_hf = accel_scale / (accel_scale + 1.0)
        ppg = ((1 - w_hf) * smooth + w_hf * hf_noise).astype(np.float32)
        acc = (accel_scale * rng.standard_normal((100 * 10, 3))).astype(np.float32)
        # Nested per-record subdirectory, matching PhysioNet's REAL BUT PPG
        # layout (<root>/112001/112001_PPG.*) -- see the note in
        # tests/test_but_ppg_adapter.py. Writing these flat at <root> matched
        # the adapter's old (wrong) assumption and hid a real path bug.
        rec_dir = root / rec_id
        rec_dir.mkdir()
        wfdb.wrsamp(f"{rec_id}_PPG", fs=30, units=["NU"], sig_name=["PPG"],
                    p_signal=ppg[:, None], write_dir=str(rec_dir), fmt=["16"])
        wfdb.wrsamp(f"{rec_id}_ACC", fs=100, units=["g", "g", "g"],
                    sig_name=["ACC_X", "ACC_Y", "ACC_Z"], p_signal=acc,
                    write_dir=str(rec_dir), fmt=["16", "16", "16"])
        rows.append({"signal_id": rec_id, "quality": quality, "hr": 70.0})
    pd.DataFrame(rows).to_csv(root / "quality-hr-ann.csv", index=False)
    return root


def test_fit_motion_noise_amplitude_returns_one_value_per_severity(fake_but_ppg_with_accel, reference_ppg):
    amplitudes = fit_motion_noise_amplitude(fake_but_ppg_with_accel, reference_ppg)
    assert set(amplitudes.keys()) == set(DEGRADATION_SEVERITIES)
    for sev in DEGRADATION_SEVERITIES:
        assert amplitudes[sev] > 0


def test_fit_motion_noise_amplitude_increases_with_severity(fake_but_ppg_with_accel, reference_ppg):
    amplitudes = fit_motion_noise_amplitude(fake_but_ppg_with_accel, reference_ppg)
    ordered = [amplitudes[s] for s in sorted(DEGRADATION_SEVERITIES)]
    assert ordered == sorted(ordered)   # monotonically non-decreasing with severity


def test_solved_amplitudes_reproduce_the_real_noise_quantiles(fake_but_ppg_with_accel, reference_ppg):
    from trustbio.degradation.calibrate import _achieved_ratio, real_noise_quantiles
    targets, n_real = real_noise_quantiles(fake_but_ppg_with_accel)
    assert n_real == 12 and targets[0.1] <= targets[0.3] <= targets[0.6]
    amplitudes = fit_motion_noise_amplitude(fake_but_ppg_with_accel, reference_ppg)
    for sev in DEGRADATION_SEVERITIES:
        achieved = _achieved_ratio(reference_ppg, amplitudes[sev])
        # bisection hits the target unless it is outside the achievable range
        if 1e-3 < amplitudes[sev] < 20.0:
            assert abs(achieved - targets[sev]) < 0.02 * max(targets[sev], 0.05)


def test_fit_requires_reference_windows(fake_but_ppg_with_accel):
    with pytest.raises(ValueError):
        fit_motion_noise_amplitude(fake_but_ppg_with_accel, [])


def test_fit_writes_cache_and_details_files(fake_but_ppg_with_accel, reference_ppg, tmp_path, monkeypatch):
    cache_path = tmp_path / "noise_amplitude_cache.json"
    monkeypatch.setattr(
        "trustbio.degradation.calibrate.NOISE_AMPLITUDE_CACHE_PATH", cache_path,
    )
    fit_motion_noise_amplitude(fake_but_ppg_with_accel, reference_ppg, cache=True)
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text())
    assert set(float(k) for k in cached.keys()) == set(DEGRADATION_SEVERITIES)
    details = json.loads((tmp_path / "noise_amplitude_calibration.json").read_text())
    assert details["n_real_recordings"] == 12 and details["n_reference_windows"] == 6
    assert set(details["targets"]) == {"0.1", "0.3", "0.6"}


def test_fit_never_writes_the_production_cache(fake_but_ppg_with_accel, reference_ppg, tmp_path):
    from trustbio.degradation import calibrate
    fit_motion_noise_amplitude(fake_but_ppg_with_accel, reference_ppg)          # default cache=True
    assert calibrate.NOISE_AMPLITUDE_CACHE_PATH.parent == tmp_path, (
        "the autouse fixture must redirect the cache path; without it this call "
        "overwrote features_cache/noise_amplitude_cache.json with fixture numbers"
    )
