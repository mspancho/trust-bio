import json

import numpy as np
import pandas as pd
import pytest
import wfdb

from trustbio.config import DEGRADATION_SEVERITIES
from trustbio.degradation.calibrate import fit_motion_noise_amplitude


@pytest.fixture
def fake_but_ppg_with_accel(tmp_path):
    root = tmp_path / "but-ppg"
    root.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    # Recordings with progressively higher accel magnitude and lower quality,
    # so the fit has real signal to recover (higher motion -> more noise needed
    # to reproduce the observed quality degradation).
    for i, rec_id in enumerate(["112001", "112002", "112003", "112004"]):
        accel_scale = 1.0 + i * 2.0
        quality = 1 if i < 2 else 0   # first two "good", last two "poor"
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


def test_fit_motion_noise_amplitude_returns_one_value_per_severity(fake_but_ppg_with_accel):
    amplitudes = fit_motion_noise_amplitude(fake_but_ppg_with_accel)
    assert set(amplitudes.keys()) == set(DEGRADATION_SEVERITIES)
    for sev in DEGRADATION_SEVERITIES:
        assert amplitudes[sev] > 0


def test_fit_motion_noise_amplitude_increases_with_severity(fake_but_ppg_with_accel):
    amplitudes = fit_motion_noise_amplitude(fake_but_ppg_with_accel)
    ordered = [amplitudes[s] for s in sorted(DEGRADATION_SEVERITIES)]
    assert ordered == sorted(ordered)   # monotonically non-decreasing with severity


def test_fit_writes_cache_file(fake_but_ppg_with_accel, tmp_path, monkeypatch):
    cache_path = tmp_path / "noise_amplitude_cache.json"
    monkeypatch.setattr(
        "trustbio.degradation.calibrate.NOISE_AMPLITUDE_CACHE_PATH", cache_path,
    )
    fit_motion_noise_amplitude(fake_but_ppg_with_accel, cache=True)
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text())
    assert set(float(k) for k in cached.keys()) == set(DEGRADATION_SEVERITIES)
