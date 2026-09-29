import numpy as np
import pytest

from trustbio.taxonomy.sqi import (
    calibrate_ppg_ref, combined_sqi_trace, ecg_sqi_trace, mean_ppg_oob_ratio,
    ppg_oob_ratio_trace, ppg_sqi_trace,
)

FS = 125


def _ppg(fs=FS, seconds=10, seed=0):
    """Smooth pulse-like waveform: fundamental + harmonic inside the pulse band."""
    t = np.arange(seconds * fs) / fs
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
            + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def _ecg(fs=FS, seconds=10, seed=0):
    """Spiky QRS-like train (one sharp spike every 0.8 s) on a quiet baseline."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(seconds * fs)
    x[:: int(0.8 * fs)] = 1.0
    return x.astype(np.float32)


def test_traces_have_one_value_per_second_at_any_rate():
    for fs in (30, 125, 1000):
        assert len(ecg_sqi_trace(_ecg(fs), fs)) == 10
        assert len(ppg_sqi_trace(_ppg(fs), fs, ref=0.5)) == 10


def test_clean_ppg_scores_high_and_the_ratio_is_rate_consistent():
    r30, r125 = mean_ppg_oob_ratio(_ppg(30), 30), mean_ppg_oob_ratio(_ppg(125), 125)
    assert r30 < 0.15 and r125 < 0.15 and abs(r30 - r125) < 0.1
    assert (ppg_sqi_trace(_ppg(), FS, ref=0.5) > 0.7).all()


def test_spiky_ecg_is_not_penalised():
    assert ecg_sqi_trace(_ecg(), FS).tolist() == [1.0] * 10


def test_flat_ecg_span_scores_zero_exactly_where_it_is_flat():
    x = _ecg(); x[3 * FS:6 * FS] = 0.0                  # lead-off seconds 3,4,5
    s = ecg_sqi_trace(x, FS)
    assert s.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]


def test_noisy_ppg_span_drops_only_where_noise_is():
    x = _ppg(); rng = np.random.default_rng(1)
    x[2 * FS:5 * FS] += (0.8 * np.std(x) * rng.standard_normal(3 * FS)).astype(np.float32)
    s = ppg_sqi_trace(x, FS, ref=0.5)
    assert (s[2:5] < 0.5).all()
    assert (np.r_[s[:2], s[5:]] > 0.7).all()
    r = ppg_oob_ratio_trace(x, FS)
    assert r[2:5].min() > r[:2].max() and r[2:5].min() > r[5:].max()


def test_flat_ppg_span_scores_zero():
    x = _ppg(); x[7 * FS:9 * FS] = 0.0
    s = ppg_sqi_trace(x, FS, ref=0.5)
    assert s[7:9].tolist() == [0.0, 0.0] and (s[:7] > 0.7).all()


def test_combined_is_elementwise_min():
    a, b = np.array([1.0, 0.2, 0.9]), np.array([0.5, 0.8, 0.9, 0.1])
    assert combined_sqi_trace(a, b).tolist() == [0.5, 0.2, 0.9]


def test_calibrate_ppg_ref_is_the_quantile_with_floor_and_default():
    r = np.linspace(0.0, 1.0, 1001)
    assert abs(calibrate_ppg_ref(r, 0.95) - 0.95) < 1e-9
    assert calibrate_ppg_ref(np.array([]), 0.95) == 0.5
    assert calibrate_ppg_ref(np.zeros(10), 0.95) == 1e-3
    assert np.isfinite(calibrate_ppg_ref(np.array([0.1, np.nan, 0.3]), 0.5))


def test_mean_oob_ratio_increases_with_added_noise():
    x = _ppg()
    noisy = x + (0.5 * np.std(x) * np.random.default_rng(2).standard_normal(len(x))).astype(np.float32)
    assert mean_ppg_oob_ratio(noisy, FS) > mean_ppg_oob_ratio(x, FS)
