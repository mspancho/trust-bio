import numpy as np
import pytest

from trustbio.taxonomy.sqi import (
    calibrate_hf_ref, combined_sqi_trace, hf_noise_ratio, mean_hf_ratio, sqi_trace,
)

FS = 125


def _clean(fs=FS, seconds=10, seed=0):
    t = np.arange(seconds * fs) / fs
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
            + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_trace_has_one_value_per_second_at_any_rate():
    assert len(sqi_trace(_clean(125), 125, hf_ref=0.5)) == 10
    assert len(sqi_trace(_clean(30), 30, hf_ref=0.5)) == 10
    assert len(sqi_trace(_clean(1000), 1000, hf_ref=0.5)) == 10


def test_clean_signal_scores_high_everywhere():
    s = sqi_trace(_clean(), FS, hf_ref=0.5)
    assert s.min() > 0.7 and s.max() <= 1.0


def test_flat_span_scores_zero_exactly_where_it_is_flat():
    x = _clean(); x[3 * FS:6 * FS] = 0.0                # lead-off seconds 3,4,5
    s = sqi_trace(x, FS, hf_ref=0.5)
    assert s[3:6].tolist() == [0.0, 0.0, 0.0]
    assert (s[:3] > 0.7).all() and (s[6:] > 0.7).all()


def test_noisy_span_drops_only_where_noise_is():
    x = _clean(); rng = np.random.default_rng(1)
    x[2 * FS:5 * FS] += (0.8 * np.std(x) * rng.standard_normal(3 * FS)).astype(np.float32)
    s = sqi_trace(x, FS, hf_ref=0.5)
    assert (s[2:5] < 0.5).all()
    assert (np.r_[s[:2], s[5:]] > 0.7).all()


def test_combined_is_elementwise_min():
    a, b = np.array([1.0, 0.2, 0.9]), np.array([0.5, 0.8, 0.9, 0.1])
    assert combined_sqi_trace(a, b).tolist() == [0.5, 0.2, 0.9]


def test_calibrate_hf_ref_separates_native_good_from_poor():
    rng = np.random.default_rng(0)
    good, poor = rng.normal(0.10, 0.02, 300), rng.normal(0.50, 0.05, 300)
    ref = calibrate_hf_ref(good, poor)
    # SQI < 0.5 <=> hf_ratio > ref/2, so ref/2 must sit between the two populations
    assert 0.15 < ref / 2 < 0.45
    calls_poor = np.mean(poor > ref / 2); calls_good = np.mean(good > ref / 2)
    assert calls_poor > 0.95 and calls_good < 0.05


def test_mean_hf_ratio_is_higher_for_noisier_signal():
    x = _clean(); noisy = x + (0.5 * np.std(x) * np.random.default_rng(2).standard_normal(len(x))).astype(np.float32)
    assert mean_hf_ratio(noisy, FS) > mean_hf_ratio(x, FS)
    assert hf_noise_ratio(np.zeros(100), FS) == 0.0
