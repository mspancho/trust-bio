import numpy as np
import pytest

from trustbio.config import DEGRADATION_SEVERITIES, DEGRADATION_KINDS
from trustbio.degradation.inject import (
    inject_motion_artifact, inject_lead_off, inject_missing_ppg, apply_degradation,
)


@pytest.fixture
def clean_signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(2500).astype(np.float32)   # 10s @ 250Hz


def test_inject_motion_artifact_changes_signal_and_scales_with_severity(clean_signal):
    rng = np.random.default_rng(1)
    amplitudes = {0.1: 0.05, 0.3: 0.2, 0.6: 0.5}
    low = inject_motion_artifact(clean_signal.copy(), fs=250, severity=0.1,
                                  rng=np.random.default_rng(1), noise_amplitudes=amplitudes)
    high = inject_motion_artifact(clean_signal.copy(), fs=250, severity=0.6,
                                   rng=np.random.default_rng(1), noise_amplitudes=amplitudes)
    assert not np.allclose(low, clean_signal)
    assert not np.allclose(high, clean_signal)
    # higher severity injects noise into a larger contiguous span
    low_diff_span = np.sum(~np.isclose(low, clean_signal))
    high_diff_span = np.sum(~np.isclose(high, clean_signal))
    assert high_diff_span > low_diff_span


def test_inject_lead_off_zeroes_a_contiguous_span(clean_signal):
    rng = np.random.default_rng(2)
    out = inject_lead_off(clean_signal.copy(), severity=0.3, rng=rng)
    n_zero = np.sum(out == 0.0)
    expected = int(0.3 * len(clean_signal))
    assert abs(n_zero - expected) <= 1   # rounding tolerance
    # the zeroed region must be contiguous
    zero_idx = np.where(out == 0.0)[0]
    assert zero_idx.max() - zero_idx.min() + 1 == len(zero_idx)


def test_inject_missing_ppg_returns_none(clean_signal):
    assert inject_missing_ppg(clean_signal) is None
    assert inject_missing_ppg(None) is None


def test_apply_degradation_dispatches_by_kind(clean_signal):
    rng = np.random.default_rng(3)
    ecg, ppg = clean_signal.copy(), clean_signal.copy()
    noise_amplitudes = {s: 0.3 for s in DEGRADATION_SEVERITIES}

    ecg_out, ppg_out = apply_degradation(ecg, ppg, fs=250, kind="motion_artifact",
                                          severity=0.3, rng=rng,
                                          noise_amplitudes=noise_amplitudes)
    assert ppg_out is not None

    ecg_out2, ppg_out2 = apply_degradation(ecg, ppg, fs=250, kind="lead_off",
                                            severity=0.3, rng=rng)
    assert np.sum(ecg_out2 == 0.0) > 0

    ecg_out3, ppg_out3 = apply_degradation(ecg, ppg, fs=250, kind="missing_ppg",
                                            severity=0.3, rng=rng)
    assert ppg_out3 is None
    assert ecg_out3 is not None


def test_apply_degradation_rejects_unknown_kind(clean_signal):
    with pytest.raises(KeyError):
        apply_degradation(clean_signal, clean_signal, fs=250, kind="not_a_kind",
                           severity=0.3, rng=np.random.default_rng(0))


def test_all_degradation_kinds_are_dispatchable(clean_signal):
    rng = np.random.default_rng(4)
    noise_amplitudes = {s: 0.3 for s in DEGRADATION_SEVERITIES}
    for kind in DEGRADATION_KINDS:
        for sev in DEGRADATION_SEVERITIES:
            apply_degradation(clean_signal.copy(), clean_signal.copy(), fs=250,
                               kind=kind, severity=sev, rng=rng,
                               noise_amplitudes=noise_amplitudes)
