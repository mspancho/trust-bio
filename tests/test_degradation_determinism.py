"""A degraded waveform must be a pure function of (seed, visit, condition,
modality): the fault-taxonomy analysis recomputes SQI traces on the SAME
waveform the model saw, in a different process, possibly in a different order."""
import numpy as np
import pytest

from trustbio.degradation.inject import make_degraded_loader, visit_rng

AMPS = {0.1: 0.2, 0.3: 0.3, 0.6: 0.4}


def _loader():
    def load(visit_id, modality):
        rng = np.random.default_rng(abs(hash((visit_id, modality))) % (2**32))
        return rng.standard_normal(1250).astype(np.float32), 125
    return load


def test_visit_rng_is_stable_and_condition_specific():
    a = visit_rng(0, "p1_w3", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    b = visit_rng(0, "p1_w3", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    c = visit_rng(0, "p1_w4", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    d = visit_rng(0, "p1_w3", "lead_off", 0.6, "ecg").integers(0, 10**9, 5)
    e = visit_rng(1, "p1_w3", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist() and a.tolist() != d.tolist() and a.tolist() != e.tolist()


@pytest.mark.parametrize("kind,corrupted,untouched", [
    ("lead_off", "ecg", "ppg"), ("motion_artifact", "ppg", "ecg"),
])
def test_same_window_degrades_identically_regardless_of_call_order(kind, corrupted, untouched):
    first = make_degraded_loader(_loader(), kind, 0.3, seed=0, noise_amplitudes=AMPS)
    second = make_degraded_loader(_loader(), kind, 0.3, seed=0, noise_amplitudes=AMPS)
    x1, _ = first("v1", corrupted)
    first("v2", corrupted); first("v3", corrupted)          # advance any shared state
    second("v9", corrupted)
    x2, _ = second("v1", corrupted)
    assert np.array_equal(x1, x2)
    clean, _ = _loader()("v1", corrupted)
    assert not np.array_equal(x1, clean)
    passthrough, _ = first("v1", untouched)
    assert np.array_equal(passthrough, _loader()("v1", untouched)[0])


def test_different_windows_get_different_spans():
    loader = make_degraded_loader(_loader(), "lead_off", 0.3, seed=0, noise_amplitudes=AMPS)
    starts = []
    for v in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        x, _ = loader(v, "ecg")
        starts.append(int(np.argmax(x == 0.0)))
    assert len(set(starts)) > 1


def test_none_kind_is_identity_and_missing_ppg_raises():
    base = _loader()
    assert make_degraded_loader(base, None, None, seed=0) is base
    loader = make_degraded_loader(base, "missing_ppg", 0.3, seed=0)
    assert np.array_equal(loader("v1", "ecg")[0], base("v1", "ecg")[0])
    with pytest.raises(ValueError):
        loader("v1", "ppg")
