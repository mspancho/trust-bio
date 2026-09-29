import numpy as np
import pandas as pd
import pytest

from scripts.build_fault_features import (
    degraded_pair, known_condition, load_condition_features, raw_signals, window_rows,
)
from trustbio.degradation.inject import make_degraded_loader
from trustbio.pipeline import DatasetHandle, extract_features_for_model
from trustbio.store import FeatureStore
from trustbio.taxonomy.disagreement import fit_hr_probe

AMPS = {0.1: 0.2, 0.3: 0.3, 0.6: 0.4}
FS = 125


def _signal(visit_id, modality):
    rng = np.random.default_rng(abs(hash((visit_id, modality))) % (2**32))
    t = np.arange(10 * FS) / FS
    return (np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32), FS


def _handle(n=12):
    ids = [f"p{i:02d}_w0" for i in range(n)]
    visits = pd.DataFrame({"visit_id": ids, "subject_id": [v.split("_")[0] for v in ids],
                           "split": ["train"] * 8 + ["val"] * 2 + ["test"] * 2})
    splits = {s: visits[visits.split == s][["visit_id"]].reset_index(drop=True) for s in ("train", "val", "test")}
    cohort = type("C", (), {"visits": visits})()
    return DatasetHandle(name="toy", cohort=cohort, splits=splits, load_signal=_signal, label_table={})


def test_degraded_pair_reproduces_what_the_loader_fed_the_model():
    loader = make_degraded_loader(_signal, "lead_off", 0.3, seed=0, noise_amplitudes=AMPS)
    ecg_l, _ = loader("p03_w0", "ecg"); ppg_l, _ = loader("p03_w0", "ppg")
    ecg, efs = _signal("p03_w0", "ecg"); ppg, pfs = _signal("p03_w0", "ppg")
    ecg_d, ppg_d = degraded_pair(ecg, efs, ppg, pfs, "p03_w0", "lead_off", 0.3, seed=0, amps=AMPS)
    assert np.array_equal(ecg_d, ecg_l) and np.array_equal(ppg_d, ppg_l)
    same_ecg, same_ppg = degraded_pair(ecg, efs, ppg, pfs, "p03_w0", None, None, 0, AMPS)
    assert same_ecg is ecg and same_ppg is ppg


def test_known_condition_mapping():
    assert known_condition("pulsedb_mimic", "motion_artifact", {}) == "motion_artifact"
    assert known_condition("pulsedb_mimic", None, {}) == "clean"
    assert known_condition("pulsedb_vital", None, {}) == "structural"
    assert known_condition("mimic_ext_ppg", None, {"stratum": "ppg_poor"}) == "natural_ppg_poor"
    assert known_condition("but_ppg", None, {"quality": 0}) == "real_motion"
    assert known_condition("but_ppg", None, {"quality": 1}) == "consumer_clean"
    with pytest.raises(ValueError):
        known_condition("nope", None, {})


def test_window_rows_end_to_end_on_a_toy_store(tmp_path):
    handle = _handle()
    root = tmp_path / "store"
    for cond, kind, sev in [("clean", None, None), ("lead_off_0.3", "lead_off", 0.3)]:
        degraded = DatasetHandle(name=handle.name, cohort=handle.cohort, splits=handle.splits,
                                 load_signal=make_degraded_loader(_signal, kind, sev, seed=0, noise_amplitudes=AMPS),
                                 label_table={})
        for model in ("moment-base", "chronos-bolt-small"):
            extract_features_for_model(model, degraded, FeatureStore(root / cond / "toy"), duration_sec=10,
                                       device="cpu", allow_fallback=True, force_fallback=True)
    fa = load_condition_features(root, "clean", "toy", "moment-base", "ecg_ppg_mean", 10)
    fb = load_condition_features(root, "clean", "toy", "chronos-bolt-small", "ecg_ppg_mean", 10)
    assert len(fa) == 12 and fa.index.is_unique
    rng = np.random.default_rng(0); y = rng.normal(70, 10, len(fa))
    pa = fit_hr_probe(fa.to_numpy(), y, fa.to_numpy(), y); pb = fit_hr_probe(fb.to_numpy(), y, fb.to_numpy(), y)
    signals = raw_signals(handle, 10)
    native = handle.cohort.visits.set_index("visit_id").to_dict("index")
    clean = window_rows(signals, "clean", None, None, "pulsedb_mimic", fa, fb, pa, pb, 0.5, 0.5, native, 0, AMPS, 1.0)
    fa_d = load_condition_features(root, "lead_off_0.3", "toy", "moment-base", "ecg_ppg_mean", 10)
    fb_d = load_condition_features(root, "lead_off_0.3", "toy", "chronos-bolt-small", "ecg_ppg_mean", 10)
    lead = window_rows(signals, "lead_off_0.3", "lead_off", 0.3, "pulsedb_mimic", fa_d, fb_d, pa, pb, 0.5, 0.5, native, 0, AMPS, 1.0)
    assert len(clean) == 12 and len(lead) == 12
    c, l = pd.DataFrame(clean), pd.DataFrame(lead)
    assert {"known_condition", "subject_id", "ecg_sqi_value", "ppg_sqi_value", "sqi_drop_duration", "disagreement_raw"} <= set(c.columns)
    assert (c.known_condition == "clean").all() and (l.known_condition == "lead_off").all()
    assert l.ecg_sqi_value.mean() < c.ecg_sqi_value.mean()
    assert np.allclose(l.ppg_sqi_value, c.ppg_sqi_value)          # lead-off never touches PPG
    assert (l.sqi_drop_duration >= 2).all()                         # 30% of 10 s, per-second trace
