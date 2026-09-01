import numpy as np
import pandas as pd

from trustbio.store import FeatureStore
from trustbio.eval.transport import run_cross_institution_eval, both_directions


def _fake_stores_and_labels(tmp_path, seed_a=0, seed_b=1, n=40, dim=8):
    # One store PER institution, mirroring extract_features.py's
    # <store>/<dataset>/ scoping (a shared root let concurrent extraction
    # jobs overwrite each other's files).
    mimic_store = FeatureStore(tmp_path / "features_cache" / "pulsedb_mimic")
    vital_store = FeatureStore(tmp_path / "features_cache" / "pulsedb_vital")
    rng_a, rng_b = np.random.default_rng(seed_a), np.random.default_rng(seed_b)

    def make_split(rng, n_split, prefix):
        ids = [f"{prefix}{i}" for i in range(n_split)]
        X = rng.standard_normal((n_split, dim)).astype(np.float32)
        y = pd.DataFrame(index=pd.Index(ids, name="visit_id"))
        y["hr_regression"] = rng.normal(70, 10, n_split)
        y["sbp_regression"] = rng.normal(120, 15, n_split)
        y["dbp_regression"] = rng.normal(80, 10, n_split)
        return ids, X, y

    mimic_labels, vital_labels = {}, {}
    for split, n_split in [("train", 24), ("val", 8), ("test", 8)]:
        ids, X, y = make_split(rng_a, n_split, "mimic_")
        mimic_store.save("moment-base", "ecg", 10, split, ids, X)
        mimic_labels[split] = y

        ids_b, X_b, y_b = make_split(rng_b, n_split, "vital_")
        vital_store.save("moment-base", "ecg", 10, split, ids_b, X_b)
        vital_labels[split] = y_b

    return mimic_store, vital_store, mimic_labels, vital_labels


def test_run_cross_institution_eval_produces_bp_and_hr_records(tmp_path):
    mimic_store, vital_store, mimic_labels, vital_labels = _fake_stores_and_labels(tmp_path)
    records = run_cross_institution_eval(
        "moment-base", "ecg", mimic_store, vital_store,
        source_labels=mimic_labels, target_labels=vital_labels,
        duration_sec=10, seed=0,
    )
    tasks_seen = {r["task"] for r in records}
    assert tasks_seen == {"hr_regression", "sbp_regression", "dbp_regression"}
    for r in records:
        assert np.isfinite(r["score"]) or np.isnan(r["score"])


def test_both_directions_tags_direction_correctly(tmp_path):
    mimic_store, vital_store, mimic_labels, vital_labels = _fake_stores_and_labels(tmp_path)
    records = both_directions(
        "moment-base", "ecg", mimic_store, vital_store,
        mimic_labels, vital_labels, duration_sec=10, seed=0,
    )
    directions = {r["direction"] for r in records}
    assert directions == {"mimic_to_vital", "vital_to_mimic"}
    n_tasks = 3
    assert len(records) == n_tasks * 2
