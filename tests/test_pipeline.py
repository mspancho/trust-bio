import numpy as np
import pandas as pd
import pytest

from trustbio.config import ALL_TASKS, MODALITIES
from trustbio.pipeline import (
    DatasetHandle, extract_features_for_model, assemble_matrices,
    has_cached_features, run_evaluation,
)
from trustbio.store import FeatureStore


def _toy_dataset_handle(n_visits=20, seed=0):
    rng = np.random.default_rng(seed)
    visit_ids = [f"v{i}" for i in range(n_visits)]
    splits = {
        "train": pd.DataFrame({"visit_id": visit_ids[:12]}),
        "val": pd.DataFrame({"visit_id": visit_ids[12:16]}),
        "test": pd.DataFrame({"visit_id": visit_ids[16:]}),
    }

    def load_signal(visit_id, modality):
        return rng.standard_normal(2500).astype(np.float32), 250   # 10s @ 250Hz

    label_table = {}
    for split_name, df in splits.items():
        idx = pd.Index(df["visit_id"], name="visit_id")
        y = pd.DataFrame(index=idx)
        for task in ALL_TASKS:
            if task.kind == "regression":
                y[task.name] = rng.normal(70, 10, len(idx))
            else:
                y[task.name] = rng.integers(0, 2, len(idx)).astype(float)
        label_table[split_name] = y

    return DatasetHandle(
        name="toy", cohort=None, splits=splits, load_signal=load_signal,
        label_table=label_table,
    )


def test_extract_features_for_model_uses_fallback_and_caches(tmp_path):
    dataset = _toy_dataset_handle()
    store = FeatureStore(tmp_path / "features_cache")
    ran = extract_features_for_model(
        "moment-base", dataset, store, duration_sec=10, device="cpu",
        allow_fallback=True, force_fallback=True,
    )
    assert ran is True
    for modality in MODALITIES:
        assert store.exists("moment-base", modality, 10, "train")
        assert store.exists("moment-base", modality, 10, "test")


def test_has_cached_features_true_after_extraction(tmp_path):
    dataset = _toy_dataset_handle()
    store = FeatureStore(tmp_path / "features_cache")
    extract_features_for_model(
        "moment-base", dataset, store, duration_sec=10, device="cpu",
        allow_fallback=True, force_fallback=True,
    )
    assert has_cached_features("moment-base", store, duration_sec=10)


def test_has_cached_features_false_when_never_extracted(tmp_path):
    store = FeatureStore(tmp_path / "features_cache")
    assert not has_cached_features("moment-base", store, duration_sec=10)


def test_run_evaluation_returns_records_per_modality(tmp_path):
    dataset = _toy_dataset_handle()
    store = FeatureStore(tmp_path / "features_cache")
    extract_features_for_model(
        "moment-base", dataset, store, duration_sec=10, device="cpu",
        allow_fallback=True, force_fallback=True,
    )
    out = run_evaluation(
        "moment-base", dataset, store, eval_split="test", duration_sec=10,
    )
    assert set(out.keys()) == set(MODALITIES)
    for modality, records in out.items():
        assert len(records) > 0


def test_run_evaluation_returns_none_for_unavailable_model(tmp_path):
    dataset = _toy_dataset_handle()
    store = FeatureStore(tmp_path / "features_cache")
    out = run_evaluation("csfm-base", dataset, store, eval_split="test", duration_sec=10)
    assert out is None
