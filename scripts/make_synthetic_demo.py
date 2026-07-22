#!/usr/bin/env python
"""End-to-end smoke test: exercises cohort -> extract (fallback) -> transport
eval -> taxonomy, entirely with synthetic in-memory data, no real datasets or
model weights needed. Proves the pipeline connects end to end; the numbers are
meaningless as science."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from trustbio.config import ALL_TASKS, MODALITIES
from trustbio.pipeline import DatasetHandle, extract_features_for_model, run_evaluation
from trustbio.store import FeatureStore
from trustbio.taxonomy.cluster import cluster_fault_segments, name_clusters, confusion_against_known_conditions
from trustbio.taxonomy.features import extract_fault_features, features_to_matrix


def _synthetic_dataset_handle(n_visits: int, seed: int) -> DatasetHandle:
    rng = np.random.default_rng(seed)
    visit_ids = [f"v{i}" for i in range(n_visits)]
    n_tr, n_val = int(n_visits * 0.6), int(n_visits * 0.2)
    splits = {
        "train": pd.DataFrame({"visit_id": visit_ids[:n_tr]}),
        "val": pd.DataFrame({"visit_id": visit_ids[n_tr:n_tr + n_val]}),
        "test": pd.DataFrame({"visit_id": visit_ids[n_tr + n_val:]}),
    }

    def load_signal(visit_id, modality):
        return rng.standard_normal(2500).astype(np.float32), 250

    label_table = {}
    for split_name, df in splits.items():
        idx = pd.Index(df["visit_id"], name="visit_id")
        y = pd.DataFrame(index=idx)
        for task in ALL_TASKS:
            y[task.name] = (
                rng.normal(70, 10, len(idx)) if task.kind == "regression"
                else rng.integers(0, 2, len(idx)).astype(float)
            )
        label_table[split_name] = y

    return DatasetHandle(name="synthetic", cohort=None, splits=splits,
                          load_signal=load_signal, label_table=label_table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-visits", type=int, default=120)
    args = ap.parse_args()

    print("[demo] building synthetic dataset handle...")
    dataset = _synthetic_dataset_handle(args.n_visits, seed=0)

    print("[demo] extracting features with the deterministic fallback extractor...")
    store = FeatureStore("/tmp/trustbio_synthetic_demo_cache")
    extract_features_for_model(
        "moment-base", dataset, store, duration_sec=10, device="cpu",
        allow_fallback=True, force_fallback=True, overwrite=True,
    )

    print("[demo] running evaluation...")
    out = run_evaluation("moment-base", dataset, store, eval_split="test", duration_sec=10)
    for modality, records in out.items():
        scores = [r["score"] for r in records if np.isfinite(r["score"])]
        print(f"  {modality}: {len(records)} records, mean score {np.mean(scores):.3f}")

    print("[demo] running fault taxonomy on synthetic feature vectors...")
    rng = np.random.default_rng(1)
    feats = [
        extract_fault_features(
            sqi_trace=rng.uniform(0, 1, 10), accel_trace=None, fs=1,
            source_db="synthetic", model_a_pred=70.0, model_b_pred=70.0 + rng.normal(0, 5),
            disagreement_scale=5.0,
        )
        for _ in range(30)
    ]
    X, _names = features_to_matrix(feats)
    labels = cluster_fault_segments(X, seed=0)
    print(f"  cluster sizes: {np.bincount(labels)}")

    print("[demo] SUCCESS: pipeline connects end to end (numbers are meaningless).")


if __name__ == "__main__":
    main()
