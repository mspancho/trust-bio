"""End-to-end pipeline orchestration, generalized across datasets.

Unlike signal-mcmed-msp's pipeline.py (which hardcodes MC-MED's cohort/label
loading), this version operates on a `DatasetHandle` — a uniform wrapper any
of the three adapters (PulseDB, MIMIC-III-Ext-PPG, BUT PPG) produces — so
stage 3 (extraction) and stage 4 (evaluation) are dataset-agnostic. The
per-visit orchestration logic (preprocess -> segment -> encode -> pool -> late
fusion, caching, graceful model-unavailability skip) is unchanged from
signal-mcmed-msp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import FULL_DURATION_SEC, MODALITIES
from .features.aggregate import build_visit_features
from .features.registry import get_extractor
from .eval.probe import FeatureMatrices, evaluate_model, standardize
from .store import FeatureStore

SignalLoader = Callable[[str, str], "tuple[np.ndarray, int]"]


@dataclass
class DatasetHandle:
    """Uniform interface every dataset adapter is wrapped into."""
    name: str
    cohort: object          # the adapter's Cohort (unused by pipeline.py directly;
                             # kept for callers that need cohort.counts etc.)
    splits: dict[str, pd.DataFrame]   # {"train": df, "val": df, "test": df}, each with a "visit_id" column
    load_signal: SignalLoader
    label_table: dict[str, pd.DataFrame]   # {"train": labels_df, ...}, indexed by visit_id


class ModelUnavailable(RuntimeError):
    """Raised (and caught) so the pipeline can skip a model gracefully rather
    than crash — important for restricted-access models such as CSFM."""


def extract_features_for_model(
    model_name: str,
    dataset: DatasetHandle,
    store: FeatureStore,
    duration_sec: int = FULL_DURATION_SEC,
    device: str = "cpu",
    allow_fallback: bool = False,
    checkpoint: str | None = None,
    overwrite: bool = False,
    skip_if_unavailable: bool = True,
    force_fallback: bool = False,
) -> bool:
    """Stage 3: compute and cache visit-level ECG/PPG/fusion vectors for one
    (model, dataset) pair. See signal-mcmed-msp/signalmcmed/pipeline.py for the
    original single-dataset version this generalizes."""
    try:
        extractor = get_extractor(
            model_name, device=device, allow_fallback=allow_fallback,
            checkpoint=checkpoint, force_fallback=force_fallback,
        ).load()
    except Exception as e:
        msg = f"[skip] {model_name}: cannot load with real weights ({e})"
        if skip_if_unavailable and not allow_fallback:
            print(msg)
            return False
        raise ModelUnavailable(msg) from e

    for split, df in dataset.splits.items():
        if (
            not overwrite
            and all(store.exists(model_name, m, duration_sec, split) for m in MODALITIES)
        ):
            continue

        per_modality = {m: [] for m in MODALITIES}
        kept_ids = []
        for visit_id in df["visit_id"].astype(str):
            try:
                ecg_raw, ecg_fs = dataset.load_signal(visit_id, "ecg")
                ppg_raw, ppg_fs = dataset.load_signal(visit_id, "ppg")
                vecs = build_visit_features(
                    extractor, ecg_raw, ecg_fs, ppg_raw, ppg_fs, duration_sec
                )
            except Exception:
                continue
            kept_ids.append(visit_id)
            for m in MODALITIES:
                per_modality[m].append(vecs[m])

        dim = extractor.feature_dim
        for m in MODALITIES:
            store.save(
                model_name, m, duration_sec, split,
                kept_ids,
                np.stack(per_modality[m]) if per_modality[m] else np.empty((0, dim)),
            )
    return True


def assemble_matrices(
    model_name: str,
    modality: str,
    dataset: DatasetHandle,
    store: FeatureStore,
    duration_sec: int = FULL_DURATION_SEC,
) -> FeatureMatrices:
    """Load cached features for the three splits, align to labels, standardise."""
    Xs, ys = {}, {}
    for split in ("train", "val", "test"):
        ids, feats = store.load(model_name, modality, duration_sec, split)
        y = dataset.label_table[split].reindex(pd.Index(ids, name="visit_id"))
        Xs[split], ys[split] = feats, y

    Xtr, Xval, Xte = standardize(Xs["train"], Xs["val"], Xs["test"])
    return FeatureMatrices(
        X_train=Xtr, X_val=Xval, X_test=Xte,
        y_train=ys["train"], y_val=ys["val"], y_test=ys["test"],
    )


def has_cached_features(
    model_name: str,
    store: FeatureStore,
    duration_sec: int = FULL_DURATION_SEC,
    modalities: Iterable[str] = MODALITIES,
    min_visits: int = 1,
) -> bool:
    for modality in modalities:
        for split in ("train", "val", "test"):
            if not store.exists(model_name, modality, duration_sec, split):
                return False
            try:
                ids, feats = store.load(model_name, modality, duration_sec, split)
            except Exception:
                return False
            if len(ids) < min_visits or feats.size == 0:
                return False
    return True


def run_evaluation(
    model_name: str,
    dataset: DatasetHandle,
    store: FeatureStore,
    eval_split: str = "test",
    duration_sec: int = FULL_DURATION_SEC,
    modalities: Iterable[str] = MODALITIES,
    seed: int = 0,
) -> dict[str, list[dict]] | None:
    """Stage 4: probe + score for one model across modalities. Returns None if
    the model has no cached features (so callers can skip it rather than crash)."""
    if not has_cached_features(model_name, store, duration_sec, modalities):
        return None
    out = {}
    for modality in modalities:
        feats = assemble_matrices(model_name, modality, dataset, store, duration_sec)
        out[modality] = evaluate_model(feats, eval_split=eval_split, seed=seed)
    return out
