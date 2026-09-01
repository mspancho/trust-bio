"""Cross-institution transportability evaluation: train/select hyperparameters
on one PulseDB source institution, score zero-shot on the other, in both
directions. This is the paper draft's central Results subsection ("Domain-FM
superiority and fusion benefit transport to an independent, cross-institution
cohort") — the source/target split IS PulseDB's native MIMIC/Vital partition,
not an arbitrary train/test split within one institution.

Feature caching convention: each institution's cached vectors live in their
OWN FeatureStore root (extract_features.py writes to <store>/<dataset>/, e.g.
<store>/pulsedb_mimic/ and <store>/pulsedb_vital/), and this module takes a
source store and a target store explicitly. An earlier design multiplexed both
institutions into one store via `vital_`-prefixed split names, but the tagging
was never wired into extraction, so both institutions wrote the SAME untagged
paths -- concurrent pilot array tasks interleaved and tore each other's npz
files. Separate roots make that collision impossible by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    HP_SELECTION_CLS_TASK, HP_SELECTION_REG_TASK, LOGREG_CS, RIDGE_ALPHAS,
    TASKS_BY_NAME, tasks_for_dataset,
)
from ..store import FeatureStore
from .metrics import auroc, pearson
from .probe import _fit_one, _predict


def _select_hp_on_source_val(
    kind: str, X_train, y_train_col, X_val, y_val_col, rng: np.random.Generator,
):
    grid = RIDGE_ALPHAS if kind == "regression" else LOGREG_CS
    metric = pearson if kind == "regression" else auroc
    fin_tr = np.isfinite(y_train_col)
    Xtr, ytr = X_train[fin_tr], y_train_col[fin_tr]
    if len(np.unique(ytr)) < 2 and kind == "classification":
        return grid[len(grid) // 2]
    best_val, best_reg = -np.inf, grid[len(grid) // 2]
    for reg in grid:
        model = _fit_one(kind, Xtr, ytr, reg)
        score = metric(y_val_col, _predict(kind, model, X_val))
        if np.isfinite(score) and score > best_val:
            best_val, best_reg = score, reg
    return best_reg


def run_cross_institution_eval(
    model_name: str,
    modality: str,
    source_store: FeatureStore,
    target_store: FeatureStore,
    source_labels: dict[str, pd.DataFrame],
    target_labels: dict[str, pd.DataFrame],
    duration_sec: int,
    seed: int = 0,
) -> list[dict]:
    """Fit on source's train split, select hyperparameters on source's val
    split, score zero-shot on target's full (train+val+test) pooled label set.

    `source_store`/`target_store` are the two institutions' per-dataset
    FeatureStore roots (see module docstring).
    """
    src_train_ids, X_src_train = source_store.load(model_name, modality, duration_sec, "train")
    src_val_ids, X_src_val = source_store.load(model_name, modality, duration_sec, "val")

    tgt_frames = []
    for split in ("train", "val", "test"):
        ids, X = target_store.load(model_name, modality, duration_sec, split)
        y = target_labels[split].reindex(pd.Index(ids, name="visit_id"))
        tgt_frames.append((ids, X, y))
    tgt_ids = np.concatenate([f[0] for f in tgt_frames])
    X_tgt = np.concatenate([f[1] for f in tgt_frames], axis=0)
    y_tgt = pd.concat([f[2] for f in tgt_frames])

    y_src_train = source_labels["train"].reindex(pd.Index(src_train_ids, name="visit_id"))
    y_src_val = source_labels["val"].reindex(pd.Index(src_val_ids, name="visit_id"))

    rng = np.random.default_rng(seed)
    alpha = _select_hp_on_source_val(
        "regression", X_src_train, y_src_train[HP_SELECTION_REG_TASK].to_numpy(float),
        X_src_val, y_src_val[HP_SELECTION_REG_TASK].to_numpy(float), rng,
    )
    has_cls_task = HP_SELECTION_CLS_TASK in y_src_train.columns
    c_value = (
        _select_hp_on_source_val(
            "classification", X_src_train, y_src_train[HP_SELECTION_CLS_TASK].to_numpy(float),
            X_src_val, y_src_val[HP_SELECTION_CLS_TASK].to_numpy(float), rng,
        ) if has_cls_task else LOGREG_CS[len(LOGREG_CS) // 2]
    )

    records = []
    for task_name in tasks_for_dataset("pulsedb"):
        task = TASKS_BY_NAME[task_name.name] if hasattr(task_name, "name") else TASKS_BY_NAME[task_name]
        if task.name not in y_src_train.columns:
            continue
        reg_value = alpha if task.kind == "regression" else c_value
        metric = pearson if task.kind == "regression" else auroc

        ytr = y_src_train[task.name].to_numpy(float)
        fin = np.isfinite(ytr)
        if fin.sum() < 5:
            score = float("nan")
        else:
            model = _fit_one(task.kind, X_src_train[fin], ytr[fin], reg_value)
            y_eval = y_tgt[task.name].to_numpy(float)
            score = metric(y_eval, _predict(task.kind, model, X_tgt))

        records.append(dict(task=task.name, kind=task.kind, score=score))
    return records


def both_directions(
    model_name: str,
    modality: str,
    mimic_store: FeatureStore,
    vital_store: FeatureStore,
    mimic_labels: dict[str, pd.DataFrame],
    vital_labels: dict[str, pd.DataFrame],
    duration_sec: int,
    seed: int = 0,
) -> list[dict]:
    """Run the cross-institution eval in both directions, tagging `direction`."""
    mimic_to_vital = run_cross_institution_eval(
        model_name, modality, mimic_store, vital_store,
        source_labels=mimic_labels, target_labels=vital_labels,
        duration_sec=duration_sec, seed=seed,
    )
    for r in mimic_to_vital:
        r["direction"] = "mimic_to_vital"

    vital_to_mimic = run_cross_institution_eval(
        model_name, modality, vital_store, mimic_store,
        source_labels=vital_labels, target_labels=mimic_labels,
        duration_sec=duration_sec, seed=seed,
    )
    for r in vital_to_mimic:
        r["direction"] = "vital_to_mimic"

    return mimic_to_vital + vital_to_mimic
