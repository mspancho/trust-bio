"""Evaluation metrics: Pearson correlation (regression) and AUROC (classification)."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    r, _ = pearsonr(y_true[mask], y_pred[mask])
    return float(r)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    yt = y_true[mask]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, y_score[mask]))


def metric_for_kind(kind: str):
    return pearson if kind == "regression" else auroc
