"""HR probes for the `model_disagreement` fault feature.

Two linear probes -- one on the best domain FM's features, one on the best
time-series FM's -- are fit on CLEAN PulseDB-MIMIC training features with the
same ridge/alpha-selection protocol as the main evaluation. Their absolute
disagreement on a window, divided by its spread on clean in-distribution
windows, is the taxonomy's structural-shift signal: two models trained on the
same data disagreeing sharply on a window whose SQI looks fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..eval.probe import _fit_one, _predict, select_hyperparameter


@dataclass
class HRProbe:
    scaler: StandardScaler
    model: object
    alpha: float
    n_train: int


def fit_hr_probe(X_train, y_train, X_val, y_val, seed: int = 0,
                 max_train: int = 500_000, max_val: int = 100_000) -> HRProbe:
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train, dtype=float); y_val = np.asarray(y_val, dtype=float)
    tr = np.flatnonzero(np.isfinite(y_train)); va = np.flatnonzero(np.isfinite(y_val))
    if len(tr) > max_train:
        tr = rng.choice(tr, max_train, replace=False)
    if len(va) > max_val:
        va = rng.choice(va, max_val, replace=False)
    scaler = StandardScaler().fit(X_train[tr])
    Xtr, Xva = scaler.transform(X_train[tr]), scaler.transform(X_val[va])
    alpha = select_hyperparameter("regression", Xtr, y_train[tr], Xva, y_val[va], rng,
                                  train_idx=np.arange(len(tr)))
    model = _fit_one("regression", Xtr, y_train[tr], alpha)
    return HRProbe(scaler=scaler, model=model, alpha=float(alpha), n_train=int(len(tr)))


def predict_hr(probe: HRProbe, X) -> np.ndarray:
    return np.asarray(_predict("regression", probe.model, probe.scaler.transform(X)), dtype=float)


def disagreement_scale(pred_a, pred_b) -> float:
    return max(float(np.std(np.asarray(pred_a, float) - np.asarray(pred_b, float))), 1e-6)
