"""Linear-probing evaluation protocol.

Methods "Downstream Prediction Models" + "Evaluation Metrics & Reporting":

  * Ridge regression for regression tasks; logistic regression for
    classification. scikit-learn defaults otherwise.
  * Feature vectors standardised using train-split mean/std; the same transform
    is applied to val and test.
  * A *single* regularisation hyperparameter per FM and task type, selected on
    val (ridge alpha via age regression; logistic C via sex classification),
    then frozen across all tasks. Selection is done separately per modality and
    per signal duration.
  * Data-efficiency sweep: train on 10/25/50/100% of train visits, sampling
    with replacement, 5 repeats. Report mean +/- std over repeats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..config import (
    HP_SELECTION_CLS_TASK,
    HP_SELECTION_REG_TASK,
    LOGREG_CS,
    N_RESAMPLE_REPEATS,
    RIDGE_ALPHAS,
    TASKS_BY_NAME,
    TRAIN_FRACTIONS,
)
from .metrics import auroc, pearson


@dataclass
class FeatureMatrices:
    """Standardised feature matrices and label tables for one modality."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: "pd.DataFrame"   # noqa: F821 (typing only)
    y_val: "pd.DataFrame"     # noqa: F821
    y_test: "pd.DataFrame"    # noqa: F821


def standardize(X_train, X_val, X_test):
    """Fit StandardScaler on train, apply to all splits."""
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def _fit_one(kind: str, X, y, reg_value: float):
    mask = np.isfinite(y)
    if kind == "regression":
        model = Ridge(alpha=reg_value)
    else:
        model = LogisticRegression(C=reg_value, max_iter=1000)
    model.fit(X[mask], y[mask])
    return model


def _predict(kind: str, model, X):
    if kind == "regression":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def select_hyperparameter(
    kind: str,
    X_train, y_train,
    X_val, y_val,
    rng: np.random.Generator,
    train_idx: np.ndarray,
):
    """Pick the regularisation value maximising the val metric on the
    representative selection task, using the resampled train indices."""
    grid = RIDGE_ALPHAS if kind == "regression" else LOGREG_CS
    metric = pearson if kind == "regression" else auroc

    Xtr = X_train[train_idx]
    ytr = y_train[train_idx]
    fin = np.isfinite(ytr)
    Xtr, ytr = Xtr[fin], ytr[fin]
    if len(np.unique(ytr)) < 2 and kind == "classification":
        return grid[len(grid) // 2]

    best_val, best_reg = -np.inf, grid[len(grid) // 2]
    for reg in grid:
        model = _fit_one(kind, Xtr, ytr, reg)
        score = metric(y_val, _predict(kind, model, X_val))
        if np.isfinite(score) and score > best_val:
            best_val, best_reg = score, reg
    return best_reg


def fit_predict_task(
    task_name: str,
    X_train, y_train_col,
    X_eval, y_eval_col,
    reg_value: float,
    train_idx: np.ndarray,
):
    """Fit a single task's linear model on resampled train, score on eval."""
    task = TASKS_BY_NAME[task_name]
    metric = pearson if task.kind == "regression" else auroc

    Xtr = X_train[train_idx]
    ytr = y_train_col[train_idx]
    fin = np.isfinite(ytr)
    if fin.sum() < 5 or (task.kind == "classification" and len(np.unique(ytr[fin])) < 2):
        return float("nan")
    model = _fit_one(task.kind, Xtr[fin], ytr[fin], reg_value)
    return metric(y_eval_col, _predict(task.kind, model, X_eval))


def _resample_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    size = max(1, int(round(n * frac)))
    return rng.integers(0, n, size=size)   # sampling with replacement


def evaluate_model(
    feats: FeatureMatrices,
    eval_split: str = "test",
    train_fractions=TRAIN_FRACTIONS,
    n_repeats: int = N_RESAMPLE_REPEATS,
    seed: int = 0,
):
    """Run the full protocol for one model + modality.

    Returns a long-form list of dicts:
        {task, kind, category, train_frac, repeat, score}
    one per (task, train fraction, repeat).
    """
    X_train = feats.X_train
    X_eval = feats.X_val if eval_split == "val" else feats.X_test
    y_train = feats.y_train
    y_eval = feats.y_val if eval_split == "val" else feats.y_test

    n_train = len(X_train)
    records = []

    for frac in train_fractions:
        for repeat in range(n_repeats):
            rng = np.random.default_rng((seed, int(frac * 100), repeat).__hash__() & 0xFFFFFFFF)
            train_idx = _resample_indices(n_train, frac, rng)

            # Per task-type regularisation, selected on the representative task.
            # Not every dataset's label table has every task's column (e.g.
            # but_ppg has only hr_regression, mimic_ext_ppg lacks sbp/dbp) --
            # only select a hyperparameter for a task type that's actually
            # present, mirroring eval/transport.py's same guard.
            has_reg_task = HP_SELECTION_REG_TASK in y_train.columns
            has_cls_task = HP_SELECTION_CLS_TASK in y_train.columns

            alpha = (
                select_hyperparameter(
                    "regression",
                    X_train, y_train[HP_SELECTION_REG_TASK].to_numpy(float),
                    feats.X_val, feats.y_val[HP_SELECTION_REG_TASK].to_numpy(float),
                    rng, train_idx,
                ) if has_reg_task else None
            )
            c_value = (
                select_hyperparameter(
                    "classification",
                    X_train, y_train[HP_SELECTION_CLS_TASK].to_numpy(float),
                    feats.X_val, feats.y_val[HP_SELECTION_CLS_TASK].to_numpy(float),
                    rng, train_idx,
                ) if has_cls_task else None
            )

            for task_name, task in TASKS_BY_NAME.items():
                if task_name not in y_train.columns:
                    continue
                reg_value = alpha if task.kind == "regression" else c_value
                score = fit_predict_task(
                    task_name,
                    X_train, y_train[task_name].to_numpy(float),
                    X_eval, y_eval[task_name].to_numpy(float),
                    reg_value, train_idx,
                )
                records.append(
                    dict(
                        task=task_name,
                        kind=task.kind,
                        category=task.category,
                        train_frac=frac,
                        repeat=repeat,
                        score=score,
                    )
                )
    return records
