import numpy as np
import pandas as pd

from trustbio.eval.probe import FeatureMatrices, evaluate_model
from trustbio.eval.report import aggregate_categories, build_main_table, rank_models
from trustbio.config import ALL_TASKS, CATEGORIES


def _toy_feature_matrices(n=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)

    def labels(idx):
        y = pd.DataFrame(index=idx)
        y["hr_regression"] = rng.normal(75, 10, len(idx))
        y["sbp_regression"] = rng.normal(120, 15, len(idx))
        y["dbp_regression"] = rng.normal(80, 10, len(idx))
        y["rhythm_cls"] = rng.integers(0, 2, len(idx)).astype(float)
        return y

    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    idx_tr = np.arange(n_tr)
    idx_val = np.arange(n_tr, n_tr + n_val)
    idx_te = np.arange(n_tr + n_val, n)
    return FeatureMatrices(
        X_train=X[idx_tr], X_val=X[idx_val], X_test=X[idx_te],
        y_train=labels(idx_tr), y_val=labels(idx_val), y_test=labels(idx_te),
    )


def test_evaluate_model_returns_one_record_per_task_frac_repeat():
    feats = _toy_feature_matrices()
    records = evaluate_model(feats, eval_split="test", train_fractions=[0.5, 1.0], n_repeats=2)
    assert len(records) == len(ALL_TASKS) * 2 * 2
    assert {r["task"] for r in records} == {t.name for t in ALL_TASKS}


def test_aggregate_categories_and_main_table():
    feats = _toy_feature_matrices()
    records = evaluate_model(feats, eval_split="test", train_fractions=[1.0], n_repeats=2)
    agg = aggregate_categories(records)
    assert set(agg["category"]) == set(CATEGORIES)

    table = build_main_table({"model_a": records, "model_b": records})
    assert list(table.columns) == CATEGORIES
    assert set(table.index) == {"model_a", "model_b"}

    ranks = rank_models(table)
    assert set(ranks.index) == {"model_a", "model_b"}
