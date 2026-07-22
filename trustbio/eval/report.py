"""Aggregation, ranking, and table construction.

Methods "Evaluation Metrics & Reporting":
  * Group the 20 tasks into 5 categories: age, sex, ED disposition, lab
    regression (mean over 8), ICD-10 (mean over 9).
  * For each resampling repeat, aggregate across tasks within a category AND
    across train-visit percentages -> one value per category per repeat.
  * Report mean +/- std across the 5 repeats -> 5 aggregate values per
    (model, modality).
  * Rank models within each category, then average ranks across categories,
    separately per modality and jointly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CATEGORIES


def aggregate_categories(records: list[dict]) -> pd.DataFrame:
    """Collapse per-(task, frac, repeat) scores into per-(category, repeat),
    averaging over tasks within a category and over train-visit percentages.

    Returns a tidy DataFrame: columns [category, repeat, score].
    """
    df = pd.DataFrame(records)
    # Step 1: per (category, repeat, train_frac) mean over tasks.
    per_frac = (
        df.groupby(["category", "repeat", "train_frac"])["score"]
        .mean()
        .reset_index()
    )
    # Step 2: average over train_frac -> per (category, repeat).
    per_repeat = (
        per_frac.groupby(["category", "repeat"])["score"].mean().reset_index()
    )
    return per_repeat


def category_mean_std(records: list[dict]) -> pd.DataFrame:
    """Mean +/- std across repeats, one row per category."""
    per_repeat = aggregate_categories(records)
    summary = (
        per_repeat.groupby("category")["score"]
        .agg(["mean", "std"])
        .reindex(CATEGORIES)
        .reset_index()
    )
    return summary


def build_main_table(
    results_by_model: dict[str, list[dict]],
) -> pd.DataFrame:
    """Wide table: rows = models, columns = the five category means.

    `results_by_model` maps model name -> records (for one modality).
    """
    rows = {}
    for model, records in results_by_model.items():
        summ = category_mean_std(records).set_index("category")
        rows[model] = summ["mean"]
    table = pd.DataFrame(rows).T
    return table[CATEGORIES]


def build_main_table_std(results_by_model: dict[str, list[dict]]) -> pd.DataFrame:
    rows = {}
    for model, records in results_by_model.items():
        summ = category_mean_std(records).set_index("category")
        rows[model] = summ["std"]
    table = pd.DataFrame(rows).T
    return table[CATEGORIES]


def rank_models(main_table: pd.DataFrame) -> pd.Series:
    """Mean rank across the five categories (higher metric = better = rank 1)."""
    # Rank descending within each category column.
    ranks = main_table.rank(ascending=False, method="min")
    return ranks.mean(axis=1).sort_values()


def rank_models_by_modality(
    tables_by_modality: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Mean category-rank per model for each modality, plus the joint ranking.

    Joint ranking ranks across all (model, modality) rows together, as in
    Table main_results_..._test_rank.
    """
    out = {}
    for modality, table in tables_by_modality.items():
        out[modality] = rank_models(table)
    per_modality = pd.DataFrame(out)

    # Joint: stack all modalities, rank within category across every row.
    stacked = pd.concat(
        {m: t for m, t in tables_by_modality.items()}, names=["modality", "model"]
    )
    joint_ranks = stacked.rank(ascending=False, method="min").mean(axis=1)
    joint = joint_ranks.groupby("model").mean().rename("joint")
    per_modality["joint"] = joint
    return per_modality
