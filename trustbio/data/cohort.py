"""Shared subject-disjoint cohort/split utilities.

Generalizes the leakage-check and split logic that signal-mcmed-msp implements
once for MC-MED (dataset.py's _assert_no_patient_leakage /
chronological_split) so all three new dataset adapters (PulseDB,
MIMIC-III-Ext-PPG, BUT PPG) share one tested implementation instead of each
reimplementing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cohort:
    """A table of eligible visits/segments with a subject-disjoint split
    assignment. `visits` must have at least `visit_id`, `subject_id`, `split`."""
    visits: pd.DataFrame

    def split(self, name: str) -> pd.DataFrame:
        return self.visits[self.visits["split"] == name].reset_index(drop=True)

    @property
    def counts(self) -> dict[str, int]:
        return self.visits["split"].value_counts().to_dict()


def assert_no_subject_leakage(
    df: pd.DataFrame, subject_col: str = "subject_id", split_col: str = "split"
) -> None:
    """Raise ValueError if any subject appears in more than one split."""
    per_subject_splits = df.groupby(subject_col)[split_col].nunique()
    leaked = per_subject_splits[per_subject_splits > 1]
    if len(leaked):
        raise ValueError(
            f"{len(leaked)} subjects appear in multiple splits "
            f"(e.g. {list(leaked.index[:5])}); splits must be subject-disjoint."
        )


def chronological_or_random_split(
    visits: pd.DataFrame,
    subject_col: str,
    time_col: str | None = None,
    frac_train: float = 0.70,
    frac_val: float = 0.15,
    seed: int = 0,
) -> pd.Series:
    """Assign each row's subject wholesale to train/val/test.

    If `time_col` is given, subjects are ordered chronologically by their
    earliest row (matching MC-MED's chronological-split convention); otherwise
    subjects are shuffled with `seed` (used for PulseDB/BUT PPG/MIMIC-III-Ext-PPG,
    which have no meaningful visit ordering across sources).
    """
    if time_col is not None:
        subject_order = visits.groupby(subject_col)[time_col].min().sort_values().index
    else:
        subjects = visits[subject_col].unique()
        rng = np.random.default_rng(seed)
        subject_order = pd.Index(rng.permutation(subjects))

    n = len(subject_order)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    assignment = {}
    for i, sid in enumerate(subject_order):
        if i < n_train:
            assignment[sid] = "train"
        elif i < n_train + n_val:
            assignment[sid] = "val"
        else:
            assignment[sid] = "test"
    return visits[subject_col].map(assignment)
