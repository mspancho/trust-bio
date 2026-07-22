import pandas as pd
import pytest

from trustbio.data.cohort import Cohort, assert_no_subject_leakage, chronological_or_random_split


def test_cohort_split_and_counts():
    df = pd.DataFrame({
        "visit_id": ["v1", "v2", "v3", "v4"],
        "subject_id": ["s1", "s1", "s2", "s3"],
        "split": ["train", "train", "val", "test"],
    })
    cohort = Cohort(visits=df)
    assert len(cohort.split("train")) == 2
    assert len(cohort.split("val")) == 1
    assert cohort.counts == {"train": 2, "val": 1, "test": 1}


def test_assert_no_subject_leakage_raises_on_leakage():
    df = pd.DataFrame({
        "subject_id": ["s1", "s1", "s2"],
        "split": ["train", "val", "test"],  # s1 appears in both train and val
    })
    with pytest.raises(ValueError):
        assert_no_subject_leakage(df)


def test_assert_no_subject_leakage_passes_when_disjoint():
    df = pd.DataFrame({
        "subject_id": ["s1", "s1", "s2", "s3"],
        "split": ["train", "train", "val", "test"],
    })
    assert_no_subject_leakage(df) is None


def test_chronological_or_random_split_is_subject_disjoint():
    df = pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(100)],
        "time": list(range(100)),
    })
    split = chronological_or_random_split(df, subject_col="subject_id", frac_train=0.7, frac_val=0.15, seed=0)
    df2 = df.assign(split=split)
    assert_no_subject_leakage(df2) is None
    counts = df2["split"].value_counts()
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15
