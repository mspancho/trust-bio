import argparse

import pandas as pd
import pytest

from scripts._dataset_builders import _empty_labels, build_dataset_handle


def _fake_cohort_csv(cache_dir, source="mimic"):
    df = pd.DataFrame({
        "visit_id": ["p000001_w0", "p000001_w1", "p000002_w0"],
        "subject_id": ["p000001", "p000001", "p000002"],
        "source": [source] * 3,
        "split": ["train", "train", "test"],
    })
    df.to_csv(cache_dir / f"pulsedb_{source}_cohort.csv", index=False)
    return df


def test_empty_labels_keep_visit_index_and_have_no_columns():
    splits = {"train": pd.DataFrame({"visit_id": ["a", "b"]}),
              "test": pd.DataFrame({"visit_id": ["c"]})}
    labels = _empty_labels(splits)
    assert labels["train"].index.tolist() == ["a", "b"]
    assert labels["train"].shape == (2, 0)
    assert labels["test"].index.name == "visit_id"
    with pytest.raises(KeyError):
        labels["train"]["hr_regression"]


def test_build_dataset_handle_without_labels_never_opens_subject_files(tmp_path):
    _fake_cohort_csv(tmp_path)
    args = argparse.Namespace(pulsedb_root=tmp_path / "no-such-root", cohort_cache=tmp_path)
    handle = build_dataset_handle("pulsedb_mimic", args, with_labels=False)
    assert handle.name == "pulsedb_mimic"
    assert handle.splits["train"]["visit_id"].tolist() == ["p000001_w0", "p000001_w1"]
    assert handle.splits["test"]["visit_id"].tolist() == ["p000002_w0"]
    assert handle.label_table["train"].shape == (2, 0)
    # Labels are derived from each window's ECG, so asking for them with no
    # subject files on disk must fail -- which proves the flag is what skipped it.
    with pytest.raises(Exception):
        build_dataset_handle("pulsedb_mimic", args, with_labels=True)
