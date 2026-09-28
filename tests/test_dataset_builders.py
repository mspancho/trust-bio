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


def test_mimic_ext_handle_reads_metadata_subset_from_cohort_cache(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    meta = pd.DataFrame({
        "signal_file_name": ["r1", "r2", "r3"], "subject_id": [1, 2, 3],
        "folder_path": ["p00/p1/r1", "p00/p2/r2", "p00/p3/r3"],
        "vector_10s_pleth_sqi": ["[1, 1, 1]"] * 3, "vector_10s_ecg_sqi": ["[1, 1, 1]"] * 3,
        "median_30s_hr": [70.0, 71.0, 72.0], "event_rhythm": ["SR", "AF", "SR"],
    })
    meta.to_csv(cache / "mimic_ext_ppg_metadata.csv", index=False)
    args = argparse.Namespace(mimic_ext_ppg_root=tmp_path / "no-such-root", cohort_cache=cache)
    handle = build_dataset_handle("mimic_ext_ppg", args, with_labels=True)
    all_ids = sorted(v for df in handle.splits.values() for v in df["visit_id"])
    assert all_ids == ["r1", "r2", "r3"]
    assert handle.label_table[next(iter(handle.splits))].columns.tolist() == ["hr_regression", "rhythm_cls"]
