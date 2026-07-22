"""Tests for the PulseDB adapter (corrected schema: per-window visits).

PulseDB's real per-subject segment file (Segment_Files/PulseDB_MIMIC/p*.mat)
is a MATLAB v7.3/HDF5 struct `Subj_Wins` holding ECG_Raw/PPG_Raw/ABP_Raw as
independent fields shaped (n_windows, 1, 1250) -- NOT a combined `Signals`
array with a row-index convention. Each window has its own SegSBP/SegDBP and
IncludeFlag QC flag. See docs/pulsedb_structure_notes.md (Task 7, empirically
confirmed) for the full ground truth.

This test suite uses a synthetic .npz fixture that mimics that corrected
per-window shape (multiple windows per subject, per-window SBP/DBP, an
IncludeFlag array) so the adapter's window-unpacking and QC-filtering logic is
actually exercised, rather than a flat one-row-per-subject fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustbio.data.pulsedb import (
    PulseDBPaths,
    build_pulsedb_cohort,
    build_pulsedb_label_table,
    make_pulsedb_signal_loader,
    PULSEDB_FS,
    PULSEDB_SEGMENT_SEC,
)


def _make_subject_npz(path, subject_id, n_windows, rng, include_flag=None,
                       sbp_values=None, ecg_windows=None):
    """Write one synthetic per-subject .npz mimicking the real per-window
    Subj_Wins schema: ECG_Raw/PPG_Raw/ABP_Raw shaped (n_windows, 1, n_samples),
    SegSBP/SegDBP shaped (n_windows,), IncludeFlag shaped (n_windows,)."""
    n_samples = PULSEDB_SEGMENT_SEC * PULSEDB_FS
    if ecg_windows is not None:
        ecg = ecg_windows
    else:
        ecg = rng.standard_normal((n_windows, 1, n_samples)).astype(np.float32)
    ppg = rng.standard_normal((n_windows, 1, n_samples)).astype(np.float32)
    abp = (80 + 20 * rng.standard_normal((n_windows, 1, n_samples))).astype(np.float32)
    if sbp_values is None:
        sbp_values = np.array([120.0 + i for i in range(n_windows)])
    dbp = np.array([80.0 + i for i in range(n_windows)])
    if include_flag is None:
        include_flag = np.ones(n_windows, dtype=bool)
    np.savez(
        path,
        ecg_raw=ecg, ppg_raw=ppg, abp_raw=abp,
        seg_sbp=np.asarray(sbp_values, dtype=np.float64),
        seg_dbp=np.asarray(dbp, dtype=np.float64),
        include_flag=np.asarray(include_flag, dtype=bool),
        subject_id=subject_id,
        age=np.array([50.0]),
    )


@pytest.fixture
def fake_pulsedb_root(tmp_path):
    """Synthetic PulseDB root with multiple windows per subject, per-window
    SBP/DBP, and an IncludeFlag QC array -- mirrors the real Subj_Wins schema
    (confirmed in docs/pulsedb_structure_notes.md), not a flat per-subject
    fixture."""
    root = tmp_path / "pulsedb"
    for source, subdir in [("mimic", "PulseDB_MIMIC"), ("vital", "PulseDB_Vital")]:
        d = root / "Segment_Files" / subdir
        d.mkdir(parents=True)
        rng = np.random.default_rng(0 if source == "mimic" else 1)
        # subject 0: 5 windows, all included
        _make_subject_npz(
            d / f"{source}_s0.npz", f"{source}_s0", n_windows=5, rng=rng,
        )
        # subject 1: 4 windows, one excluded by IncludeFlag (window 2)
        _make_subject_npz(
            d / f"{source}_s1.npz", f"{source}_s1", n_windows=4, rng=rng,
            include_flag=np.array([True, True, False, True]),
        )
        # subject 2: 3 windows, all included
        _make_subject_npz(
            d / f"{source}_s2.npz", f"{source}_s2", n_windows=3, rng=rng,
        )
    return root


@pytest.fixture
def distinct_window_root(tmp_path):
    """A single-subject root where window 0 and window 1 have visibly
    different ECG content, to confirm the signal loader indexes the correct
    window rather than always returning window 0."""
    root = tmp_path / "pulsedb"
    d = root / "Segment_Files" / "PulseDB_MIMIC"
    d.mkdir(parents=True)
    n_samples = PULSEDB_SEGMENT_SEC * PULSEDB_FS
    ecg = np.zeros((2, 1, n_samples), dtype=np.float32)
    ecg[0, 0, :] = 1.0  # window 0: constant 1.0
    ecg[1, 0, :] = -7.0  # window 1: constant -7.0, unmistakably different
    rng = np.random.default_rng(42)
    _make_subject_npz(
        d / "mimic_sX.npz", "mimic_sX", n_windows=2, rng=rng, ecg_windows=ecg,
    )
    return root


def test_build_pulsedb_cohort_is_subject_disjoint_and_filters_include_flag(fake_pulsedb_root):
    cohort = build_pulsedb_cohort(fake_pulsedb_root, source="mimic", file_ext="npz")
    assert set(cohort.visits["source"]) == {"mimic"}
    # 5 + 4 + 3 = 12 windows total, minus 1 excluded by IncludeFlag = 11
    assert len(cohort.visits) == 11
    assert cohort.visits["split"].isin(["train", "val", "test"]).all()

    # the excluded window must not appear as a visit
    assert "mimic_s1_w2" not in set(cohort.visits["visit_id"])
    # the included windows of that same subject must still be present
    assert {"mimic_s1_w0", "mimic_s1_w1", "mimic_s1_w3"}.issubset(set(cohort.visits["visit_id"]))

    # all windows of a given subject must land in exactly one split (no leakage)
    per_subject_splits = cohort.visits.groupby("subject_id")["split"].nunique()
    assert (per_subject_splits == 1).all()
    assert set(cohort.visits["subject_id"]) == {"mimic_s0", "mimic_s1", "mimic_s2"}


def test_pulsedb_signal_loader_reads_correct_window_not_always_window_zero(distinct_window_root):
    load = make_pulsedb_signal_loader(distinct_window_root, source="mimic", file_ext="npz")

    ecg0, fs0 = load("mimic_sX_w0", "ecg")
    ecg1, fs1 = load("mimic_sX_w1", "ecg")

    assert fs0 == PULSEDB_FS
    assert fs1 == PULSEDB_FS
    assert len(ecg0) == PULSEDB_SEGMENT_SEC * PULSEDB_FS
    assert len(ecg1) == PULSEDB_SEGMENT_SEC * PULSEDB_FS

    assert np.allclose(ecg0, 1.0)
    assert np.allclose(ecg1, -7.0)
    assert not np.allclose(ecg0, ecg1)


def test_pulsedb_signal_loader_reads_ppg(fake_pulsedb_root):
    load = make_pulsedb_signal_loader(fake_pulsedb_root, source="mimic", file_ext="npz")
    ppg, ppg_fs = load("mimic_s0_w0", "ppg")
    assert ppg_fs == PULSEDB_FS
    assert len(ppg) == PULSEDB_SEGMENT_SEC * PULSEDB_FS


def test_pulsedb_label_table_pulls_per_window_sbp_dbp(fake_pulsedb_root):
    cohort = build_pulsedb_cohort(fake_pulsedb_root, source="mimic", file_ext="npz")
    visit_ids = cohort.visits["visit_id"].tolist()
    labels = build_pulsedb_label_table(
        fake_pulsedb_root, source="mimic", visit_ids=visit_ids, file_ext="npz",
    )
    assert set(labels.columns) == {"hr_regression", "sbp_regression", "dbp_regression"}
    assert labels["sbp_regression"].notna().all()
    assert labels["dbp_regression"].notna().all()

    # subject s0's windows have SegSBP = [120, 121, 122, 123, 124] (per fixture
    # construction) -- confirm the label table reflects each window's own
    # value, not a single subject-wide constant.
    assert labels.loc["mimic_s0_w0", "sbp_regression"] == pytest.approx(120.0)
    assert labels.loc["mimic_s0_w1", "sbp_regression"] == pytest.approx(121.0)
    assert labels.loc["mimic_s0_w4", "sbp_regression"] == pytest.approx(124.0)
    assert labels.loc["mimic_s0_w0", "sbp_regression"] != labels.loc["mimic_s0_w1", "sbp_regression"]


def test_pulsedb_label_table_hr_derived_from_ecg(fake_pulsedb_root):
    cohort = build_pulsedb_cohort(fake_pulsedb_root, source="mimic", file_ext="npz")
    visit_ids = cohort.visits["visit_id"].tolist()
    labels = build_pulsedb_label_table(
        fake_pulsedb_root, source="mimic", visit_ids=visit_ids, file_ext="npz",
    )
    # random-noise ECG won't always yield clean peaks, but at minimum HR
    # column must exist and be numeric (finite where derivable).
    assert "hr_regression" in labels.columns
    assert labels["hr_regression"].dtype.kind == "f"


def test_pulsedb_paths_properties(tmp_path):
    paths = PulseDBPaths(tmp_path)
    assert paths.mimic_dir == tmp_path / "Segment_Files" / "PulseDB_MIMIC"
    assert paths.vital_dir == tmp_path / "Segment_Files" / "PulseDB_Vital"
