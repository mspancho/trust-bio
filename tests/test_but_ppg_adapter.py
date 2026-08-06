# tests/test_but_ppg_adapter.py
import numpy as np
import pandas as pd
import pytest
import wfdb

from trustbio.data.but_ppg import (
    build_but_ppg_cohort, build_but_ppg_label_table,
    make_but_ppg_signal_loader, load_but_ppg_accelerometer,
)


@pytest.fixture
def fake_but_ppg_root(tmp_path):
    """Mirrors PhysioNet's REAL BUT PPG layout: each six-digit recording lives
    in its own subdirectory (`<root>/100001/100001_PPG.{dat,hea}`), with only
    the annotation CSVs at the root. An earlier version of this fixture wrote
    every record flat at the root, which matched what the adapter used to
    assume but NOT the real dataset -- so it hid a real path bug until BUT PPG
    was actually fetched. Keep this nested."""
    root = tmp_path / "but-ppg"
    root.mkdir()
    rows = []
    for i, rec_id in enumerate(["100001", "100002", "112001"]):  # last has ACC
        rng = np.random.default_rng(i)
        rec_dir = root / rec_id
        rec_dir.mkdir()
        ppg = rng.standard_normal(30 * 10).astype(np.float32)
        ecg = rng.standard_normal(1000 * 10).astype(np.float32)
        wfdb.wrsamp(f"{rec_id}_PPG", fs=30, units=["NU"], sig_name=["PPG"],
                    p_signal=ppg[:, None], write_dir=str(rec_dir), fmt=["16"])
        wfdb.wrsamp(f"{rec_id}_ECG", fs=1000, units=["mV"], sig_name=["ECG"],
                    p_signal=ecg[:, None], write_dir=str(rec_dir), fmt=["16"])
        if rec_id == "112001":
            acc = rng.standard_normal((100 * 10, 3)).astype(np.float32)
            wfdb.wrsamp(f"{rec_id}_ACC", fs=100, units=["g", "g", "g"],
                        sig_name=["ACC_X", "ACC_Y", "ACC_Z"],
                        p_signal=acc, write_dir=str(rec_dir), fmt=["16", "16", "16"])
        rows.append({"signal_id": rec_id, "quality": 1 if i != 1 else 0, "hr": 70.0 + i})
    quality_hr = pd.DataFrame(rows)
    quality_hr.to_csv(root / "quality-hr-ann.csv", index=False)
    return root, quality_hr


def test_build_cohort_subject_from_first_three_digits(fake_but_ppg_root):
    root, quality_hr = fake_but_ppg_root
    cohort = build_but_ppg_cohort(root, quality_hr_csv=quality_hr)
    assert set(cohort.visits["visit_id"]) == {"100001", "100002", "112001"}
    assert set(cohort.visits["subject_id"]) == {"100", "112"}
    assert "quality" in cohort.visits.columns


def test_signal_loader_reads_ppg_and_ecg_at_native_rates(fake_but_ppg_root):
    root, _ = fake_but_ppg_root
    load = make_but_ppg_signal_loader(root)
    ppg, ppg_fs = load("100001", "ppg")
    ecg, ecg_fs = load("100001", "ecg")
    assert ppg_fs == 30
    assert ecg_fs == 1000
    assert len(ppg) == 30 * 10
    assert len(ecg) == 1000 * 10


def test_accelerometer_present_only_for_recent_ids(fake_but_ppg_root):
    root, _ = fake_but_ppg_root
    assert load_but_ppg_accelerometer(root, "100001") is None
    acc, acc_fs = load_but_ppg_accelerometer(root, "112001")
    assert acc_fs == 100
    assert acc.shape[1] == 3


def test_label_table_has_hr(fake_but_ppg_root):
    root, quality_hr = fake_but_ppg_root
    labels = build_but_ppg_label_table(quality_hr, visit_ids=["100001", "100002", "112001"])
    assert list(labels.columns) == ["hr_regression"]
    assert labels.loc["100001", "hr_regression"] == 70.0


def test_cohort_accepts_real_csv_column_names(fake_but_ppg_root):
    """PhysioNet's published quality-hr-ann.csv header is `ID,Quality,HR`
    (title case, BOM-prefixed), not the lowercase names the other fixtures
    use. Earlier code only survived this by positional-column luck."""
    root, quality_hr = fake_but_ppg_root
    real_style = quality_hr.rename(
        columns={"signal_id": "ID", "quality": "Quality", "hr": "HR"}
    )
    cohort = build_but_ppg_cohort(root, quality_hr_csv=real_style)
    assert set(cohort.visits["visit_id"]) == {"100001", "100002", "112001"}
    assert "quality" in cohort.visits.columns

    labels = build_but_ppg_label_table(
        real_style, visit_ids=["100001", "100002", "112001"]
    )
    assert list(labels.columns) == ["hr_regression"]
    assert labels["hr_regression"].notna().all()


def test_cohort_excludes_malformed_transposed_headers(fake_but_ppg_root):
    """48 real BUT PPG records declare `nsig nsamp` transposed (300 signals of
    1 sample). wfdb returns a single garbage sample instead of raising, so the
    cohort builder must drop them rather than let them poison features."""
    root, quality_hr = fake_but_ppg_root
    # Corrupt 100002's header the way the real dataset does.
    hea = root / "100002" / "100002_PPG.hea"
    lines = hea.read_text().splitlines()
    name, nsig, fs, nsamp = lines[0].split()[:4]
    lines[0] = f"{name} {nsamp} {fs} 1"   # transpose nsig/nsamp
    hea.write_text("\n".join(lines) + "\n")

    cohort = build_but_ppg_cohort(root, quality_hr_csv=quality_hr)
    assert "100002" not in set(cohort.visits["visit_id"])
    assert set(cohort.visits["visit_id"]) == {"100001", "112001"}

    kept = build_but_ppg_cohort(
        root, quality_hr_csv=quality_hr, drop_malformed_headers=False
    )
    assert "100002" in set(kept.visits["visit_id"])
