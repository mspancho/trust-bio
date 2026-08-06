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


def _make_release1_record(root, rec_id, n_samples, fs, values):
    """Rewrite `rec_id`'s PPG record in BUT PPG release-1's broken encoding:
    header declares `<n_samples> <fs> 1` (one spec line PER SAMPLE, carrying the
    sample value in its gain/baseline fields) and the .dat is all zeros."""
    rec_dir = root / rec_id
    hea = rec_dir / f"{rec_id}_PPG.hea"
    lines = [f"{rec_id}_PPG {n_samples} {fs} 1"]
    for v in values:
        # physical = (0 - baseline)/gain  ->  choose baseline=-v, gain=1
        lines.append(f"{rec_id}_PPG.dat 16 1.0({int(-round(v))})/a.u. 0 0 0 0 0 ")
    hea.write_text("\n".join(lines) + "\n")
    (rec_dir / f"{rec_id}_PPG.dat").write_bytes(b"\x00\x00" * n_samples)


def test_release1_records_are_reconstructed_not_dropped(fake_but_ppg_root):
    """The 48 real release-1 records store their waveform in the header's
    per-sample gain fields with an all-zero .dat. They must be RECONSTRUCTED and
    retained -- dropping them would cost 24% of subjects (50 -> 38) for 1.2% of
    records, since every recording of 12 subjects is affected."""
    import numpy as np
    from trustbio.data.but_ppg import is_release1_record, make_but_ppg_signal_loader

    root, quality_hr = fake_but_ppg_root
    fs, n = 30, 300
    # A clean 1 Hz sine at 30 Hz -> recoverable, physiologically periodic.
    values = 100.0 + 50.0 * np.sin(2 * np.pi * np.arange(n) / fs)
    _make_release1_record(root, "100002", n, fs, values)

    assert is_release1_record(root, "100002") is True
    assert is_release1_record(root, "112001") is False

    cohort = build_but_ppg_cohort(root, quality_hr_csv=quality_hr)
    # nothing dropped
    assert set(cohort.visits["visit_id"]) == {"100001", "100002", "112001"}
    v = cohort.visits.set_index("visit_id")
    assert bool(v.loc["100002", "reconstructed"]) is True
    assert bool(v.loc["112001", "reconstructed"]) is False
    # fidelity recorded for reconstructed rows, NaN for the others
    assert np.isnan(v.loc["112001", "reconstruction_hr_err_pct"])

    # the reconstruction recovers the full sample count, not a single sample
    load = make_but_ppg_signal_loader(root)
    sig, got_fs = load("100002", "ppg")
    assert got_fs == fs
    assert len(sig) == n, f"expected {n} samples, got {len(sig)} (the old bug)"
    assert np.ptp(sig) > 0, "reconstructed signal must not be constant"
