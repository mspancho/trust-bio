import numpy as np
import pandas as pd
import pytest
import wfdb

from trustbio.data.mimic_ext_ppg import (
    build_mimic_ext_ppg_cohort, build_mimic_ext_ppg_label_table, first_sqi_code,
    make_mimic_ext_ppg_signal_loader, parse_sqi_vector,
)


@pytest.fixture
def fake_metadata_and_waveforms(tmp_path):
    root = tmp_path / "mimic-iii-ext-ppg"
    rows = []
    for i in range(8):
        patient = f"p{i:06d}"
        folder = f"p0{i//4}/{patient}"
        seg_name = f"{patient}_seg1"
        (root / folder).mkdir(parents=True, exist_ok=True)
        fs = 125
        n = fs * 30
        rng = np.random.default_rng(i)
        pleth = rng.standard_normal(n).astype(np.float32)
        ecg = rng.standard_normal(n).astype(np.float32)
        wfdb.wrsamp(
            seg_name, fs=fs, units=["mV", "NU"], sig_name=["II", "PLETH"],
            p_signal=np.stack([ecg, pleth], axis=1), write_dir=str(root / folder),
            fmt=["16", "16"],
        )
        rows.append({
            # Mirror the REAL metadata.csv convention: folder_path is the FULL
            # record path (directory + record name, no trailing slash, no
            # extension), and signal_file_name repeats only its last component:
            #   folder_path      = "p04/p044018/3000060_0002_0_2"
            #   signal_file_name = "3000060_0002_0_2"
            # This fixture previously set folder_path=<dir>+"/", which made the
            # adapter's (incorrect) folder_path/signal_file_name join look right
            # here while failing on every real record.
            # And segment_id is a small integer that REPEATS across records
            # (2, 4, ...); only signal_file_name is unique.
            "segment_id": i % 3, "signal_file_name": seg_name,
            "folder_path": f"{folder}/{seg_name}",
            "subject_id": i, "event_rhythm": "SR" if i % 2 == 0 else "AF",
            "median_30s_hr": 70.0 + i,
            "vector_10s_pleth_sqi": "[1, 1, 0]" if i % 4 else "[0, 1, 1]",
            "vector_10s_ecg_sqi": "[1, -2, 1]" if i % 2 else "[1, 1, 1]",
            "strat_fold": i % 10,
        })
    meta = pd.DataFrame(rows)
    meta.to_csv(root / "metadata.csv", index=False)
    return root, meta


def test_build_cohort_is_subject_disjoint(fake_metadata_and_waveforms):
    root, meta = fake_metadata_and_waveforms
    cohort = build_mimic_ext_ppg_cohort(root, metadata_csv=meta)
    assert len(cohort.visits) == 8
    assert "vector_10s_pleth_sqi" in cohort.visits.columns
    assert cohort.visits["split"].isin(["train", "val", "test"]).all()


def test_signal_loader_reads_ecg_and_ppg(fake_metadata_and_waveforms):
    root, meta = fake_metadata_and_waveforms
    load = make_mimic_ext_ppg_signal_loader(root, meta)
    ecg, ecg_fs = load("p000000_seg1", "ecg")
    ppg, ppg_fs = load("p000000_seg1", "ppg")
    assert ecg_fs == 125
    assert ppg_fs == 125
    assert len(ecg) == 125 * 30
    assert len(ppg) == 125 * 30


def test_label_table_maps_rhythm_and_hr(fake_metadata_and_waveforms):
    root, meta = fake_metadata_and_waveforms
    labels = build_mimic_ext_ppg_label_table(meta, visit_ids=meta["signal_file_name"].tolist())
    assert set(labels.columns) == {"hr_regression", "rhythm_cls"}
    assert labels.loc["p000000_seg1", "rhythm_cls"] == 0.0   # SR
    assert labels.loc["p000001_seg1", "rhythm_cls"] == 1.0   # AF
    assert labels["hr_regression"].notna().all()


def test_signal_loader_reads_missing_samples_as_flat_line(tmp_path):
    root = tmp_path / "mimic-iii-ext-ppg"
    (root / "p00/p000009").mkdir(parents=True)
    fs, n = 125, 125 * 30
    rng = np.random.default_rng(9)
    ecg = rng.standard_normal(n).astype(np.float32); ecg[100:300] = np.nan     # a 1.6-s dropout
    pleth = rng.standard_normal(n).astype(np.float32)
    wfdb.wrsamp("p000009_seg1", fs=fs, units=["mV", "NU"], sig_name=["II", "PLETH"],
                p_signal=np.stack([ecg, pleth], axis=1), write_dir=str(root / "p00/p000009"), fmt=["16", "16"])
    meta = pd.DataFrame([{"segment_id": 0, "signal_file_name": "p000009_seg1",
                          "folder_path": "p00/p000009/p000009_seg1", "subject_id": 9}])
    load = make_mimic_ext_ppg_signal_loader(root, meta)
    ecg_out, _ = load("p000009_seg1", "ecg")
    assert np.isfinite(ecg_out).all()
    assert (ecg_out[100:300] == 0.0).all() and np.abs(ecg_out[:100]).sum() > 0


def test_cohort_visit_ids_are_unique_record_names(fake_metadata_and_waveforms):
    root, meta = fake_metadata_and_waveforms
    cohort = build_mimic_ext_ppg_cohort(root, metadata_csv=meta)
    assert cohort.visits["visit_id"].is_unique
    assert set(cohort.visits["visit_id"]) == set(meta["signal_file_name"])


def test_parse_sqi_vector_handles_real_formats():
    assert parse_sqi_vector("[1, 1, -2]").tolist() == [1.0, 1.0, -2.0]
    v = parse_sqi_vector("[np.float64(86.21), np.float64(87.21), nan]")
    assert v[:2].tolist() == [86.21, 87.21] and np.isnan(v[2])
    assert np.isnan(parse_sqi_vector("[nan, nan, nan]")).all()
    assert first_sqi_code("[0, 1, 1]") == 0.0
    assert np.isnan(first_sqi_code("[nan, 1, 1]"))
    assert np.isnan(first_sqi_code(float("nan")))
