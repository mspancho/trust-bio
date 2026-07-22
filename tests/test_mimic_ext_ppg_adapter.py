import numpy as np
import pandas as pd
import pytest
import wfdb

from trustbio.data.mimic_ext_ppg import (
    build_mimic_ext_ppg_cohort, build_mimic_ext_ppg_label_table,
    make_mimic_ext_ppg_signal_loader,
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
            "segment_id": seg_name, "signal_file_name": seg_name, "folder_path": folder + "/",
            "subject_id": i, "event_rhythm": "SR" if i % 2 == 0 else "AF",
            "median_30s_hr": 70.0 + i, "vector_10s_pleth_sqi": "[1, 1, 1]",
            "vector_10s_ecg_sqi": "[1, 1, 1]", "strat_fold": i % 10,
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
    labels = build_mimic_ext_ppg_label_table(meta, visit_ids=meta["segment_id"].tolist())
    assert set(labels.columns) == {"hr_regression", "rhythm_cls"}
    assert labels.loc["p000000_seg1", "rhythm_cls"] == 0.0   # SR
    assert labels.loc["p000001_seg1", "rhythm_cls"] == 1.0   # AF
    assert labels["hr_regression"].notna().all()
