"""BUT PPG adapter: smartphone-camera PPG with a hospital-grade reference ECG
and (for IDs >= 112001) synchronized triaxial accelerometry, plus a native
binary signal-quality label — the sharpest available proxy for the clinical
(reference ECG) to consumer-device (smartphone PPG) gap, and the primary
source of REAL (not synthetic) motion artifact for Task 11's calibration.

Record ID convention (confirmed via PhysioNet page): a 6-digit ID where the
first 3 digits are the subject and the last 3 are the measurement number
within that subject (e.g. "100001" = subject 100, measurement 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .cohort import Cohort, assert_no_subject_leakage, chronological_or_random_split

DEFAULT_ROOT = Path(
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/but-ppg/"
    "physionet.org/files/butppg/2.0.0"
)

# Accelerometer channel was only added starting at this record ID (confirmed
# via PhysioNet page: "ACC signals ... only for recordings 112001 onwards").
ACC_MIN_RECORD_ID = 112001


def _load_quality_hr(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "quality-hr-ann.csv")


def _subject_id_from_record(record_id: str) -> str:
    return record_id[:3]


def build_but_ppg_cohort(
    root: str | Path = DEFAULT_ROOT,
    quality_hr_csv: pd.DataFrame | None = None,
    seed: int = 0,
) -> Cohort:
    """One visit per recording (6-digit signal_id). Splits are subject-disjoint
    (first 3 digits of the ID). Passes through the native `quality` label so
    Task 13's fault taxonomy can consume it directly."""
    root = Path(root)
    qhr = quality_hr_csv if quality_hr_csv is not None else _load_quality_hr(root)
    qhr = qhr.rename(columns={"signal_id": "visit_id"}) if "signal_id" in qhr.columns else qhr.rename(columns={qhr.columns[0]: "visit_id"})
    df = qhr[["visit_id"]].copy()
    df["visit_id"] = df["visit_id"].astype(str).str.zfill(6)
    df["subject_id"] = df["visit_id"].map(_subject_id_from_record)
    df["quality"] = qhr["quality"].to_numpy() if "quality" in qhr.columns else qhr.iloc[:, 1].to_numpy()
    df["split"] = chronological_or_random_split(
        df, subject_col="subject_id", time_col=None, seed=seed,
    )
    assert_no_subject_leakage(df)
    return Cohort(visits=df.reset_index(drop=True))


def make_but_ppg_signal_loader(root: str | Path = DEFAULT_ROOT):
    """SignalLoader closure: (visit_id, modality) -> (raw_signal, fs). Reads
    `{visit_id}_PPG` (30 Hz) or `{visit_id}_ECG` (1000 Hz) WFDB records."""
    root = Path(root)

    def load(visit_id: str, modality: str):
        suffix = {"ppg": "PPG", "ecg": "ECG"}[modality]
        rec = wfdb.rdrecord(str(root / f"{visit_id}_{suffix}"))
        sig = np.asarray(rec.p_signal)[:, 0].astype(np.float32)
        return sig, rec.fs

    return load


def load_but_ppg_accelerometer(
    root: str | Path, visit_id: str,
) -> tuple[np.ndarray, int] | None:
    """Return (n_samples, 3) triaxial accelerometer array + fs, or None if
    this recording predates the accelerometer channel (ID < 112001)."""
    if int(visit_id) < ACC_MIN_RECORD_ID:
        return None
    root = Path(root)
    rec = wfdb.rdrecord(str(root / f"{visit_id}_ACC"))
    return np.asarray(rec.p_signal).astype(np.float32), rec.fs


def build_but_ppg_label_table(
    quality_hr: pd.DataFrame, visit_ids: list[str],
) -> pd.DataFrame:
    """hr_regression from the reference HR column in quality-hr-ann.csv."""
    qhr = quality_hr.copy()
    id_col = "signal_id" if "signal_id" in qhr.columns else qhr.columns[0]
    hr_col = "hr" if "hr" in qhr.columns else qhr.columns[-1]
    qhr[id_col] = qhr[id_col].astype(str).str.zfill(6)
    qhr = qhr.set_index(id_col)
    out = pd.DataFrame(index=pd.Index(visit_ids, name="visit_id"))
    out["hr_regression"] = qhr[hr_col].reindex(visit_ids).astype(float).to_numpy()
    return out
