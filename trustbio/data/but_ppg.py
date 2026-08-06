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
    # utf-8-sig strips the UTF-8 BOM that PhysioNet's real CSVs carry -- without
    # it the first column parses as "﻿ID" and every by-name lookup misses.
    return pd.read_csv(root / "quality-hr-ann.csv", encoding="utf-8-sig")


def _normalize_quality_hr_columns(qhr: pd.DataFrame) -> pd.DataFrame:
    """Map the real CSV's column names onto the ones this module uses.

    The published quality-hr-ann.csv header is `ID,Quality,HR` (title case, and
    BOM-prefixed). Earlier code relied on positional fallbacks
    (`columns[0]`, `.iloc[:, 1]`, `columns[-1]`) which happened to work only
    because the column ORDER matched -- a silent trap if the upstream file ever
    reorders. Match case-insensitively by name instead.
    """
    rename = {}
    for col in qhr.columns:
        key = str(col).strip().lstrip("﻿").lower()
        if key in ("id", "signal_id"):
            rename[col] = "signal_id"
        elif key == "quality":
            rename[col] = "quality"
        elif key in ("hr", "heart_rate"):
            rename[col] = "hr"
    return qhr.rename(columns=rename)


def _subject_id_from_record(record_id: str) -> str:
    return record_id[:3]


def _has_usable_header(root: Path, visit_id: str) -> bool:
    """False for records whose WFDB header is malformed upstream.

    48 of BUT PPG v2.0.0's 3,888 records (all IDs < 112001, i.e. the
    pre-accelerometer era) ship a header with `nsig` and `nsamp` TRANSPOSED --
    e.g. `100001_PPG 300 30 1` declares 300 signals of 1 sample each, when the
    600-byte .dat provably holds 300 int16 samples of one signal. wfdb does not
    error on these; it faithfully returns a single-sample array, so the bad
    records would silently poison downstream features. Detect and drop them.

    Tracked follow-up: check PhysioNet errata / upstream authors so the paper's
    Methods can cite the real cause rather than just reporting an exclusion.
    """
    hea = _record_path(root, visit_id, "PPG").with_suffix(".hea")
    try:
        fields = hea.read_text(errors="replace").splitlines()[0].split()
    except (OSError, IndexError):
        return False
    if len(fields) < 4:
        return False
    try:
        nsig, _fs, nsamp = int(fields[1]), int(fields[2]), int(fields[3])
    except ValueError:
        return False
    return not (nsamp == 1 and nsig > 1)


def build_but_ppg_cohort(
    root: str | Path = DEFAULT_ROOT,
    quality_hr_csv: pd.DataFrame | None = None,
    seed: int = 0,
    drop_malformed_headers: bool = True,
) -> Cohort:
    """One visit per recording (6-digit signal_id). Splits are subject-disjoint
    (first 3 digits of the ID). Passes through the native `quality` label so
    Task 13's fault taxonomy can consume it directly.

    `drop_malformed_headers` (default True) excludes records whose upstream WFDB
    header has nsig/nsamp transposed -- see `_has_usable_header`. Those records
    load as a single garbage sample rather than raising, so keeping them would
    silently corrupt features. The count dropped is printed so it can be
    reported in Methods. Pass False only to reproduce the unfiltered cohort.
    """
    root = Path(root)
    qhr = quality_hr_csv if quality_hr_csv is not None else _load_quality_hr(root)
    qhr = _normalize_quality_hr_columns(qhr)
    if "signal_id" not in qhr.columns:
        raise ValueError(
            "quality-hr-ann.csv has no recognizable ID column "
            f"(saw {list(qhr.columns)}); expected 'ID' or 'signal_id'."
        )

    df = qhr[["signal_id"]].rename(columns={"signal_id": "visit_id"}).copy()
    df["visit_id"] = df["visit_id"].astype(str).str.zfill(6)
    df["subject_id"] = df["visit_id"].map(_subject_id_from_record)
    df["quality"] = qhr["quality"].to_numpy()

    if drop_malformed_headers:
        n_before = len(df)
        keep = df["visit_id"].map(lambda v: _has_usable_header(root, v))
        df = df[keep].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(
                f"[but_ppg] excluded {n_dropped}/{n_before} records with malformed "
                "upstream WFDB headers (nsig/nsamp transposed; they load as a "
                "single sample). Report this exclusion in Methods."
            )

    df["split"] = chronological_or_random_split(
        df, subject_col="subject_id", time_col=None, seed=seed,
    )
    assert_no_subject_leakage(df)
    return Cohort(visits=df.reset_index(drop=True))


def _record_path(root: Path, visit_id: str, suffix: str) -> Path:
    """Path to one WFDB record, WITHOUT its extension.

    BUT PPG stores each recording in its own six-digit subdirectory (confirmed
    against the live PhysioNet page): `<root>/100001/100001_PPG.{dat,hea}`,
    NOT a flat `<root>/100001_PPG.{dat,hea}`. Only the two annotation CSVs
    (quality-hr-ann.csv, subject-info.csv) live at the root itself.
    """
    return root / visit_id / f"{visit_id}_{suffix}"


def make_but_ppg_signal_loader(root: str | Path = DEFAULT_ROOT):
    """SignalLoader closure: (visit_id, modality) -> (raw_signal, fs). Reads
    `{visit_id}/{visit_id}_PPG` (30 Hz) or `{visit_id}/{visit_id}_ECG`
    (1000 Hz) WFDB records."""
    root = Path(root)

    def load(visit_id: str, modality: str):
        suffix = {"ppg": "PPG", "ecg": "ECG"}[modality]
        rec = wfdb.rdrecord(str(_record_path(root, visit_id, suffix)))
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
    rec = wfdb.rdrecord(str(_record_path(root, visit_id, "ACC")))
    return np.asarray(rec.p_signal).astype(np.float32), rec.fs


def build_but_ppg_label_table(
    quality_hr: pd.DataFrame, visit_ids: list[str],
) -> pd.DataFrame:
    """hr_regression from the reference HR column in quality-hr-ann.csv."""
    qhr = _normalize_quality_hr_columns(quality_hr.copy())
    missing = {"signal_id", "hr"} - set(qhr.columns)
    if missing:
        raise ValueError(
            f"quality-hr-ann.csv missing required column(s) {sorted(missing)}; "
            f"saw {list(qhr.columns)}. Expected an ID column and an HR column."
        )
    id_col, hr_col = "signal_id", "hr"
    qhr[id_col] = qhr[id_col].astype(str).str.zfill(6)
    qhr = qhr.set_index(id_col)
    out = pd.DataFrame(index=pd.Index(visit_ids, name="visit_id"))
    out["hr_regression"] = qhr[hr_col].reindex(visit_ids).astype(float).to_numpy()
    return out
