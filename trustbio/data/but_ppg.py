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

import re as _re
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


# One "<file> 16 <gain>(<baseline>)/<units>" signal-spec line of a WFDB header.
_SPEC_RE = _re.compile(r"\s16\s+(-?[\d.]+)\((-?\d+)\)/(\S+)")


def _read_header_specs(hea: Path) -> tuple[list[str], list[float], list[int]]:
    """Return (first_line_fields, gains, baselines) from a WFDB .hea."""
    try:
        lines = hea.read_text(errors="replace").splitlines()
    except OSError:
        return [], [], []
    if not lines:
        return [], [], []
    gains, baselines = [], []
    for line in lines[1:]:
        m = _SPEC_RE.search(line)
        if m:
            gains.append(float(m.group(1)))
            baselines.append(int(m.group(2)))
    return lines[0].split(), gains, baselines


def is_release1_record(root: Path, visit_id: str, suffix: str = "PPG") -> bool:
    """True if this record uses BUT PPG release 1's broken header encoding.

    Per PhysioNet's Release Notes, release 1 covered IDs 100001-111004 and
    release 2 added 112001 onwards. All 48 release-1 records ship a header whose
    first line reads e.g. `100001_PPG 300 30 1` -- i.e. "300 signals, 30 Hz,
    1 sample". That is NOT a transposed channel count: release 1 wrote one
    signal-spec line PER SAMPLE, and the sample values themselves live in those
    lines' gain/baseline fields, while the .dat payload is all zeros.
    Release-2 headers are correct (`112001_PPG 3 30 300`).

    Detected by the signature `nsamp == 1 and nsig > 1`, which no well-formed
    single-channel record produces. PhysioNet publishes no errata for this
    (checked 2026-08-06).
    """
    fields, _gains, _baselines = _read_header_specs(
        _record_path(root, visit_id, suffix).with_suffix(".hea")
    )
    if len(fields) < 4:
        return False
    try:
        nsig, nsamp = int(fields[1]), int(fields[3])
    except ValueError:
        return False
    return nsamp == 1 and nsig > 1


def _read_release1_signal(root: Path, visit_id: str, suffix: str) -> tuple[np.ndarray, int]:
    """Reconstruct a release-1 record's waveform from its header gain fields.

    The .dat is (usually) all zeros; the waveform is carried by the per-sample
    gain/baseline pairs, so the standard WFDB conversion
    `physical = (raw - baseline) / gain` applied element-wise recovers it. This
    was validated against an independent label: record 100001's recovered PPG
    yields 81.8 bpm vs. the 83 bpm reference HR in quality-hr-ann.csv (1.4%
    error). See `reconstruction_hr_error_pct` for per-record fidelity.
    """
    hea = _record_path(root, visit_id, suffix).with_suffix(".hea")
    fields, gains, baselines = _read_header_specs(hea)
    if not gains:
        raise ValueError(f"no parsable signal specs in {hea}")
    fs = int(round(float(fields[2])))

    dat = _record_path(root, visit_id, suffix).with_suffix(".dat")
    raw = np.fromfile(dat, dtype="<i2").astype(np.float64)
    n = len(gains)
    # Release 1's .dat is normally all zeros; when it does carry samples (10 of
    # the 48 records have a non-zero ECG .dat) use them, else treat as zeros.
    if len(raw) == n and np.any(raw):
        digital = raw
    else:
        digital = np.zeros(n, dtype=np.float64)

    g = np.asarray(gains, dtype=np.float64)
    b = np.asarray(baselines, dtype=np.float64)
    g[g == 0] = np.nan  # never divide by a zero gain
    sig = (digital - b) / g
    return np.nan_to_num(sig, nan=0.0).astype(np.float32), fs


def reconstruction_hr_error_pct(
    root: Path, visit_id: str, reference_hr: float,
) -> float:
    """|recovered HR - reference HR| / reference HR * 100, or NaN if unavailable.

    Fidelity check for release-1 reconstructions: estimates heart rate from the
    recovered PPG by autocorrelation and compares it to the record's own
    reference annotation. Across the 48 release-1 records the median error is
    ~4%, but the distribution has a long tail (mean ~37%; ~60% within 10%), so
    downstream consumers that need trustworthy morphology -- notably the
    degradation calibration in `trustbio/degradation/calibrate.py` -- should
    filter on this rather than assume every reconstruction is faithful.
    """
    if not reference_hr or not np.isfinite(reference_hr) or reference_hr <= 0:
        return float("nan")
    try:
        sig, fs = _read_release1_signal(Path(root), visit_id, "PPG")
    except (OSError, ValueError):
        return float("nan")
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < fs or not np.any(np.isfinite(sig)):
        return float("nan")
    sig = sig - sig.mean()
    if not np.any(sig):
        return float("nan")
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    lo, hi = int(fs * 60 / 180), int(fs * 60 / 40)   # 40..180 bpm
    hi = min(hi, len(ac))
    if hi <= lo:
        return float("nan")
    period = lo + int(np.argmax(ac[lo:hi]))
    if period <= 0:
        return float("nan")
    est = 60.0 * fs / period
    return float(abs(est - reference_hr) / reference_hr * 100.0)


def build_but_ppg_cohort(
    root: str | Path = DEFAULT_ROOT,
    quality_hr_csv: pd.DataFrame | None = None,
    seed: int = 0,
    annotate_reconstruction: bool = True,
) -> Cohort:
    """One visit per recording (6-digit signal_id). Splits are subject-disjoint
    (first 3 digits of the ID). Passes through the native `quality` label so
    Task 13's fault taxonomy can consume it directly.

    ALL 3,888 records are retained, including the 48 release-1 records whose
    headers are malformed upstream -- their waveforms are reconstructed from the
    header's per-sample gain fields (see `is_release1_record`), which preserves
    the full 50-subject cohort. Excluding them would have cost 24% of subjects
    (50 -> 38) for only 1.2% of records, since every recording of 12 release-1
    subjects is affected.

    With `annotate_reconstruction` (default True) the cohort gains two columns:
      `reconstructed`            -- True for the 48 release-1 records
      `reconstruction_hr_err_pct`-- |recovered HR - reference HR| / reference,
                                    NaN for non-reconstructed records
    Reconstruction fidelity varies (median ~4% error, but a long tail), so
    consumers needing trustworthy morphology -- notably the degradation
    calibration -- should filter on the error column rather than assume every
    reconstruction is faithful. Nothing is dropped here.
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

    if annotate_reconstruction:
        df["reconstructed"] = df["visit_id"].map(
            lambda v: is_release1_record(root, v)
        )
        ref_hr = dict(zip(df["visit_id"], pd.to_numeric(qhr["hr"], errors="coerce"))) \
            if "hr" in qhr.columns else {}
        df["reconstruction_hr_err_pct"] = [
            reconstruction_hr_error_pct(root, v, ref_hr.get(v, float("nan")))
            if rec else float("nan")
            for v, rec in zip(df["visit_id"], df["reconstructed"])
        ]
        n_rec = int(df["reconstructed"].sum())
        if n_rec:
            err = df.loc[df["reconstructed"], "reconstruction_hr_err_pct"]
            ok = int((err < 10).sum())
            print(
                f"[but_ppg] reconstructed {n_rec}/{len(df)} release-1 records from "
                f"header gain fields ({ok}/{n_rec} within 10% of reference HR; "
                f"median err {err.median():.1f}%). All records retained; filter on "
                "`reconstruction_hr_err_pct` where reconstruction fidelity matters."
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
        # Release-1 records can't be read by wfdb (their header declares 1
        # sample and their .dat is zeros) -- reconstruct from the header's
        # per-sample gain fields instead. See is_release1_record().
        if is_release1_record(root, visit_id, suffix):
            return _read_release1_signal(root, visit_id, suffix)
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
