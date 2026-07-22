"""PulseDB adapter: cross-institution ECG+PPG+ABP source (MIMIC-III vs. VitalDB).

PulseDB ships as one MATLAB v7.3 (HDF5) .mat file per subject under
Segment_Files/PulseDB_MIMIC/ and Segment_Files/PulseDB_Vital/ (Boston ICU vs.
Seoul surgical/perioperative source, respectively -- this split IS the
cross-institution transportability axis used by eval/transport.py).

Real-file field layout is confirmed empirically in Task 7 (see
docs/pulsedb_structure_notes.md, verified twice down to floating-point
precision) and is DIFFERENT from the plan's original assumed schema:

- The top-level key is `Subj_Wins`, a struct (not a `Signals` array with a
  row-index convention -- there is no SIGNAL_ROW_ECG/PPG/ABP scheme).
- `ECG_Raw`, `PPG_Raw`, `ABP_Raw` are independent fields, each shaped
  (n_windows, 1, 1250) -- one 10-second @ 125 Hz window per row.
- `SegSBP`, `SegDBP` are PER-WINDOW (shape (n_windows,)), not one value per
  subject.
- `IncludeFlag` is a per-window boolean QC/inclusion flag; PulseDB's natural
  unit is the WINDOW, not the subject-file, so this adapter builds one visit
  per (subject, window) pair that passes IncludeFlag, with `subject_id` kept
  constant across a subject's windows so subject-disjoint splitting still
  groups correctly.

Labels: PulseDB provides per-window SBP/DBP directly (its original purpose is
cuff-less blood-pressure estimation). Heart rate is not a stored field, so it
is derived from the ECG window via a simple peak-interval estimate at load
time (Methods: this is the same "derive HR from the signal" approach used for
BUT PPG and MIMIC-III-Ext-PPG, keeping the hr_regression task's supervision
consistent across all three datasets rather than trusting three different
upstream HR-estimation pipelines).

`file_ext="npz"` is accepted purely to let tests substitute a synthetic
fixture without needing mat73/the real download; the real path is
`file_ext="mat"` (default), read via `mat73.loadmat`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal

from .cohort import Cohort, assert_no_subject_leakage, chronological_or_random_split

# Confirmed empirically in Task 7 (docs/pulsedb_structure_notes.md): 125 Hz,
# 10-second (1250-sample) segments. There is no SIGNAL_ROW_ECG/PPG/ABP -- the
# real schema keys ECG/PPG/ABP by field name (ECG_Raw/PPG_Raw/ABP_Raw), not by
# row index into a combined array.
PULSEDB_FS = 125
PULSEDB_SEGMENT_SEC = 10


@dataclass
class PulseDBPaths:
    root: Path

    @property
    def mimic_dir(self) -> Path:
        return self.root / "Segment_Files" / "PulseDB_MIMIC"

    @property
    def vital_dir(self) -> Path:
        return self.root / "Segment_Files" / "PulseDB_Vital"

    def source_dir(self, source: str) -> Path:
        return self.mimic_dir if source == "mimic" else self.vital_dir


def _list_subject_files(paths: PulseDBPaths, source: str, file_ext: str) -> list[Path]:
    return sorted(paths.source_dir(source).glob(f"*.{file_ext}"))


def _subject_id_from_path(path: Path) -> str:
    return path.stem


def _load_subject_windows(path: Path, file_ext: str) -> dict:
    """Return the per-subject window data for one segment file:
    {"ecg": (n_windows, n_samples), "ppg": (n_windows, n_samples),
     "sbp": (n_windows,), "dbp": (n_windows,), "include_flag": (n_windows,)}

    Handles both the real .mat (Subj_Wins struct, via mat73) and the
    synthetic .npz test fixture.
    """
    if file_ext == "npz":
        data = np.load(path)
        return {
            "ecg": np.asarray(data["ecg_raw"])[:, 0, :].astype(np.float32),
            "ppg": np.asarray(data["ppg_raw"])[:, 0, :].astype(np.float32),
            "sbp": np.asarray(data["seg_sbp"], dtype=np.float64).ravel(),
            "dbp": np.asarray(data["seg_dbp"], dtype=np.float64).ravel(),
            "include_flag": np.asarray(data["include_flag"], dtype=bool).ravel(),
        }
    from mat73 import loadmat
    raw = loadmat(str(path))
    wins = raw["Subj_Wins"]
    return {
        "ecg": np.asarray(wins["ECG_Raw"])[:, 0, :].astype(np.float32),
        "ppg": np.asarray(wins["PPG_Raw"])[:, 0, :].astype(np.float32),
        "sbp": np.asarray(wins["SegSBP"], dtype=np.float64).ravel(),
        "dbp": np.asarray(wins["SegDBP"], dtype=np.float64).ravel(),
        "include_flag": np.asarray(wins["IncludeFlag"], dtype=bool).ravel(),
    }


def _parse_visit_id(visit_id: str) -> tuple[str, int]:
    """Split a visit_id like 'p000160_w3' back into ('p000160', 3)."""
    subject_id, _, win_part = visit_id.rpartition("_w")
    return subject_id, int(win_part)


def _estimate_hr_from_ecg(ecg: np.ndarray, fs: int) -> float:
    """Simple peak-interval heart-rate estimate: bandpass -> find_peaks -> HR."""
    if len(ecg) < fs * 2:
        return float("nan")
    b, a = scipy.signal.butter(3, [0.5, 40], btype="bandpass", fs=fs)
    filtered = scipy.signal.filtfilt(b, a, ecg)
    peaks, _ = scipy.signal.find_peaks(filtered, distance=fs * 0.33)  # <=180 bpm
    if len(peaks) < 2:
        return float("nan")
    ibi_sec = np.diff(peaks) / fs
    return float(60.0 / np.mean(ibi_sec))


def build_pulsedb_cohort(
    root: str | Path, source: str, file_ext: str = "mat", seed: int = 0,
) -> Cohort:
    """Build a subject-disjoint, per-window cohort for one PulseDB source
    ("mimic" or "vital").

    PulseDB's natural unit is a WINDOW, not a subject-file: each subject file
    has many windows, each with its own SegSBP/SegDBP/ECG/PPG/ABP segment.
    One visit is created per (subject, window) pair that passes the native
    `IncludeFlag` QC filter, with `visit_id` encoding both the subject and the
    window index (e.g. "p000160_w3") and `subject_id` held constant across a
    subject's windows so that all of one subject's windows land in the same
    split (splits are subject-disjoint, not window-disjoint).

    Splits are random (no meaningful chronological ordering exists across a
    de-identified multi-subject file set), 70/15/15 train/val/test.
    """
    paths = PulseDBPaths(Path(root))
    files = _list_subject_files(paths, source, file_ext)

    rows = []
    for f in files:
        subject_id = _subject_id_from_path(f)
        windows = _load_subject_windows(f, file_ext)
        include_flag = windows["include_flag"]
        n_windows = len(include_flag)
        for w in range(n_windows):
            if not include_flag[w]:
                continue
            rows.append({
                "visit_id": f"{subject_id}_w{w}",
                "subject_id": subject_id,
                "source": source,
            })

    df = pd.DataFrame(rows, columns=["visit_id", "subject_id", "source"])
    df["split"] = chronological_or_random_split(
        df, subject_col="subject_id", time_col=None, seed=seed,
    )
    assert_no_subject_leakage(df)
    return Cohort(visits=df.reset_index(drop=True))


def make_pulsedb_signal_loader(root: str | Path, source: str, file_ext: str = "mat"):
    """SignalLoader closure: (visit_id, modality) -> (raw_signal, fs).

    `visit_id` (e.g. "p000160_w3") is parsed back into (subject file, window
    index) to index into that window's ECG_Raw/PPG_Raw row. `modality` is
    "ecg" or "ppg" (ABP is available via the label table, not as a loadable
    modality, since no FM in the registry consumes blood pressure waveforms
    directly).
    """
    paths = PulseDBPaths(Path(root))
    cache: dict[str, dict] = {}

    def load(visit_id: str, modality: str):
        subject_id, win_idx = _parse_visit_id(visit_id)
        if subject_id not in cache:
            path = paths.source_dir(source) / f"{subject_id}.{file_ext}"
            if not path.exists():
                raise FileNotFoundError(f"no PulseDB file for subject {subject_id} at {path}")
            cache[subject_id] = _load_subject_windows(path, file_ext)
        return cache[subject_id][modality][win_idx], PULSEDB_FS

    return load


def build_pulsedb_label_table(
    root: str | Path, source: str, visit_ids: list[str], file_ext: str = "mat",
) -> pd.DataFrame:
    """Wide label table: hr_regression (derived from that window's ECG),
    sbp_regression, dbp_regression (that window's SegSBP/SegDBP), indexed by
    visit_id."""
    paths = PulseDBPaths(Path(root))
    cache: dict[str, dict] = {}
    rows = []
    for vid in visit_ids:
        subject_id, win_idx = _parse_visit_id(vid)
        if subject_id not in cache:
            path = paths.source_dir(source) / f"{subject_id}.{file_ext}"
            cache[subject_id] = _load_subject_windows(path, file_ext)
        windows = cache[subject_id]
        hr = _estimate_hr_from_ecg(windows["ecg"][win_idx], PULSEDB_FS)
        rows.append({
            "visit_id": vid,
            "hr_regression": hr,
            "sbp_regression": windows["sbp"][win_idx],
            "dbp_regression": windows["dbp"][win_idx],
        })
    return pd.DataFrame(rows).set_index("visit_id")
