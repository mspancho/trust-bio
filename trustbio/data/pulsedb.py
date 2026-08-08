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

# Matches the lab's shared dataset convention (see the on-disk locations table
# in the implementation plan) and Task 7's real fetch script default.
DEFAULT_ROOT = Path("/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb")


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
            "ecg": _as_window_matrix(data["ecg_raw"]),
            "ppg": _as_window_matrix(data["ppg_raw"]),
            "sbp": np.asarray(data["seg_sbp"], dtype=np.float64).ravel(),
            "dbp": np.asarray(data["seg_dbp"], dtype=np.float64).ravel(),
            "include_flag": np.asarray(data["include_flag"], dtype=bool).ravel(),
        }
    from mat73 import loadmat
    raw = loadmat(str(path))
    wins = raw["Subj_Wins"]
    return {
        "ecg": _as_window_matrix(wins["ECG_Raw"]),
        "ppg": _as_window_matrix(wins["PPG_Raw"]),
        "sbp": np.asarray(wins["SegSBP"], dtype=np.float64).ravel(),
        "dbp": np.asarray(wins["SegDBP"], dtype=np.float64).ravel(),
        "include_flag": np.asarray(wins["IncludeFlag"], dtype=bool).ravel(),
    }


def _as_window_matrix(arr) -> np.ndarray:
    """Normalise a PulseDB signal field to (n_windows, n_samples) float32.

    Real PulseDB is NOT uniformly 3-D, despite Task 7's two-file inspection
    suggesting `(n_windows, 1, 1250)`. Measured across a 150-file stratified
    sample of the real 2,423-file PulseDB_MIMIC set
    (scripts/probe_pulsedb_shapes.py):

        3-D (n_windows, 1, 1250) : 141 files
        1-D (1250,)              :   9 files   <- all with IncludeFlag size 1

    The 1-D files are SINGLE-WINDOW subjects: MATLAB/mat73 drops the leading
    singleton dimensions, so `(1, 1, 1250)` arrives as `(1250,)`. Hardcoding
    `[:, 0, :]` therefore raised
    `IndexError: too many indices for array` on ~6% of subjects, which is why
    the real cohort could not be built at all.

    Last-axis length was 1250 for every file sampled, so samples always live on
    the final axis and any middle axis is a squeezable channel dimension.
    """
    a = np.asarray(arr)
    if a.ndim == 1:            # single window, fully squeezed -> (1, n_samples)
        a = a[None, :]
    elif a.ndim == 3:          # (n_windows, 1, n_samples) -> drop channel axis
        a = a[:, 0, :]
    elif a.ndim != 2:          # 2-D is already (n_windows, n_samples)
        raise ValueError(
            f"unexpected PulseDB signal array with ndim={a.ndim}, shape={a.shape}"
        )
    return a.astype(np.float32)


def _parse_visit_id(visit_id: str) -> tuple[str, int]:
    """Split a visit_id like 'p000160_w3' back into ('p000160', 3)."""
    subject_id, _, win_part = visit_id.rpartition("_w")
    return subject_id, int(win_part)


#: Plausible human heart rate, used to reject spurious inter-beat intervals.
_HR_MIN_BPM, _HR_MAX_BPM = 25.0, 220.0


def _estimate_hr_from_ecg(ecg: np.ndarray, fs: int) -> float:
    """QRS-interval heart-rate estimate, validated against reference HR.

    The previous implementation (0.5-40 Hz band, `find_peaks` with only a
    distance constraint, MEAN inter-beat interval) produced labels containing NO
    heart-rate information. Measured against BUT PPG's 400 human-annotated
    good-quality records:

        old: mean 133.0 bpm, MAE 60.5 bpm, 7.0% within 10%, corr -0.023
        new: mean  73.2 bpm, MAE  1.0 bpm, 97.8% within 10%, corr +0.944
        (reference mean 72.5 bpm)

    and against PulseDB's own ABP pulse rate at its native 125 Hz:

        old: mean 134.6 bpm, MAE 65.9 bpm,   0% within 10%, corr +0.823
        new: mean  68.7 bpm, MAE  0.3 bpm, 100% within 10%, corr +0.994
        (ABP anchor mean 68.7 bpm)

    The old estimator's +0.823 correlation against ABP while reading ~2x too
    high is the signature of T-wave double-counting: it tracked heart rate but
    counted roughly two beats per cardiac cycle.

    Three changes, each load-bearing:
      * 5-15 Hz band isolates QRS energy; 0.5-40 Hz passed T-waves through.
      * a prominence threshold rejects T-waves and noise, where the old
        `find_peaks` accepted ANY local maximum as a beat.
      * MEDIAN of physiologically plausible intervals, so a handful of spurious
        peaks cannot drag the estimate, unlike the old mean over all intervals.
    """
    if len(ecg) < fs * 2:
        return float("nan")

    lo, hi = 5.0, min(15.0, fs / 2.0 - 0.1)   # keep below Nyquist (62.5 Hz @125)
    if hi <= lo:
        return float("nan")
    b, a = scipy.signal.butter(3, [lo, hi], btype="bandpass", fs=fs)
    filtered = scipy.signal.filtfilt(b, a, ecg)

    scale = np.std(filtered)
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    filtered = filtered / scale

    # abs(): R peaks invert depending on lead polarity, so match either sign.
    peaks, _ = scipy.signal.find_peaks(
        np.abs(filtered),
        distance=max(1, int(fs * 60.0 / _HR_MAX_BPM)),
        prominence=1.0,                        # in std units, post-normalisation
    )
    if len(peaks) < 2:
        return float("nan")

    ibi_sec = np.diff(peaks) / fs
    ibi_sec = ibi_sec[(ibi_sec > 60.0 / _HR_MAX_BPM) & (ibi_sec < 60.0 / _HR_MIN_BPM)]
    if len(ibi_sec) == 0:
        return float("nan")
    return float(60.0 / np.median(ibi_sec))


def _read_include_flag_only(path: Path, file_ext: str) -> np.ndarray:
    """Read ONLY the IncludeFlag of one subject file.

    Cohort building needs nothing but the QC flag, yet `_load_subject_windows`
    materialises every signal array too -- ~287 MB per file on the real data,
    measured at 13.0 s/file, which is why a full scan takes ~8.7 h per
    institution. mat73 has no lazy access, but h5py can read a single HDF5
    dataset without touching the rest of the file.
    """
    if file_ext == "npz":
        with np.load(path) as data:
            return np.asarray(data["include_flag"], dtype=bool).ravel()
    import h5py

    with h5py.File(path, "r") as f:
        # PulseDB v7.3 files are HDF5: /Subj_Wins/IncludeFlag
        grp = f[list(f.keys())[0]] if "Subj_Wins" not in f else f["Subj_Wins"]
        return np.asarray(grp["IncludeFlag"][()], dtype=bool).ravel()


def cohort_cache_path(store: str | Path, source: str) -> Path:
    """Where a built PulseDB cohort is cached (see `build_pulsedb_cohort`)."""
    return Path(store) / f"pulsedb_{source}_cohort.csv"


def build_pulsedb_cohort(
    root: str | Path, source: str, file_ext: str = "mat", seed: int = 0,
    cache: str | Path | None = None, rebuild: bool = False,
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

    CACHING (`cache=<dir>`): building this cohort means opening every subject
    file, which on the real data is ~2,423 files / 411 GB for MIMIC alone --
    measured at 8.7 h serial, and ~19 h for both institutions. Doing that once
    per pipeline stage is infeasible, so pass `cache` to write/read a CSV and
    pay the scan only once. `rebuild=True` forces a re-scan.
    """
    cache_file = cohort_cache_path(cache, source) if cache is not None else None
    if cache_file is not None and cache_file.exists() and not rebuild:
        df = pd.read_csv(cache_file, dtype={"visit_id": str, "subject_id": str})
        assert_no_subject_leakage(df)
        return Cohort(visits=df.reset_index(drop=True))

    paths = PulseDBPaths(Path(root))
    files = _list_subject_files(paths, source, file_ext)

    rows = []
    for f in files:
        subject_id = _subject_id_from_path(f)
        # Only the QC flag is needed here -- reading the signal arrays too would
        # move ~287 MB/file for nothing.
        include_flag = _read_include_flag_only(f, file_ext)
        for w in range(len(include_flag)):
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

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"[pulsedb] cached {len(df):,} windows / "
              f"{df['subject_id'].nunique():,} subjects -> {cache_file}")

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


def label_cache_path(store: str | Path, source: str) -> Path:
    """Where a built PulseDB label table is cached (see build_pulsedb_label_table)."""
    return Path(store) / f"pulsedb_{source}_labels.csv"


def build_pulsedb_label_table(
    root: str | Path, source: str, visit_ids: list[str], file_ext: str = "mat",
    cache: str | Path | None = None, rebuild: bool = False,
) -> pd.DataFrame:
    """Wide label table: hr_regression (derived from that window's ECG),
    sbp_regression, dbp_regression (that window's SegSBP/SegDBP), indexed by
    visit_id.

    CACHING (`cache=<dir>`): hr_regression is DERIVED, not stored -- every window
    needs its ECG loaded and peak-detected, which means opening each subject's
    ~287 MB .mat. Measured on the 100-subject pilot: 700 s for PulseDB_MIMIC and
    268 s for Vital, ~16 min total, and that cost would be repeated by EVERY task
    in an extraction array. Pass `cache` to compute once and reuse; a cached
    table is filtered to the requested visit_ids, and any visit missing from the
    cache triggers a full rebuild so a stale cache can't silently drop windows.
    """
    cache_file = label_cache_path(cache, source) if cache is not None else None
    if cache_file is not None and cache_file.exists() and not rebuild:
        cached = pd.read_csv(cache_file, dtype={"visit_id": str}).set_index("visit_id")
        wanted = pd.Index(visit_ids, name="visit_id")
        if wanted.isin(cached.index).all():
            return cached.reindex(wanted)
        print(f"[pulsedb] label cache {cache_file.name} is missing "
              f"{(~wanted.isin(cached.index)).sum():,} requested visits; rebuilding")

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
    if not rows:
        return pd.DataFrame(
            columns=["hr_regression", "sbp_regression", "dbp_regression"],
            index=pd.Index([], name="visit_id"),
            dtype="float64",
        )
    out = pd.DataFrame(rows).set_index("visit_id")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_file)
        print(f"[pulsedb] cached {len(out):,} labels -> {cache_file}")

    return out
