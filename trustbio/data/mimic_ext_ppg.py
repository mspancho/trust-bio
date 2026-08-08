"""MIMIC-III-Ext-PPG adapter: quality-annotated ICU PPG+ECG with native rhythm
labels and pre-computed signal quality indices (confirmed schema — see Task 9
docstring in the implementation plan for the exact metadata.csv columns).

Rhythm classification is scoped to sinus rhythm (SR) vs. atrial fibrillation
(AF) — the two most prevalent labels in event_rhythm — with all other rhythms
(STACH, VPACE, SBRAD, etc.) treated as missing for this task, matching the
"rhythm_cls" task definition in config.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .cohort import Cohort, assert_no_subject_leakage, chronological_or_random_split

DEFAULT_ROOT = Path(
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iii-ext-ppg/"
    "physionet.org/files/mimic-iii-ext-ppg/1.1.0"
)


def _load_metadata(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "metadata.csv")


def build_mimic_ext_ppg_cohort(
    root: str | Path = DEFAULT_ROOT,
    metadata_csv: pd.DataFrame | None = None,
    seed: int = 0,
) -> Cohort:
    """One visit per segment (segment_id). Splits are subject-disjoint (by
    `subject_id`, which is the patient), random since strat_fold's own 10-fold
    assignment is reserved for the paper's own CV protocol rather than reused
    verbatim here. Passes through the native SQI columns unchanged so Task 13's
    fault taxonomy can read them directly from the cohort table."""
    root = Path(root)
    meta = metadata_csv if metadata_csv is not None else _load_metadata(root)
    df = meta[[
        "segment_id", "subject_id", "vector_10s_pleth_sqi", "vector_10s_ecg_sqi",
    ]].rename(columns={"segment_id": "visit_id"}).copy()
    df["split"] = chronological_or_random_split(
        df, subject_col="subject_id", time_col=None, seed=seed,
    )
    assert_no_subject_leakage(df)
    return Cohort(visits=df.reset_index(drop=True))


def make_mimic_ext_ppg_signal_loader(root: str | Path, metadata: pd.DataFrame | None = None):
    """SignalLoader closure: (visit_id, modality) -> (raw_signal, fs).

    Reads the WFDB record named by `signal_file_name` under `folder_path`;
    channel "II" is ECG, "PLETH" is PPG (confirmed WFDB sig_name convention,
    matching the same convention already used for MC-MED in signal-mcmed-msp).
    """
    root = Path(root)
    meta = metadata if metadata is not None else _load_metadata(root)
    meta = meta.set_index("segment_id")
    cache: dict[str, dict] = {}

    def load(visit_id: str, modality: str):
        if visit_id not in cache:
            row = meta.loc[visit_id]
            # `folder_path` is the FULL record path, not a directory: real values
            # look like "p04/p044018/3000060_0002_0_2", and `signal_file_name`
            # merely repeats its last component. Joining the two (as this did
            # originally) builds a doubled path that does not exist. Verified
            # against the live PhysioNet server by HTTP status:
            #   p04/p044018/3000060_0002_0_2.hea                  -> 200
            #   p04/p044018/3000060_0002_0_23000060_0002_0_2.hea  -> 404
            #   p04/p044018/3000060_0002_0_2/3000060_0002_0_2.hea -> 404
            # wfdb.rdrecord takes the path WITHOUT the .hea/.dat extension.
            rec_path = root / str(row["folder_path"])
            rec = wfdb.rdrecord(str(rec_path))
            sig_names = [s.upper() for s in rec.sig_name]
            ecg_idx = sig_names.index("II")
            ppg_idx = sig_names.index("PLETH")
            cache[visit_id] = {
                "ecg": np.asarray(rec.p_signal)[:, ecg_idx].astype(np.float32),
                "ppg": np.asarray(rec.p_signal)[:, ppg_idx].astype(np.float32),
                "fs": rec.fs,
            }
        entry = cache[visit_id]
        return entry[modality], entry["fs"]

    return load


_RHYTHM_MAP = {"SR": 0.0, "AF": 1.0}


def build_mimic_ext_ppg_label_table(
    metadata: pd.DataFrame, visit_ids: list[str],
) -> pd.DataFrame:
    """hr_regression from median_30s_hr; rhythm_cls from event_rhythm, scoped
    to SR (0) vs. AF (1), NaN for any other rhythm label."""
    meta = metadata.set_index("segment_id").reindex(visit_ids)
    out = pd.DataFrame(index=pd.Index(visit_ids, name="visit_id"))
    out["hr_regression"] = pd.to_numeric(meta["median_30s_hr"], errors="coerce").to_numpy()
    out["rhythm_cls"] = meta["event_rhythm"].map(_RHYTHM_MAP).astype(float).to_numpy()
    return out


def parse_sqi_vector(sqi_str: str) -> np.ndarray:
    """Parse a stringified SQI vector column (e.g. "[1, 1, 0]") into an array.
    Used by taxonomy/features.py (Task 13) to read the native SQI columns."""
    return np.asarray(ast.literal_eval(sqi_str), dtype=float)
