"""Per-segment fault-taxonomy feature extraction.

Five features distinguish transient motion artifact, persistent lead-off, and
structural site/device shift without any diagnostic label:
  - sqi_value: mean signal-quality index over the segment (low = degraded).
  - sqi_drop_duration: longest contiguous run of low-SQI samples (short bursts
    suggest transient motion; sustained low-SQI suggests persistent lead-off).
  - accel_corr: correlation between the SQI-drop indicator and accelerometer
    magnitude, when an accelerometer channel is available (0.0 otherwise) —
    high correlation implicates motion as the cause of any quality drop.
  - source_db: the originating dataset/source-institution string, passed
    through as a categorical feature (structural shift is partly a property of
    *which* source a segment came from).
  - model_disagreement: |domain-FM prediction - time-series-FM prediction|,
    z-scored by `disagreement_scale` — segments where two models trained on
    the same data disagree sharply despite a clean SQI are the structural-
    shift signature the paper draft describes (Results: "quality indices ...
    also flag segments that are clean but out-of-distribution").
  - ecg_sqi_value / ppg_sqi_value: mean per-modality quality. Added because
    lead-off (flat ECG) and motion artifact (noisy PPG) share span lengths at
    equal injected severity and differ only in WHICH channel dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentFaultFeatures:
    sqi_value: float
    sqi_drop_duration: float
    accel_corr: float
    source_db: str
    model_disagreement: float
    ecg_sqi_value: float = float("nan")
    ppg_sqi_value: float = float("nan")


FEATURE_NAMES = ["sqi_value", "sqi_drop_duration", "accel_corr", "source_db",
                 "model_disagreement", "ecg_sqi_value", "ppg_sqi_value"]


_LOW_SQI_THRESHOLD = 0.5


def _longest_low_sqi_run(sqi_trace: np.ndarray) -> int:
    is_low = sqi_trace < _LOW_SQI_THRESHOLD
    if not is_low.any():
        return 0
    longest = current = 0
    for val in is_low:
        current = current + 1 if val else 0
        longest = max(longest, current)
    return longest


def _accel_sqi_correlation(sqi_trace: np.ndarray, accel_trace: np.ndarray | None) -> float:
    if accel_trace is None or len(accel_trace) != len(sqi_trace) or np.std(accel_trace) == 0:
        return 0.0
    is_low = (sqi_trace < _LOW_SQI_THRESHOLD).astype(float)
    if np.std(is_low) == 0:
        return 0.0
    corr = np.corrcoef(is_low, accel_trace)[0, 1]
    return float(0.0 if np.isnan(corr) else corr)


def extract_fault_features(
    sqi_trace: np.ndarray,
    accel_trace: np.ndarray | None,
    fs: int,
    source_db: str,
    model_a_pred: float,
    model_b_pred: float,
    disagreement_scale: float,
    ecg_sqi_trace: np.ndarray | None = None,
    ppg_sqi_trace: np.ndarray | None = None,
) -> SegmentFaultFeatures:
    """`sqi_trace` is the combined (min over modalities) per-second quality;
    the optional per-modality traces add which channel lost quality -- the
    only thing that separates a flat ECG electrode from a noisy PPG at equal
    span length."""
    sqi_trace = np.asarray(sqi_trace, dtype=float)
    return SegmentFaultFeatures(
        sqi_value=float(np.mean(sqi_trace)),
        sqi_drop_duration=float(_longest_low_sqi_run(sqi_trace)),
        accel_corr=_accel_sqi_correlation(sqi_trace, accel_trace),
        source_db=source_db,
        model_disagreement=float(abs(model_a_pred - model_b_pred) / disagreement_scale),
        ecg_sqi_value=float(np.mean(ecg_sqi_trace)) if ecg_sqi_trace is not None else float("nan"),
        ppg_sqi_value=float(np.mean(ppg_sqi_trace)) if ppg_sqi_trace is not None else float("nan"),
    )


def features_to_matrix(
    features: list[SegmentFaultFeatures], columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Numeric matrix over `columns` (default FEATURE_NAMES). `source_db` is
    encoded as an integer category code, ordered by sorted name."""
    names = list(columns) if columns is not None else list(FEATURE_NAMES)
    unknown = [c for c in names if c not in FEATURE_NAMES]
    if unknown:
        raise KeyError(f"unknown feature columns {unknown}; choose from {FEATURE_NAMES}")
    source_code = {s: i for i, s in enumerate(sorted({f.source_db for f in features}))}
    rows = []
    for f in features:
        values = {
            "sqi_value": f.sqi_value, "sqi_drop_duration": f.sqi_drop_duration,
            "accel_corr": f.accel_corr, "source_db": float(source_code[f.source_db]),
            "model_disagreement": f.model_disagreement,
            "ecg_sqi_value": f.ecg_sqi_value, "ppg_sqi_value": f.ppg_sqi_value,
        }
        rows.append([values[c] for c in names])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)), names
