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
) -> SegmentFaultFeatures:
    return SegmentFaultFeatures(
        sqi_value=float(np.mean(sqi_trace)),
        sqi_drop_duration=float(_longest_low_sqi_run(sqi_trace)),
        accel_corr=_accel_sqi_correlation(sqi_trace, accel_trace),
        source_db=source_db,
        model_disagreement=float(abs(model_a_pred - model_b_pred) / disagreement_scale),
    )


def features_to_matrix(
    features: list[SegmentFaultFeatures],
) -> tuple[np.ndarray, list[str]]:
    """Encode `source_db` as an integer category code (stable ordering by
    first appearance) alongside the four numeric features."""
    names = ["sqi_value", "sqi_drop_duration", "accel_corr", "source_db", "model_disagreement"]
    sources = [f.source_db for f in features]
    unique_sources = sorted(set(sources))
    source_code = {s: i for i, s in enumerate(unique_sources)}
    rows = [
        [f.sqi_value, f.sqi_drop_duration, f.accel_corr,
         float(source_code[f.source_db]), f.model_disagreement]
        for f in features
    ]
    return np.asarray(rows, dtype=np.float64), names
