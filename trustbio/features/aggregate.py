"""Visit-level multimodal fusion.

Methods "Feature Extraction & Aggregation":
  * each modality is mean-pooled to a visit vector inside the extractor
    (FeatureExtractor.encode_modality),
  * late fusion = average of the ECG-only and PPG-only visit vectors, giving an
    ECG + PPG vector of the same dimensionality.
"""
from __future__ import annotations

import numpy as np

from ..config import FULL_DURATION_SEC
from .base import FeatureExtractor


def fuse_late(ecg_vec: np.ndarray, ppg_vec: np.ndarray) -> np.ndarray:
    """Late feature-level fusion: element-wise average of the two visit vectors."""
    return 0.5 * (ecg_vec + ppg_vec)


def build_visit_features(
    extractor: FeatureExtractor,
    ecg_raw: np.ndarray,
    ecg_fs: int,
    ppg_raw: np.ndarray,
    ppg_fs: int,
    duration_sec: int = FULL_DURATION_SEC,
) -> dict[str, np.ndarray]:
    """Produce ECG-only, PPG-only, and ECG+PPG visit vectors for one visit.

    `ecg_raw` / `ppg_raw` are the *raw* single-lead windows (at their native
    sampling rates); the extractor preprocesses, segments, encodes and pools
    each internally. Returns {"ecg", "ppg", "ecg_ppg_mean"}.
    """
    ecg_vec = extractor.encode_modality(ecg_raw, ecg_fs, "ecg", duration_sec)
    ppg_vec = extractor.encode_modality(ppg_raw, ppg_fs, "ppg", duration_sec)
    return {
        "ecg": ecg_vec,
        "ppg": ppg_vec,
        "ecg_ppg_mean": fuse_late(ecg_vec, ppg_vec),
    }
