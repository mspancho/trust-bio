"""Hand-crafted domain-feature baselines (vendored from the original repo).

  * ECG: 54-d vector via the vendored `extract_ecg_feature` (NeuroKit2:
    morphology + HRV + SQI), 10-s segments @250 Hz.
  * PPG: 306-d vector via the vendored pyPPG `extract_ppg_features`, 60-s
    segments @250 Hz (pyPPG requires >= 20 s; 60 s chosen on val).

Both are extracted per segment and mean-pooled, exactly like the FM
representations. In the ECG+PPG setting the unimodal vectors are late-averaged
by the shared fusion logic (so ecg-domain and ppg-domain are separate models,
matching the original's separate extract-features scripts).
"""
from __future__ import annotations

import numpy as np

from ..config import ECG_DOMAIN_DIM, PPG_DOMAIN_DIM, FMSpec, PPG_DOMAIN_DEFAULT_SEC
from .base import FeatureExtractor
from ..vendored_utils import extract_ecg_feature, extract_ppg_features


class ECGDomainFeatures(FeatureExtractor):
    """Vendored NeuroKit2 ECG features, 10-s segments @250 Hz."""

    segment_sec = 10

    def load(self):
        return self

    def encode_segments(self, segments: np.ndarray, modality: str) -> np.ndarray:
        feats = []
        for seg in segments:
            try:
                f = extract_ecg_feature(seg.astype(np.float64), fs=self.sampling_freq)
            except Exception:
                f = np.zeros(ECG_DOMAIN_DIM, dtype=np.float32)
            feats.append(_fit_dim(f, ECG_DOMAIN_DIM))
        return np.stack(feats).astype(np.float32)


class PPGDomainFeatures(FeatureExtractor):
    """Vendored pyPPG features, >=20-s segments (60 s default) @250 Hz."""

    def __init__(self, spec: FMSpec, device: str = "cpu",
                 segment_sec: int = PPG_DOMAIN_DEFAULT_SEC):
        super().__init__(spec, device)
        self.segment_sec = segment_sec

    def load(self):
        try:
            import pyPPG  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"pyPPG required for PPG domain features ({e})")
        return self

    def encode_segments(self, segments: np.ndarray, modality: str) -> np.ndarray:
        feats = []
        for seg in segments:
            try:
                f = extract_ppg_features(seg.astype(np.float64), fs=self.sampling_freq)
            except Exception:
                f = np.zeros(PPG_DOMAIN_DIM, dtype=np.float32)
            feats.append(_fit_dim(f, PPG_DOMAIN_DIM))
        return np.stack(feats).astype(np.float32)


def _fit_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    """Coerce to exactly `dim` (defensive; the vendored extractors yield `dim`)."""
    vec = np.nan_to_num(np.asarray(vec, dtype=np.float32).ravel())
    if len(vec) == dim:
        return vec
    out = np.zeros(dim, dtype=np.float32)
    out[: min(dim, len(vec))] = vec[:dim]
    return out
