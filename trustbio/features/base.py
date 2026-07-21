"""Common interface for all frozen feature extractors.

Every FM (and the domain-feature baselines) is wrapped as a `FeatureExtractor`.
The extractor owns the full per-visit recipe from the original SignalMC-MED
scripts: preprocess the raw ECG/PPG (resample to the model's expected fs, clean,
z-normalise via the vendored utils), split into the model's segment length,
encode each segment, and mean-pool to a visit vector. Models are used strictly
as frozen extractors — no fine-tuning.

Subclasses normally only implement `load()` and `encode_segments()`; the shared
`encode_visit()` handles preprocessing, segmentation, aggregation, and late
ECG+PPG fusion identically across models (matching the reference code).
"""
from __future__ import annotations

import abc

import numpy as np

from ..config import FMSpec, SEGMENT_SEC
from ..vendored_utils import get_preprocess_fns


class FeatureExtractor(abc.ABC):
    """Frozen feature extractor for one FM / baseline.

    The same extractor is applied to both ECG and PPG inputs regardless of the
    model's pretraining modality (Methods: cross-modality generalisation).
    """

    # Segment length (seconds) fed to the model. PPG domain features override
    # this to 60 s (pyPPG needs >= 20 s); everything else uses 10 s.
    segment_sec: int = SEGMENT_SEC

    def __init__(self, spec: FMSpec, device: str = "cpu"):
        self.spec = spec
        self.device = device

    @property
    def feature_dim(self) -> int:
        return self.spec.feature_dim

    @property
    def sampling_freq(self) -> int:
        return self.spec.sampling_freq

    @property
    def long_input(self) -> bool:
        return self.spec.long_input

    @abc.abstractmethod
    def load(self) -> "FeatureExtractor":
        """Load weights / build the model. Returns self for chaining."""

    @abc.abstractmethod
    def encode_segments(self, segments: np.ndarray, modality: str) -> np.ndarray:
        """Encode `segments` of shape (n_segments, seg_len) at this model's
        sampling frequency. Returns (n_segments, feature_dim)."""

    def encode_long(self, signal: np.ndarray, modality: str) -> np.ndarray:
        """Encode an entire long signal into a single (1, feature_dim) vector.

        Only used by long-input variants (e.g. xECG-10min). Default encodes the
        whole signal as one segment.
        """
        return self.encode_segments(signal[None, :], modality)

    # ------------------------------------------------------------------ #
    # Shared per-visit orchestration (matches the original extract scripts)
    # ------------------------------------------------------------------ #
    def preprocess(self, raw: np.ndarray, fs_in: int, modality: str) -> np.ndarray:
        """Vendored resample->clean->z-norm to this model's expected fs.

        Returns a 1-D float32 array of length duration*fs_out.
        """
        pp_ecg, pp_ppg = get_preprocess_fns(self.sampling_freq)
        fn = pp_ecg if modality == "ecg" else pp_ppg
        out = fn(raw, fs_in)          # (1, n) per the vendored implementation
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _segment(self, signal: np.ndarray, duration_sec: int) -> np.ndarray:
        """Split the first `duration_sec` of `signal` into non-overlapping
        `segment_sec` windows -> (n_segments, segment_sec*fs)."""
        fs = self.sampling_freq
        seg_len = self.segment_sec * fs
        n_total = duration_sec * fs
        usable = signal[:n_total]
        n = len(usable) // seg_len
        if n == 0:
            raise ValueError("signal shorter than one segment")
        return usable[: n * seg_len].reshape(n, seg_len)

    def encode_modality(self, raw: np.ndarray, fs_in: int, modality: str,
                        duration_sec: int) -> np.ndarray:
        """Raw signal -> mean-pooled visit feature vector for one modality."""
        sig = self.preprocess(raw, fs_in, modality)
        if self.long_input:
            seg = sig[: duration_sec * self.sampling_freq]
            feats = self.encode_long(seg, modality)        # (1, dim)
            return np.asarray(feats, dtype=np.float32).reshape(-1)
        segs = self._segment(sig, duration_sec)
        seg_feats = self.encode_segments(segs, modality)   # (n, dim)
        return np.asarray(seg_feats, dtype=np.float32).mean(axis=0)
