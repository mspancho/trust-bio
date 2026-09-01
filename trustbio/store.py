"""On-disk cache for visit-level feature vectors.

Feature extraction with the real FMs is expensive, so visit-level vectors are
cached per (model, modality, duration). The probing/reporting stages read from
this cache, decoupling the heavy GPU extraction from the cheap linear sweeps.

Layout:
    <root>/<model>/<modality>/<duration_sec>s/<split>.npz
    each .npz holds: visit_ids (str), features (float32, [n_visits, dim])
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, model: str, modality: str, duration_sec: int, split: str) -> Path:
        return self.root / model / modality / f"{duration_sec}s" / f"{split}.npz"

    def save(self, model, modality, duration_sec, split, visit_ids, features):
        path = self._path(model, modality, duration_sec, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so the final filename only ever names a complete
        # file: np.savez writes in place, and a writer killed mid-save (or two
        # writers racing) otherwise leaves a torn zip that np.load rejects
        # with BadZipFile -- observed when two array tasks shared one path.
        tmp = path.with_suffix(f".tmp-{os.getpid()}.npz")
        np.savez_compressed(
            tmp,
            visit_ids=np.asarray(visit_ids, dtype=str),
            features=np.asarray(features, dtype=np.float32),
        )
        tmp.replace(path)

    def load(self, model, modality, duration_sec, split):
        path = self._path(model, modality, duration_sec, split)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        return data["visit_ids"], data["features"]

    def exists(self, model, modality, duration_sec, split) -> bool:
        return self._path(model, modality, duration_sec, split).exists()
