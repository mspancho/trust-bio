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
import re
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

    # ------------------------------------------------------------------ #
    # Chunked extraction. A full-scale (model, dataset) cell is millions of  #
    # windows and 15-45 GPU-hours (ecg-domain on CPU: ~110 h) -- no single   #
    # wall-clock window survives that, and the extraction loop only saves at #
    # the end of a split. So a split is cut into contiguous chunks, each an  #
    # array task that saves <split>.chunkXXofNN.npz; a resubmitted task      #
    # skips chunks already on disk, and merge_chunks() assembles the final   #
    # <split>.npz only once every chunk is present.                          #
    # ------------------------------------------------------------------ #
    _CHUNK_RE = re.compile(r"^(?P<split>.+)\.chunk(?P<i>\d{2,})of(?P<n>\d{2,})\.npz$")

    @staticmethod
    def chunk_split(split: str, chunk: int, n_chunks: int) -> str:
        """Split name under which chunk `chunk` of `split` is stored, e.g.
        "train.chunk03of08"."""
        if n_chunks < 1 or not (0 <= chunk < n_chunks):
            raise ValueError(f"chunk {chunk} out of range for n_chunks={n_chunks}")
        return f"{split}.chunk{chunk:02d}of{n_chunks:02d}"

    def chunk_files(self, model, modality, duration_sec, split):
        """(n_chunks, {chunk_index: path}) for one split's chunk files;
        (None, {}) when there are none."""
        d = self._path(model, modality, duration_sec, split).parent
        found: dict[int, Path] = {}
        n_chunks = None
        if not d.exists():
            return None, found
        for p in d.iterdir():
            m = self._CHUNK_RE.match(p.name)
            if not m or m.group("split") != split:
                continue
            n = int(m.group("n"))
            if n_chunks is None:
                n_chunks = n
            elif n != n_chunks:
                raise ValueError(f"mixed chunk counts under {d}: {n_chunks} vs {n} ({p.name})")
            found[int(m.group("i"))] = p
        return n_chunks, found

    def merge_chunks(self, model, modality, duration_sec, split,
                     expected_ids=None, remove: bool = True) -> dict:
        """Concatenate a split's chunk files into the final <split>.npz.

        Refuses (FileNotFoundError) unless every chunk 0..n-1 is present, so
        an unfinished cell can never become a quietly truncated matrix. With
        `expected_ids` (the cohort's visit order for this split) the merged ids
        must be a subsequence of it -- same order, no duplicates, nothing
        foreign -- and the count of windows skipped upstream is reported.
        Chunk files are deleted only after the merged file has been re-read
        and its row count verified.
        """
        final = self._path(model, modality, duration_sec, split)
        expected = None if expected_ids is None else np.asarray(expected_ids, dtype=str)
        n_chunks, found = self.chunk_files(model, modality, duration_sec, split)
        if n_chunks is None:
            if final.exists():
                ids, _ = self.load(model, modality, duration_sec, split)
                skipped = None if expected is None else int(len(expected) - len(ids))
                return dict(rows=len(ids), chunks=0, already_merged=True, skipped=skipped)
            raise FileNotFoundError(f"no chunk files and no merged file for {final}")
        missing = sorted(set(range(n_chunks)) - set(found))
        if missing:
            raise FileNotFoundError(
                f"{final.parent}/{split}: {len(missing)} of {n_chunks} chunks missing: {missing}"
            )

        ids_parts, feat_parts = [], []
        for i in range(n_chunks):
            data = np.load(found[i], allow_pickle=False)
            ids_parts.append(np.asarray(data["visit_ids"], dtype=str))
            feat_parts.append(data["features"])
        dims = {f.shape[1] for f in feat_parts if f.ndim == 2}
        if len(dims) > 1:
            raise ValueError(f"inconsistent feature dims across chunks: {sorted(dims)}")
        ids = np.concatenate(ids_parts)
        feats = np.concatenate(feat_parts, axis=0)
        del ids_parts, feat_parts

        skipped = None
        if expected is not None:
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"duplicate visit_ids in merged {split}")
            foreign = np.setdiff1d(ids, expected)
            if len(foreign):
                raise ValueError(
                    f"{len(foreign)} merged visit_ids are not in the cohort split "
                    f"(e.g. {foreign[:3].tolist()})"
                )
            if not np.array_equal(expected[np.isin(expected, ids)], ids):
                raise ValueError(f"merged {split} rows are not in cohort order")
            skipped = int(len(expected) - len(ids))

        self.save(model, modality, duration_sec, split, ids, feats)
        check_ids, check_feats = self.load(model, modality, duration_sec, split)
        if len(check_ids) != len(ids) or check_feats.shape != feats.shape:
            raise RuntimeError(f"verification failed after merging {final}")
        if remove:
            for p in found.values():
                p.unlink()
        return dict(rows=len(ids), chunks=n_chunks, already_merged=False, skipped=skipped)
