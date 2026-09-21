#!/usr/bin/env python
"""Stage 3b CLI: merge one cell's chunk files into final per-split matrices.

Chunked extraction (extract_features.py --chunk/--n-chunks) leaves
<store>/<dataset>/<model>/<modality>/<dur>s/<split>.chunkXXofNN.npz files.
This concatenates them into <split>.npz -- refusing, for the WHOLE cell, if
any chunk of any modality/split is missing, so an unfinished extraction can
never be merged into a quietly truncated matrix. The merged visit_ids are
checked against the cohort's own order (no duplicates, nothing foreign, same
sequence) and the number of windows skipped upstream is reported.

    python scripts/merge_feature_chunks.py --store features_cache/full \
        --dataset pulsedb_mimic --model xecg-10min --duration-sec 10 \
        --cohort-cache features_cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trustbio.config import MODALITIES
from trustbio.pipeline import DatasetHandle
from trustbio.store import FeatureStore

if __package__:
    from ._dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle
else:
    from _dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle


def merge_cell(store_root: Path, dataset: DatasetHandle, model: str,
               duration_sec: int, keep_chunks: bool = False) -> int:
    """Merge every (modality, split) of one (model, dataset) cell. Returns 0
    when the cell is fully merged, 1 when it was refused as incomplete."""
    store = FeatureStore(Path(store_root) / dataset.name)

    # Refuse-before-write: check every (modality, split) first, so a cell is
    # merged entirely or not at all.
    problems = []
    for split in dataset.splits:
        for m in MODALITIES:
            if store.exists(model, m, duration_sec, split):
                continue
            n, found = store.chunk_files(model, m, duration_sec, split)
            if n is None:
                problems.append(f"{m}/{split}: no chunk files and no merged file")
                continue
            missing = sorted(set(range(n)) - set(found))
            if missing:
                problems.append(f"{m}/{split}: missing chunks {missing} of {n}")
    if problems:
        print(f"[merge] REFUSED {model}/{dataset.name}: cell incomplete", flush=True)
        for p in problems:
            print(f"    {p}", flush=True)
        return 1

    for split, df in dataset.splits.items():
        expected = df["visit_id"].astype(str).tolist()
        for m in MODALITIES:
            stats = store.merge_chunks(model, m, duration_sec, split,
                                       expected_ids=expected, remove=not keep_chunks)
            msg = f"[merge] {model}/{dataset.name}/{m}/{split}: rows={stats['rows']:,}"
            msg += " (already merged)" if stats["already_merged"] else f" from {stats['chunks']} chunks"
            frac = stats["skipped"] / max(len(expected), 1)
            msg += f", skipped upstream {stats['skipped']:,} ({frac:.2%})"
            if frac > 0.01:
                msg = "WARNING: " + msg
            print(msg, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    add_dataset_root_args(ap)
    ap.add_argument("--store", required=True,
                    help="BASE store root (features live under <store>/<dataset>/)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration-sec", type=int, default=600)
    ap.add_argument("--keep-chunks", action="store_true",
                    help="leave chunk files in place after merging")
    args = ap.parse_args()

    dataset = build_dataset_handle(args.dataset, args, with_labels=False)
    return merge_cell(Path(args.store), dataset, args.model, args.duration_sec,
                      keep_chunks=args.keep_chunks)


if __name__ == "__main__":
    sys.exit(main())
