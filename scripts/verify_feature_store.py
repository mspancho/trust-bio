#!/usr/bin/env python
"""Verify a merged feature store against its cohorts (stage 3c).

For every <store>/<dataset>/<model>/<modality>/<dur>s/<split>.npz: the row
count and visit_id ORDER must equal the cohort split, the feature dim must
match the registry, every value must be finite, rows must be distinct, and
the matrix must not be silently zero (the pilot once produced 155k x 54 of
pure zeros with green logs). ecg-domain's ppg column is exempt from the
distinct/non-zero checks: an ECG R-peak extractor on PPG legitimately bails.

Exit 1 if anything fails, so it can gate downstream jobs via
--dependency=afterok. Run it as a batch job -- the largest files are ~11 GB.

    python scripts/verify_feature_store.py --store features_cache/full \
        --cohort-cache features_cache --duration-sec 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trustbio.config import FM_REGISTRY, MODALITIES
from trustbio.store import FeatureStore


def check_file(store, model, modality, duration_sec, split, expected_ids, dim):
    ids, X = store.load(model, modality, duration_sec, split)
    problems = []
    if X.shape != (len(expected_ids), dim):
        problems.append(f"shape {X.shape} != ({len(expected_ids)}, {dim})")
    if ids.tolist() != expected_ids:
        problems.append("visit_ids differ from cohort order")
    if not np.isfinite(X).all():
        problems.append("non-finite values")
    exempt = model == "ecg-domain" and modality == "ppg"
    if not exempt and len(X):
        head = X[: min(200, len(X))]
        distinct = len({row.tobytes() for row in head})
        if distinct < 0.75 * len(head):
            problems.append(f"only {distinct}/{len(head)} distinct rows in the first {len(head)}")
        nz = np.count_nonzero(X[:5000]) / X[:5000].size
        if nz < 0.2:
            problems.append(f"only {nz:.1%} non-zero values (silent zero-fill?)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="BASE store root (features under <store>/<dataset>/)")
    ap.add_argument("--cohort-cache", type=Path, required=True,
                    help="directory holding pulsedb_<source>_cohort.csv")
    ap.add_argument("--duration-sec", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=["pulsedb_mimic", "pulsedb_vital"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    n_files, n_bad = 0, 0
    for ds in args.datasets:
        cohort = pd.read_csv(args.cohort_cache / f"{ds}_cohort.csv",
                             usecols=["visit_id", "split"], dtype=str)
        expected = {s: cohort.loc[cohort["split"] == s, "visit_id"].tolist() for s in args.splits}
        store = FeatureStore(Path(args.store) / ds)
        models = sorted(p.name for p in (Path(args.store) / ds).iterdir() if p.is_dir())
        for model in models:
            dim = FM_REGISTRY[model].feature_dim
            for modality in MODALITIES:
                for split in args.splits:
                    n_files += 1
                    try:
                        problems = check_file(store, model, modality, args.duration_sec,
                                              split, expected[split], dim)
                    except Exception as exc:  # noqa: BLE001 -- report and keep checking
                        problems = [f"{type(exc).__name__}: {exc}"]
                    tag = f"{ds}/{model}/{modality}/{split}"
                    if problems:
                        n_bad += 1
                        print(f"BAD  {tag}: " + "; ".join(problems), flush=True)
                    else:
                        print(f"ok   {tag}", flush=True)
    print(f"\nchecked {n_files} files, {n_bad} bad", flush=True)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
