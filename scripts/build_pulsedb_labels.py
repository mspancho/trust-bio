#!/usr/bin/env python
"""Stage 2b: build and cache the PulseDB label tables ONCE.

hr_regression is derived from each window's ECG, so this opens every subject
file: ~4.5 ms/window measured on the pilot, i.e. ~4.7 h for the full 3.79M-
window MIMIC cohort and ~2 h for Vital. Every eval stage then reads the cached
CSV in seconds. Extraction does not need labels (extract_features.py builds
its handle with with_labels=False), so this runs alongside the extraction
arrays rather than gating them.

    python scripts/build_pulsedb_labels.py --store features_cache --source mimic
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from trustbio.data.pulsedb import (
    DEFAULT_ROOT, build_pulsedb_cohort, build_pulsedb_label_table, label_cache_path,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--store", type=Path, required=True,
                    help="directory holding the cohort CSVs; label CSVs are written beside them")
    ap.add_argument("--source", choices=["mimic", "vital", "both"], default="both")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    for src in (["mimic", "vital"] if args.source == "both" else [args.source]):
        cohort = build_pulsedb_cohort(args.root, source=src, cache=args.store)
        ids = cohort.visits["visit_id"].astype(str).tolist()
        t0 = time.time()
        table = build_pulsedb_label_table(args.root, src, ids, cache=args.store, rebuild=args.rebuild)
        hr = table["hr_regression"].to_numpy(dtype=float)
        print(f"\n=== pulsedb_{src} labels ===")
        print(f"  windows      {len(table):,} (cohort {len(ids):,})")
        print(f"  built in     {(time.time() - t0) / 60:.1f} min")
        print(f"  median hr {np.nanmedian(hr):.1f} bpm, hr NaN {int(np.isnan(hr).sum()):,}")
        print(f"  median sbp {np.nanmedian(table['sbp_regression']):.1f} / "
              f"dbp {np.nanmedian(table['dbp_regression']):.1f} mmHg")
        print(f"  cache        {label_cache_path(args.store, src)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
