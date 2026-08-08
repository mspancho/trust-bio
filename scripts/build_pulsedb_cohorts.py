#!/usr/bin/env python
"""Build and cache the PulseDB cohorts once, so downstream stages don't rescan.

Building a cohort opens every subject file. With the h5py fast path (reading
only IncludeFlag rather than every signal array) this is far cheaper than the
13.0 s/file measured via mat73, but it is still a full pass over thousands of
files -- so do it ONCE here and let every later stage read the cached CSV.
"""
import argparse, time
from pathlib import Path
from trustbio.data.pulsedb import build_pulsedb_cohort, cohort_cache_path

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path,
                default=Path("/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb"))
ap.add_argument("--store", type=Path, required=True)
ap.add_argument("--source", choices=["mimic", "vital", "both"], default="both")
ap.add_argument("--rebuild", action="store_true")
a = ap.parse_args()

sources = ["mimic", "vital"] if a.source == "both" else [a.source]
for src in sources:
    t0 = time.time()
    c = build_pulsedb_cohort(a.root, source=src, cache=a.store, rebuild=a.rebuild)
    v = c.visits
    print(f"\n=== pulsedb_{src} ===")
    print(f"  built in       {(time.time()-t0)/60:.1f} min")
    print(f"  windows        {len(v):,}")
    print(f"  subjects       {v['subject_id'].nunique():,}")
    print(f"  windows/subj   {len(v)/max(v['subject_id'].nunique(),1):.0f}")
    print(f"  splits         {c.counts}")
    per = v.groupby('subject_id')['split'].nunique()
    print(f"  leakage check  subjects in >1 split = {(per>1).sum()} (must be 0)")
    print(f"  cache          {cohort_cache_path(a.store, src)}")
