#!/usr/bin/env python
"""Verify the real PulseDB cohort builds and loads. Batch job only (mat73 OOMs
a 4 GB shell; these files average ~170 MB each)."""
import argparse, collections, time
from pathlib import Path
import numpy as np
from trustbio.data.pulsedb import (
    build_pulsedb_cohort, build_pulsedb_label_table, make_pulsedb_signal_loader,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb")
ap.add_argument("--source", default="mimic")
a = ap.parse_args()

t0 = time.time()
cohort = build_pulsedb_cohort(Path(a.root), source=a.source)
build_s = time.time() - t0
v = cohort.visits
print(f"\ncohort built in {build_s/60:.1f} min")
print(f"windows(visits) = {len(v):,}")
print(f"subjects        = {v['subject_id'].nunique():,}")
print(f"splits          = {cohort.counts}")
print(f"windows/subject : mean {len(v)/v['subject_id'].nunique():.1f}")
per = v.groupby("subject_id")["split"].nunique()
print(f"subjects in >1 split = {(per>1).sum()}  (MUST be 0)")

load = make_pulsedb_signal_loader(Path(a.root), source=a.source)
ids = v["visit_id"].sample(min(60, len(v)), random_state=0).tolist()
shapes = collections.Counter()
t1 = time.time()
for vid in ids:
    e, fe = load(vid, "ecg"); p, fp = load(vid, "ppg")
    shapes[(len(e), fe, len(p), fp)] += 1
load_s = time.time() - t1
print(f"\nsignal shapes over {len(ids)} random windows: {shapes.most_common(3)}")
print(f"load time: {load_s:.1f}s for {len(ids)} windows "
      f"({load_s/len(ids)*1000:.0f} ms/window, cached by subject)")

labels = build_pulsedb_label_table(Path(a.root), a.source, ids)
print(f"\nlabel columns: {list(labels.columns)}")
for c in labels.columns:
    s = labels[c]
    print(f"  {c:16s} n={s.notna().sum():3d}  mean={s.mean():7.1f}  "
          f"range=[{s.min():.0f}, {s.max():.0f}]")
