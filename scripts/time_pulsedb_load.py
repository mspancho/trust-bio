#!/usr/bin/env python
"""Measure per-file cost of PulseDB cohort building, to size it honestly.

build_pulsedb_cohort opens every subject file to read IncludeFlag; a 4h job
timed out before finishing 2,423 files, so measure the real rate on a small
sample and extrapolate instead of guessing at a wall limit.
"""
import glob, time, sys
import numpy as np
from trustbio.data.pulsedb import _load_subject_windows

d = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb/Segment_Files/PulseDB_MIMIC"
files = sorted(glob.glob(d + "/*.mat"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
step = max(1, len(files) // N)
sample = files[::step][:N]

times, wins, sizes = [], [], []
for f in sample:
    t0 = time.time()
    w = _load_subject_windows(__import__("pathlib").Path(f), "mat")
    dt = time.time() - t0
    times.append(dt); wins.append(len(w["include_flag"]))
    sizes.append(__import__("os").path.getsize(f) / 1e6)
    print(f"  {f.split('/')[-1]}: {dt:5.1f}s  {len(w['include_flag']):4d} windows  "
          f"{sizes[-1]:6.0f} MB", flush=True)
    del w

t = np.array(times)
print(f"\nper-file: mean {t.mean():.1f}s  median {np.median(t):.1f}s  max {t.max():.1f}s")
print(f"windows/file: mean {np.mean(wins):.0f}  total est {np.mean(wins)*len(files):,.0f}")
print(f"file size: mean {np.mean(sizes):.0f} MB")
print(f"\nEXTRAPOLATION for all {len(files)} files:")
print(f"  serial      : {t.mean()*len(files)/3600:.1f} h")
print(f"  16 parallel : {t.mean()*len(files)/3600/16:.1f} h")
