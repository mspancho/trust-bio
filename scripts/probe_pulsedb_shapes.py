#!/usr/bin/env python
"""Probe the real shape variation across PulseDB's per-subject .mat files.

Task 7 inspected only 2 sample files and recorded `(n_windows, 1, 1250)` for the
signal fields. That assumption is baked into `trustbio/data/pulsedb.py` as a
hardcoded `[:, 0, :]` index -- and it does NOT hold across the full 2,423-file
PulseDB_MIMIC set: loading the real cohort dies with

    IndexError: too many indices for array:
    array is 1-dimensional, but 3 were indexed

This script characterises what shapes actually occur, so the adapter can be
fixed against measured reality rather than another guess.

MUST run as a batch job with real memory: mat73.loadmat reads a whole .mat into
RAM and these average ~170 MB each (411 GB / 2,423 files), which OOM-kills the
4 GB interactive shell. Also note this file lives in the repo (shared storage),
not /tmp -- a compute node has its own local /tmp and will not see a script
written on the login node.

    sbatch --mem=32G --time=2:00:00 --wrap="... python scripts/probe_pulsedb_shapes.py"
"""
from __future__ import annotations

import argparse
import collections
import glob
import os

import numpy as np

SIGNAL_KEYS = ["ECG_Raw", "PPG_Raw", "ABP_Raw"]
META_KEYS = ["SegSBP", "SegDBP", "IncludeFlag", "SubjectID"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb/"
                "Segment_Files/PulseDB_MIMIC",
    )
    ap.add_argument("--n", type=int, default=150,
                    help="how many files to sample (stratified across the list)")
    args = ap.parse_args()

    import mat73

    files = sorted(glob.glob(os.path.join(args.dir, "*.mat")))
    print(f"total files in {args.dir}: {len(files)}", flush=True)
    if not files:
        return 1

    step = max(1, len(files) // args.n)
    sample = files[::step][: args.n]
    print(f"sampling {len(sample)} files (every {step}th)\n", flush=True)

    ndim_counts: collections.Counter = collections.Counter()
    combos: collections.Counter = collections.Counter()
    lastdim: collections.Counter = collections.Counter()
    odd: list = []
    errors: list = []

    for i, f in enumerate(sample, 1):
        base = os.path.basename(f)
        try:
            raw = mat73.loadmat(f)
            wins = raw[list(raw.keys())[0]]
            dims = {}
            for k in SIGNAL_KEYS + META_KEYS:
                if k not in wins:
                    dims[k] = None
                    continue
                a = np.asarray(wins[k])
                dims[k] = a.ndim
            ecg = np.asarray(wins["ECG_Raw"])
            ndim_counts[ecg.ndim] += 1
            lastdim[ecg.shape[-1] if ecg.ndim else None] += 1
            combos[tuple(dims[k] for k in SIGNAL_KEYS)] += 1
            nflag = int(np.asarray(wins["IncludeFlag"]).size)
            if ecg.ndim != 3:
                odd.append((base, ecg.ndim, tuple(ecg.shape), f"n_flag={nflag}"))
            del raw, wins
        except Exception as exc:  # noqa: BLE001 -- we want to catalogue every failure
            errors.append((base, f"{type(exc).__name__}: {exc}"[:90]))
        if i % 25 == 0:
            print(f"  ...{i}/{len(sample)}", flush=True)

    print(f"\n=== ECG_Raw ndim distribution ===\n{dict(ndim_counts)}")
    print(f"\n=== (ECG,PPG,ABP) ndim combos ===\n{dict(combos)}")
    print(f"\n=== ECG_Raw last-axis length (sample count?) ===\n{dict(lastdim)}")
    print(f"\n=== non-3D files: {len(odd)} ===")
    for o in odd[:15]:
        print("   ", o)
    print(f"\n=== load errors: {len(errors)} ===")
    for e in errors[:10]:
        print("   ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
