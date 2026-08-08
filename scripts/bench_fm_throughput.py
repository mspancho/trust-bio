#!/usr/bin/env python
"""Measure real FM inference throughput, to size feature extraction honestly.

Everything so far has been sized from my estimates, which have been wrong by up
to 20x. This benchmarks each available model on REAL PulseDB windows and reports
ms/window/model, so the sampling decision is arithmetic rather than guesswork.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from trustbio.config import available_models, FM_REGISTRY
from trustbio.features.registry import get_extractor
from trustbio.data.pulsedb import make_pulsedb_signal_loader

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", type=Path, required=True)
ap.add_argument("--root", type=Path,
                default=Path("/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb"))
ap.add_argument("--source", default="mimic")
ap.add_argument("--n-windows", type=int, default=64)
ap.add_argument("--n-subjects", type=int, default=4,
                help="spread the windows over this many subjects; the loader "
                     "caches per subject, so this mirrors real extraction")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

import torch
dev = a.device if torch.cuda.is_available() else "cpu"
print(f"device: {dev}"
      f"{' (' + torch.cuda.get_device_name(0) + ')' if dev=='cuda' else ''}", flush=True)

# Sample windows from a FEW SUBJECTS, not uniformly at random. The loader
# caches by subject, and a uniform sample of 64 windows out of 3.79M hits ~62
# distinct subjects -- defeating the cache and charging a fresh ~287 MB file
# read per window (measured: 18.9 s/window, which is a benchmark artifact, not
# the pipeline's real cost). Real extraction walks a subject's windows together.
_all = pd.read_csv(a.cohort, dtype=str)
_subjects = _all["subject_id"].drop_duplicates().head(a.n_subjects).tolist()
df = (_all[_all["subject_id"].isin(_subjects)]
      .groupby("subject_id", sort=False)
      .head(max(1, a.n_windows // max(1, len(_subjects)))))
print(f"sampling {len(df)} windows from {len(_subjects)} subjects "
      f"(subject-wise, matching real extraction order)", flush=True)
load = make_pulsedb_signal_loader(a.root, source=a.source)

print(f"loading {a.n_windows} real windows...", flush=True)
t0 = time.time()
sigs = []
for vid in df["visit_id"]:
    ecg, fs = load(vid, "ecg")
    sigs.append((np.asarray(ecg, dtype=np.float32), fs))
load_s = time.time() - t0
print(f"  {load_s:.1f}s ({load_s/len(sigs)*1000:.1f} ms/window, subject-cached)\n", flush=True)

models = available_models()
print(f"benchmarking {len(models)} models: {models}\n", flush=True)
rows = []
for m in models:
    try:
        ex = get_extractor(m, device=dev, allow_fallback=False).load()
        ex.encode_modality(sigs[0][0], sigs[0][1], "ecg", 10)      # warm up
        t0 = time.time()
        for sig, fs in sigs:
            ex.encode_modality(sig, fs, "ecg", 10)
        dt = time.time() - t0
        ms = dt / len(sigs) * 1000
        rows.append((m, ms))
        print(f"  {m:20s} {ms:8.2f} ms/window   ({dt:.1f}s for {len(sigs)})", flush=True)
        del ex
        if dev == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {m:20s} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)

if rows:
    total_ms = sum(ms for _, ms in rows)
    print(f"\nsum over {len(rows)} models: {total_ms:.1f} ms/window (1 modality)")
    for label, w in [("PulseDB both institutions", 5_245_454),
                     ("250 subj/inst", 1565*250 + 495*250),
                     ("500 subj/inst", 1565*500 + 495*500)]:
        h = w * total_ms / 1000 / 3600
        print(f"  {label:28s} {w:10,d} windows -> {h:7.1f} GPU-h  ({h/24:.1f} days)")
    print("\n(x3 if all modalities ecg/ppg/fusion are extracted separately)")
