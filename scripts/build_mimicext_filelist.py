#!/usr/bin/env python
"""Precompute the MIMIC-III-Ext-PPG download list ONCE, as a compact manifest.

Every fetch shard previously called pd.read_csv on the 4.92 GB metadata.csv to
recover two columns. Measured consequence: each shard held ~4.25 GB resident,
and because slurm packed four shards onto one node, that was ~17 GB of identical
data on a single host before a byte was downloaded -- and ~10 minutes of CPU per
shard just to parse.

The needed information is tiny: one relative record path per line. Write it once
here (~6.4M short lines, a few hundred MB as text, read lazily line-by-line by
each shard), so shards start downloading immediately at ~50 MB RSS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iii-ext-ppg/"
    "physionet.org/files/mimic-iii-ext-ppg/1.1.0"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    print(f"reading {a.root/'metadata.csv'} (one pass)", flush=True)
    meta = pd.read_csv(
        a.root / "metadata.csv",
        usecols=["folder_path"],
        dtype={"folder_path": "string"},
    ).dropna()

    # folder_path is the FULL record path (verified against the live server:
    # "<folder_path>.hea" -> 200, the folder_path/signal_file_name join -> 404).
    paths = meta["folder_path"].astype(str).drop_duplicates()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"wrote {len(paths):,} record paths -> {a.out} "
          f"({a.out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
