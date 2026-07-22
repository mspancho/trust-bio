#!/usr/bin/env python
"""Fetch MIMIC-III-Ext-PPG via authenticated PhysioNet session (credentialed
access already confirmed granted). Mirrors the session-cookie login flow used
earlier in this project (CSRF token -> POST login -> cookie-authenticated GET),
downloading only metadata + a configurable subset of waveform files (the full
6.3M-segment waveform set is large; --max-patients bounds it for a first pass)."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path(
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iii-ext-ppg/"
    "physionet.org/files/mimic-iii-ext-ppg/1.1.0"
)
BASE_URL = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0/"


def _wget(url: str, dest: Path, user: str, password: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "--user", user, "--password", password, "-O", str(dest), url],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--max-patients", type=int, default=None,
                     help="limit to the first N patient folders from RECORDS "
                          "(omit to fetch all 6,188)")
    args = ap.parse_args()

    user = os.environ["PHYSIONET_USERNAME"]
    password = os.environ["PHYSIONET_PASSWORD"]

    args.root.mkdir(parents=True, exist_ok=True)
    for fname in ("README.md", "RECORDS", "metadata.csv", "SHA256SUMS.txt"):
        _wget(BASE_URL + fname, args.root / fname, user, password)
    print(f"[fetch_mimic_ext_ppg] metadata files written to {args.root}")

    records = (args.root / "RECORDS").read_text().splitlines()
    if args.max_patients:
        records = records[: args.max_patients]
    print(f"[fetch_mimic_ext_ppg] fetching waveforms for {len(records)} patient folders "
          f"(this reads metadata.csv to find each segment's signal_file_name)")

    import pandas as pd
    meta = pd.read_csv(args.root / "metadata.csv")
    for patient_dir in records:
        sub = meta[meta["folder_path"].str.startswith(patient_dir)]
        for _, row in sub.iterrows():
            for ext in (".hea", ".dat"):
                # folder_path already ends in "/" (see trustbio/data/mimic_ext_ppg.py);
                # the actual file is signal_file_name + ext under that folder.
                rel = row["folder_path"] + row["signal_file_name"] + ext
                url = BASE_URL + rel
                dest = args.root / rel
                if not dest.exists():
                    _wget(url, dest, user, password)
    print("[fetch_mimic_ext_ppg] done")


if __name__ == "__main__":
    main()
