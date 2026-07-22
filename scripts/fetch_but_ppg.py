#!/usr/bin/env python
"""Fetch BUT PPG (fully open, Creative Commons Attribution 4.0, no
credentialing) via authenticated PhysioNet session (login not required for
open-tier resources, but the same session-cookie flow works uniformly)."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path(
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/but-ppg/"
    "physionet.org/files/butppg/2.0.0"
)
BASE_URL = "https://physionet.org/files/butppg/2.0.0/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    # wget -r mirrors the whole open-access directory tree in one shot (no
    # credentials needed for this resource, unlike MIMIC-III-Ext-PPG).
    subprocess.run(
        ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=4",
         "-P", str(args.root), BASE_URL],
        check=True,
    )
    print(f"[fetch_but_ppg] mirrored to {args.root}")


if __name__ == "__main__":
    main()
