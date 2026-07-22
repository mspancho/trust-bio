#!/usr/bin/env python
"""Fetch PulseDB into the lab's shared datasets directory.

PulseDB has no PhysioNet page. Its GitHub README (github.com/pulselabteam/
PulseDB) lists three official mirrors hosted directly by the PulseDB/Rutgers
team: Box, Google Drive, and OneDrive. A Kaggle "mirror" also exists, but it
is a third party's derived, non-canonical "Supplementary Subset" release
(CC-BY-NC-SA-4.0) -- trust-bio is a public MIT-licensed repo, so this script
deliberately uses the official Google Drive mirror instead:

    https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV

The folder layout (confirmed by listing, 2026-07-22) is:

    Info_Files/                        -- PulseDB_Info.mat, Train_Info.mat,
                                           AAMI_Cal_Info.mat, AAMI_Test_Info.mat,
                                           CalBased_Test_Info.mat, CalFree_Test_Info.mat
    Segment_Files/PulseDB_MIMIC/       -- one .mat per subject, e.g. p000160.mat
    Segment_Files/PulseDB_Vital/       -- one .mat per subject (VitalDB-derived)
    Subset_Files/                      -- (present, smaller pre-built subsets)
    Supplementary_Info_Files/          -- VitalDB_*_Info.mat variants
    LICENSE_PulseDB_MIMIC.txt, LICENSE_PulseDB_Vital.txt

This uses `gdown` (https://github.com/wkentaro/gdown), a pip package that can
list and download public Google Drive folders/files without OAuth.

IMPORTANT -- Google Drive rate limiting: Google enforces an anonymous
per-file download quota on public files that is shared across *everyone* who
downloads that file (not just this script or this machine). Heavily-used
public datasets like PulseDB routinely trip this quota, at which point
Google serves an HTML "Quota exceeded" / "Too many users have viewed or
downloaded this file recently" page instead of the actual file -- regardless
of tool (gdown, curl, browser) or account. This script retries transient
failures a few times, but a persistent quota error is a Google Drive-side
condition that this script cannot route around; when it happens repeatedly
across many distinct file IDs, it means the mirror is temporarily saturated
for anonymous access and the run should be retried later (sometimes it clears
within minutes, sometimes it takes longer per Google's own messaging, up to
~24h).

Usage:
    # Full dataset (real use -- large: thousands of per-subject .mat files):
    python scripts/fetch_pulsedb.py

    # Bounded sample for fast local verification (Info_Files + a handful of
    # MIMIC segment files only -- NOT the full dataset):
    python scripts/fetch_pulsedb.py --sample-only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gdown

DEFAULT_ROOT = Path("/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb")
FOLDER_URL = "https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV"

# How many per-subject .mat files to grab from Segment_Files/PulseDB_MIMIC
# when --sample-only is set. Kept small so this task's own verification stays
# fast; omit --sample-only to fetch everything for real use.
SAMPLE_SEGMENT_COUNT = 3

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10


def list_folder(url: str) -> list:
    """List every file in the PulseDB Drive folder (recursively) without
    downloading anything. Returns gdown.download_folder's
    GoogleDriveFileToDownload objects, each with .path (relative path inside
    the folder) and .id (Drive file id)."""
    return gdown.download_folder(url, skip_download=True, quiet=False)


def download_one(file_id: str, dest: Path) -> bool:
    """Download a single file by Drive id with retries. Returns True on
    success, False if all retries were exhausted (e.g. persistent Google
    Drive quota error) -- callers should treat False as a real failure to
    surface, not silently ignore."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = gdown.download(id=file_id, output=str(dest), quiet=False)
            if out is not None and dest.exists() and dest.stat().st_size > 0:
                return True
            last_err = RuntimeError("gdown.download returned no output")
        except Exception as exc:  # noqa: BLE001 -- we want to retry any gdown error
            last_err = exc
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[fetch_pulsedb]   attempt {attempt}/{MAX_RETRIES} failed "
                f"for {dest.name} ({last_err}); retrying in {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    print(
        f"[fetch_pulsedb]   FAILED after {MAX_RETRIES} attempts: {dest.name}: "
        f"{last_err}",
        file=sys.stderr,
    )
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--sample-only",
        action="store_true",
        help=(
            "Only fetch Info_Files/ plus a small handful of .mat files from "
            "Segment_Files/PulseDB_MIMIC/, instead of the full dataset. "
            "Omit this flag to fetch everything for real use."
        ),
    )
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_pulsedb] listing {FOLDER_URL}")
    try:
        entries = list_folder(FOLDER_URL)
    except Exception as exc:
        print(f"[fetch_pulsedb] FAILED to list Drive folder: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[fetch_pulsedb] folder listing returned {len(entries)} entries")

    if args.sample_only:
        selected = [e for e in entries if e.path.startswith("Info_Files/")]
        segment_entries = [
            e for e in entries if e.path.startswith("Segment_Files/PulseDB_MIMIC/")
        ]
        selected += segment_entries[:SAMPLE_SEGMENT_COUNT]
        print(
            f"[fetch_pulsedb] --sample-only: selected {len(selected)} files "
            f"(all of Info_Files/, first {SAMPLE_SEGMENT_COUNT} of "
            f"Segment_Files/PulseDB_MIMIC/)"
        )
    else:
        selected = entries
        print(f"[fetch_pulsedb] full fetch: selected all {len(selected)} files")

    n_ok, n_fail = 0, 0
    failures = []
    for entry in selected:
        dest = args.root / entry.path
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[fetch_pulsedb] skip (already present): {entry.path}")
            n_ok += 1
            continue
        print(f"[fetch_pulsedb] downloading {entry.path} (id={entry.id})")
        ok = download_one(entry.id, dest)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append(entry.path)

    print("\n[fetch_pulsedb] ===== summary =====")
    print(f"[fetch_pulsedb] root: {args.root}")
    for subdir in sorted({Path(e.path).parts[0] for e in selected}):
        n = len(list((args.root / subdir).rglob("*.mat"))) if (args.root / subdir).exists() else 0
        print(f"[fetch_pulsedb] {subdir}: {n} .mat files on disk")
    print(f"[fetch_pulsedb] ok={n_ok} fail={n_fail} (of {len(selected)} selected)")
    if failures:
        print("[fetch_pulsedb] failed files:", file=sys.stderr)
        for f in failures:
            print(f"[fetch_pulsedb]   - {f}", file=sys.stderr)
        print(
            "[fetch_pulsedb] NOTE: repeated failures across many distinct "
            "files usually mean Google Drive's anonymous per-file download "
            "quota is currently saturated for this folder (a Drive-side "
            "condition affecting all anonymous downloaders, not specific to "
            "this script/machine) -- retry later.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
