#!/usr/bin/env python
"""Fetch PulseDB from the Box mirror (preferred over Google Drive).

PulseDB's README lists three official mirrors. We use **Box**, not Google Drive,
because Drive enforces a per-file anonymous download quota that is shared across
every downloader of a given public file worldwide -- during this project's first
PulseDB sample fetch, 6 of 9 files hit that quota and failed after retries, and
no amount of retrying fixes a Drive-side limit. Box instead serves the dataset
as 26 large multipart archives over plain HTTPS with byte-range support
(verified: HTTP 206 on a range request), so downloads are resumable and
quota-free.

Layout: 16 parts for PulseDB_MIMIC + 10 for PulseDB_Vital. Per the upstream
README, unzipping only the FIRST part of each set reconstructs the whole folder
(the remaining parts are picked up automatically by 7z as archive volumes).

    python scripts/fetch_pulsedb_box.py                  # download + extract
    python scripts/fetch_pulsedb_box.py --source mimic   # one source only
    python scripts/fetch_pulsedb_box.py --no-extract     # download only

Resumable: re-running skips already-complete parts and resumes partial ones
(`curl -C -`), so an interrupted job can simply be re-submitted.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_ROOT = Path("/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb")
README_URL = "https://raw.githubusercontent.com/pulselabteam/PulseDB/main/README.md"

# Parsed from the upstream README rather than hardcoded, so a mirror refresh
# upstream doesn't silently leave us fetching dead links.
_CURL_RE = re.compile(
    r'curl\s+-L\s+-o\s+"(?P<name>[^"]+)"\s+-C\s+-\s+'
    r'"(?P<url>https://rutgers\.box\.com/shared/static/[^"]+)"'
)

EXPECTED_PARTS = {"mimic": 16, "vital": 10}


def parse_parts(readme_text: str) -> list[tuple[str, str]]:
    """Return [(filename, url), ...] for every Box part in the README."""
    return [(m.group("name"), m.group("url")) for m in _CURL_RE.finditer(readme_text)]


def fetch_readme() -> str:
    with urllib.request.urlopen(README_URL, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def download_part(name: str, url: str, dest_dir: Path) -> bool:
    """curl one part with resume. Returns True on success."""
    dest = dest_dir / name
    cmd = [
        "curl", "-L", "-C", "-", "--fail", "--retry", "5",
        "--retry-delay", "10", "--retry-connrefused",
        "-o", str(dest), url,
    ]
    print(f"[pulsedb-box] {name} -> {dest}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[pulsedb-box]   FAILED (curl exit {proc.returncode}): {name}",
              file=sys.stderr, flush=True)
        return False
    return True


def extract(first_part: Path, out_dir: Path) -> bool:
    """Unzip a multipart set by pointing 7z at its FIRST part."""
    if shutil.which("7z") is None:
        print("[pulsedb-box] 7z not found; skipping extraction "
              "(download is complete -- extract manually)", file=sys.stderr)
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pulsedb-box] extracting {first_part.name} -> {out_dir}", flush=True)
    proc = subprocess.run(
        ["7z", "x", "-y", f"-o{out_dir}", str(first_part)],
        stdout=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        print(f"[pulsedb-box]   extraction FAILED for {first_part.name} "
              f"(7z exit {proc.returncode})", file=sys.stderr, flush=True)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--source", choices=["mimic", "vital", "both"], default="both")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    parts_dir = args.root / "_box_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    parts = parse_parts(fetch_readme())
    if not parts:
        print("[pulsedb-box] could not parse any Box URLs from the upstream README",
              file=sys.stderr)
        return 2

    wanted = []
    for name, url in parts:
        which = "mimic" if "MIMIC" in name else "vital"
        if args.source in ("both", which):
            wanted.append((which, name, url))

    counts = {k: sum(1 for w, _, _ in wanted if w == k) for k in ("mimic", "vital")}
    for src, n in counts.items():
        if n and n != EXPECTED_PARTS[src]:
            print(f"[pulsedb-box] WARNING: parsed {n} {src} parts, "
                  f"expected {EXPECTED_PARTS[src]} -- upstream may have changed",
                  file=sys.stderr)
    print(f"[pulsedb-box] {len(wanted)} parts to fetch "
          f"(mimic={counts['mimic']}, vital={counts['vital']})", flush=True)

    failed = [name for _, name, url in wanted if not download_part(name, url, parts_dir)]
    if failed:
        print(f"[pulsedb-box] {len(failed)} part(s) failed: {failed}", file=sys.stderr)
        print("[pulsedb-box] re-run to resume; extraction skipped while parts "
              "are missing", file=sys.stderr)
        return 1

    if args.no_extract:
        print(f"[pulsedb-box] downloaded {len(wanted)} parts to {parts_dir}; "
              "extraction skipped (--no-extract)")
        return 0

    seg = args.root / "Segment_Files"
    ok = True
    for src, first in (("mimic", "PulseDB_MIMIC.zip.001"),
                       ("vital", "PulseDB_Vital.zip.001")):
        if args.source in ("both", src) and (parts_dir / first).exists():
            ok &= extract(parts_dir / first, seg)

    for src, folder in (("mimic", "PulseDB_MIMIC"), ("vital", "PulseDB_Vital")):
        if args.source in ("both", src):
            d = seg / folder
            n = len(list(d.glob("*.mat"))) if d.exists() else 0
            print(f"[pulsedb-box] {folder}: {n} .mat files")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
