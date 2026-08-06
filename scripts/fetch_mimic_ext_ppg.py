#!/usr/bin/env python
"""Fetch MIMIC-III-Ext-PPG from PhysioNet (credentialed access required).

PhysioNet serves credentialed datasets behind a **Django session login**, not
HTTP Basic auth. Verified against the live site: `--user/--password` (Basic)
returns 403 on `files/mimic-iii-ext-ppg/1.1.0/...` -- identically to sending no
credentials at all, i.e. Basic is ignored outright -- whereas a CSRF-token +
POST login followed by cookie-authenticated GETs returns 200 and real data.
An earlier version of this script used Basic auth and would have failed on its
very first file despite valid credentials.

Credentials come from the environment (PHYSIONET_USERNAME / PHYSIONET_PASSWORD);
put them in the gitignored `.env` at the repo root and source it:

    set -a; . ./.env; set +a
    python scripts/fetch_mimic_ext_ppg.py --max-patients 20

Credentials are never logged, echoed, or placed on a command line (which would
expose them via `ps`); they are POSTed once by an in-process urllib call.

The full waveform set is very large, so `--max-patients N` bounds a first pass.
Downloads are incremental: existing files are skipped, so an interrupted run can
simply be re-submitted.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ROOT = Path(
    "/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iii-ext-ppg/"
    "physionet.org/files/mimic-iii-ext-ppg/1.1.0"
)
BASE_URL = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0/"
LOGIN_URL = "https://physionet.org/login/"

METADATA_FILES = ("README.md", "RECORDS", "metadata.csv", "SHA256SUMS.txt")
_CSRF_RE = re.compile(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"')


def make_authenticated_opener(username: str, password: str):
    """Log in to PhysioNet and return a cookie-carrying urllib opener.

    Raises RuntimeError if the login does not appear to have succeeded, so a
    credential problem surfaces immediately rather than as thousands of silent
    403s partway through a long fetch.
    """
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "trust-bio/1.0 (research data fetch)")]

    with opener.open(LOGIN_URL, timeout=120) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    m = _CSRF_RE.search(page)
    if not m:
        raise RuntimeError("could not find a CSRF token on PhysioNet's login page")

    body = urllib.parse.urlencode({
        "csrfmiddlewaretoken": m.group(1),
        "username": username,
        "password": password,
    }).encode()
    req = urllib.request.Request(LOGIN_URL, data=body, headers={"Referer": LOGIN_URL})
    with opener.open(req, timeout=120) as resp:
        landed = resp.geturl()

    if any(c.name == "sessionid" for c in jar):
        return opener
    raise RuntimeError(
        "PhysioNet login did not yield a session cookie "
        f"(landed on {landed}). Check PHYSIONET_USERNAME / PHYSIONET_PASSWORD."
    )


def download(opener, url: str, dest: Path) -> bool:
    """GET one file. Returns True on success, False on a reported failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with opener.open(url, timeout=300) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"[fetch_mimic_ext_ppg]   FAILED {url}: {exc}", file=sys.stderr, flush=True)
        return False
    tmp.replace(dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--max-patients", type=int, default=None,
                    help="limit to the first N patient folders from RECORDS "
                         "(omit to fetch all)")
    ap.add_argument("--metadata-only", action="store_true",
                    help="fetch only README/RECORDS/metadata.csv/SHA256SUMS")
    args = ap.parse_args()

    try:
        username = os.environ["PHYSIONET_USERNAME"]
        password = os.environ["PHYSIONET_PASSWORD"]
    except KeyError as exc:
        print(f"[fetch_mimic_ext_ppg] missing env var {exc}; "
              "run `set -a; . ./.env; set +a` first", file=sys.stderr)
        return 2

    opener = make_authenticated_opener(username, password)
    print("[fetch_mimic_ext_ppg] PhysioNet session established", flush=True)

    args.root.mkdir(parents=True, exist_ok=True)
    for fname in METADATA_FILES:
        if not download(opener, BASE_URL + fname, args.root / fname):
            print(f"[fetch_mimic_ext_ppg] could not fetch {fname}; aborting",
                  file=sys.stderr)
            return 1
    print(f"[fetch_mimic_ext_ppg] metadata written to {args.root}", flush=True)

    if args.metadata_only:
        return 0

    import pandas as pd

    records = [r for r in (args.root / "RECORDS").read_text().splitlines() if r.strip()]
    if args.max_patients:
        records = records[: args.max_patients]
    meta = pd.read_csv(args.root / "metadata.csv")
    print(f"[fetch_mimic_ext_ppg] {len(records)} patient folders; "
          f"metadata.csv has {len(meta)} segments", flush=True)

    n_ok = n_skip = n_fail = 0
    for i, patient_dir in enumerate(records, 1):
        sub = meta[meta["folder_path"].astype(str).str.startswith(patient_dir)]
        for _, row in sub.iterrows():
            for ext in (".hea", ".dat"):
                # folder_path already ends in "/" (see trustbio/data/mimic_ext_ppg.py);
                # the file itself is signal_file_name + ext inside that folder.
                rel = f"{row['folder_path']}{row['signal_file_name']}{ext}"
                dest = args.root / rel
                if dest.exists():
                    n_skip += 1
                    continue
                if download(opener, BASE_URL + rel, dest):
                    n_ok += 1
                else:
                    n_fail += 1
        if i % 10 == 0 or i == len(records):
            print(f"[fetch_mimic_ext_ppg] {i}/{len(records)} folders "
                  f"(downloaded {n_ok}, skipped {n_skip}, failed {n_fail})", flush=True)

    print(f"[fetch_mimic_ext_ppg] done: {n_ok} downloaded, {n_skip} already present, "
          f"{n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
