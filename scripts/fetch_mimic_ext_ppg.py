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
import time
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
    """GET one file, RESUMING a partial .part if present.

    PhysioNet honours byte ranges (verified: a range request on metadata.csv
    returns `content-range: bytes 0-0/4919648345`), and that matters a lot here
    -- metadata.csv alone is 4.92 GB and downloads at only ~5-11 MB/min, so a
    restart-from-zero cannot finish inside a 6-hour wall limit. Without Range
    support a timed-out job loops forever, re-fetching the same first gigabytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Already complete? Do NOT re-download. Without this the metadata loop
    # re-fetched metadata.csv (4.92 GB) and SHA256SUMS.txt (1.27 GB) on EVERY
    # run -- at PhysioNet's ~12 MB/min that burns hours before a single waveform
    # is touched, and then overwrites a known-good file. Observed: a fetch that
    # looked "stalled at 0 waveforms" for an hour was in fact 1.8 GB into
    # re-downloading a metadata.csv we already had complete on disk.
    if dest.exists() and dest.stat().st_size > 0:
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        with opener.open(req, timeout=300) as resp:
            # 206 = server honoured the range; 200 = it ignored it and is
            # sending the whole file, so we must start over rather than append.
            if have and resp.status == 200:
                print(f"[fetch_mimic_ext_ppg]   server ignored Range; restarting "
                      f"{dest.name}", flush=True)
                have = 0
            mode = "ab" if have else "wb"
            if have:
                print(f"[fetch_mimic_ext_ppg]   resuming {dest.name} at "
                      f"{have/1e9:.2f} GB", flush=True)
            with open(tmp, mode) as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
    except urllib.error.HTTPError as exc:
        # 416 = we already hold the whole file; treat as complete.
        if exc.code == 416 and have:
            tmp.replace(dest)
            return True
        print(f"[fetch_mimic_ext_ppg]   FAILED {url}: {exc}", file=sys.stderr, flush=True)
        return False
    except (urllib.error.URLError, OSError) as exc:
        # Keep the .part so the next run can resume from here.
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
    ap.add_argument("--shard", type=int, default=0,
                    help="this task's index within --n-shards (0-based)")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="split patient folders across N concurrent tasks. Each "
                         "shard fetches a disjoint slice, so tasks never race "
                         "for the same file.")
    ap.add_argument("--record-list", type=Path, default=None,
                    help="precomputed newline-separated record paths (see "
                         "scripts/build_mimicext_filelist.py). Strongly "
                         "preferred over re-parsing the 4.92 GB metadata.csv: "
                         "each shard doing that held ~4.25 GB RSS and burned "
                         "~10 min of CPU before downloading anything.")
    ap.add_argument("--throttle-ms", type=int, default=0,
                    help="sleep this long between file requests. Be a polite "
                         "client: physionet.org is shared infrastructure and "
                         "these are ~6.4M small files, so the load is request "
                         "rate, not bandwidth.")
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

    # Fast path: a precomputed record list. Each line is a full record path
    # (no extension), so no metadata parsing is needed at all.
    if args.record_list is not None:
        with open(args.record_list) as fh:
            rels = [ln.strip() for ln in fh if ln.strip()]
        if args.max_patients:
            # bound by PATIENT folder, not by record, so --max-patients keeps
            # its meaning: take whole folders until the limit is reached.
            seen, keep = set(), []
            for r in rels:
                folder = "/".join(r.split("/")[:2])
                if folder not in seen:
                    if len(seen) >= args.max_patients:
                        break
                    seen.add(folder)
                keep.append(r)
            rels = keep
        if args.n_shards > 1:
            rels = rels[args.shard :: args.n_shards]
        print(f"[fetch_mimic_ext_ppg] shard {args.shard}/{args.n_shards}: "
              f"{len(rels):,} records from {args.record_list.name}", flush=True)

        n_ok = n_skip = n_fail = 0
        for i, rel_base in enumerate(rels, 1):
            for ext in (".hea", ".dat"):
                rel = rel_base + ext
                dest = args.root / rel
                if dest.exists() and dest.stat().st_size > 0:
                    n_skip += 1
                    continue
                if download(opener, BASE_URL + rel, dest):
                    n_ok += 1
                else:
                    n_fail += 1
                if args.throttle_ms:
                    time.sleep(args.throttle_ms / 1000.0)
            if i % 200 == 0 or i == len(rels):
                print(f"[fetch_mimic_ext_ppg] {i:,}/{len(rels):,} records "
                      f"(downloaded {n_ok:,}, skipped {n_skip:,}, failed {n_fail:,})",
                      flush=True)
        print(f"[fetch_mimic_ext_ppg] done: {n_ok:,} downloaded, {n_skip:,} present, "
              f"{n_fail:,} failed")
        return 1 if n_fail else 0

    records = [r for r in (args.root / "RECORDS").read_text().splitlines() if r.strip()]
    if args.max_patients:
        records = records[: args.max_patients]
    if args.n_shards > 1:
        # Deterministic stride so shards are disjoint and every folder is
        # covered exactly once, with no coordination between tasks.
        records = records[args.shard :: args.n_shards]
        print(f"[fetch_mimic_ext_ppg] shard {args.shard}/{args.n_shards}: "
              f"{len(records):,} patient folders", flush=True)

    # Read ONLY the two columns we need. metadata.csv is 4.92 GB / 6.4M rows;
    # a full pd.read_csv of it inside a 16 GB job stalls for over an hour
    # without downloading a single byte (observed: two python procs pegged at
    # 0.0% CPU, ~18 MB RSS, no output past the startup banner). usecols keeps
    # the frame to the ~2 string columns actually referenced below.
    meta = pd.read_csv(
        args.root / "metadata.csv",
        usecols=["folder_path", "signal_file_name"],
        dtype={"folder_path": "string", "signal_file_name": "string"},
    )
    print(f"[fetch_mimic_ext_ppg] {len(records)} patient folders; "
          f"metadata.csv has {len(meta)} segments", flush=True)

    # Group segments by patient folder ONCE, rather than re-scanning all 6.4M
    # rows per patient. The old `meta[...str.startswith(patient_dir)]` inside
    # the loop was O(n_rows x n_patients) -- ~40 BILLION string comparisons for
    # 6.4M rows x 6,188 patients -- which is why the fetch sat for 75 minutes at
    # 0% CPU without downloading anything. RECORDS lines look like "p00/p000020/",
    # so the patient key is the folder_path prefix up to its second "/".
    wanted = {r.rstrip("/") + "/" for r in records}
    meta = meta.dropna(subset=["folder_path", "signal_file_name"])
    patient_key = (
        meta["folder_path"].astype(str)
        .str.split("/").str[:2].str.join("/") + "/"
    )
    by_patient = {
        k: v for k, v in meta.groupby(patient_key, sort=False) if k in wanted
    }
    print(f"[fetch_mimic_ext_ppg] {sum(len(v) for v in by_patient.values()):,} segments "
          f"across {len(by_patient):,} of {len(records):,} requested folders", flush=True)

    n_ok = n_skip = n_fail = 0
    for i, patient_dir in enumerate(records, 1):
        sub = by_patient.get(patient_dir.rstrip("/") + "/")
        if sub is None:
            continue
        for folder_path, signal_file_name in zip(
            sub["folder_path"].astype(str), sub["signal_file_name"].astype(str)
        ):
            for ext in (".hea", ".dat"):
                # VERIFIED against the live server: folder_path is the FULL
                # record path already (e.g. "p04/p044018/3000060_0002_0_2") and
                # signal_file_name repeats its last component, so the file is
                # simply folder_path + ext. Confirmed by HTTP status:
                #   p04/p044018/3000060_0002_0_2.hea                     -> 200
                #   p04/p044018/3000060_0002_0_23000060_0002_0_2.hea     -> 404
                #   p04/p044018/3000060_0002_0_2/3000060_0002_0_2.hea    -> 404
                rel = f"{folder_path}{ext}"
                dest = args.root / rel
                if dest.exists():
                    n_skip += 1
                    continue
                if download(opener, BASE_URL + rel, dest):
                    n_ok += 1
                else:
                    n_fail += 1
                if args.throttle_ms:
                    time.sleep(args.throttle_ms / 1000.0)
        if i % 10 == 0 or i == len(records):
            print(f"[fetch_mimic_ext_ppg] {i}/{len(records)} folders "
                  f"(downloaded {n_ok}, skipped {n_skip}, failed {n_fail})", flush=True)

    print(f"[fetch_mimic_ext_ppg] done: {n_ok} downloaded, {n_skip} already present, "
          f"{n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
