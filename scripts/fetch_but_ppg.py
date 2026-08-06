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

# The URL path segments wget will recreate under its -P target, given that we
# let it build the default "<host>/<url path>" tree (see the note in main()).
_MIRROR_SUFFIX = ("physionet.org", "files", "butppg", "2.0.0")


def _mirror_base(root: Path) -> Path:
    """The directory to hand wget as -P so that its natural
    "<host>/<url path>" output tree lands exactly on `root`.

    For the default root this is .../datasets/but-ppg/ (i.e. `root` with the
    four `_MIRROR_SUFFIX` segments removed). If `root` doesn't end in that
    suffix (a caller passed some other layout), fall back to `root` itself so
    the download still succeeds -- just nested one level deeper -- rather than
    silently computing a wrong parent via blind `.parent` chaining.
    """
    parts = root.parts
    if len(parts) >= len(_MIRROR_SUFFIX) and parts[-len(_MIRROR_SUFFIX):] == _MIRROR_SUFFIX:
        return Path(*parts[: -len(_MIRROR_SUFFIX)])
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--quiet", action="store_true",
                    help="suppress wget's per-file progress output")
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    # Mirror the open-access directory tree (no credentials needed for this
    # resource, unlike MIMIC-III-Ext-PPG). BUT PPG stores each six-digit
    # recording in its own subdirectory, so the local tree MUST stay nested:
    #   <root>/100001/100001_PPG.{dat,hea}, ... plus the two root-level CSVs.
    #
    # We deliberately do NOT use -nH/--cut-dirs here. wget's default behaviour
    # rebuilds the natural "<host>/<url path>" tree under -P, which lands
    # exactly on DEFAULT_ROOT (.../but-ppg/physionet.org/files/butppg/2.0.0/).
    # An earlier version used "-nH --cut-dirs=4", which stripped one level too
    # many: every record subdirectory's listing page collapsed onto a single
    # <root>/index.html that overwrote itself while the crawler's in-memory URL
    # set grew, and the job was OOM-killed having saved zero data files.
    #   -np            : never ascend above 2.0.0/ (stay inside this dataset)
    #   -R index.html* : don't keep the directory-listing pages themselves
    #   -N -c          : timestamp/resume, so re-running is incremental
    mirror_base = _mirror_base(args.root)
    cmd = [
        "wget", "-r", "-N", "-c", "-np",
        "-R", "index.html*",
        "--no-verbose" if args.quiet else "--progress=dot:giga",
        "-P", str(mirror_base),
        BASE_URL,
    ]
    print(f"[fetch_but_ppg] mirror base: {mirror_base}")
    print(f"[fetch_but_ppg] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    n_records = len([p for p in args.root.glob("*/") if p.is_dir()])
    print(f"[fetch_but_ppg] mirrored to {args.root} ({n_records} record dirs)")


if __name__ == "__main__":
    main()
