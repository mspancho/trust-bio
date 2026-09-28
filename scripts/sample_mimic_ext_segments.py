#!/usr/bin/env python
"""Draw the MIMIC-III-Ext-PPG taxonomy sample: segments stratified by the
NATIVE signal-quality code of their first 10-s sub-window (the sub-window the
models see at duration_sec=10), spread across subjects, restricted to records
that actually contain lead II.

    clean     pleth_sqi0 == 1 and ecg_sqi0 == 1
    ppg_poor  pleth_sqi0 == 0
    ecg_poor  pleth_sqi0 == 1 and ecg_sqi0 <= 0   (0, -2, -3 ... undocumented negative codes)

Writes <out-dir>/mimic_ext_ppg_metadata.csv (the sampled rows, ALL columns,
so downstream handles never touch the 4.9 GB metadata.csv again) and
<out-dir>/mimic_ext_ppg_cohort.csv (subject-disjoint split + stratum columns).
Run as a batch job: reading metadata.csv in chunks needs ~10 GB and ~5 min.

    python scripts/sample_mimic_ext_segments.py --out-dir features_cache/taxonomy \
        --n-per-stratum 800 --max-per-subject 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from trustbio.data.mimic_ext_ppg import DEFAULT_ROOT, build_mimic_ext_ppg_cohort, first_sqi_code

STRATA = ("clean", "ppg_poor", "ecg_poor")


def _stratum(p0: float, e0: float) -> str | None:
    if np.isnan(p0) or np.isnan(e0):
        return None
    if p0 == 0:
        return "ppg_poor"
    if p0 == 1 and e0 == 1:
        return "clean"
    if p0 == 1 and e0 <= 0:
        return "ecg_poor"
    return None


def select_segments(meta_chunks, n_per_stratum: int, max_per_subject: int, seed: int) -> pd.DataFrame:
    """Reservoir-style selection over an iterable of metadata chunks: every row
    gets a random key; per stratum keep the smallest keys subject to the
    per-subject cap. Deterministic for a given seed and chunk order."""
    rng = np.random.default_rng(seed)
    pools = {s: [] for s in STRATA}
    for chunk in meta_chunks:
        chunk = chunk.copy()
        chunk["pleth_sqi0"] = chunk["vector_10s_pleth_sqi"].map(first_sqi_code)
        chunk["ecg_sqi0"] = chunk["vector_10s_ecg_sqi"].map(first_sqi_code)
        chunk["stratum"] = [_stratum(p, e) for p, e in zip(chunk["pleth_sqi0"], chunk["ecg_sqi0"])]
        chunk["_key"] = rng.random(len(chunk))
        for s in STRATA:
            part = chunk[chunk["stratum"] == s]
            if len(part):
                pools[s].append(part.nsmallest(20 * n_per_stratum, "_key"))
    out = []
    per_subject: dict = {}          # cap is GLOBAL across strata: spread subjects
    for s in STRATA:
        if not pools[s]:
            continue
        cand = pd.concat(pools[s]).sort_values("_key")
        keep = []
        for i, row in cand.iterrows():
            c = per_subject.get(row["subject_id"], 0)
            if c >= max_per_subject:
                continue
            per_subject[row["subject_id"]] = c + 1
            keep.append(i)
            if len(keep) >= n_per_stratum:
                break
        out.append(cand.loc[keep])
    result = pd.concat(out).drop(columns=["_key"]).reset_index(drop=True)
    assert result["signal_file_name"].is_unique
    return result


def has_lead_ii(root: Path, folder_path: str) -> bool:
    try:
        return "II" in [s.upper() for s in wfdb.rdheader(str(root / folder_path)).sig_name]
    except Exception:  # noqa: BLE001 -- unreadable header == unusable record
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-per-stratum", type=int, default=800)
    ap.add_argument("--max-per-subject", type=int, default=4)
    ap.add_argument("--oversample", type=float, default=1.5,
                    help="draw this many times n-per-stratum before the lead-II check")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = pd.read_csv(args.root / "metadata.csv", chunksize=args.chunksize, low_memory=False)
    picked = select_segments(chunks, int(args.n_per_stratum * args.oversample),
                             args.max_per_subject, args.seed)
    picked["has_ecg"] = [has_lead_ii(args.root, fp) for fp in picked["folder_path"]]
    picked = picked[picked["has_ecg"]].drop(columns=["has_ecg"])
    picked = picked.groupby("stratum", group_keys=False).head(args.n_per_stratum).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    picked.to_csv(args.out_dir / "mimic_ext_ppg_metadata.csv", index=False)
    cohort = build_mimic_ext_ppg_cohort(args.root, metadata_csv=picked, seed=args.seed).visits
    extra = picked.set_index(picked["signal_file_name"].astype(str))[["stratum", "pleth_sqi0", "ecg_sqi0"]]
    cohort = cohort.join(extra, on="visit_id")
    cohort.to_csv(args.out_dir / "mimic_ext_ppg_cohort.csv", index=False)
    print(f"strata: {cohort['stratum'].value_counts().to_dict()}  subjects: {cohort['subject_id'].nunique()}")
    print(f"splits: {cohort['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
