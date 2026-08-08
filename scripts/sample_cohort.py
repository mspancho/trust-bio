#!/usr/bin/env python
"""Draw a reproducible SUBJECT-level sample from a cached cohort.

Sampling happens at the SUBJECT level, never the window level: every window of a
chosen subject is kept, and no subject is split across train/val/test. Sampling
windows independently would leak the same subject into multiple splits and
inflate every score.

The full cohort CSV is left untouched, so the same cache serves both a pilot and
the eventual full run.

    python scripts/sample_cohort.py \
        --cohort features_cache/pulsedb_mimic_cohort.csv \
        --out    features_cache/pilot100/pulsedb_mimic_cohort.csv \
        --n-subjects 100 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_subjects(df: pd.DataFrame, n_subjects: int, seed: int) -> pd.DataFrame:
    """Keep all windows of `n_subjects` randomly chosen subjects.

    Samples subjects WITHIN each split, so the pilot preserves the full cohort's
    train/val/test structure instead of accidentally drawing (say) only training
    subjects and leaving the eval splits empty.
    """
    if "split" not in df.columns:
        chosen = (
            df["subject_id"].drop_duplicates()
            .sample(n=min(n_subjects, df["subject_id"].nunique()), random_state=seed)
        )
        return df[df["subject_id"].isin(set(chosen))]

    # Proportional allocation across splits, so split ratios survive sampling.
    per_split = df.groupby("split")["subject_id"].nunique()
    total = per_split.sum()
    keep: set[str] = set()
    for split, n_in_split in per_split.items():
        want = max(1, round(n_subjects * n_in_split / total))
        subs = (
            df.loc[df["split"] == split, "subject_id"].drop_duplicates()
            .sample(n=min(want, n_in_split), random_state=seed)
        )
        keep.update(subs)
    return df[df["subject_id"].isin(keep)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-subjects", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = pd.read_csv(a.cohort, dtype={"visit_id": str, "subject_id": str})
    out = sample_subjects(df, a.n_subjects, a.seed)

    # A subject must never appear in more than one split.
    leaked = out.groupby("subject_id")["split"].nunique()
    n_leaked = int((leaked > 1).sum()) if len(leaked) else 0
    if n_leaked:
        raise SystemExit(f"ERROR: {n_leaked} subjects span multiple splits")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    print(f"{a.cohort.name}: {df['subject_id'].nunique():,} subjects / {len(df):,} windows")
    print(f"  -> sampled {out['subject_id'].nunique():,} subjects / {len(out):,} windows "
          f"({len(out)/max(len(df),1)*100:.1f}%)")
    print(f"  splits: {out['split'].value_counts().to_dict()}")
    print(f"  leakage check: subjects in >1 split = {n_leaked} (must be 0)")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
