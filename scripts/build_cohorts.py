#!/usr/bin/env python
"""Stage 1 CLI: build subject-disjoint cohorts for all three datasets and
cache them to disk (so downstream stages read a cached CSV instead of
re-scanning each dataset's raw files)."""
from __future__ import annotations

import argparse
from pathlib import Path

from trustbio.data.pulsedb import DEFAULT_ROOT as PULSEDB_ROOT, build_pulsedb_cohort
from trustbio.data.mimic_ext_ppg import DEFAULT_ROOT as MIMIC_EXT_PPG_ROOT, build_mimic_ext_ppg_cohort
from trustbio.data.but_ppg import DEFAULT_ROOT as BUT_PPG_ROOT, build_but_ppg_cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulsedb-root", type=Path, default=Path("/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb"))
    ap.add_argument("--mimic-ext-ppg-root", type=Path, default=MIMIC_EXT_PPG_ROOT)
    ap.add_argument("--but-ppg-root", type=Path, default=BUT_PPG_ROOT)
    ap.add_argument("--store", type=Path, required=True)
    args = ap.parse_args()
    args.store.mkdir(parents=True, exist_ok=True)

    mimic_cohort = build_pulsedb_cohort(args.pulsedb_root, source="mimic")
    mimic_cohort.visits.to_csv(args.store / "pulsedb_mimic_cohort.csv", index=False)
    print(f"PulseDB-MIMIC cohort: {mimic_cohort.counts}")

    vital_cohort = build_pulsedb_cohort(args.pulsedb_root, source="vital")
    vital_cohort.visits.to_csv(args.store / "pulsedb_vital_cohort.csv", index=False)
    print(f"PulseDB-Vital cohort: {vital_cohort.counts}")

    mimic_ext_ppg_cohort = build_mimic_ext_ppg_cohort(args.mimic_ext_ppg_root)
    mimic_ext_ppg_cohort.visits.to_csv(args.store / "mimic_ext_ppg_cohort.csv", index=False)
    print(f"MIMIC-III-Ext-PPG cohort: {mimic_ext_ppg_cohort.counts}")

    but_ppg_cohort = build_but_ppg_cohort(args.but_ppg_root)
    but_ppg_cohort.visits.to_csv(args.store / "but_ppg_cohort.csv", index=False)
    print(f"BUT PPG cohort: {but_ppg_cohort.counts}")


if __name__ == "__main__":
    main()
