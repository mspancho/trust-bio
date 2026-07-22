#!/usr/bin/env python
"""Stage 4a CLI: cross-institution transportability eval (PulseDB MIMIC<->Vital)
for every available model, writing a long-form CSV of per-task, per-direction
scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trustbio.config import available_models
from trustbio.data.pulsedb import build_pulsedb_cohort, build_pulsedb_label_table
from trustbio.eval.transport import both_directions
from trustbio.store import FeatureStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulsedb-root", type=Path, required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--duration-sec", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mimic_cohort = build_pulsedb_cohort(args.pulsedb_root, source="mimic")
    vital_cohort = build_pulsedb_cohort(args.pulsedb_root, source="vital")
    mimic_labels = {
        s: build_pulsedb_label_table(args.pulsedb_root, "mimic", mimic_cohort.split(s)["visit_id"].tolist())
        for s in ("train", "val", "test")
    }
    vital_labels = {
        s: build_pulsedb_label_table(args.pulsedb_root, "vital", vital_cohort.split(s)["visit_id"].tolist())
        for s in ("train", "val", "test")
    }

    store = FeatureStore(args.store)
    all_records = []
    for model_name in available_models():
        for modality in ("ecg", "ppg", "ecg_ppg_mean"):
            try:
                records = both_directions(
                    model_name, modality, store, mimic_labels, vital_labels,
                    duration_sec=args.duration_sec,
                )
            except FileNotFoundError:
                print(f"[skip] {model_name}/{modality}: features not cached")
                continue
            for r in records:
                r["model"] = model_name
                r["modality"] = modality
            all_records.extend(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(out_path, index=False)
    print(f"wrote {len(all_records)} records to {out_path}")


if __name__ == "__main__":
    main()
