#!/usr/bin/env python
"""Stage 4c CLI: standard in-dataset linear-probe benchmark + reporting for
one dataset (not the cross-institution comparison — see run_transport_eval.py
for that). Useful for sanity-checking a single dataset/model pair before
running the full cross-institution or taxonomy analyses. Builds its
DatasetHandle the same way extract_features.py does (via _dataset_builders.py)
rather than depending on a pickle file, so it can run standalone against any
of the four dataset choices without a prior stage producing that pickle."""
from __future__ import annotations

import argparse
from pathlib import Path

from trustbio.config import available_models
from trustbio.eval.report import build_main_table, rank_models
from trustbio.pipeline import run_evaluation
from trustbio.store import FeatureStore

from _dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    add_dataset_root_args(ap)
    ap.add_argument("--store", required=True)
    ap.add_argument("--duration-sec", type=int, default=600)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = build_dataset_handle(args.dataset, args)
    print(f"{args.dataset} cohort: {dataset.cohort.counts}")

    store = FeatureStore(args.store)
    results_by_model = {}
    for model_name in available_models():
        out = run_evaluation(
            model_name, dataset, store, eval_split=args.eval_split,
            duration_sec=args.duration_sec,
        )
        if out is None:
            print(f"[skip] {model_name}: no cached features")
            continue
        results_by_model[model_name] = out["ecg_ppg_mean"]

    table = build_main_table(results_by_model)
    ranks = rank_models(table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path)
    ranks.to_csv(out_path.with_name(out_path.stem + "_ranks.csv"))
    print(table)
    print(ranks)


if __name__ == "__main__":
    main()
