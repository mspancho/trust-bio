#!/usr/bin/env python
"""Stage 3 CLI: extract and cache visit-level features for one (model,
dataset) pair, optionally under a specified degradation condition."""
from __future__ import annotations

import argparse
from pathlib import Path

from trustbio.config import is_model_available
from trustbio.pipeline import extract_features_for_model
from trustbio.store import FeatureStore

if __package__:
    from ._dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle
else:
    from _dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    add_dataset_root_args(ap)
    ap.add_argument("--store", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration-sec", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--degrade-kind", default=None,
                     choices=[None, "motion_artifact", "lead_off", "missing_ppg"])
    ap.add_argument("--degrade-severity", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.allow_fallback and not is_model_available(args.model, args.checkpoint):
        print(f"[skip] {args.model}: no weights available. Skipping cleanly.")
        return

    dataset = build_dataset_handle(
        args.dataset, args, degrade_kind=args.degrade_kind,
        degrade_severity=args.degrade_severity, seed=args.seed,
    )
    print(f"{args.dataset} cohort: {dataset.cohort.counts}")

    # --store is the BASE cache root; each dataset gets its own subdirectory.
    # FeatureStore paths carry no dataset component, so two datasets written to
    # the same root silently share <model>/<modality>/<duration>s/<split>.npz --
    # concurrent array tasks then interleave or tear each other's files (this
    # happened: the 100-subject pilot produced split-mixed and BadZipFile npz).
    extract_features_for_model(
        args.model, dataset, FeatureStore(Path(args.store) / args.dataset),
        duration_sec=args.duration_sec, device=args.device,
        allow_fallback=args.allow_fallback, checkpoint=args.checkpoint,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
