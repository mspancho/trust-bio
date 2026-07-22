#!/usr/bin/env python
"""Build the extraction manifest: one line per (model, dataset) cell that is
currently available (real weights present, or --include-fallback set)."""
from __future__ import annotations

import argparse
from pathlib import Path

from trustbio.config import MAIN_TEST_MODELS, is_model_available

DATASETS = ["pulsedb_mimic", "pulsedb_vital", "mimic_ext_ppg", "but_ppg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = []
    for model in MAIN_TEST_MODELS:
        if not is_model_available(model):
            print(f"[manifest] {model}: unavailable, excluding from manifest")
            continue
        for dataset in DATASETS:
            lines.append(f"{model} {dataset} {args.duration}")

    Path(args.out).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} (model, dataset, duration) cells to {args.out}")


if __name__ == "__main__":
    main()
