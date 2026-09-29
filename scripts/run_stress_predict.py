#!/usr/bin/env python
"""Stage 5a: degradation stress test, prediction pass for ONE model.

    python scripts/run_stress_predict.py --model xecg-10min --out-dir results/stress

Fits the transport-style probes (ridge; alpha on source val heart rate) on each
PulseDB institution's clean full-scale store with the taxonomy cohort's
subjects held out, then predicts the taxonomy cohort's windows under every
condition in the taxonomy store, for both PulseDB institutions (within- and
cross-source) and MIMIC-III-Ext-PPG (heart rate only). Writes
    <out-dir>/predictions_<model>.csv.gz   long: model, modality, source, target,
                                           condition, kind, severity, visit_id,
                                           task, y_true, y_pred
    <out-dir>/alphas_<model>.json          alpha / row counts per modality x source
One sbatch array task per model: scripts/run_stress.sbatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from trustbio.config import MODALITIES
from trustbio.data.mimic_ext_ppg import build_mimic_ext_ppg_label_table
from trustbio.eval.stress import TARGET_TASKS, fit_probes, load_pulsedb_labels, load_source, predict_cells

SOURCES = ("mimic", "vital")
TARGETS = ("pulsedb_mimic", "pulsedb_vital", "mimic_ext_ppg")


def taxonomy_subjects(taxonomy_cohort_cache: Path, source: str) -> set[str]:
    df = pd.read_csv(Path(taxonomy_cohort_cache) / f"pulsedb_{source}_cohort.csv", dtype=str)
    return set(df["subject_id"])


def target_labels(target: str, label_cache: Path, taxonomy_cohort_cache: Path) -> pd.DataFrame:
    if target.startswith("pulsedb_"):
        return load_pulsedb_labels(label_cache, target.split("_", 1)[1])
    cache = Path(taxonomy_cohort_cache)
    meta = pd.read_csv(cache / "mimic_ext_ppg_metadata.csv", low_memory=False)
    ids = pd.read_csv(cache / "mimic_ext_ppg_cohort.csv", dtype=str)["visit_id"].tolist()
    return build_mimic_ext_ppg_label_table(meta, ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--full-store", type=Path, default=Path("features_cache/full"))
    ap.add_argument("--label-cache", type=Path, default=Path("features_cache"))
    ap.add_argument("--taxonomy-store", type=Path, default=Path("features_cache/taxonomy_store"))
    ap.add_argument("--taxonomy-cohort-cache", type=Path, default=Path("features_cache/taxonomy"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=list(MODALITIES))
    ap.add_argument("--duration-sec", type=int, default=10)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = {t: target_labels(t, args.label_cache, args.taxonomy_cohort_cache) for t in args.targets}
    source_labels = {s: labels[f"pulsedb_{s}"] if f"pulsedb_{s}" in labels else load_pulsedb_labels(args.label_cache, s)
                     for s in args.sources}
    frames, alphas = [], {}
    for modality in args.modalities:
        for source in args.sources:
            t0 = time.time()
            src = load_source(args.full_store, source, args.model, modality, args.duration_sec, source_labels[source],
                              exclude_subjects=taxonomy_subjects(args.taxonomy_cohort_cache, source),
                              max_rows=args.max_train_rows, seed=args.seed)
            probes, alpha = fit_probes(src, seed=args.seed)
            n_train, n_excl = int(len(src["train"]["ids"])), src["train"]["n_excluded"]
            alphas[f"{modality}/{source}"] = dict(alpha=alpha, n_train=n_train, n_excluded_train=n_excl,
                                                  tasks=sorted(probes))
            print(f"[stress] {args.model}/{modality}/{source}: {len(probes)} probes on {n_train:,} rows "
                  f"(alpha={alpha:.4g}; {n_excl:,} taxonomy-subject rows held out) in {time.time() - t0:.0f}s",
                  flush=True)
            del src
            for target in args.targets:
                try:
                    frame = predict_cells(probes, args.taxonomy_store, target, args.model, modality,
                                          args.duration_sec, labels[target], TARGET_TASKS[target])
                except FileNotFoundError as exc:
                    print(f"[stress] skip {args.model}/{modality}/{source}->{target}: {exc}", flush=True)
                    continue
                frame.insert(0, "source", source)
                frame.insert(0, "modality", modality)
                frame.insert(0, "model", args.model)
                frames.append(frame)
                print(f"[stress]   -> {target}: {len(frame):,} predictions over "
                      f"{frame['condition'].nunique()} conditions", flush=True)
    if not frames:
        print("[stress] nothing predicted", flush=True)
        return 1
    out = pd.concat(frames, ignore_index=True)
    path = args.out_dir / f"predictions_{args.model}.csv.gz"
    out.to_csv(path, index=False)
    (args.out_dir / f"alphas_{args.model}.json").write_text(json.dumps(alphas, indent=2))
    print(f"[stress] wrote {len(out):,} rows -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
