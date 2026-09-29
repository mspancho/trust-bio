#!/usr/bin/env python
"""Stage 4b-ii: fault-taxonomy clustering, validation and assignment (Table 3).

Reads results/taxonomy/fault_features.csv (scripts/build_fault_features.py).
Fits KMeans(k=3) on the rows whose known condition is motion_artifact /
lead_off / structural, names clusters by majority vote, and reports how often
each condition lands in its own cluster -- overall, by dataset, by severity,
with subject-bootstrap CIs -- for four feature sets (the full set and three
ablations, because source_db makes the structural class partly circular).
Every other row (clean controls, MIMIC-ext natural degradation, BUT PPG real
motion / consumer-clean) is assigned to the nearest fitted centroid.

    python scripts/run_taxonomy.py --features-csv results/taxonomy/fault_features.csv --out-dir results/taxonomy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trustbio.taxonomy.cluster import (
    bootstrap_recall, condition_recall, confusion_against_known_conditions, fit_fault_clusters, silhouette,
)
from trustbio.taxonomy.features import FEATURE_NAMES

FIT_CONDITIONS = ("motion_artifact", "lead_off", "structural")
FEATURE_SETS = {
    "all": list(FEATURE_NAMES),
    "no_source_db": [f for f in FEATURE_NAMES if f != "source_db"],
    "no_disagreement": [f for f in FEATURE_NAMES if f != "model_disagreement"],
    "sqi_only": ["sqi_value", "sqi_drop_duration", "ecg_sqi_value", "ppg_sqi_value"],
}


def _matrix(df: pd.DataFrame, columns: list[str], source_codes: dict[str, int]) -> np.ndarray:
    X = df[[c for c in columns]].copy()
    if "source_db" in columns:
        X["source_db"] = df["source_db"].map(source_codes).astype(float)
    return X.to_numpy(dtype=float)


def _recall_rows(feature_set, fit, assigned, n_boot, seed) -> list[dict]:
    rows = []
    known = fit["known_condition"].tolist()
    subj = fit["subject_id"].astype(str).tolist()
    groups = [("all", "all", np.ones(len(fit), bool))]
    groups += [(ds, "all", (fit["dataset"] == ds).to_numpy()) for ds in sorted(fit["dataset"].unique())]
    groups += [("all", str(sv), (fit["severity"] == sv).to_numpy()) for sv in sorted(fit["severity"].dropna().unique())]
    for ds, sev, mask in groups:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        ci = bootstrap_recall([assigned[i] for i in idx], [known[i] for i in idx], [subj[i] for i in idx],
                              n_boot=n_boot, seed=seed)
        for cond, (point, lo, hi) in ci.items():
            n = int(sum(known[i] == cond for i in idx))
            rows.append(dict(feature_set=feature_set, condition=cond, dataset=ds, severity=sev,
                             n=n, recall=point, ci_lo=lo, ci_hi=hi))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.features_csv, dtype={"visit_id": str, "subject_id": str})
    source_codes = {s: i for i, s in enumerate(sorted(table["source_db"].astype(str).unique()))}
    summary, recall_rows = {}, []
    for set_name, columns in FEATURE_SETS.items():
        usable = table.dropna(subset=[c for c in columns if c != "source_db"])
        fit = usable[usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
        rest = usable[~usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
        fc = fit_fault_clusters(_matrix(fit, columns, source_codes), fit["known_condition"].tolist(), seed=args.seed)
        fit_names = [fc.names[int(c)] for c in fc.labels_]
        rest_names = fc.predict_names(_matrix(rest, columns, source_codes)) if len(rest) else []
        confusion = confusion_against_known_conditions(fc.labels_, fit["known_condition"].tolist(), fc.names)
        confusion.to_csv(args.out_dir / f"confusion_{set_name}.csv")
        recall_rows += _recall_rows(set_name, fit, fit_names, args.n_boot, args.seed)
        assignments = pd.concat([
            fit.assign(cluster=fit_names, in_fit=True), rest.assign(cluster=rest_names, in_fit=False),
        ])[["visit_id", "dataset", "condition", "severity", "known_condition", "cluster", "in_fit"]]
        assignments.to_csv(args.out_dir / f"assignments_{set_name}.csv", index=False)
        held_out = (assignments[~assignments.in_fit].groupby(["known_condition", "cluster"]).size()
                    .unstack(fill_value=0))
        summary[set_name] = dict(
            columns=columns, n_fit=int(len(fit)), dropped_nan=int(len(table) - len(usable)),
            silhouette=silhouette(_matrix(fit, columns, source_codes), fc.labels_),
            cluster_names={int(k): v for k, v in fc.names.items()},
            recall=condition_recall(fit_names, fit["known_condition"].tolist()),
            held_out_assignment={k: {kk: int(vv) for kk, vv in row.items()} for k, row in held_out.iterrows()},
        )
        print(f"[taxonomy] {set_name}: names={fc.names} recall={summary[set_name]['recall']} "
              f"silhouette={summary[set_name]['silhouette']:.3f}", flush=True)

    pd.DataFrame(recall_rows).to_csv(args.out_dir / "table3_recall.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[taxonomy] wrote table3_recall.csv, confusion_*.csv, assignments_*.csv, summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
