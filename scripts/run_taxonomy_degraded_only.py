#!/usr/bin/env python
"""Stage 4b-iv: two-stage taxonomy -- detect degraded windows by SQI, then
cluster ONLY those, with k chosen by silhouette.

run_taxonomy.py clustered every fit-set window with k fixed at 3 and found
the clusters splitting by severity, with structural (clean cross-site
windows) never winning a cluster. The paper's wording is "we clustered
degraded segments", so this asks the question that way:

  1. detect: a window is degraded if the SQI says so. Two rules --
     `drop`:    the combined per-second SQI dipped below 0.5 at least once;
     `outlier`: ppg_sqi_value or ecg_sqi_value below the 2.5th percentile, or
                model_disagreement above the 97.5th percentile, of the clean
                PulseDB-MIMIC controls (bounds from controls only, never from
                condition labels).
  2. type: KMeans on the detected fit rows, k in 2..6 chosen by silhouette on
     a 5,000-row subsample; every cluster named by its majority known
     condition (several clusters may share a name -- one fault at several
     severities is a legitimate outcome). Held-out rows are typed by nearest
     centroid.

Recall is reported per condition and severity as detection_rate x
typing_recall, so the two stages can be read separately.

    python scripts/run_taxonomy_degraded_only.py --features-csv results/taxonomy/fault_features.csv --out-dir results/taxonomy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from trustbio.taxonomy.cluster import fit_fault_clusters

if __package__:
    from .run_taxonomy import FEATURE_SETS, FIT_CONDITIONS, _matrix
else:
    from run_taxonomy import FEATURE_SETS, FIT_CONDITIONS, _matrix

HELD_OUT_ORDER = ["clean", "natural_clean", "natural_ppg_poor", "natural_ecg_poor", "real_motion", "consumer_clean"]
K_RANGE = (2, 3, 4, 5, 6)


def clean_control_bounds(table: pd.DataFrame) -> dict:
    ctrl = table[(table["dataset"] == "pulsedb_mimic") & (table["condition"] == "clean")]
    return dict(ppg_lo=float(ctrl["ppg_sqi_value"].quantile(0.025)),
                ecg_lo=float(ctrl["ecg_sqi_value"].quantile(0.025)),
                dis_hi=float(ctrl["model_disagreement"].quantile(0.975)))


def detect(table: pd.DataFrame, rule: str, bounds: dict) -> np.ndarray:
    if rule == "drop":
        return (table["sqi_drop_duration"] > 0).to_numpy()
    if rule == "outlier":
        return ((table["ppg_sqi_value"] < bounds["ppg_lo"]) | (table["ecg_sqi_value"] < bounds["ecg_lo"])
                | (table["model_disagreement"] > bounds["dis_hi"])).to_numpy()
    raise ValueError(f"unknown rule {rule!r}")


def choose_k(X: np.ndarray, seed: int = 0, sample_size: int = 5000) -> tuple[int, dict]:
    Xs = StandardScaler().fit_transform(X)
    sils = {}
    for k in K_RANGE:
        if len(Xs) <= k:
            continue
        fc = fit_fault_clusters(X, None, seed=seed, n_clusters=k)
        if len(set(fc.labels_.tolist())) < 2:
            continue
        sils[k] = float(silhouette_score(Xs, fc.labels_, sample_size=min(sample_size, len(Xs)), random_state=seed))
    best = max(sils, key=sils.get)
    return best, sils


def majority_names(labels: np.ndarray, known: list[str]) -> dict[int, str]:
    """Each cluster takes its majority known condition; names may repeat."""
    return {int(c): Counter(k for k, l in zip(known, labels) if l == c).most_common(1)[0][0]
            for c in sorted(set(labels.tolist()))}


def _recall_rows(rule, set_name, fit, detected, typed_ok) -> list[dict]:
    """typed_ok: per fit row, True if detected AND its cluster carries its name."""
    rows = []
    groups = [("all", np.ones(len(fit), bool))]
    groups += [(str(sv), (fit["severity"] == sv).to_numpy()) for sv in sorted(fit["severity"].dropna().unique())]
    for sev, mask in groups:
        for cond in sorted(set(fit["known_condition"])):
            m = mask & (fit["known_condition"] == cond).to_numpy()
            n_total, n_det = int(m.sum()), int((m & detected).sum())
            if n_total == 0:
                continue
            det_rate = n_det / n_total
            typing = float((m & typed_ok).sum() / n_det) if n_det else float("nan")
            rows.append(dict(rule=rule, feature_set=set_name, condition=cond, severity=sev, n_total=n_total,
                             n_detected=n_det, detection_rate=det_rate, typing_recall=typing,
                             overall_recall=det_rate * typing if n_det else 0.0))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rules", nargs="+", default=["drop", "outlier"], choices=["drop", "outlier"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.features_csv, dtype={"visit_id": str, "subject_id": str})
    source_codes = {s: i for i, s in enumerate(sorted(table["source_db"].astype(str).unique()))}
    bounds = clean_control_bounds(table)
    print(f"[degraded-only] clean-control bounds: {bounds}", flush=True)
    summary, recall_rows = {r: {} for r in args.rules}, []

    for rule in args.rules:
        for set_name, columns in FEATURE_SETS.items():
            usable = table.dropna(subset=[c for c in columns if c != "source_db"]).reset_index(drop=True)
            fit = usable[usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
            rest = usable[~usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
            det_fit = detect(fit, rule, bounds)
            sub = fit[det_fit].reset_index(drop=True)
            if sub["known_condition"].nunique() < 2 or len(sub) < 10:
                print(f"[degraded-only] {rule}/{set_name}: only {len(sub)} detected rows; skipping", flush=True)
                continue
            X = _matrix(sub, columns, source_codes)
            k, sils = choose_k(X, seed=args.seed)
            fc = fit_fault_clusters(X, None, seed=args.seed, n_clusters=k)
            names = majority_names(fc.labels_, sub["known_condition"].tolist())
            sub_names = np.asarray([names[int(l)] for l in fc.labels_])
            typed_ok = np.zeros(len(fit), bool)
            typed_ok[np.flatnonzero(det_fit)] = sub_names == sub["known_condition"].to_numpy()
            recall_rows += _recall_rows(rule, set_name, fit, det_fit, typed_ok)
            confusion = (pd.crosstab(pd.Series(sub_names, name="cluster_name"), sub["known_condition"])
                         .reindex(columns=sorted(FIT_CONDITIONS), fill_value=0))
            confusion.to_csv(args.out_dir / f"degraded_only_confusion_{rule}_{set_name}.csv")

            held_rows = []
            det_rest = detect(rest, rule, bounds) if len(rest) else np.zeros(0, bool)
            for grp in [g for g in HELD_OUT_ORDER if g in set(rest["known_condition"])]:
                m = (rest["known_condition"] == grp).to_numpy()
                row = dict(known_condition=grp, n=int(m.sum()), detection_rate=float((m & det_rest).sum() / m.sum()))
                shares = {c: 0.0 for c in sorted(FIT_CONDITIONS)}
                if (m & det_rest).any():
                    pred = fc.predict(_matrix(rest[m & det_rest], columns, source_codes))
                    cnt = Counter(names[int(p)] for p in pred)
                    shares.update({c: v / sum(cnt.values()) for c, v in cnt.items()})
                row.update(shares)
                held_rows.append(row)
            pd.DataFrame(held_rows).set_index("known_condition").to_csv(
                args.out_dir / f"degraded_only_heldout_{rule}_{set_name}.csv")

            summary[rule][set_name] = dict(k=int(k), silhouette_by_k={str(kk): v for kk, v in sils.items()},
                                           cluster_names={str(c): n for c, n in names.items()},
                                           n_detected=int(len(sub)),
                                           detected_by_condition=sub["known_condition"].value_counts().to_dict())
            print(f"[degraded-only] {rule}/{set_name}: detected {len(sub):,}/{len(fit):,}; k={k} "
                  f"(silhouette {sils[k]:.3f}); names={names}", flush=True)

    pd.DataFrame(recall_rows).to_csv(args.out_dir / "degraded_only_recall.csv", index=False)
    (args.out_dir / "degraded_only_summary.json").write_text(json.dumps(dict(bounds=bounds, **summary), indent=2, default=float))
    print(f"[degraded-only] wrote degraded_only_recall.csv, degraded_only_confusion_*.csv, "
          f"degraded_only_heldout_*.csv, degraded_only_summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
