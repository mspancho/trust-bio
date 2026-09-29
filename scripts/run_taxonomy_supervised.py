#!/usr/bin/env python
"""Stage 4b-iii: supervised separability ceiling for the fault taxonomy.

The paper draft's Methods names two options for the taxonomy -- an
unsupervised clustering, or "a supervised classifier trained against the
known synthetic-condition labels with generalization tested on the real-
degradation datasets". run_taxonomy.py did the former (KMeans, k=3). This
does the latter on the SAME fault-feature table and feature sets, so the two
answer complementary questions: "do the classes fall out on their own?" vs
"are they separable at all with these features?".

  * Fit set: rows whose known condition is motion_artifact / lead_off /
    structural (same as the clustering). Subject-grouped 5-fold CV (groups
    are dataset/subject); recall per condition, by dataset and by severity,
    from out-of-fold predictions, with the same subject-bootstrap CIs.
  * Held-out groups (clean controls, MIMIC-ext natural strata, BUT PPG real
    motion / consumer clean) get class shares from a model fit on all fit rows.
  * The structural question is also asked without the mild synthetic rows
    confounding it: clean Vital vs clean MIMIC, out-of-fold AUROC per set.

    python scripts/run_taxonomy_supervised.py --features-csv results/taxonomy/fault_features.csv \
        --out-dir results/taxonomy --models logreg hgb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustbio.taxonomy.cluster import bootstrap_recall, condition_recall

if __package__:
    from .run_taxonomy import FEATURE_SETS, FIT_CONDITIONS, _matrix
else:
    from run_taxonomy import FEATURE_SETS, FIT_CONDITIONS, _matrix

HELD_OUT_ORDER = ["clean", "natural_clean", "natural_ppg_poor", "natural_ecg_poor", "real_motion", "consumer_clean"]


def make_model(name: str, seed: int = 0):
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                              random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose logreg or hgb")


def _groups(df: pd.DataFrame) -> np.ndarray:
    return (df["dataset"].astype(str) + "/" + df["subject_id"].astype(str)).to_numpy()


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str,
                    seed: int = 0, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold class predictions with subject-grouped folds."""
    n_splits = min(n_splits, len(np.unique(groups)))
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model(model_name, seed).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred.astype(str)


def oof_scores_binary(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0,
                      n_splits: int = 5) -> np.ndarray:
    n_splits = min(n_splits, len(np.unique(groups)))
    scores = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model("logreg", seed).fit(X[tr], y[tr])
        scores[te] = model.predict_proba(X[te])[:, list(model.classes_).index(1)]
    return scores


def _recall_rows(model_name, feature_set, fit, pred, n_boot, seed) -> list[dict]:
    rows = []
    known = fit["known_condition"].tolist()
    subj = _groups(fit).tolist()
    groups = [("all", "all", np.ones(len(fit), bool))]
    groups += [(ds, "all", (fit["dataset"] == ds).to_numpy()) for ds in sorted(fit["dataset"].unique())]
    groups += [("all", str(sv), (fit["severity"] == sv).to_numpy()) for sv in sorted(fit["severity"].dropna().unique())]
    for ds, sev, mask in groups:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        ci = bootstrap_recall([pred[i] for i in idx], [known[i] for i in idx], [subj[i] for i in idx],
                              n_boot=n_boot, seed=seed)
        for cond, (point, lo, hi) in ci.items():
            rows.append(dict(model=model_name, feature_set=feature_set, condition=cond, dataset=ds,
                             severity=sev, n=int(sum(known[i] == cond for i in idx)),
                             recall=point, ci_lo=lo, ci_hi=hi))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["logreg", "hgb"], choices=["logreg", "hgb"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.features_csv, dtype={"visit_id": str, "subject_id": str})
    source_codes = {s: i for i, s in enumerate(sorted(table["source_db"].astype(str).unique()))}
    recall_rows, summary = [], {m: {} for m in args.models}
    summary["structural_vs_clean_auroc"] = {}

    for set_name, columns in FEATURE_SETS.items():
        usable = table.dropna(subset=[c for c in columns if c != "source_db"])
        fit = usable[usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
        rest = usable[~usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
        X_fit, y_fit, g_fit = _matrix(fit, columns, source_codes), fit["known_condition"].to_numpy(str), _groups(fit)
        for model_name in args.models:
            pred = oof_predictions(X_fit, y_fit, g_fit, model_name, args.seed)
            recall_rows += _recall_rows(model_name, set_name, fit, pred, args.n_boot, args.seed)
            confusion = (pd.crosstab(pd.Series(pred, name="predicted"), pd.Series(y_fit, name="known"))
                         .reindex(index=sorted(FIT_CONDITIONS), columns=sorted(FIT_CONDITIONS), fill_value=0))
            confusion.to_csv(args.out_dir / f"supervised_confusion_{model_name}_{set_name}.csv")
            full = make_model(model_name, args.seed).fit(X_fit, y_fit)
            held = pd.DataFrame(index=pd.Index([], name="known_condition"))
            if len(rest):
                rest_pred = full.predict(_matrix(rest, columns, source_codes)).astype(str)
                held = (pd.crosstab(rest["known_condition"], pd.Series(rest_pred, name="predicted"), normalize="index")
                        .reindex(index=[c for c in HELD_OUT_ORDER if c in set(rest["known_condition"])],
                                 columns=sorted(FIT_CONDITIONS), fill_value=0.0))
            held.to_csv(args.out_dir / f"supervised_heldout_{model_name}_{set_name}.csv")
            summary[model_name][set_name] = dict(
                columns=columns, n_fit=int(len(fit)),
                macro_f1=float(f1_score(y_fit, pred, average="macro")),
                recall=condition_recall(pred.tolist(), y_fit.tolist()),
            )
            print(f"[supervised] {model_name}/{set_name}: macro-F1={summary[model_name][set_name]['macro_f1']:.3f} "
                  f"recall={ {k: round(v, 3) for k, v in summary[model_name][set_name]['recall'].items()} }", flush=True)

        # Structural without the mild synthetic rows in the way: clean Vital vs clean MIMIC.
        binary = usable[usable["known_condition"].isin(["structural", "clean"])].reset_index(drop=True)
        if len(binary) and binary["known_condition"].nunique() == 2:
            yb = (binary["known_condition"] == "structural").astype(int).to_numpy()
            scores = oof_scores_binary(_matrix(binary, columns, source_codes), yb, _groups(binary), args.seed)
            summary["structural_vs_clean_auroc"][set_name] = float(roc_auc_score(yb, scores))
            print(f"[supervised] structural-vs-clean AUROC ({set_name}) = {summary['structural_vs_clean_auroc'][set_name]:.3f}", flush=True)

    pd.DataFrame(recall_rows).to_csv(args.out_dir / "supervised_recall.csv", index=False)
    (args.out_dir / "supervised_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[supervised] wrote supervised_recall.csv, supervised_confusion_*.csv, supervised_heldout_*.csv, "
          f"supervised_summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
