#!/usr/bin/env python
"""Stage 5b: degradation stress test, tables and figures.

    python scripts/run_stress_analysis.py --pred-dir results/stress \
        --fault-features results/taxonomy/fault_features.csv --out-dir results/stress

Pools predictions_<model>.csv.gz from the prediction pass, then writes
  stress_scores.csv           r / MAE per probe x condition, deltas vs clean
  stress_rank_stability.csv   Spearman of the models' ranking vs their clean ranking
  stress_gap.csv              r(domain model) - r(time-series model) per condition
  stress_fusion.csv           fusion vs best unimodal per condition
  stress_detection_flags.csv  per taxonomy window: drop / outlier / supervised flags
  stress_harm_coverage.csv    per-window harm joined with the flags
  stress_summary.json         headline numbers + fidelity check vs full_transport.csv
  figures/fig2{a,b,c,d}_*_<task>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from trustbio.eval.stress_analysis import (  # noqa: E402
    AFFECTED, CLEAN, fusion_table, gap_table, harm_coverage, paired_harm, rank_stability, score_table,
)

if __package__:
    from .run_taxonomy import FEATURE_SETS
    from .run_taxonomy_degraded_only import clean_control_bounds, detect
else:
    from run_taxonomy import FEATURE_SETS
    from run_taxonomy_degraded_only import clean_control_bounds, detect

DETECTOR_FEATURES = FEATURE_SETS["no_source_db"]
POSITIVE = ("motion_artifact", "lead_off")
NEGATIVE = ("clean", "structural", "natural_clean")
RULES = ("drop", "outlier", "supervised")


def supervised_flags(table: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(degraded) > 0.5 from a gradient-boosted binary detector on
    the label-free features: synthetic motion/lead-off rows vs clean controls,
    subject-grouped 5-fold. Rows outside those classes get the full fit."""
    X = table[DETECTOR_FEATURES].to_numpy(float)
    y = table["known_condition"].isin(POSITIVE).to_numpy().astype(int)
    labelled = table["known_condition"].isin(POSITIVE + NEGATIVE).to_numpy()
    groups = (table["dataset"].astype(str) + "/" + table["subject_id"].astype(str)).to_numpy()
    p = np.full(len(table), np.nan)
    idx = np.flatnonzero(labelled)
    n_splits = min(5, len(np.unique(groups[idx])))
    for tr, te in GroupKFold(n_splits=n_splits).split(X[idx], y[idx], groups[idx]):
        clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed)
        p[idx[te]] = clf.fit(X[idx][tr], y[idx][tr]).predict_proba(X[idx][te])[:, 1]
    if (~labelled).any():
        clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed)
        p[~labelled] = clf.fit(X[idx], y[idx]).predict_proba(X[~labelled])[:, 1]
    return p > 0.5


def detection_flags(table: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    bounds = clean_control_bounds(table)
    out = table[["dataset", "condition", "visit_id", "known_condition"]].copy()
    out["det_drop"] = detect(table, "drop", bounds)
    out["det_outlier"] = detect(table, "outlier", bounds)
    out["det_supervised"] = supervised_flags(table, seed)
    return out.reset_index(drop=True)


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    files = sorted(Path(pred_dir).glob("predictions_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no predictions_*.csv.gz under {pred_dir}")
    return pd.concat([pd.read_csv(f, dtype={"visit_id": str}) for f in files], ignore_index=True)


def _pair_label(source: str, target: str) -> str:
    inst = target.replace("pulsedb_", "")
    return f"{source}->{inst}" + (" (within)" if inst == source else "")


def _affected_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[[m in AFFECTED.get(k, ()) for k, m in zip(df["kind"], df["modality"])]]


def plot_severity_curves(scores: pd.DataFrame, task: str, path: Path) -> None:
    s = scores[(scores["task"] == task) & (scores["kind"] != "missing_ppg")]
    pairs = sorted({(r.source, r.target) for r in s.itertuples()})
    kinds = ["motion_artifact", "lead_off"]
    fig, axes = plt.subplots(len(pairs), len(kinds), figsize=(4.2 * len(kinds), 2.6 * len(pairs)),
                             squeeze=False, sharey=True)
    for i, (src, tgt) in enumerate(pairs):
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            g = s[(s["source"] == src) & (s["target"] == tgt) & (s["kind"].isin([kind, CLEAN]))]
            for model, gm in g.groupby("model"):
                for modality, ls in zip(AFFECTED[kind], ("-", "--")):
                    gg = gm[gm["modality"] == modality].sort_values("severity")
                    ax.plot(gg["severity"], gg["r"], ls, marker="o", ms=3, label=f"{model} / {modality}")
            ax.set_title(f"{kind}: {_pair_label(src, tgt)}", fontsize=9)
            ax.set_xlabel("severity (fraction of window)")
            if j == 0:
                ax.set_ylabel(f"{task} Pearson r")
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap(gap: pd.DataFrame, task: str, path: Path, domain: str, ts: str) -> None:
    g = gap[(gap["task"] == task) & (gap["kind"] != "missing_ppg")]
    pairs = sorted({(r.source, r.target) for r in g.itertuples()})
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 2.8), squeeze=False, sharey=True)
    for i, (src, tgt) in enumerate(pairs):
        ax = axes[0, i]
        gp = g[(g["source"] == src) & (g["target"] == tgt)]
        for kind in ("motion_artifact", "lead_off"):
            for modality in AFFECTED[kind]:
                gg = gp[(gp["kind"].isin([kind, CLEAN])) & (gp["modality"] == modality)].sort_values("severity")
                ax.plot(gg["severity"], gg["gap"], marker="o", ms=3, label=f"{kind} / {modality}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(_pair_label(src, tgt), fontsize=9)
        ax.set_xlabel("severity")
        if i == 0:
            ax.set_ylabel(f"r({domain}) - r({ts})")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fusion(fus: pd.DataFrame, task: str, path: Path) -> None:
    f = fus[fus["task"] == task]
    pairs = sorted({(r.source, r.target) for r in f.itertuples()})
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 2.8), squeeze=False, sharey=True)
    for i, (src, tgt) in enumerate(pairs):
        ax = axes[0, i]
        fp = f[(f["source"] == src) & (f["target"] == tgt)]
        for model, fm in fp.groupby("model"):
            for kind, marker in (("motion_artifact", "o"), ("lead_off", "s")):
                gg = fm[fm["kind"].isin([kind, CLEAN])].sort_values("severity")
                ax.plot(gg["severity"], gg["fusion_minus_best"], marker=marker, ms=3, lw=0.8,
                        label=f"{model} / {kind}")
            mp = fm[fm["kind"] == "missing_ppg"]
            if len(mp):
                ax.plot([1.0], mp["fusion_minus_best"], "x", ms=5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(_pair_label(src, tgt), fontsize=9)
        ax.set_xlabel("severity (x = missing PPG)")
        if i == 0:
            ax.set_ylabel("r(fusion) - r(best unimodal)")
    axes[0, 0].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_harm_coverage(cov: pd.DataFrame, task: str, path: Path) -> None:
    """Affected-modality cells only, averaged over models and source/target pairs."""
    c = _affected_rows(cov[cov["task"] == task])
    if c.empty:
        return
    agg = (c.groupby(["kind", "severity", "modality", "rule"])
           [["harm_share_caught", "material_recall", "material_rate", "mean_harm"]].mean().reset_index())
    cells = sorted({(k, s, m) for k, s, m in zip(agg["kind"], agg["severity"], agg["modality"])})
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(cells)), 3.2))
    width = 0.8 / len(RULES)
    for r_i, rule in enumerate(RULES):
        vals = [agg[(agg.kind == k) & (agg.severity == s) & (agg.modality == m) & (agg.rule == rule)]
                ["harm_share_caught"].mean() for k, s, m in cells]
        ax.bar(np.arange(len(cells)) + (r_i - 1) * width, vals, width, label=f"{rule}: share of harm caught")
    mat = [agg[(agg.kind == k) & (agg.severity == s) & (agg.modality == m)]["material_rate"].mean()
           for k, s, m in cells]
    ax.plot(np.arange(len(cells)), mat, "k_", ms=14, label="fraction of windows harmed > 5 units")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([f"{k}\n{s} / {m}" for k, s, m in cells], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(task)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fidelity_vs_transport(scores: pd.DataFrame, transport_csv: Path) -> dict:
    """Clean cross-source HR r here vs results/full_transport.csv (same recipe)."""
    if not transport_csv.exists():
        return {}
    t = pd.read_csv(transport_csv)
    t = t[t["task"] == "hr_regression"].set_index(["model", "modality", "direction"])["score"]
    out = {}
    s = scores[(scores["task"] == "hr_regression") & (scores["condition"] == CLEAN)]
    for r in s.itertuples():
        inst = r.target.replace("pulsedb_", "")
        if not r.target.startswith("pulsedb_") or inst == r.source:
            continue
        key = (r.model, r.modality, f"{r.source}_to_{inst}")
        if key in t.index:
            out[f"{r.model}/{r.modality}/{key[2]}"] = dict(stress_r=float(r.r), transport_r=float(t[key]),
                                                          abs_diff=float(abs(r.r - t[key])))
    return out


def headline(scores: pd.DataFrame, cov: pd.DataFrame) -> dict:
    """Mean over models (and source/target pairs) of delta_r and harm coverage,
    affected modality only, per task x kind x severity."""
    out = {}
    for (task, kind, sev, mod), g in _affected_rows(scores).groupby(["task", "kind", "severity", "modality"]):
        out.setdefault(task, {})[f"{kind}/{sev}/{mod}"] = dict(delta_r_mean=float(g["delta_r"].mean()),
                                                                  delta_mae_mean=float(g["delta_mae"].mean()))
    for (task, kind, sev, mod, rule), g in _affected_rows(cov).groupby(["task", "kind", "severity", "modality", "rule"]):
        cell = out.setdefault(task, {}).setdefault(f"{kind}/{sev}/{mod}", {})
        cell[f"harm_share_caught_{rule}"] = float(g["harm_share_caught"].mean())
        cell[f"material_recall_{rule}"] = float(g["material_recall"].mean())
        cell["material_rate"] = float(g["material_rate"].mean())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--fault-features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--transport-csv", type=Path, default=Path("results/full_transport.csv"))
    ap.add_argument("--domain", default="xecg-10min")
    ap.add_argument("--ts", default="moment-base")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    pred = load_predictions(args.pred_dir)
    print(f"[stress-analysis] {len(pred):,} predictions, models {sorted(pred['model'].unique())}", flush=True)
    scores = score_table(pred)
    scores.to_csv(args.out_dir / "stress_scores.csv", index=False)
    stab = rank_stability(scores)
    stab.to_csv(args.out_dir / "stress_rank_stability.csv", index=False)
    gap = gap_table(scores, args.domain, args.ts)
    gap.to_csv(args.out_dir / "stress_gap.csv", index=False)
    fus = fusion_table(scores)
    fus.to_csv(args.out_dir / "stress_fusion.csv", index=False)

    table = pd.read_csv(args.fault_features, dtype={"visit_id": str, "subject_id": str})
    flags = detection_flags(table, args.seed)
    flags.to_csv(args.out_dir / "stress_detection_flags.csv", index=False)
    harm = paired_harm(pred)
    cov = harm_coverage(harm, flags)
    cov.to_csv(args.out_dir / "stress_harm_coverage.csv", index=False)

    summary = dict(n_predictions=int(len(pred)), models=sorted(pred["model"].unique().tolist()),
                   fidelity_vs_transport=fidelity_vs_transport(scores, args.transport_csv),
                   headline=headline(scores, cov))
    (args.out_dir / "stress_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    for task in sorted(scores["task"].unique()):
        plot_severity_curves(scores, task, fig_dir / f"fig2a_severity_curves_{task}.png")
        plot_gap(gap, task, fig_dir / f"fig2b_gap_{task}.png", args.domain, args.ts)
        plot_fusion(fus, task, fig_dir / f"fig2c_fusion_{task}.png")
        plot_harm_coverage(cov, task, fig_dir / f"fig2d_harm_coverage_{task}.png")
    print(f"[stress-analysis] wrote tables, stress_summary.json and figures to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
