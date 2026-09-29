#!/usr/bin/env python
"""Figure 3 panels for the fault taxonomy (a: projection, b: confusion,
d: held-out assignment shares, e: recall vs severity). Panel c (waveform
examples) needs raw-signal access and is produced separately."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trustbio.taxonomy.features import FEATURE_NAMES  # noqa: E402

FIT = ("motion_artifact", "lead_off", "structural")
HELD_OUT_ORDER = ["clean", "natural_clean", "natural_ppg_poor", "natural_ecg_poor", "real_motion", "consumer_clean"]
COLORS = {"motion_artifact": "#d95f02", "lead_off": "#7570b3", "structural": "#1b9e77"}
COLUMN_SETS = {
    "all": list(FEATURE_NAMES),
    "no_source_db": [f for f in FEATURE_NAMES if f != "source_db"],
    "no_disagreement": [f for f in FEATURE_NAMES if f != "model_disagreement"],
    "sqi_only": ["sqi_value", "sqi_drop_duration", "ecg_sqi_value", "ppg_sqi_value"],
}


def _numeric(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = df[columns].copy()
    if "source_db" in columns:
        codes = {s: i for i, s in enumerate(sorted(df["source_db"].astype(str).unique()))}
        X["source_db"] = df["source_db"].map(codes).astype(float)
    return X.to_numpy(float)


def plot_projection(fit_df: pd.DataFrame, columns: list[str], out_png: Path) -> None:
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(_numeric(fit_df, columns)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, col, title in ((axes[0], "known_condition", "known condition"), (axes[1], "cluster", "assigned cluster")):
        for name, grp in fit_df.assign(pc1=Z[:, 0], pc2=Z[:, 1]).groupby(col):
            ax.scatter(grp.pc1, grp.pc2, s=6, alpha=0.5, label=str(name), color=COLORS.get(str(name)))
        ax.set_title(f"PCA of fault features, coloured by {title}"); ax.set_xlabel("PC1"); ax.legend(markerscale=3, fontsize=8)
    axes[0].set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_confusion(confusion: pd.DataFrame, out_png: Path) -> None:
    norm = confusion.div(confusion.sum(axis=0).replace(0, np.nan), axis=1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(norm.to_numpy(float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(norm.columns))); ax.set_xticklabels(norm.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(norm.index))); ax.set_yticklabels(norm.index)
    ax.set_xlabel("known condition"); ax.set_ylabel("assigned cluster")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm.iat[i, j]:.2f}\n(n={int(confusion.iat[i, j])})", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_heldout_shares(assignments: pd.DataFrame, out_png: Path) -> None:
    held = assignments[~assignments["in_fit"].astype(bool)]
    share = (held.groupby(["known_condition", "cluster"]).size().unstack(fill_value=0)
             .reindex([c for c in HELD_OUT_ORDER if c in set(held.known_condition)]))
    share = share.div(share.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bottom = np.zeros(len(share))
    for cl in share.columns:
        ax.bar(share.index, share[cl].to_numpy(), bottom=bottom, label=cl, color=COLORS.get(str(cl)))
        bottom += share[cl].to_numpy()
    ax.set_ylabel("share of windows"); ax.set_ylim(0, 1); ax.legend(title="assigned cluster", fontsize=8)
    ax.set_title("Held-out groups: where do they land?"); plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_severity_recall(table3: pd.DataFrame, out_png: Path) -> None:
    sub = table3[(table3.feature_set == "all") & (table3.dataset == "all") & (table3.severity != "all")]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for cond, grp in sub.groupby("condition"):
        if cond not in ("motion_artifact", "lead_off"):
            continue
        grp = grp.assign(sev=grp["severity"].astype(float)).sort_values("sev")
        ax.errorbar(grp.sev, grp.recall, yerr=[grp.recall - grp.ci_lo, grp.ci_hi - grp.recall],
                    marker="o", capsize=3, label=cond, color=COLORS.get(cond))
    ax.set_xlabel("injected severity (fraction of window)"); ax.set_ylabel("recall of own cluster"); ax.set_ylim(0, 1.02)
    ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--assignments-csv", type=Path, required=True)
    ap.add_argument("--confusion-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-set", default="all", choices=sorted(COLUMN_SETS))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(args.features_csv, dtype={"visit_id": str})
    asg = pd.read_csv(args.assignments_csv, dtype={"visit_id": str})
    columns = COLUMN_SETS[args.feature_set]
    fit = feats.merge(asg[["visit_id", "condition", "cluster", "in_fit"]], on=["visit_id", "condition"])
    fit = fit[fit["in_fit"].astype(bool)].dropna(subset=[c for c in columns if c != "source_db"])
    plot_projection(fit, columns, args.out_dir / "fig3a_projection.png")
    plot_confusion(pd.read_csv(args.confusion_csv, index_col=0), args.out_dir / "fig3b_confusion.png")
    plot_heldout_shares(asg, args.out_dir / "fig3d_structural_share.png")
    t3_path = args.features_csv.parent / "table3_recall.csv"
    if t3_path.exists():
        plot_severity_recall(pd.read_csv(t3_path, dtype={"severity": str}), args.out_dir / "fig3e_severity_recall.png")
    print(f"[plot] wrote figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
