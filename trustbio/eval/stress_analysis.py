"""Tables for the degradation stress test (paper Results §2), computed from
the per-window predictions written by scripts/run_stress_predict.py.

Everything here is a pure function of data frames so it can be unit-tested
on synthetic predictions; scripts/run_stress_analysis.py does the I/O,
detection flags and figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KEYS = ["model", "modality", "source", "target", "task"]
COND = ["condition", "kind", "severity"]
CLEAN = "clean"
MATERIAL_HARM = {"hr_regression": 5.0, "sbp_regression": 5.0, "dbp_regression": 5.0}   # bpm / mmHg
# Which probe modalities a fault can touch (inject.py: motion -> PPG, lead-off -> ECG).
AFFECTED = {"motion_artifact": ("ppg", "ecg_ppg_mean"), "lead_off": ("ecg", "ecg_ppg_mean"),
            "missing_ppg": ("ecg_ppg_mean",)}


def pearson_r(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return float("nan")
    return float(np.corrcoef(y[m], p[m])[0, 1])


def score_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Pearson r and MAE per probe x condition, with the clean baseline and deltas."""
    rows = []
    for key, g in pred.groupby(KEYS + COND, dropna=False, sort=True):
        y, p = g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float)
        fin = np.isfinite(y) & np.isfinite(p)
        rows.append(dict(zip(KEYS + COND, key), n=int(fin.sum()), r=pearson_r(y, p),
                         mae=float(np.mean(np.abs(p[fin] - y[fin]))) if fin.any() else float("nan")))
    s = pd.DataFrame(rows)
    base = s[s["condition"] == CLEAN][KEYS + ["r", "mae"]].rename(columns={"r": "r_clean", "mae": "mae_clean"})
    s = s.merge(base, on=KEYS, how="left")
    s["delta_r"] = s["r"] - s["r_clean"]
    s["delta_mae"] = s["mae"] - s["mae_clean"]
    return s


def paired_harm(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-window harm = |err_degraded| - |err_clean| for the same probe and window."""
    clean = (pred[pred["condition"] == CLEAN][KEYS + ["visit_id", "y_pred"]]
             .rename(columns={"y_pred": "y_pred_clean"}))
    deg = pred[pred["condition"] != CLEAN].merge(clean, on=KEYS + ["visit_id"], how="inner")
    deg["harm"] = (deg["y_pred"] - deg["y_true"]).abs() - (deg["y_pred_clean"] - deg["y_true"]).abs()
    return deg[KEYS + COND + ["visit_id", "y_true", "harm"]].reset_index(drop=True)


def rank_stability(scores: pd.DataFrame, metric: str = "r") -> pd.DataFrame:
    """Spearman correlation between the models' clean ranking and their ranking
    under each condition, per (modality, source, target, task)."""
    grp = ["modality", "source", "target", "task"]
    rows = []
    for key, g in scores.groupby(grp, sort=True):
        base = g[g["condition"] == CLEAN].set_index("model")[metric]
        for cond, gc in g[g["condition"] != CLEAN].groupby(COND, dropna=False, sort=True):
            cur = gc.set_index("model")[metric].reindex(base.index)
            ok = base.notna() & cur.notna()
            rho = float(spearmanr(base[ok], cur[ok]).statistic) if ok.sum() >= 3 else float("nan")
            rows.append(dict(zip(grp, key), condition=cond[0], kind=cond[1], severity=cond[2],
                             n_models=int(ok.sum()), spearman=rho,
                             top_clean=base[ok].idxmax() if ok.any() else None,
                             top_degraded=cur[ok].idxmax() if ok.any() else None))
    return pd.DataFrame(rows)


def gap_table(scores: pd.DataFrame, domain: str = "xecg-10min", ts: str = "moment-base",
              metric: str = "r") -> pd.DataFrame:
    """r(domain model) - r(time-series model) per condition, with the clean gap alongside."""
    idx = ["modality", "source", "target", "task"] + COND
    d = scores[scores["model"] == domain].set_index(idx)[metric].rename("r_domain")
    t = scores[scores["model"] == ts].set_index(idx)[metric].rename("r_ts")
    out = pd.concat([d, t], axis=1).reset_index()
    out["gap"] = out["r_domain"] - out["r_ts"]
    base = (out[out["condition"] == CLEAN][["modality", "source", "target", "task", "gap"]]
            .rename(columns={"gap": "gap_clean"}))
    return out.merge(base, on=["modality", "source", "target", "task"], how="left")


def fusion_table(scores: pd.DataFrame, metric: str = "r") -> pd.DataFrame:
    """Fusion vs the better unimodal probe under the same condition; for
    missing_ppg the comparator is the clean ECG-only probe (PPG is absent)."""
    rows = []
    for key, g in scores.groupby(["model", "source", "target", "task"], sort=True):
        by = {(r.modality, r.condition): getattr(r, metric) for r in g.itertuples()}
        meta = {r.condition: (r.kind, r.severity) for r in g.itertuples()}
        for cond in sorted({c for (_, c) in by}):
            if ("ecg_ppg_mean", cond) not in by:
                continue
            if cond == "missing_ppg":
                cands = {"ecg": by.get(("ecg", CLEAN), np.nan)}
            else:
                cands = {m: by.get((m, cond), np.nan) for m in ("ecg", "ppg")}
            best = max(cands, key=lambda m: -np.inf if np.isnan(cands[m]) else cands[m])
            rows.append(dict(zip(["model", "source", "target", "task"], key), condition=cond,
                             kind=meta[cond][0], severity=meta[cond][1], r_fusion=by[("ecg_ppg_mean", cond)],
                             best_unimodal=best, r_best_unimodal=cands[best],
                             fusion_minus_best=by[("ecg_ppg_mean", cond)] - cands[best]))
    return pd.DataFrame(rows)


def harm_coverage(harm: pd.DataFrame, flags: pd.DataFrame, rules=("drop", "outlier", "supervised"),
                  material=MATERIAL_HARM) -> pd.DataFrame:
    """Join per-window harm with the taxonomy's detection flags (by target /
    dataset, condition, visit_id). Per probe x condition x rule: detection rate,
    mean harm, mean harm among undetected windows, share of positive harm in
    detected windows, and recall of materially harmed windows (harm > tau)."""
    f = flags.rename(columns={"dataset": "target"})
    j = harm.merge(f[["target", "condition", "visit_id"] + [f"det_{r}" for r in rules]],
                   on=["target", "condition", "visit_id"], how="inner")
    rows = []
    for key, g in j.groupby(KEYS + COND, dropna=False, sort=True):
        tau = material.get(key[KEYS.index("task")], np.nan)
        h = g["harm"].to_numpy(float)
        pos = np.clip(h, 0, None)
        mat = h > tau
        for rule in rules:
            det = g[f"det_{rule}"].to_numpy(bool)
            rows.append(dict(zip(KEYS + COND, key), rule=rule, n=int(len(g)), detection_rate=float(det.mean()),
                             mean_harm=float(h.mean()),
                             mean_harm_undetected=float(h[~det].mean()) if (~det).any() else float("nan"),
                             harm_share_caught=float(pos[det].sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
                             material_rate=float(mat.mean()),
                             material_recall=float(det[mat].mean()) if mat.any() else float("nan")))
    return pd.DataFrame(rows)
