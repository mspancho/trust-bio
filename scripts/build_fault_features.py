#!/usr/bin/env python
"""Stage 4b-i: assemble the fault-feature table for the taxonomy.

For every (condition, dataset, window) in the taxonomy store this recomputes
the exact waveform the models were fed (clean, or degraded via the same
per-window generator), derives per-second SQI traces from it, looks up the
two HR-probe predictions from the stored features, and writes one row. The
probes are fit here on CLEAN full-scale PulseDB-MIMIC train features; hf_ref
(one per modality) is calibrated against MIMIC-III-Ext-PPG's native SQI on
the clean condition. Run as a batch job (loads two full-scale train
matrices, ~64 GB).

    python scripts/build_fault_features.py --store features_cache/taxonomy_store \
        --cohort-cache features_cache/taxonomy --full-store features_cache/full \
        --full-cohort-cache features_cache --out-dir results/taxonomy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustbio.data.but_ppg import load_but_ppg_accelerometer
from trustbio.data.mimic_ext_ppg import first_sqi_code, native_stratum
from trustbio.degradation.calibrate import load_cached_noise_amplitude
from trustbio.degradation.inject import apply_degradation, visit_rng
from trustbio.eval.metrics import auroc
from trustbio.store import FeatureStore
from trustbio.taxonomy.disagreement import disagreement_scale, fit_hr_probe, predict_hr
from trustbio.taxonomy.features import SegmentFaultFeatures, extract_fault_features, features_to_matrix
from trustbio.taxonomy.sqi import DEFAULT_HF_REF, calibrate_hf_ref, combined_sqi_trace, mean_hf_ratio, sqi_trace

if __package__:
    from ._dataset_builders import add_dataset_root_args, build_dataset_handle
else:
    from _dataset_builders import add_dataset_root_args, build_dataset_handle

CONDITIONS = [("clean", None, None)] + [
    (f"{k}_{s}", k, s) for k in ("motion_artifact", "lead_off") for s in (0.1, 0.3, 0.6)
]
DEGRADED_DATASETS = ["pulsedb_mimic", "pulsedb_vital", "mimic_ext_ppg"]
CLEAN_ONLY_DATASETS = ["but_ppg"]
FIT_CONDITIONS = ("motion_artifact", "lead_off", "structural")
NATIVE_COLUMNS = ("subject_id", "stratum", "pleth_sqi0", "ecg_sqi0", "quality")


def known_condition(dataset: str, kind: str | None, native: dict) -> str:
    """See the plan's condition table: synthetic kinds name themselves; clean
    windows are labelled by where they come from."""
    if kind is not None:
        return kind
    if dataset == "pulsedb_mimic":
        return "clean"
    if dataset == "pulsedb_vital":
        return "structural"
    if dataset == "mimic_ext_ppg":
        return {"clean": "natural_clean", "ppg_poor": "natural_ppg_poor",
                "ecg_poor": "natural_ecg_poor"}[str(native.get("stratum"))]
    if dataset == "but_ppg":
        return "real_motion" if int(native.get("quality")) == 0 else "consumer_clean"
    raise ValueError(f"unknown dataset {dataset!r}")


def degraded_pair(ecg, ecg_fs, ppg, ppg_fs, visit_id, kind, severity, seed, amps):
    """Exactly what make_degraded_loader hands a model for this window (same
    per-window generator, same modality order)."""
    if kind is None or severity is None:
        return ecg, ppg
    ecg_out, _ = apply_degradation(ecg, None, ecg_fs, kind, severity,
                                   visit_rng(seed, str(visit_id), kind, severity, "ecg"), amps)
    _, ppg_out = apply_degradation(np.zeros_like(ppg), ppg, ppg_fs, kind, severity,
                                   visit_rng(seed, str(visit_id), kind, severity, "ppg"), amps)
    return ecg_out, ppg_out


def load_condition_features(store_root, cond, dataset, model, modality, duration_sec) -> pd.DataFrame:
    """All splits of one cell, indexed by visit_id."""
    store = FeatureStore(Path(store_root) / cond / dataset)
    parts = []
    for split in ("train", "val", "test"):
        if store.exists(model, modality, duration_sec, split):
            ids, X = store.load(model, modality, duration_sec, split)
            parts.append(pd.DataFrame(X, index=pd.Index(ids.astype(str), name="visit_id")))
    if not parts:
        raise FileNotFoundError(f"no features under {store.root} for {model}/{modality}")
    return pd.concat(parts)


def native_annotations(cohort_cache, dataset: str, visits: pd.DataFrame) -> dict:
    """{visit_id: {...}} of per-window native annotations: subject_id, and for
    MIMIC-ext stratum / pleth_sqi0 / ecg_sqi0, for BUT PPG quality. Prefers
    the taxonomy cohort CSV the samplers wrote (it carries the stratum
    columns; a handle rebuilds its own table from the metadata subset and
    does not), and otherwise derives the MIMIC-ext codes from the native SQI
    vectors."""
    df = None
    if cohort_cache is not None:
        p = Path(cohort_cache) / f"{dataset}_cohort.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"visit_id": str, "subject_id": str})
    if df is None:
        df = visits.copy()
    df["visit_id"] = df["visit_id"].astype(str)
    if dataset == "mimic_ext_ppg":
        if "pleth_sqi0" not in df.columns and "vector_10s_pleth_sqi" in df.columns:
            df["pleth_sqi0"] = df["vector_10s_pleth_sqi"].map(first_sqi_code)
        if "ecg_sqi0" not in df.columns and "vector_10s_ecg_sqi" in df.columns:
            df["ecg_sqi0"] = df["vector_10s_ecg_sqi"].map(first_sqi_code)
        if "stratum" not in df.columns:
            df["stratum"] = [native_stratum(p, e) for p, e in zip(df["pleth_sqi0"], df["ecg_sqi0"])]
    return df.set_index("visit_id").to_dict("index")


def raw_signals(handle, duration_sec: int) -> dict:
    """{visit_id: (ecg, ecg_fs, ppg, ppg_fs)} for every window of the handle,
    truncated to the first duration_sec seconds like encode_modality does."""
    out = {}
    for df in handle.splits.values():
        for vid in df["visit_id"].astype(str):
            ecg, efs = handle.load_signal(vid, "ecg")
            ppg, pfs = handle.load_signal(vid, "ppg")
            out[vid] = (np.asarray(ecg[: duration_sec * int(efs)], np.float32), int(efs),
                        np.asarray(ppg[: duration_sec * int(pfs)], np.float32), int(pfs))
    return out


def accel_traces(root, visit_ids, n_seconds: int, window_sec: float) -> dict:
    """{visit_id: per-second mean accelerometer magnitude} for BUT PPG."""
    out = {}
    for vid in visit_ids:
        res = load_but_ppg_accelerometer(root, vid)
        if res is None:
            continue
        acc, fs = res
        mag = np.linalg.norm(np.asarray(acc, float), axis=1)
        n = max(1, int(round(window_sec * fs)))
        per = [float(np.mean(mag[i * n:(i + 1) * n])) for i in range(len(mag) // n)]
        if len(per) >= n_seconds:
            out[vid] = np.asarray(per[:n_seconds])
    return out


def window_rows(signals, cond, kind, sev, dataset, feats_a, feats_b, probe_a, probe_b,
                hf_ref_ecg, hf_ref_ppg, native, seed, amps, window_sec, accel=None) -> list[dict]:
    ids = [v for v in signals if v in feats_a.index and v in feats_b.index]
    pa = predict_hr(probe_a, feats_a.loc[ids].to_numpy()) if ids else np.array([])
    pb = predict_hr(probe_b, feats_b.loc[ids].to_numpy()) if ids else np.array([])
    rows = []
    for vid, a, b in zip(ids, pa, pb):
        ecg, efs, ppg, pfs = signals[vid]
        ecg_d, ppg_d = degraded_pair(ecg, efs, ppg, pfs, vid, kind, sev, seed, amps)
        e_sqi = sqi_trace(ecg_d, efs, hf_ref_ecg, window_sec)
        p_sqi = sqi_trace(ppg_d, pfs, hf_ref_ppg, window_sec)
        comb = combined_sqi_trace(e_sqi, p_sqi)
        acc = accel.get(vid) if accel else None
        acc = acc[: len(comb)] if acc is not None and len(acc) >= len(comb) else None
        f = extract_fault_features(comb, acc, fs=int(round(1 / window_sec)), source_db=dataset,
                                   model_a_pred=float(a), model_b_pred=float(b), disagreement_scale=1.0,
                                   ecg_sqi_trace=e_sqi, ppg_sqi_trace=p_sqi)
        nat = native.get(vid, {}) if native else {}
        row = dict(dataset=dataset, visit_id=vid, condition=cond, kind=kind or "",
                   severity=float(sev) if sev is not None else np.nan,
                   known_condition=known_condition(dataset, kind, nat),
                   pred_a=float(a), pred_b=float(b), disagreement_raw=abs(float(a) - float(b)),
                   sqi_value=f.sqi_value, sqi_drop_duration=f.sqi_drop_duration, accel_corr=f.accel_corr,
                   source_db=f.source_db, ecg_sqi_value=f.ecg_sqi_value, ppg_sqi_value=f.ppg_sqi_value)
        row.update({k: nat.get(k) for k in NATIVE_COLUMNS})
        rows.append(row)
    return rows


def _calibrate_refs(signals, native):
    """hf_ref per modality from MIMIC-ext's native first-sub-window codes."""
    ppg_good, ppg_poor, ecg_good, ecg_poor = [], [], [], []
    for vid, (ecg, efs, ppg, pfs) in signals.items():
        nat = native.get(vid, {})
        p0, e0 = nat.get("pleth_sqi0"), nat.get("ecg_sqi0")
        if p0 is not None and not pd.isna(p0):
            (ppg_good if p0 == 1 else ppg_poor).append(mean_hf_ratio(ppg, pfs))
        if e0 is not None and not pd.isna(e0) and (p0 == 1):
            (ecg_good if e0 == 1 else ecg_poor).append(mean_hf_ratio(ecg, efs))

    def one(good, poor, name):
        if len(good) < 20 or len(poor) < 20:
            print(f"[fault-features] WARNING: too few native {name} labels ({len(good)}/{len(poor)}); hf_ref={DEFAULT_HF_REF}")
            return DEFAULT_HF_REF, float("nan")
        y = np.r_[np.zeros(len(good)), np.ones(len(poor))]
        s = np.r_[good, poor]
        return calibrate_hf_ref(np.asarray(good), np.asarray(poor)), float(auroc(y, s))

    ref_ppg, auc_ppg = one(ppg_good, ppg_poor, "PLETH")
    ref_ecg, auc_ecg = one(ecg_good, ecg_poor, "ECG")
    return ref_ecg, ref_ppg, auc_ecg, auc_ppg


def main() -> int:
    ap = argparse.ArgumentParser()
    add_dataset_root_args(ap)
    ap.add_argument("--store", required=True, help="taxonomy store root (with <cond>/<dataset>/ below)")
    ap.add_argument("--full-store", required=True, help="full-scale clean store root (probe training)")
    ap.add_argument("--full-cohort-cache", type=Path, required=True, help="dir with pulsedb_mimic_labels.csv")
    ap.add_argument("--out-dir", type=Path, default=Path("results/taxonomy"))
    ap.add_argument("--domain-model", default="xecg-10min")
    ap.add_argument("--ts-model", default="moment-base")
    ap.add_argument("--modality", default="ecg_ppg_mean")
    ap.add_argument("--duration-sec", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=500_000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    amps = load_cached_noise_amplitude()

    # 1. Probes on clean full-scale PulseDB-MIMIC train features.
    labels = pd.read_csv(args.full_cohort_cache / "pulsedb_mimic_labels.csv", dtype={"visit_id": str}).set_index("visit_id")
    probes = {}
    for name in (args.domain_model, args.ts_model):
        full = FeatureStore(Path(args.full_store) / "pulsedb_mimic")
        ids_tr, X_tr = full.load(name, args.modality, args.duration_sec, "train")
        ids_va, X_va = full.load(name, args.modality, args.duration_sec, "val")
        y_tr = labels["hr_regression"].reindex(ids_tr.astype(str)).to_numpy(float)
        y_va = labels["hr_regression"].reindex(ids_va.astype(str)).to_numpy(float)
        probes[name] = fit_hr_probe(X_tr, y_tr, X_va, y_va, seed=args.seed, max_train=args.max_train)
        print(f"[fault-features] probe {name}: alpha={probes[name].alpha:g} n_train={probes[name].n_train:,}", flush=True)
        del X_tr, X_va

    # 2. Raw signals, native annotations, accelerometer per dataset.
    signals, natives, accel = {}, {}, {}
    for ds in DEGRADED_DATASETS + CLEAN_ONLY_DATASETS:
        h = build_dataset_handle(ds, args, with_labels=False)
        signals[ds] = raw_signals(h, args.duration_sec)
        natives[ds] = native_annotations(getattr(args, "cohort_cache", None), ds, h.cohort.visits)
        missing = [v for v in signals[ds] if v not in natives[ds]]
        if missing:
            raise RuntimeError(f"{ds}: {len(missing)} windows have no native annotation row (e.g. {missing[:3]})")
        print(f"[fault-features] {ds}: {len(signals[ds]):,} windows loaded", flush=True)
    accel["but_ppg"] = accel_traces(args.but_ppg_root, list(signals["but_ppg"]), args.duration_sec, args.window_sec)

    # 3. hf_ref per modality from MIMIC-ext native SQI.
    ref_ecg, ref_ppg, auc_ecg, auc_ppg = _calibrate_refs(signals["mimic_ext_ppg"], natives["mimic_ext_ppg"])
    print(f"[fault-features] hf_ref ecg={ref_ecg:.3f} (AUROC vs native {auc_ecg:.3f}) ppg={ref_ppg:.3f} (AUROC {auc_ppg:.3f})", flush=True)

    # 4. Rows for every (condition, dataset).
    rows = []
    for cond, kind, sev in CONDITIONS:
        for ds in DEGRADED_DATASETS + (CLEAN_ONLY_DATASETS if kind is None else []):
            fa = load_condition_features(args.store, cond, ds, args.domain_model, args.modality, args.duration_sec)
            fb = load_condition_features(args.store, cond, ds, args.ts_model, args.modality, args.duration_sec)
            rows.extend(window_rows(signals[ds], cond, kind, sev, ds, fa, fb,
                                    probes[args.domain_model], probes[args.ts_model], ref_ecg, ref_ppg,
                                    natives[ds], args.seed, amps, args.window_sec, accel.get(ds)))
            print(f"[fault-features] {cond}/{ds}: {len(rows):,} rows so far", flush=True)
    table = pd.DataFrame(rows)

    # 5. Disagreement scaled by its spread on clean in-distribution windows.
    ctrl = table[(table.dataset == "pulsedb_mimic") & (table.condition == "clean")]
    scale = disagreement_scale(ctrl["pred_a"], ctrl["pred_b"])
    table["model_disagreement"] = table["disagreement_raw"] / scale
    table.to_csv(args.out_dir / "fault_features.csv", index=False)

    fit = table[table.known_condition.isin(FIT_CONDITIONS)].reset_index(drop=True)
    feats = [SegmentFaultFeatures(r.sqi_value, r.sqi_drop_duration, r.accel_corr, r.source_db,
                                  r.model_disagreement, r.ecg_sqi_value, r.ppg_sqi_value) for r in fit.itertuples()]
    X, names = features_to_matrix(feats)
    np.savez(args.out_dir / "fault_features.npz", X=X, known_conditions=fit["known_condition"].to_numpy(str),
             feature_names=np.asarray(names), visit_id=fit["visit_id"].to_numpy(str),
             dataset=fit["dataset"].to_numpy(str), subject_id=fit["subject_id"].astype(str).to_numpy())
    config = dict(hf_ref_ecg=ref_ecg, hf_ref_ppg=ref_ppg, sqi_auroc_ecg=auc_ecg, sqi_auroc_ppg=auc_ppg,
                  disagreement_scale=scale, domain_model=args.domain_model, ts_model=args.ts_model,
                  modality=args.modality, seed=args.seed, window_sec=args.window_sec,
                  probe_alpha={k: v.alpha for k, v in probes.items()}, probe_n_train={k: v.n_train for k, v in probes.items()},
                  n_rows=int(len(table)), n_fit_rows=int(len(fit)),
                  rows_per_condition={f"{c}/{d}": int(n) for (c, d), n in table.groupby(["condition", "dataset"]).size().items()})
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[fault-features] wrote {len(table):,} rows ({len(fit):,} in the fit set) to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
