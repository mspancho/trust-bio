"""Degradation stress test (paper Results §2): does a fault change what a
frozen foundation model's probe predicts, and by how much?

Probes are fit exactly as in the transport evaluation (eval/transport.py):
ridge on one PulseDB institution's clean full-scale train split, alpha chosen
on its val split by heart-rate Pearson r, one alpha for every regression task.
They are then applied to the taxonomy cohort's embeddings under every
condition in the taxonomy store -- clean, motion_artifact_<sev>,
lead_off_<sev> -- for both PulseDB institutions (within-source and
cross-source) and for MIMIC-III-Ext-PPG (heart rate only). The same windows
were embedded clean and degraded, so every prediction is paired and the
analysis (eval/stress_analysis.py) can measure per-window harm, not just a
score drop.

The taxonomy cohort's subjects are removed from the probe's train and val rows
so the within-source numbers are held-out-subject numbers.

`missing_ppg` is derived, not extracted: the fusion (ecg_ppg_mean) probe is fed
the clean ECG-only vector, i.e. a deployment that falls back to the ECG encoder
when the PPG channel is absent.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..store import FeatureStore
from .probe import _fit_one, _predict
from .transport import _select_hp_on_source_val

PULSEDB_TASKS = ("hr_regression", "sbp_regression", "dbp_regression")
MIMIC_EXT_TASKS = ("hr_regression",)
TARGET_TASKS = {"pulsedb_mimic": PULSEDB_TASKS, "pulsedb_vital": PULSEDB_TASKS, "mimic_ext_ppg": MIMIC_EXT_TASKS}
FUSION = "ecg_ppg_mean"
MISSING_PPG = "missing_ppg"
_CONDITION_RE = re.compile(r"^(?P<kind>[a-z_]+)_(?P<sev>\d+(?:\.\d+)?)$")
_PRED_COLUMNS = ["target", "condition", "kind", "severity", "visit_id", "task", "y_true", "y_pred"]


def parse_condition(name: str) -> tuple[str, float]:
    """'clean' -> ('clean', 0.0); 'motion_artifact_0.3' -> ('motion_artifact', 0.3);
    'missing_ppg' -> ('missing_ppg', 1.0): the whole channel is gone."""
    if name == "clean":
        return "clean", 0.0
    if name == MISSING_PPG:
        return MISSING_PPG, 1.0
    m = _CONDITION_RE.match(name)
    if not m:
        raise ValueError(f"unrecognised condition directory {name!r}")
    return m.group("kind"), float(m.group("sev"))


def subject_of(visit_id: str) -> str:
    """PulseDB visit ids are '<subject>_w<window>'."""
    return str(visit_id).rsplit("_w", 1)[0]


def load_pulsedb_labels(label_cache: str | Path, source: str) -> pd.DataFrame:
    return (pd.read_csv(Path(label_cache) / f"pulsedb_{source}_labels.csv", dtype={"visit_id": str})
            .set_index("visit_id"))


def load_source(full_store: str | Path, source: str, model: str, modality: str, duration_sec: int,
                labels: pd.DataFrame, exclude_subjects=(), max_rows: int | None = None, seed: int = 0) -> dict:
    """Train/val rows of one institution's clean full-scale store, minus
    `exclude_subjects`; the train split may be subsampled to `max_rows`."""
    store = FeatureStore(Path(full_store) / f"pulsedb_{source}")
    excl = set(exclude_subjects)
    out = {}
    for split in ("train", "val"):
        ids, X = store.load(model, modality, duration_sec, split)
        ids = np.asarray(ids, dtype=str)
        keep = np.ones(len(ids), dtype=bool)
        if excl:
            keep &= ~pd.Series(ids).map(subject_of).isin(excl).to_numpy()
        idx = np.flatnonzero(keep)
        if split == "train" and max_rows is not None and len(idx) > max_rows:
            idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_rows, replace=False))
        if len(idx) != len(ids):
            ids, X = ids[idx], X[idx]
        out[split] = dict(ids=ids, X=X, y=labels.reindex(pd.Index(ids, name="visit_id")),
                          n_excluded=int((~keep).sum()))
    return out


def fit_probes(src: dict, tasks=PULSEDB_TASKS, seed: int = 0) -> tuple[dict, float]:
    """One ridge per task at the alpha chosen on val heart rate, as transport.py does."""
    tr, va = src["train"], src["val"]
    alpha = _select_hp_on_source_val(
        "regression", tr["X"], tr["y"]["hr_regression"].to_numpy(float),
        va["X"], va["y"]["hr_regression"].to_numpy(float), np.random.default_rng(seed))
    probes = {}
    for task in tasks:
        if task not in tr["y"].columns:
            continue
        y = tr["y"][task].to_numpy(float)
        fin = np.isfinite(y)
        if fin.sum() < 5:
            continue
        probes[task] = _fit_one("regression", tr["X"] if fin.all() else tr["X"][fin], y[fin], alpha)
    return probes, float(alpha)


def list_conditions(taxonomy_store: str | Path, target: str, model: str, modality: str,
                    duration_sec: int) -> list[str]:
    root = Path(taxonomy_store)
    return sorted(p.name for p in root.iterdir() if (p / target / model / modality / f"{duration_sec}s").is_dir())


def load_cell(cell_root: str | Path, model: str, modality: str, duration_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """All splits of one taxonomy-store cell, concatenated."""
    store = FeatureStore(cell_root)
    parts = [store.load(model, modality, duration_sec, s) for s in ("train", "val", "test")
             if store.exists(model, modality, duration_sec, s)]
    if not parts:
        raise FileNotFoundError(f"no split files under {Path(cell_root) / model / modality / f'{duration_sec}s'}")
    return (np.concatenate([np.asarray(p[0], dtype=str) for p in parts]),
            np.concatenate([p[1] for p in parts], axis=0))


def predict_frame(probes: dict, ids, X, labels: pd.DataFrame, tasks, target: str, condition: str) -> pd.DataFrame:
    kind, severity = parse_condition(condition)
    y = labels.reindex(pd.Index(ids, name="visit_id"))
    frames = [pd.DataFrame(dict(target=target, condition=condition, kind=kind, severity=severity,
                                visit_id=ids, task=task, y_true=y[task].to_numpy(float),
                                y_pred=_predict("regression", probes[task], X)))
              for task in tasks if task in probes and task in y.columns]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_PRED_COLUMNS)


def predict_cells(probes: dict, taxonomy_store: str | Path, target: str, model: str, modality: str,
                  duration_sec: int, labels: pd.DataFrame, tasks) -> pd.DataFrame:
    """Predictions for every condition of one target cell; for the fusion
    modality also the derived missing_ppg condition (clean ECG-only vector)."""
    root = Path(taxonomy_store)
    conditions = list_conditions(root, target, model, modality, duration_sec)
    if not conditions:
        raise FileNotFoundError(f"no {target}/{model}/{modality} cells under {root}")
    frames = [predict_frame(probes, *load_cell(root / cond / target, model, modality, duration_sec),
                            labels, tasks, target, cond) for cond in conditions]
    if modality == FUSION and (root / "clean" / target / model / "ecg" / f"{duration_sec}s").is_dir():
        ids, X = load_cell(root / "clean" / target, model, "ecg", duration_sec)
        frames.append(predict_frame(probes, ids, X, labels, tasks, target, MISSING_PPG))
    return pd.concat(frames, ignore_index=True)
