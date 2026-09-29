# Degradation Stress Test (Results §2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how each foundation model's linear probes degrade under the synthetic faults already embedded in the taxonomy store, and how much of that harm the label-free fault taxonomy (plan `2026-09-28-fault-taxonomy-analysis.md`) would have caught.

**Architecture:** Two stages. (a) A prediction pass, one SLURM array task per model: ridge probes are fit on one PulseDB institution's clean full-scale store exactly as the transport evaluation does (`trustbio/eval/transport.py`), with the 60 taxonomy-cohort subjects held out, then applied to the taxonomy cohort's embeddings under every condition in `features_cache/taxonomy_store/` (clean, motion_artifact_{0.1,0.3,0.6}, lead_off_{0.1,0.3,0.6}) for both institutions and MIMIC-III-Ext-PPG, writing per-window predictions. (b) An analysis pass on the pooled predictions: score-vs-severity tables, the domain-vs-time-series gap and ranking stability, fusion vs best unimodal, and a paired per-window harm analysis joined with the taxonomy's detection rules (`drop`, `outlier`, and a supervised detector) to answer "does label-free detection cover the harmful regime?".

**Tech Stack:** Python 3.11 in the pinned `trust-bio` conda env (numpy 1.26, pandas, scikit-learn 1.7, scipy 1.13, matplotlib); SLURM on HMS O2 (`short` partition); existing `FeatureStore`, `trustbio.eval.probe`, `trustbio.eval.transport`.

## Global Constraints

- Metric is Pearson r, as everywhere else in the repo (`trustbio/eval/metrics.py`); MAE is reported alongside because harm is measured per window in label units (bpm, mmHg).
- Probes are fit like the transport evaluation: `Ridge` on the source train split, alpha from `RIDGE_ALPHAS` chosen on the source val split by heart-rate r, one alpha for all regression tasks (`trustbio/eval/transport.py::_select_hp_on_source_val`, `trustbio/eval/probe.py::_fit_one`). No new hyperparameters.
- No new feature extraction. Inputs are `features_cache/full/pulsedb_{mimic,vital}/<model>/<modality>/10s/{train,val}.npz`, `features_cache/taxonomy_store/<condition>/<dataset>/<model>/<modality>/10s/{train,val,test}.npz` (verified, 1,386 files), labels `features_cache/pulsedb_{mimic,vital}_labels.csv`, cohorts `features_cache/taxonomy/*.csv`, and `results/taxonomy/fault_features.csv`.
- Subject-level honesty: the taxonomy cohort's subjects (`features_cache/taxonomy/pulsedb_<src>_cohort.csv`, 60 per institution) are removed from the probe's train and val rows, so within-source scores are held-out-subject scores.
- `motion_artifact` corrupts PPG only and `lead_off` corrupts ECG only (`trustbio/degradation/inject.py::apply_degradation`), so ECG-only probes under motion and PPG-only probes under lead-off see identical embeddings; those cells are kept (they verify the pairing gives exactly zero harm) but are not "affected" cells.
- `missing_ppg` is derived, not extracted: the fusion probe (`ecg_ppg_mean`) is fed the clean ECG-only vector, i.e. a deployment that falls back to the ECG encoder output when PPG is absent. Severity recorded as 1.0.
- Models: `xecg-10min moment-base chronos-bolt-small ecgfounder papagei dbeta ecg-domain`; modalities `ecg ppg ecg_ppg_mean`; tasks `hr_regression sbp_regression dbp_regression` on PulseDB targets, `hr_regression` on MIMIC-ext.
- Compute: one array task per model, `short` partition, 8 CPUs, 64 GB, 6 h (the transport job took 8.5 h / 101 GB for all 42 fit-sets in one process; each task here does 6 fit-sets against small targets). The login node kills ~2–3 GB python processes, so nothing that loads a full-scale matrix runs there; the analysis pass loads ~4M prediction rows and also runs as a job.
- Hooks: never `rm -rf` a path ending in an unexpanded variable; never combine deletions with `sbatch` in one command; spell out job ids rather than bracket ranges; no command may name the dotenv file.
- Results go under `results/stress/` (gitignored); code, tests, sbatch files, README and plan notes are committed to `main`.

---

## Design decisions (read before any task)

- **Why paired predictions, not just scores.** A score drop says a fault hurts on average. The paper's question is whether the windows a label-free detector flags are the windows the fault actually hurt. That needs per-window harm, `|err_degraded| − |err_clean|` for the same probe on the same window, joined by `(dataset, condition, visit_id)` to the taxonomy's per-window SQI features. The taxonomy store embedded the same windows clean and degraded, so this join is exact.
- **Detection rules reused, not re-tuned.** `drop` and `outlier` come from `scripts/run_taxonomy_degraded_only.py` (`clean_control_bounds`, `detect`). The supervised ceiling is a binary gradient-boosted detector (synthetic motion/lead-off rows vs clean controls: `clean`, `structural`, `natural_clean`) with subject-grouped out-of-fold probabilities on the 8 label-free features (`FEATURE_SETS["no_source_db"]`), thresholded at 0.5.
- **Two coverage statistics.** `harm_share_caught` = Σ positive harm in detected windows / Σ positive harm (harm-weighted recall); `material_recall` = P(detected | harm > 5 bpm or 5 mmHg) (how many windows whose estimate moved materially were flagged). Both are per probe × condition × rule; the figure averages over models for the affected modality.
- **Ranking stability** is Spearman between the seven models' clean r and their r under each condition, per (modality, source→target, task). The domain-vs-general gap is `r(xecg-10min) − r(moment-base)`, the top models of each class at both sites.
- **Fidelity check.** The clean cross-source HR r from the prediction pass should approximately reproduce `results/full_transport.csv` (same probe recipe; target is a 60-subject sample and the probe lost 60 subjects, so agreement within ~0.05 is expected, exact equality is not).

## File structure

- Create `trustbio/eval/stress.py` — probe fitting on the full store with subject exclusion; prediction over taxonomy-store cells; condition parsing; derived `missing_ppg`.
- Create `trustbio/eval/stress_analysis.py` — pure table functions: `score_table`, `paired_harm`, `rank_stability`, `gap_table`, `fusion_table`, `harm_coverage`.
- Create `scripts/run_stress_predict.py` — CLI for one model (labels, cohorts, loop over modality × source × target, writes `results/stress/predictions_<model>.csv.gz` + `alphas_<model>.json`).
- Create `scripts/run_stress_analysis.py` — CLI: pools predictions, builds detection flags (rules + supervised), writes tables, summary JSON, figures.
- Create `scripts/run_stress.sbatch` (array over models) and `scripts/run_stress_analysis.sbatch` (dependent).
- Test `tests/test_eval_stress.py`, `tests/test_stress_analysis.py`.
- Modify `README.md` (new section), this plan's Outcome section, memory.

---

### Task 1: Probe fitting and cell prediction (`trustbio/eval/stress.py`)

**Files:**
- Create: `trustbio/eval/stress.py`
- Test: `tests/test_eval_stress.py`

**Interfaces:**
- Consumes: `FeatureStore.load/exists` (`trustbio/store.py`), `_fit_one`, `_predict` (`trustbio/eval/probe.py`), `_select_hp_on_source_val` (`trustbio/eval/transport.py`).
- Produces: `parse_condition(name) -> (kind: str, severity: float)`; `subject_of(visit_id) -> str`; `load_pulsedb_labels(label_cache, source) -> DataFrame[hr_regression, sbp_regression, dbp_regression] indexed by visit_id`; `load_source(full_store, source, model, modality, duration_sec, labels, exclude_subjects=(), max_rows=None, seed=0) -> {"train": {"ids","X","y","n_excluded"}, "val": {...}}`; `fit_probes(src, tasks=PULSEDB_TASKS, seed=0) -> (dict[task, Ridge], alpha)`; `load_cell(cell_root, model, modality, duration_sec) -> (ids, X)`; `predict_frame(probes, ids, X, labels, tasks, target, condition) -> DataFrame[target, condition, kind, severity, visit_id, task, y_true, y_pred]`; `predict_cells(probes, taxonomy_store, target, model, modality, duration_sec, labels, tasks) -> DataFrame`; constants `PULSEDB_TASKS`, `MIMIC_EXT_TASKS`, `TARGET_TASKS`, `FUSION`, `MISSING_PPG`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_stress.py
import json

import numpy as np
import pandas as pd
import pytest

from trustbio.store import FeatureStore


def _ids(subjects, n_windows):
    return [f"{s}_w{w}" for s in subjects for w in range(n_windows)]


def _labels(ids, X, rng):
    w = rng.normal(size=X.shape[1])
    hr = 70 + 10 * X @ w + rng.normal(0, 0.5, len(ids))
    return pd.DataFrame(dict(visit_id=ids, hr_regression=hr, sbp_regression=120 + hr / 2, dbp_regression=60 + hr / 4))


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(0)
    dim, model = 6, "m"
    tr_subj = [f"s{i:03d}" for i in range(50)]
    va_subj = [f"s{i:03d}" for i in range(50, 60)]
    tax_subj = tr_subj[:5]                       # taxonomy cohort = first 5 train subjects
    full = FeatureStore(tmp_path / "full" / "pulsedb_mimic")
    frames = []
    X_by_split = {}
    for split, subj in (("train", tr_subj), ("val", va_subj)):
        ids = _ids(subj, 10)
        X = rng.normal(size=(len(ids), dim)).astype(np.float32)
        X_by_split[split] = (ids, X)
    # one label vector, same weights for every split
    w = rng.normal(size=dim)
    for split, (ids, X) in X_by_split.items():
        hr = 70 + 10 * X @ w + rng.normal(0, 0.5, len(ids))
        frames.append(pd.DataFrame(dict(visit_id=ids, hr_regression=hr, sbp_regression=120 + hr / 2, dbp_regression=60 + hr / 4)))
        for modality in ("ecg", "ppg", "ecg_ppg_mean"):
            full.save(model, modality, 10, split, ids, X)
    pd.concat(frames).to_csv(tmp_path / "pulsedb_mimic_labels.csv", index=False)
    (tmp_path / "tax").mkdir()
    pd.DataFrame(dict(visit_id=_ids(tax_subj, 10), subject_id=[s for s in tax_subj for _ in range(10)],
                      source="mimic", split="train")).to_csv(tmp_path / "tax" / "pulsedb_mimic_cohort.csv", index=False)
    # taxonomy store: the 5 taxonomy subjects' clean train vectors, then degraded copies
    tax_ids = _ids(tax_subj, 10)
    tr_ids, X_tr = X_by_split["train"]
    X_tax = X_tr[[tr_ids.index(v) for v in tax_ids]]
    noisy = (X_tax + rng.normal(0, 3, X_tax.shape)).astype(np.float32)
    cells = {
        ("clean", "ecg"): X_tax, ("clean", "ppg"): X_tax, ("clean", "ecg_ppg_mean"): X_tax,
        ("lead_off_0.6", "ecg"): noisy, ("lead_off_0.6", "ppg"): X_tax, ("lead_off_0.6", "ecg_ppg_mean"): noisy,
        ("motion_artifact_0.3", "ecg"): X_tax, ("motion_artifact_0.3", "ppg"): noisy, ("motion_artifact_0.3", "ecg_ppg_mean"): noisy,
    }
    for (cond, modality), X in cells.items():
        FeatureStore(tmp_path / "taxonomy_store" / cond / "pulsedb_mimic").save(model, modality, 10, "test", tax_ids, X)
    return tmp_path


def test_parse_condition_and_subject():
    from trustbio.eval.stress import parse_condition, subject_of
    assert parse_condition("clean") == ("clean", 0.0)
    assert parse_condition("motion_artifact_0.3") == ("motion_artifact", 0.3)
    assert parse_condition("lead_off_0.6") == ("lead_off", 0.6)
    assert parse_condition("missing_ppg") == ("missing_ppg", 1.0)
    with pytest.raises(ValueError):
        parse_condition("weird")
    assert subject_of("p001049_w12") == "p001049"


def test_load_source_excludes_taxonomy_subjects(synthetic):
    from trustbio.eval.stress import load_pulsedb_labels, load_source
    labels = load_pulsedb_labels(synthetic, "mimic")
    src = load_source(synthetic / "full", "mimic", "m", "ecg", 10, labels, exclude_subjects={"s000", "s001", "s002", "s003", "s004"})
    assert len(src["train"]["ids"]) == 450 and src["train"]["n_excluded"] == 50
    assert not any(v.startswith("s000_") for v in src["train"]["ids"])
    assert len(src["val"]["ids"]) == 100 and src["val"]["n_excluded"] == 0
    assert src["train"]["y"].index.tolist() == src["train"]["ids"].tolist()
    sub = load_source(synthetic / "full", "mimic", "m", "ecg", 10, labels, max_rows=100, seed=1)
    assert len(sub["train"]["ids"]) == 100 and len(sub["val"]["ids"]) == 100


def test_predict_cli_end_to_end(synthetic):
    from scripts.run_stress_predict import main
    out = synthetic / "out"
    rc = main(["--model", "m", "--full-store", str(synthetic / "full"), "--label-cache", str(synthetic),
               "--taxonomy-store", str(synthetic / "taxonomy_store"), "--taxonomy-cohort-cache", str(synthetic / "tax"),
               "--out-dir", str(out), "--sources", "mimic", "--targets", "pulsedb_mimic"])
    assert rc == 0
    pred = pd.read_csv(out / "predictions_m.csv.gz", dtype={"visit_id": str})
    assert set(pred.columns) >= {"model", "modality", "source", "target", "condition", "kind", "severity", "visit_id", "task", "y_true", "y_pred"}
    conds = pred.groupby("modality")["condition"].apply(set)
    assert conds["ecg"] == {"clean", "lead_off_0.6", "motion_artifact_0.3"}
    assert conds["ecg_ppg_mean"] == {"clean", "lead_off_0.6", "motion_artifact_0.3", "missing_ppg"}
    hr = pred[pred.task == "hr_regression"]

    def r(modality, cond):
        g = hr[(hr.modality == modality) & (hr.condition == cond)]
        return np.corrcoef(g.y_true, g.y_pred)[0, 1]

    assert r("ecg", "clean") > 0.9
    assert r("ecg", "lead_off_0.6") < r("ecg", "clean") - 0.2
    # PPG is untouched by lead-off: identical embeddings give identical predictions
    a = hr[(hr.modality == "ppg") & (hr.condition == "clean")].set_index("visit_id").y_pred
    b = hr[(hr.modality == "ppg") & (hr.condition == "lead_off_0.6")].set_index("visit_id").y_pred
    assert np.allclose(a, b.reindex(a.index))
    assert (pred[pred.condition == "missing_ppg"].severity == 1.0).all()
    assert len(pred[(pred.condition == "missing_ppg")]) == 50 * 3
    alphas = json.loads((out / "alphas_m.json").read_text())
    assert alphas["ecg/mimic"]["n_train"] == 450 and alphas["ecg/mimic"]["n_excluded_train"] == 50
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run --no-capture-output -n trust-bio python -m pytest tests/test_eval_stress.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trustbio.eval.stress'`.

- [ ] **Step 3: Create `trustbio/eval/stress.py`**

```python
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


def list_conditions(taxonomy_store: str | Path, target: str, model: str, modality: str, duration_sec: int) -> list[str]:
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
```

- [ ] **Step 4: Create `scripts/run_stress_predict.py`** (the CLI the third test drives)

```python
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
            alphas[f"{modality}/{source}"] = dict(alpha=alpha, n_train=n_train, n_excluded_train=n_excl, tasks=sorted(probes))
            print(f"[stress] {args.model}/{modality}/{source}: {len(probes)} probes on {n_train:,} rows "
                  f"(alpha={alpha:.4g}; {n_excl:,} taxonomy-subject rows held out) in {time.time() - t0:.0f}s", flush=True)
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
```

- [ ] **Step 5: Run the tests** — `conda run --no-capture-output -n trust-bio python -m pytest tests/test_eval_stress.py -q` → `3 passed`.

- [ ] **Step 6: Commit** — `git add trustbio/eval/stress.py scripts/run_stress_predict.py tests/test_eval_stress.py && git commit -m "feat: degradation stress test, probe prediction pass (plan Task 1)"`.

### Task 2: Launch the prediction array

**Files:**
- Create: `scripts/run_stress.sbatch`, `scripts/run_stress_analysis.sbatch`

**Interfaces:**
- Consumes: `scripts/run_stress_predict.py` CLI (Task 1); `scripts/run_stress_analysis.py` CLI (Task 4, may not exist yet when the array is launched: the analysis job is submitted only after Task 4).
- Produces: `results/stress/predictions_<model>.csv.gz` × 7, `results/stress/alphas_<model>.json` × 7, logs `logs/stress_<array>_<task>.out`.

- [ ] **Step 1: Write `scripts/run_stress.sbatch`**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=trustbio-stress
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/stress_%A_%a.out
#SBATCH --error=logs/stress_%A_%a.err
#
# Degradation stress test, prediction pass: one array task per model. Each task
# fits 6 probe sets (3 modalities x 2 source institutions) on the clean
# full-scale store and predicts the taxonomy store's cells. The full-scale
# train matrix is up to 11 GB (2.69M x 1024 float32) and ridge copies it, so
# 64 GB; the transport job (all 42 fit-sets in one process) took 8.5 h / 101 GB.
set -euo pipefail
MODELS=(xecg-10min moment-base chronos-bolt-small ecgfounder papagei dbeta ecg-domain)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}/stress"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
conda run --no-capture-output -n "${ENV_NAME}" python scripts/run_stress_predict.py \
  --model "${MODEL}" --out-dir "${OUT}"
```

- [ ] **Step 2: Write `scripts/run_stress_analysis.sbatch`**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=trustbio-stress-analysis
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/stress_analysis_%j.out
#SBATCH --error=logs/stress_analysis_%j.err
#
# Pools the per-model prediction files (~4M rows) into the Results §2 tables and
# figures. Submit with --dependency=afterok:<array job id> of run_stress.sbatch.
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}/stress"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"
conda run --no-capture-output -n "${ENV_NAME}" python scripts/run_stress_analysis.py \
  --pred-dir "${OUT}" --fault-features results/taxonomy/fault_features.csv --out-dir "${OUT}"
```

- [ ] **Step 3: Submit the array** — `sbatch scripts/run_stress.sbatch` (from the repo root). Record the job id. Watch `logs/stress_<id>_6.out` (ecg-domain, 54-d, finishes first) for the `[stress] ... probes on N rows` lines; each task should print 6 of them then `wrote ... rows`.

- [ ] **Step 4: Commit** — `git add scripts/run_stress.sbatch scripts/run_stress_analysis.sbatch && git commit -m "chore: stress-test sbatch array and analysis job (plan Task 2)"`.

### Task 3: Analysis tables (`trustbio/eval/stress_analysis.py`)

**Files:**
- Create: `trustbio/eval/stress_analysis.py`
- Test: `tests/test_stress_analysis.py`

**Interfaces:**
- Consumes: the long prediction frame from Task 1 (`model, modality, source, target, condition, kind, severity, visit_id, task, y_true, y_pred`) and a detection-flag frame (`dataset, condition, visit_id, det_drop, det_outlier, det_supervised`) built by Task 4.
- Produces: `pearson_r(y, p) -> float`; `score_table(pred) -> DataFrame[KEYS + COND + n, r, mae, r_clean, mae_clean, delta_r, delta_mae]`; `paired_harm(pred) -> DataFrame[KEYS + COND + visit_id, y_true, harm]`; `rank_stability(scores, metric="r") -> DataFrame[modality, source, target, task, condition, kind, severity, n_models, spearman, top_clean, top_degraded]`; `gap_table(scores, domain, ts, metric="r") -> DataFrame[..., r_domain, r_ts, gap, gap_clean]`; `fusion_table(scores, metric="r") -> DataFrame[model, source, target, task, condition, kind, severity, r_fusion, best_unimodal, r_best_unimodal, fusion_minus_best]`; `harm_coverage(harm, flags, rules=("drop","outlier","supervised"), material=MATERIAL_HARM) -> DataFrame[KEYS + COND + rule, n, detection_rate, mean_harm, mean_harm_undetected, harm_share_caught, material_rate, material_recall]`; constants `KEYS`, `COND`, `CLEAN`, `MATERIAL_HARM`, `AFFECTED`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stress_analysis.py
import numpy as np
import pandas as pd


def _pred(seed=0):
    """3 models of decreasing quality x 3 modalities, one source/target/task.
    lead_off_0.6 harms ecg + fusion (noisier prediction), motion_artifact_0.3
    harms ppg + fusion; the untouched modality is IDENTICAL to clean."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i:02d}_w{w}" for i in range(20) for w in range(5)]
    y = 70 + rng.normal(0, 10, len(ids))
    rows = []
    for m, noise in (("A", 1.0), ("B", 3.0), ("C", 6.0)):
        base = {mod: y + rng.normal(0, noise, len(ids)) for mod in ("ecg", "ppg", "ecg_ppg_mean")}
        for mod in base:
            for cond, kind, sev, hit in (("clean", "clean", 0.0, ()), ("lead_off_0.6", "lead_off", 0.6, ("ecg", "ecg_ppg_mean")),
                                         ("motion_artifact_0.3", "motion_artifact", 0.3, ("ppg", "ecg_ppg_mean"))):
                pred = base[mod] + (rng.normal(0, 8, len(ids)) if mod in hit else 0.0)
                rows += [dict(model=m, modality=mod, source="mimic", target="pulsedb_mimic", condition=cond, kind=kind,
                              severity=sev, visit_id=v, task="hr_regression", y_true=yy, y_pred=p)
                         for v, yy, p in zip(ids, y, pred)]
        # missing_ppg for fusion only: fusion probe on ECG-only vector
        for v, yy, p in zip(ids, y, base["ecg"] + 1.0):
            rows.append(dict(model=m, modality="ecg_ppg_mean", source="mimic", target="pulsedb_mimic", condition="missing_ppg",
                             kind="missing_ppg", severity=1.0, visit_id=v, task="hr_regression", y_true=yy, y_pred=p))
    return pd.DataFrame(rows)


def test_score_table_and_deltas():
    from trustbio.eval.stress_analysis import score_table
    s = score_table(_pred())
    a = s[(s.model == "A") & (s.modality == "ecg")].set_index("condition")
    assert a.loc["clean", "r"] > 0.95 and a.loc["clean", "delta_r"] == 0.0
    assert a.loc["lead_off_0.6", "delta_r"] < -0.05 and a.loc["lead_off_0.6", "delta_mae"] > 1.0
    assert abs(a.loc["motion_artifact_0.3", "delta_r"]) < 1e-12          # ecg untouched by motion
    assert {"n", "r", "mae", "r_clean", "mae_clean", "delta_r", "delta_mae"} <= set(s.columns)


def test_paired_harm_is_zero_for_untouched_modality():
    from trustbio.eval.stress_analysis import paired_harm
    h = paired_harm(_pred())
    assert "clean" not in set(h.condition)
    ecg_motion = h[(h.modality == "ecg") & (h.condition == "motion_artifact_0.3")]
    assert np.allclose(ecg_motion.harm, 0.0)
    ecg_lead = h[(h.modality == "ecg") & (h.condition == "lead_off_0.6") & (h.model == "A")]
    assert ecg_lead.harm.mean() > 1.0 and len(ecg_lead) == 100


def test_rank_stability_gap_and_fusion():
    from trustbio.eval.stress_analysis import fusion_table, gap_table, rank_stability, score_table
    s = score_table(_pred())
    st = rank_stability(s)
    row = st[(st.modality == "ecg") & (st.condition == "lead_off_0.6")].iloc[0]
    assert row.n_models == 3 and row.spearman == 1.0 and row.top_clean == "A"
    g = gap_table(s, domain="A", ts="C")
    ge = g[(g.modality == "ecg")].set_index("condition")
    assert ge.loc["clean", "gap"] > 0 and (ge.gap_clean == ge.loc["clean", "gap"]).all()
    f = fusion_table(s)
    fa = f[f.model == "A"].set_index("condition")
    assert fa.loc["lead_off_0.6", "best_unimodal"] == "ppg"           # ppg untouched by lead-off
    assert fa.loc["motion_artifact_0.3", "best_unimodal"] == "ecg"
    assert fa.loc["missing_ppg", "best_unimodal"] == "ecg"
    assert np.isclose(fa.loc["missing_ppg", "fusion_minus_best"], fa.loc["missing_ppg", "r_fusion"] - fa.loc["clean", "r_best_unimodal"]
                      if fa.loc["clean", "best_unimodal"] == "ecg" else fa.loc["missing_ppg", "fusion_minus_best"])


def test_harm_coverage_math():
    from trustbio.eval.stress_analysis import harm_coverage, paired_harm
    p = _pred()
    h = paired_harm(p)
    key = h[(h.model == "A") & (h.modality == "ecg") & (h.condition == "lead_off_0.6")]
    flags = pd.DataFrame(dict(dataset="pulsedb_mimic", condition="lead_off_0.6", visit_id=key.visit_id.to_numpy(),
                              det_drop=(key.harm > 0).to_numpy(), det_outlier=False, det_supervised=(key.harm > 5.0).to_numpy()))
    cov = harm_coverage(h, flags).set_index(["model", "modality", "condition", "rule"])
    drop = cov.loc[("A", "ecg", "lead_off_0.6", "drop")]
    assert drop.harm_share_caught == 1.0 and drop.material_recall == 1.0 and drop.n == 100
    outl = cov.loc[("A", "ecg", "lead_off_0.6", "outlier")]
    assert outl.harm_share_caught == 0.0 and outl.material_recall == 0.0 and outl.detection_rate == 0.0
    sup = cov.loc[("A", "ecg", "lead_off_0.6", "supervised")]
    assert sup.material_recall == 1.0 and 0.0 < sup.harm_share_caught < 1.0
    assert ("A", "ecg", "motion_artifact_0.3", "drop") not in cov.index     # no flags for that condition -> not joined
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: trustbio.eval.stress_analysis`.

- [ ] **Step 3: Create `trustbio/eval/stress_analysis.py`**

```python
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
AFFECTED = {"motion_artifact": ("ppg", "ecg_ppg_mean"), "lead_off": ("ecg", "ecg_ppg_mean"), "missing_ppg": ("ecg_ppg_mean",)}


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
    clean = pred[pred["condition"] == CLEAN][KEYS + ["visit_id", "y_pred"]].rename(columns={"y_pred": "y_pred_clean"})
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
            rows.append(dict(zip(grp, key), condition=cond[0], kind=cond[1], severity=cond[2], n_models=int(ok.sum()),
                             spearman=rho, top_clean=base[ok].idxmax() if ok.any() else None,
                             top_degraded=cur[ok].idxmax() if ok.any() else None))
    return pd.DataFrame(rows)


def gap_table(scores: pd.DataFrame, domain: str = "xecg-10min", ts: str = "moment-base", metric: str = "r") -> pd.DataFrame:
    """r(domain model) - r(time-series model) per condition, with the clean gap alongside."""
    idx = ["modality", "source", "target", "task"] + COND
    d = scores[scores["model"] == domain].set_index(idx)[metric].rename("r_domain")
    t = scores[scores["model"] == ts].set_index(idx)[metric].rename("r_ts")
    out = pd.concat([d, t], axis=1).reset_index()
    out["gap"] = out["r_domain"] - out["r_ts"]
    base = out[out["condition"] == CLEAN][["modality", "source", "target", "task", "gap"]].rename(columns={"gap": "gap_clean"})
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
            cands = {"ecg": by.get(("ecg", CLEAN), np.nan)} if cond == "missing_ppg" else \
                    {m: by.get((m, cond), np.nan) for m in ("ecg", "ppg")}
            best = max(cands, key=lambda m: -np.inf if np.isnan(cands[m]) else cands[m])
            rows.append(dict(zip(["model", "source", "target", "task"], key), condition=cond, kind=meta[cond][0],
                             severity=meta[cond][1], r_fusion=by[("ecg_ppg_mean", cond)], best_unimodal=best,
                             r_best_unimodal=cands[best], fusion_minus_best=by[("ecg_ppg_mean", cond)] - cands[best]))
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
```

- [ ] **Step 4: Run the tests** — `conda run --no-capture-output -n trust-bio python -m pytest tests/test_stress_analysis.py -q` → `4 passed`.

- [ ] **Step 5: Commit** — `git add trustbio/eval/stress_analysis.py tests/test_stress_analysis.py && git commit -m "feat: stress-test analysis tables (plan Task 3)"`.

### Task 4: Analysis CLI with detection flags and figures (`scripts/run_stress_analysis.py`)

**Files:**
- Create: `scripts/run_stress_analysis.py`
- Test: `tests/test_run_stress_analysis.py`

**Interfaces:**
- Consumes: Task 3 functions; `clean_control_bounds(table)`, `detect(table, rule, bounds)` from `scripts/run_taxonomy_degraded_only.py`; `FEATURE_SETS["no_source_db"]` from `scripts/run_taxonomy.py`.
- Produces (in `--out-dir`): `stress_scores.csv`, `stress_rank_stability.csv`, `stress_gap.csv`, `stress_fusion.csv`, `stress_harm_coverage.csv`, `stress_detection_flags.csv`, `stress_summary.json` (`n_predictions`, `models`, `fidelity_vs_transport` = per model/modality abs diff of clean cross-source HR r vs `results/full_transport.csv` when present, `headline` = mean over models of delta_r / harm_share_caught for the affected modality per kind × severity), `figures/fig2a_severity_curves_<task>.png`, `fig2b_gap_<task>.png`, `fig2c_fusion_<task>.png`, `fig2d_harm_coverage_<task>.png` for each task present.
- Functions: `supervised_flags(table, seed=0) -> np.ndarray[bool]`; `detection_flags(table, seed=0) -> DataFrame[dataset, condition, visit_id, known_condition, det_drop, det_outlier, det_supervised]`; `load_predictions(pred_dir) -> DataFrame`; `main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_stress_analysis.py
import json

import numpy as np
import pandas as pd

from tests.test_stress_analysis import _pred


def _fault_features(pred, rng):
    """One taxonomy row per (dataset, condition, visit_id) of the synthetic
    predictions, with the 8 label-free features; severe rows carry drops."""
    keys = pred[pred.condition != "missing_ppg"][["target", "condition", "kind", "severity", "visit_id"]].drop_duplicates()
    n = len(keys)
    sev = keys.severity.fillna(0).to_numpy()
    drop = np.where(sev >= 0.6, 5.0 + rng.normal(0, 0.5, n), 0.0)
    return pd.DataFrame(dict(
        dataset=keys.target.to_numpy(), condition=keys.condition.to_numpy(), visit_id=keys.visit_id.to_numpy(),
        kind=keys.kind.where(keys.kind != "clean", np.nan).to_numpy(), severity=keys.severity.where(keys.severity > 0, np.nan).to_numpy(),
        known_condition=np.where(keys.kind == "clean", "clean", keys.kind).astype(object),
        subject_id=[v.split("_w")[0] for v in keys.visit_id], source_db=keys.target.to_numpy(),
        sqi_value=1 - drop / 10 + rng.normal(0, 0.01, n), sqi_drop_duration=drop, accel_corr=0.0,
        model_disagreement=1.0 + sev + rng.normal(0, 0.1, n), ecg_sqi_value=1.0 - drop / 10,
        ppg_sqi_value=0.93 - sev / 5 + rng.normal(0, 0.01, n), ecg_drop_duration=drop, ppg_drop_duration=0.0))


def test_analysis_cli_end_to_end(tmp_path):
    from scripts.run_stress_analysis import main
    rng = np.random.default_rng(0)
    pred = _pred()
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for m, g in pred.groupby("model"):
        g.to_csv(pred_dir / f"predictions_{m}.csv.gz", index=False)
    ff = tmp_path / "fault_features.csv"
    _fault_features(pred, rng).to_csv(ff, index=False)
    out = tmp_path / "out"
    assert main(["--pred-dir", str(pred_dir), "--fault-features", str(ff), "--out-dir", str(out),
                 "--domain", "A", "--ts", "C"]) == 0
    for name in ("stress_scores.csv", "stress_rank_stability.csv", "stress_gap.csv", "stress_fusion.csv",
                 "stress_harm_coverage.csv", "stress_detection_flags.csv", "stress_summary.json"):
        assert (out / name).exists(), name
    assert (out / "figures" / "fig2a_severity_curves_hr_regression.png").exists()
    assert (out / "figures" / "fig2d_harm_coverage_hr_regression.png").exists()
    cov = pd.read_csv(out / "stress_harm_coverage.csv")
    lead = cov[(cov.condition == "lead_off_0.6") & (cov.modality == "ecg") & (cov.rule == "drop")]
    assert (lead.detection_rate == 1.0).all()                       # severe rows carry drops
    summary = json.loads((out / "stress_summary.json").read_text())
    assert summary["models"] == ["A", "B", "C"] and summary["n_predictions"] == len(pred)
    flags = pd.read_csv(out / "stress_detection_flags.csv")
    assert {"det_drop", "det_outlier", "det_supervised"} <= set(flags.columns)
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError: scripts.run_stress_analysis`.

- [ ] **Step 3: Create `scripts/run_stress_analysis.py`**

```python
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
        clf = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed).fit(X[idx], y[idx])
        p[~labelled] = clf.predict_proba(X[~labelled])[:, 1]
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


def plot_severity_curves(scores: pd.DataFrame, task: str, path: Path) -> None:
    s = scores[(scores["task"] == task) & (scores["kind"] != "missing_ppg")]
    pairs = sorted({(r.source, r.target) for r in s.itertuples()})
    kinds = ["motion_artifact", "lead_off"]
    fig, axes = plt.subplots(len(pairs), len(kinds), figsize=(4.2 * len(kinds), 2.6 * len(pairs)), squeeze=False, sharey=True)
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
                ax.plot(gg["severity"], gg["fusion_minus_best"], marker=marker, ms=3, lw=0.8, label=f"{model} / {kind}")
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
    c = cov[cov["task"] == task]
    c = c[[m in AFFECTED.get(k, ()) for k, m in zip(c["kind"], c["modality"])]]
    if c.empty:
        return
    agg = (c.groupby(["kind", "severity", "modality", "rule"])[["harm_share_caught", "material_recall", "material_rate", "mean_harm"]]
           .mean().reset_index())
    cells = sorted({(k, s, m) for k, s, m in zip(agg["kind"], agg["severity"], agg["modality"])})
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(cells)), 3.2))
    width = 0.8 / len(RULES)
    for r_i, rule in enumerate(RULES):
        vals = [agg[(agg.kind == k) & (agg.severity == s) & (agg.modality == m) & (agg.rule == rule)]["harm_share_caught"].mean()
                for k, s, m in cells]
        ax.bar(np.arange(len(cells)) + (r_i - 1) * width, vals, width, label=f"{rule}: share of harm caught")
    mat = [agg[(agg.kind == k) & (agg.severity == s) & (agg.modality == m)]["material_rate"].mean() for k, s, m in cells]
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
            out[f"{r.model}/{r.modality}/{key[2]}"] = dict(stress_r=float(r.r), transport_r=float(t[key]), abs_diff=float(abs(r.r - t[key])))
    return out


def headline(scores: pd.DataFrame, cov: pd.DataFrame) -> dict:
    """Mean over models (and source/target pairs) of delta_r and harm coverage,
    affected modality only, per task x kind x severity."""
    out = {}
    s = scores[[m in AFFECTED.get(k, ()) for k, m in zip(scores["kind"], scores["modality"])]]
    for (task, kind, sev, mod), g in s.groupby(["task", "kind", "severity", "modality"]):
        out.setdefault(task, {})[f"{kind}/{sev}/{mod}"] = dict(delta_r_mean=float(g["delta_r"].mean()),
                                                                  delta_mae_mean=float(g["delta_mae"].mean()))
    c = cov[[m in AFFECTED.get(k, ()) for k, m in zip(cov["kind"], cov["modality"])]]
    for (task, kind, sev, mod, rule), g in c.groupby(["task", "kind", "severity", "modality", "rule"]):
        out.setdefault(task, {}).setdefault(f"{kind}/{sev}/{mod}", {})[f"harm_share_caught_{rule}"] = float(g["harm_share_caught"].mean())
        out[task][f"{kind}/{sev}/{mod}"][f"material_recall_{rule}"] = float(g["material_recall"].mean())
        out[task][f"{kind}/{sev}/{mod}"]["material_rate"] = float(g["material_rate"].mean())
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
```

- [ ] **Step 4: Run the tests** — `conda run --no-capture-output -n trust-bio python -m pytest tests/test_run_stress_analysis.py tests/test_stress_analysis.py -q` → `5 passed`.

- [ ] **Step 5: Submit the analysis job once the array finishes** — `sbatch --dependency=afterok:<array id> scripts/run_stress_analysis.sbatch` (spelled-out id; submit in a command of its own).

- [ ] **Step 6: Commit** — `git add scripts/run_stress_analysis.py tests/test_run_stress_analysis.py && git commit -m "feat: stress-test analysis CLI with detection coverage and figures (plan Task 4)"`.

### Task 5: Read the results, record, commit

**Files:**
- Modify: `README.md` (new section after the taxonomy sections), this plan (Outcome section below), memory file `trustbio-taxonomy-findings.md` (or a new `trustbio-stress-findings.md`).

- [ ] **Step 1: Verify the array** — every `logs/stress_<id>_<k>.out` ends with `wrote ... rows`; `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State` shows 7 COMPLETED; `ls results/stress/predictions_*.csv.gz | wc -l` = 7.
- [ ] **Step 2: Read** `stress_summary.json` (fidelity: clean cross-source HR r within ~0.05 of `full_transport.csv`; headline delta_r and coverage per kind × severity), `stress_scores.csv` (r by severity for the affected modalities, within vs cross), `stress_rank_stability.csv` (does the seven-model ranking hold?), `stress_gap.csv`, `stress_fusion.csv`, `stress_harm_coverage.csv` (drop / outlier / supervised coverage of harm at severity 0.3 for motion on PPG/fusion; this is the sentence the user asked for), and the four figures.
- [ ] **Step 3: Record** a README section "Degradation stress test (plan 2026-09-29, Results §2)" with job ids and the numbers, the plan's Outcome section, and memory.
- [ ] **Step 4: Commit** — `git add README.md docs/superpowers/plans/2026-09-29-degradation-stress-test.md && git commit -m "docs: degradation stress test outcome (Results §2)"` and push `main`.

## Self-review notes

- **Spec coverage.** Draft Results §2 asks for performance under motion artifact, lead-off and missing-PPG at graded severity for each model class and dataset (Table 2: `stress_scores.csv` by model × modality × source→target × condition; MIMIC-ext for HR), the change in the domain-vs-time-series gap with severity (Figure 2b: `stress_gap.csv`, plus ranking stability), fusion decomposed by which modality is degraded (Figure 2c: `stress_fusion.csv`; motion degrades PPG, lead-off degrades ECG, missing-PPG removes PPG), and the link the user asked for between harm and label-free detection (`stress_harm_coverage.csv`, Figure 2d). Figure 2d of the draft (synthetic vs real degradation statistics) was already covered by the taxonomy plan's calibration.
- **Placeholder scan.** Every step shows code; the only unknowns are job ids and numbers, recorded in Task 5.
- **Type consistency.** `predict_cells` returns the eight prediction columns; the CLI prepends `model, modality, source`; `score_table`/`paired_harm`/`harm_coverage` key on `KEYS + COND` with the same names; `detection_flags` columns `dataset, condition, visit_id, det_drop, det_outlier, det_supervised` match `harm_coverage`'s join (`dataset` renamed to `target`); `parse_condition` severities (clean 0.0, missing_ppg 1.0) match what the figures plot.
- **Known risks.** (1) Memory: an array task loads one 11 GB matrix per fit-set and ridge copies it; 64 GB leaves 3× headroom. (2) Time: ~34 ridge fits per fit-set at 2.7M × 1024; the 6 h limit is ~3× the estimate; if a task times out, rerun it alone with `sbatch --array=<k> scripts/run_stress.sbatch`. (3) The `outlier` rule is device-confounded on BUT PPG (taxonomy Task 15), irrelevant here because coverage is computed on PulseDB/MIMIC-ext synthetic rows only. (4) `structural`-labelled clean Vital rows are negatives for the supervised detector; they are clean windows, so that is correct.
