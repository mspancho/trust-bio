#!/usr/bin/env python
"""Shared DatasetHandle construction for the four (dataset) choices, used by
both extract_features.py (stage 3) and run_benchmark.py (stage 4c). Factored
out here so both CLIs build a DatasetHandle the same way from CLI args, rather
than one of them depending on an artifact (e.g. a pickle file) the other
produces."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trustbio.data.pulsedb import build_pulsedb_cohort, build_pulsedb_label_table, make_pulsedb_signal_loader
from trustbio.data.mimic_ext_ppg import (
    build_mimic_ext_ppg_cohort, build_mimic_ext_ppg_label_table,
    make_mimic_ext_ppg_signal_loader,
)
from trustbio.data.but_ppg import (
    build_but_ppg_cohort, build_but_ppg_label_table, make_but_ppg_signal_loader,
)
from trustbio.degradation.inject import apply_degradation
from trustbio.pipeline import DatasetHandle

DATASET_CHOICES = ["pulsedb_mimic", "pulsedb_vital", "mimic_ext_ppg", "but_ppg"]


def add_dataset_root_args(ap):
    """Shared --pulsedb-root/--mimic-ext-ppg-root/--but-ppg-root arguments."""
    from pathlib import Path
    from trustbio.data.pulsedb import DEFAULT_ROOT as PULSEDB_ROOT
    from trustbio.data.mimic_ext_ppg import DEFAULT_ROOT as MIMIC_EXT_PPG_ROOT
    from trustbio.data.but_ppg import DEFAULT_ROOT as BUT_PPG_ROOT

    ap.add_argument("--pulsedb-root", type=Path, default=PULSEDB_ROOT)
    ap.add_argument("--mimic-ext-ppg-root", type=Path, default=MIMIC_EXT_PPG_ROOT)
    ap.add_argument("--but-ppg-root", type=Path, default=BUT_PPG_ROOT)
    return ap


def wrap_loader_with_degradation(load_signal, kind, severity, seed):
    """Wrap a SignalLoader so ECG/PPG pairs pass through apply_degradation
    before the model's own preprocessing sees them. `kind=None` disables
    degradation entirely (the clean baseline condition)."""
    if kind is None or severity is None:
        return load_signal
    rng = np.random.default_rng(seed)

    def degraded_load(visit_id, modality):
        raw, sig_fs = load_signal(visit_id, modality)
        if modality == "ecg":
            ecg_out, _ = apply_degradation(raw, None, sig_fs, kind, severity, rng)
            return ecg_out, sig_fs
        _, ppg_out = apply_degradation(np.zeros_like(raw), raw, sig_fs, kind, severity, rng)
        if ppg_out is None:
            raise ValueError("missing_ppg degradation: PPG channel dropped for this visit")
        return ppg_out, sig_fs

    return degraded_load


def build_dataset_handle(
    name: str, args, degrade_kind: str | None = None,
    degrade_severity: float | None = None, seed: int = 0,
) -> DatasetHandle:
    """Build a DatasetHandle for one of DATASET_CHOICES from parsed CLI args
    (which must include --pulsedb-root/--mimic-ext-ppg-root/--but-ppg-root, via
    add_dataset_root_args)."""
    if name == "pulsedb_mimic":
        cohort = build_pulsedb_cohort(args.pulsedb_root, source="mimic")
        splits = {s: cohort.split(s) for s in ("train", "val", "test")}
        loader = make_pulsedb_signal_loader(args.pulsedb_root, source="mimic")
        labels = {s: build_pulsedb_label_table(args.pulsedb_root, "mimic", df["visit_id"].tolist())
                  for s, df in splits.items()}
    elif name == "pulsedb_vital":
        cohort = build_pulsedb_cohort(args.pulsedb_root, source="vital")
        splits = {s: cohort.split(s) for s in ("train", "val", "test")}
        loader = make_pulsedb_signal_loader(args.pulsedb_root, source="vital")
        labels = {s: build_pulsedb_label_table(args.pulsedb_root, "vital", df["visit_id"].tolist())
                  for s, df in splits.items()}
    elif name == "mimic_ext_ppg":
        cohort = build_mimic_ext_ppg_cohort(args.mimic_ext_ppg_root)
        splits = {s: cohort.split(s) for s in ("train", "val", "test")}
        loader = make_mimic_ext_ppg_signal_loader(args.mimic_ext_ppg_root)
        meta = pd.read_csv(args.mimic_ext_ppg_root / "metadata.csv")
        labels = {s: build_mimic_ext_ppg_label_table(meta, df["visit_id"].tolist())
                  for s, df in splits.items()}
    elif name == "but_ppg":
        cohort = build_but_ppg_cohort(args.but_ppg_root)
        splits = {s: cohort.split(s) for s in ("train", "val", "test")}
        loader = make_but_ppg_signal_loader(args.but_ppg_root)
        qhr = pd.read_csv(args.but_ppg_root / "quality-hr-ann.csv")
        labels = {s: build_but_ppg_label_table(qhr, df["visit_id"].tolist())
                  for s, df in splits.items()}
    else:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASET_CHOICES}")

    if degrade_kind:
        loader = wrap_loader_with_degradation(loader, degrade_kind, degrade_severity, seed)
    return DatasetHandle(name=name, cohort=cohort, splits=splits, load_signal=loader, label_table=labels)
