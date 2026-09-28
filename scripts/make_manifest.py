#!/usr/bin/env python
"""Build an extraction manifest: one line per (model, dataset[, chunk]) cell.

Lines are `model dataset duration` or, with --chunks, `model dataset duration
chunk n_chunks`. Chunking exists because a full-scale (model, dataset) cell is
millions of windows and 15-45 GPU-hours as one job (ecg-domain on CPU: ~110 h)
-- no single wall-clock window survives that. Each chunk task finishes in a
few hours, and a resubmitted task skips chunks already on disk.

    python scripts/make_manifest.py --duration 10 --out manifest_full_gpu.txt \
        --models moment-base chronos-bolt-small dbeta ecgfounder xecg-10min papagei \
        --datasets pulsedb_mimic pulsedb_vital --chunks pulsedb_mimic=8 pulsedb_vital=3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from trustbio.config import (
    DEGRADATION_KINDS, DEGRADATION_SEVERITIES, MAIN_TEST_MODELS, is_model_available,
)

DATASETS = ["pulsedb_mimic", "pulsedb_vital", "mimic_ext_ppg", "but_ppg"]


def parse_chunks(specs: list[str]) -> dict[str, int]:
    """['pulsedb_mimic=8', ...] -> {'pulsedb_mimic': 8, ...}."""
    out: dict[str, int] = {}
    for spec in specs:
        name, sep, n = spec.partition("=")
        if not sep or name not in DATASETS or not n.isdigit() or int(n) < 1:
            raise ValueError(f"bad --chunks spec {spec!r}; expected <dataset>=<positive int>")
        out[name] = int(n)
    return out


def parse_degrade(specs: list[str]) -> list[tuple[str, str, float]]:
    """['motion_artifact=0.3', ...] -> [('motion_artifact_0.3', 'motion_artifact', 0.3), ...]."""
    out = []
    for spec in specs:
        kind, sep, sev = spec.partition("=")
        if not sep or kind not in DEGRADATION_KINDS:
            raise ValueError(f"bad --degrade spec {spec!r}; expected <kind>=<severity>")
        try:
            sev_f = float(sev)
        except ValueError as exc:
            raise ValueError(f"bad severity in {spec!r}") from exc
        if sev_f not in DEGRADATION_SEVERITIES:
            raise ValueError(f"severity {sev_f} not in {DEGRADATION_SEVERITIES}")
        out.append((f"{kind}_{sev}", kind, sev_f))
    return out


def build_lines(models: list[str], datasets: list[str], duration: int,
                chunks: dict[str, int], conditions=None) -> list[str]:
    """One line per (model, dataset[, chunk][, condition]). `conditions` is a
    list of (name, kind, severity); kind None means the clean baseline. With
    conditions, the cell's store root becomes <store>/<name> (see
    scripts/_extract_cell.sh), keeping degraded and clean features apart."""
    lines = []
    for model in models:
        for dataset in datasets:
            n = chunks.get(dataset)
            bases = ([f"{model} {dataset} {duration}"] if n is None
                     else [f"{model} {dataset} {duration} {c} {n}" for c in range(n)])
            for base in bases:
                if not conditions:
                    lines.append(base)
                    continue
                for name, kind, sev in conditions:
                    suffix = f" cond={name}" + (f" kind={kind} sev={sev}" if kind else "")
                    lines.append(base + suffix)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to include (default: every available MAIN_TEST_MODELS entry)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--chunks", nargs="*", default=[], metavar="DATASET=N",
                    help="cut each (model, DATASET) cell into N contiguous chunk tasks; "
                         "datasets not listed get one unchunked line")
    ap.add_argument("--with-clean", action="store_true",
                    help="emit a cond=clean line per cell (needed whenever --degrade is used)")
    ap.add_argument("--degrade", nargs="*", default=[], metavar="KIND=SEV",
                    help="also emit a degraded line per cell for each KIND=SEV")
    ap.add_argument("--skip-availability-check", action="store_true",
                    help="do not filter --models through is_model_available (the job checks anyway)")
    args = ap.parse_args()
    chunks = parse_chunks(args.chunks)
    conditions = ([("clean", None, None)] if args.with_clean else []) + parse_degrade(args.degrade)
    if args.degrade and not args.with_clean:
        print("[manifest] note: --degrade without --with-clean emits no clean baseline lines")

    models = []
    for model in (args.models or MAIN_TEST_MODELS):
        if not args.skip_availability_check and not is_model_available(model):
            print(f"[manifest] {model}: unavailable, excluding from manifest")
            continue
        models.append(model)

    lines = build_lines(models, args.datasets, args.duration, chunks, conditions=conditions or None)
    Path(args.out).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} cells ({len(models)} models x {len(args.datasets)} datasets) to {args.out}")


if __name__ == "__main__":
    main()
