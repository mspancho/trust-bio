# TRUST-BIO

**TR**ansportability **U**nder **S**ite/device sensor **T**axonomy for **BIO**signal foundation models

TRUST-BIO stress-tests a recent finding from [SignalMC-MED](https://arxiv.org/abs/2603.09940) — that domain-specific biosignal foundation models outperform general time-series foundation models, and that ECG+PPG fusion helps — across an independent, multi-institution cohort and under realistic signal degradation. It also characterizes degraded/shifted segments with a label-free fault taxonomy (transient motion artifact, persistent lead-off, structural site/device shift), laying groundwork for fault-type-specific mitigation.

## What this answers

1. **Does the domain-FM > time-series-FM ranking transport?** Trained/tuned on one PulseDB source institution (Boston ICU, from MIMIC-III), scored zero-shot on the other (Seoul surgical, from VitalDB) — and the reverse direction.
2. **What happens under degradation?** Motion artifact, lead-off (electrode disconnect), and missing-PPG-channel conditions, injected at three severities, with motion-artifact noise calibrated against BUT PPG's real accelerometer/quality relationship.
3. **Can degradation be diagnosed without labels?** A signal-quality-index- and accelerometer-based feature set clusters degraded/shifted segments into transient, persistent, and structural fault classes, validated post-hoc against known synthetic conditions.

See `docs/paper_draft.docx` for the full scaffolded manuscript (results sections are placeholder-marked pending real experimental runs).

## What this does NOT do (yet)

This repo establishes whether transport holds and characterizes the structure of degradation. It does **not** implement or evaluate a fault-aware mitigation system (reweighting, abstention, label-free test-time adaptation) — that is future work motivated by, but not delivered in, this study.

## Datasets

| Dataset | Role | Access |
|---|---|---|
| [PulseDB](https://github.com/pulselabteam/PulseDB) | Cross-institution transportability (MIMIC vs. Vital source) | Open via the official Google Drive mirror (see `scripts/fetch_pulsedb.py`) — the Kaggle listing that turns up for "PulseDB" is a third-party re-upload under a different, non-canonical, non-commercial license, and is deliberately not used here; see `docs/pulsedb_structure_notes.md` for the full story |
| [MIMIC-III-Ext-PPG](https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/) | Fault taxonomy (native signal quality indices + rhythm labels) | PhysioNet credentialed |
| [BUT PPG](https://physionet.org/content/butppg/2.0.0/) | Real-degradation calibration + clinical-to-smartphone-camera gap | Open (CC-BY 4.0) |

## Models

Domain-specific biosignal FMs (ECGFounder, xECG, D-BETA, PaPaGei-S), general time-series FMs (MOMENT, Chronos-Bolt), hand-crafted domain features (NeuroKit2 ECG, pyPPG), and optionally CSFM (Oxford, access-restricted — the pipeline runs a complete comparison without it and picks it up automatically once available). See `trustbio/config.py`'s `FM_REGISTRY` for the full list.

## Quick start (synthetic demo, no gated data/models needed)

```bash
pip install -e .
python scripts/make_synthetic_demo.py --n-visits 120
```

This fabricates synthetic data, runs every pipeline stage with the deterministic fallback extractor, and prints results. The numbers are meaningless as science — it only proves the pipeline connects end to end.

## Real run (HMS O2 cluster)

```bash
bash scripts/setup_env.sh        # one-time: install deps + vendor model repos
bash scripts/run_all.sh          # submits the full SLURM DAG: fetch -> cohort -> extract -> {transport, taxonomy, benchmark}
squeue -u $USER                  # watch progress
```

Environment overrides: `TRUSTBIO_ENV` (conda env, default `map-env-base`), `TRUSTBIO_STORE` (feature cache dir), `TRUSTBIO_OUT` (results dir), `TRUSTBIO_MAX_PARALLEL` (concurrent GPU array tasks, default 3).

### Fetching datasets individually

```bash
python scripts/fetch_pulsedb.py                          # no credentials needed
python scripts/fetch_but_ppg.py                           # no credentials needed
PHYSIONET_USERNAME=... PHYSIONET_PASSWORD=... \
  python scripts/fetch_mimic_ext_ppg.py                   # requires credentialed access
```

### Adding CSFM once access is granted

Place the checkpoint at `model_weights/csfm_base.pt` (or set `TRUSTBIO_CKPT_CSFM_BASE`), clone its source repo to `model_repos/Cardiac-Sensing-FM/`, and re-run — no code changes needed; `trustbio/config.py`'s `is_model_available()` picks it up automatically.

### D-BETA requires a HuggingFace token, even after your access request is approved

`Manhph2211/D-BETA` on HuggingFace is a gated repo — an approved access request alone doesn't authenticate your shell or SLURM jobs. Before running any stage that includes `dbeta`, either `export HF_TOKEN=<your-token>` (works for both interactive and `sbatch` jobs, since SLURM inherits the submitting shell's environment by default) or run `huggingface-cli login` once to cache a token. Without one, `is_model_available("dbeta")` returns `False` and D-BETA is skipped cleanly rather than crashing — check `python -c "from trustbio.config import is_model_available; print(is_model_available('dbeta'))"` if you're unsure whether it's wired up.

## Repository layout

Run `tree trustbio/` for the full module layout: `trustbio/data/` (dataset adapters), `trustbio/degradation/` (synthetic injection + real-data calibration), `trustbio/taxonomy/` (fault clustering), `trustbio/features/` + `trustbio/eval/` (feature extraction and evaluation, shared with the original signal-mcmed-msp re-implementation), `trustbio/pipeline.py` (orchestration).

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT (see `LICENSE`). Vendored external model repos (ECGFounder, xecg, papagei-foundation-model, D-BETA) retain their own upstream licenses.
