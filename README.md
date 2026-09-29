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

## Full-scale extraction (launched 2026-09-21)

Chunked, resumable SLURM arrays; see `docs/superpowers/plans/2026-09-21-full-scale-chunked-extraction.md`.

- Store: `features_cache/full/<dataset>/<model>/<modality>/10s/<split>.npz` (chunks: `<split>.chunkXXofNN.npz` until merged)
- Manifests: `manifest_full_gpu.txt` (66 GPU chunk tasks), `manifest_full_cpu.txt` (28 ecg-domain CPU tasks), `manifest_full_cells.txt` (14 cells for the merge)
- Jobs: gpu=54042608 cpu=54042609 merge=54042610 (merge waits on both arrays; it REFUSES any cell with missing chunks -- redo those chunks with `sbatch --array=<idx> scripts/extract_features[_cpu].sbatch <manifest>` then rerun that merge task)
- Labels: `scripts/build_pulsedb_labels.py --store features_cache` (jobs tb-labels-mimic 54042005 / tb-labels-vital 54042006)
- Watch: `squeue -u $USER | grep trustbio`; `grep -h kept logs/extract_<id>_*.out`

## Fault taxonomy (run 2026-09-28/29)

Plan and outcome: `docs/superpowers/plans/2026-09-28-fault-taxonomy-analysis.md`. Taxonomy cohorts in `features_cache/taxonomy/` (60 subjects x <=40 windows per PulseDB institution; 800/800/800 MIMIC-ext segments stratified by native SQI; all 3,888 BUT PPG recordings), degraded feature store in `features_cache/taxonomy_store/<condition>/<dataset>/` (clean + motion_artifact/lead_off x 0.1/0.3/0.6, all 7 models, verified 0 bad across 1,386 files), results in `results/taxonomy/` (`fault_features.csv`, `table3_recall.csv`, `confusion_*.csv`, `assignments_*.csv`, `summary.json`, `config.json`, `figures/`; first-iteration results in `results/taxonomy_v1_hfecg/`).

Jobs: cohorts 54696312; motion-noise calibration 54713465 (amplitudes 0.18/0.31/1.30 x signal std, anchored to the 50/75/95th percentile of real BUT PPG out-of-band noise); extraction 54697971 + 54697972 (with redo 54704998, 54707808, MIMIC-ext 54705061/54705062, motion 54713466/54713467); verification 54713468; analysis 54713469. Rerun end to end with `sbatch scripts/run_taxonomy.sbatch`.

Headline: KMeans(k=3) on label-free SQI/disagreement features recovers severe faults (severity 0.6: lead-off recall 0.99, motion 1.00) and not mild/moderate ones (<= 0.02); the structural (clean cross-institution) class is not identifiable (recall 0.0; disagreement AUROC 0.62). Native-SQI validation of the traces: PPG AUROC 0.65, ECG 0.51. The BUT PPG accelerometer/quality calibration originally planned has no empirical support (AUROC 0.57, Spearman -0.11) and is reported as a negative result.

### Supervised separability ceiling (plan Task 14, 2026-09-29)

`scripts/run_taxonomy_supervised.py` on the same table and feature sets (logistic regression + gradient boosting, subject-grouped 5-fold CV; outputs `results/taxonomy/supervised_*`). Without institution identity (`no_source_db`): lead-off is fully separable at severity 0.3 and 0.6 (recall 0.99 / 1.00) and not at 0.1 (0.01-0.04); motion artifact at 0.6 and 0.3 (1.00 / 0.91-0.97) and weakly at 0.1 (0.32-0.45) -- so the clustering's failure at severity 0.3 was a k=3 allocation artifact, not a feature limit. Structural is not identifiable: clean-Vital-vs-clean-MIMIC out-of-fold AUROC 0.68 (0.67 with SQI features alone; 1.00 only when `source_db` is included, which is circular), and in the 3-class fit the "structural" prediction behaves as "no detectable fault" (77% of clean MIMIC controls land there). Real-world transfer: MIMIC-ext natively poor-PPG segments are called motion 46% of the time vs 31% for natively clean ones; BUT PPG unusable vs usable recordings 78% vs 73%.
