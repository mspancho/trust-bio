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

### Degraded-only two-stage taxonomy (plan Task 15, 2026-09-29)

`scripts/run_taxonomy_degraded_only.py` asks the question the way the draft words it ("we clustered degraded segments"): first detect degraded windows with a label-free SQI rule, then cluster only those, with k in 2..6 chosen by silhouette (outputs `results/taxonomy/degraded_only_*`). Two detection rules: `drop` (the combined per-second SQI fell below 0.5 at least once) and `outlier` (PPG or ECG SQI below the 2.5th percentile, or model disagreement above the 97.5th percentile, of the clean PulseDB-MIMIC controls). Detection alone tells most of the story: `drop` catches lead-off at severity 0.3 and 0.6 in 100% of windows but motion artifact only at 0.6 (100%; 5% at 0.3, 2% at 0.1), while flagging 0.1% of clean controls and 2% of structural windows; `outlier` catches 44% of moderate motion (16% of both mild kinds) at 5% of clean controls and 17% of structural, by construction. Typing then works wherever the detected windows carry a duration signature: with the `drop` rule the silhouette picks k=3 in every feature set but SQI-only (k=4), and the clusters are short ECG flat-line (~2 s: severity-0.3 lead-off), long ECG flat-line (~5 s: severity-0.6 lead-off) and long PPG drop-out (~6 s: severity-0.6 motion), so two clusters share the name lead-off and overall recall is 0.99 for lead-off at severity 0.3, 1.00 at 0.6, and 1.00 for motion at 0.6 -- moderate lead-off, which the k=3 clustering of all windows missed entirely (0.02), is recovered at the supervised ceiling. Moderate motion is not: the `drop` rule does not see it, and the 44% that the `outlier` rule detects form a distinct "slightly noisy PPG, no drop" cluster only in SQI-only space at k=5 (overall recall 0.44), while with disagreement in the feature set k=3 lumps them with moderate lead-off (overall 0.01). Severity-0.1 rows of both kinds land in the same cluster in every set, so their non-zero recalls (0.15 lead-off or 0.16 motion, depending on the set) only record which name that cluster drew; structural windows are, by construction, not detected beyond the rules' false-positive rates (2% / 17%) and never name a cluster. Real-world transfer is carried by the detection stage: under `drop`, BUT PPG unusable recordings are flagged 32% vs 17% for usable ones, MIMIC-ext natively poor-PPG segments 5% vs 0.1% natively clean (poor-ECG 2%); under `outlier` most BUT PPG windows fall outside the clean-monitor reference whatever their label (77% of usable, 81% of unusable recordings flagged), so that rule's held-out shares say more about the device than about motion.

## Degradation stress test (run 2026-09-29, paper Results §2)

Plan and outcome: `docs/superpowers/plans/2026-09-29-degradation-stress-test.md`. `scripts/run_stress_predict.py` (array job 54724220, one task per model, 10 min to 2.5 h each at 64 GB) fits the transport-style ridge probes on each PulseDB institution's clean full-scale store with the 60 taxonomy-cohort subjects held out, then predicts the taxonomy cohort's windows under every taxonomy-store condition (clean, motion_artifact and lead_off at severity 0.1/0.3/0.6, plus a derived missing_ppg = the fusion probe fed the clean ECG-only vector) for both PulseDB institutions and MIMIC-ext (heart rate only). `scripts/run_stress_analysis.py` (job 54724562) pools the 4.5M paired predictions into `results/stress/stress_{scores,rank_stability,gap,fusion,harm_coverage,detection_flags}.csv`, `stress_summary.json` and `figures/fig2a-d_*.png`. Rerun with `sbatch scripts/run_stress.sbatch`, then `sbatch --dependency=afterok:<id> scripts/run_stress_analysis.sbatch`. Fidelity: the clean cross-source heart-rate r agrees with `results/full_transport.csv` (mean |diff| 0.045 over 42 cells, r 0.97; the target is a 60-subject sample).

Headline (heart rate, PulseDB targets, mean over 7 models and 4 source-target pairs). Motion artifact, which corrupts PPG only, does no harm at severity 0.1 or 0.3 (PPG-probe delta r 0.000 / +0.003, delta MAE 0.08 / 0.58 bpm, 0.1% / 2.7% of windows moved by more than 5 bpm) and hurts only at 0.6 (delta r -0.044, +6.8 bpm, 58% of windows; fusion +1.9 bpm, 27%), where the label-free `drop` rule flags 100% of windows and catches 100% of the harm. Lead-off, which corrupts ECG only, hurts from the mildest level: a 1-s flat-line moves 26% of ECG-probe estimates by more than 5 bpm (delta r -0.032, +1.5 bpm), a 3-s one 48% (-0.091, +5.8 bpm), a 6-s one 64% (-0.283, +14.9 bpm). Severity 0.3 and 0.6 lead-off is flagged 100% by every rule; severity 0.1 is flagged by 1.8% (`drop`), 13% (`outlier`) and 14% (supervised), catching 1% / 9% / 14% of its harm -- the harmful-but-undetected regime. The cause is resolution, not signal: the SQI trace uses fixed 1-s bins and a 1.0-s flat span straddles two of them (only 29 of 4,073 mild lead-off windows show a fully flat second), so a sub-second or run-length flat-line detector is the follow-up. The model ranking survives motion (Spearman vs the clean ranking 0.95-1.00 up to severity 0.3, 0.77 for PPG probes at 0.6) but not lead-off (0.70 / 0.52 / 0.30 on ECG probes): xECG, the best clean model, is the most lead-off-sensitive (ECG heart-rate r 0.778 -> 0.378 at 0.6, vs D-BETA 0.659 -> 0.492 and PaPaGei 0.728 -> 0.495), and its lead over MOMENT (+0.07 clean) reverses at severity 0.6 (-0.03 ECG, -0.125 fusion). Mean fusion never beats the better unimodal probe (clean -0.012; lead-off -0.02 / -0.03 / -0.08; motion -0.01 to -0.02); it halves lead-off harm (MAE +1.1 / 3.3 / 7.7 bpm vs +1.5 / 5.8 / 14.9) but at severity 0.6 switching to the PPG probe would do better. Feeding the fusion probe the ECG-only vector when PPG is missing keeps heart-rate correlation (delta r -0.003) but is miscalibrated (prediction spread doubles, -11 bpm bias for xECG, +12 bpm MAE on average) and loses blood pressure (delta r -0.21 SBP / -0.16 DBP: that information lives in PPG). Blood pressure follows the same pattern with smaller effects (SBP: lead-off ECG delta r -0.026 / -0.048 / -0.104; motion PPG -0.006 at 0.3, -0.138 at 0.6). MIMIC-ext heart rate reproduces the PulseDB pattern (lead-off ECG r 0.700 -> 0.642 / 0.500 / 0.423; motion PPG 0.637 -> 0.637 / 0.636 / 0.509).

## Manuscript

First full draft in `docs/paper/main.tex` (ML4H 2026 proceedings class, same conventions as the PISCES v2 paper), bibliography `docs/paper/ref.bib`, figure assets copied from `results/*/figures/` into `docs/paper/figures/`. Build with `docs/paper/build.sh` (tectonic; checks for unresolved citations and lists remaining `\placeholder{}` items). Every number in the draft comes from `results/full_transport.csv`, `results/full_bench_*.csv`, `results/taxonomy/` and `results/stress/`.
