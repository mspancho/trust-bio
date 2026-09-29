# Fault-Taxonomy Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the paper's third result — "Degraded segments decompose into distinguishable transient, persistent, and structural fault classes" (Table 3, Figure 3) — by clustering label-free per-segment signal-quality/motion/model-disagreement features over synthetically degraded PulseDB and MIMIC-III-Ext-PPG segments, clean cross-institution segments, MIMIC-III-Ext-PPG's natively low-SQI segments, and BUT PPG's real motion artifact, and validating the clusters against each segment's known generating condition.

**Architecture:** Phase A builds a small, reproducible degraded-feature store: a taxonomy cohort (≈2,400 windows per PulseDB institution, ≈2,400 MIMIC-III-Ext-PPG segments stratified by native SQI, all 3,888 BUT PPG recordings) is run through the existing chunk-aware extraction wrappers under seven conditions (clean, motion artifact × 3 severities, lead-off × 3 severities) for all 7 models, with degradation made deterministic per window so any later process can regenerate exactly the waveform a model saw. Phase B computes per-second SQI traces from those waveforms (calibrated once against MIMIC-III-Ext-PPG's native SQI), assembles the fault-feature table (SQI value/drop duration per modality, accelerometer correlation, source, HR-probe disagreement between the best domain FM and the best time-series FM), clusters the {motion, lead-off, structural} fit set with the existing KMeans(k=3), validates against known conditions with subject-bootstrap CIs and feature ablations, assigns the held-out controls (clean in-distribution, natural MIMIC-ext degradation, BUT PPG real motion) by nearest centroid, and renders Table 3 + Figure 3.

**Tech Stack:** Python 3.11 in conda env `trust-bio` (numpy 1.26.4, pandas, scipy 1.13, scikit-learn 1.7.2, matplotlib 3.11, seaborn 0.13, wfdb, mat73), pytest, SLURM on HMS O2 via the repo's `scripts/extract_features.sbatch` / `extract_features_cpu.sbatch` wrappers (`_extract_cell.sh`), `scripts/verify_feature_store.py`.

## Global Constraints

- Run every Python command through the pinned env: `conda run -n trust-bio python …`. Batch jobs use `conda run --no-capture-output`.
- Never name the dotenv file (or words like "credentials"/"token"/"secret") in a shell command — the `protect-secrets` hook blocks it, correctly. Jobs that need model access source it inside the sbatch body (`_extract_cell.sh` already does). No shell pattern may contain `/^…$/`-style regexes, and no glob or recursive command may run over the lab's shared top-level directory (only over the dataset roots and the repo listed below) — the `protect-paths` hook blocks these, correctly. The lab bucket path named as off-limits in the user's CLAUDE.md is never touched or named.
- Store layout `<store>/<dataset>/<model>/<modality>/<duration>s/<split>.npz`; CLIs take the BASE root and scope by dataset. Every store write goes through `FeatureStore.save` (atomic). Never merge/verify partial stores silently.
- Login nodes OOM-kill ~2–3 GB python processes: anything that loads feature matrices or the 4.9 GB `metadata.csv` runs as an sbatch job. Judge job liveness with `sstat -a -j <id>` / `sacct`, not by log emptiness (`conda run` buffers stdout).
- Shared-cluster courtesy: GPU arrays throttled `%12`; ecg-domain only on `extract_features_cpu.sbatch`.
- `results/` is gitignored: result CSVs/figures stay untracked; the README records which job produced what. Work on `main`, commit after every task, commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Repo root `/n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio`; PulseDB root `/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb`; MIMIC-III-Ext-PPG root `/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iii-ext-ppg/physionet.org/files/mimic-iii-ext-ppg/1.1.0` (`DEFAULT_ROOT` in `trustbio/data/mimic_ext_ppg.py`); BUT PPG root `/n/data1/hms/dbmi/rajpurkar/lab/datasets/but-ppg/physionet.org/files/butppg/2.0.0` (`DEFAULT_ROOT` in `trustbio/data/but_ppg.py`; 3,888 records, 3,840 with accelerometer, IDs ≥ 112001).
- Full-scale clean features (probe training data) live at `features_cache/full/pulsedb_{mimic,vital}/…` with labels at `features_cache/pulsedb_{mimic,vital}_labels.csv`; both verified on 2026-09-25.

## Design decisions (read before any task)

**Facts established while planning** (all from the repo/data on 2026-09-28):
1. `features_cache/noise_amplitude_cache.json` currently holds `{0.1: 0.735, 0.3: 0.768, 0.6: 0.817}` — written at 11:20 on 2026-09-25 by the TEST SUITE: `tests/test_degradation_calibrate.py` calls `fit_motion_noise_amplitude(fake_root)` with the default `cache=True`, which writes the production cache from a synthetic fixture. Those numbers are fixture artifacts, not a BUT PPG calibration (Task 1 fixes the test and refits on real data).
2. `wrap_loader_with_degradation` draws from ONE `np.random.Generator` shared across every window, so the corrupted span of a given window depends on call order; a second process cannot regenerate the waveform the model saw (Task 2 makes the draw a pure function of `(seed, visit_id, kind, severity, modality)`).
3. In the real `metadata.csv`, `segment_id` is a small integer (2, 4, …) repeated across records; only `signal_file_name` (e.g. `3000060_0002_0_2`) is unique. The adapter keys everything on `segment_id`, and the test fixture hid this by using unique strings there (Task 3 switches visit ids to `signal_file_name` and makes the fixture mirror reality).
4. Native SQI vectors are per 10-s sub-window of each 30-s segment; `encode_modality` uses the FIRST `duration_sec` seconds, so at `duration_sec=10` the model saw sub-window 0 and the matching native code is element `[0]`. Codes seen in a 50k-row sample: PLETH `{1, 0}`; ECG `{1, -2, 0, -3, None, -17}`; ABP mostly `None`. Negative ECG codes are undocumented in the dataset README — treat `1` as good, anything `≤ 0` as poor, keep the raw code as a column, never invent semantics for `-2`/`-3`.
5. `umap-learn` is not installed; Figure 3a uses PCA (sklearn). matplotlib/seaborn are present.
6. The best domain FM on the full-scale transport table is `xecg-10min`, the best time-series FM `moment-base`; both have `ecg_ppg_mean` features. Those are the two probes for `model_disagreement`.

**Conditions and known-condition labels** (one row per window; `dataset` ∈ pulsedb_mimic / pulsedb_vital / mimic_ext_ppg / but_ppg):

| Condition store dir | Datasets | `known_condition` | Role |
|---|---|---|---|
| `clean` | pulsedb_mimic | `clean` | in-distribution control (held out from the fit) |
| `clean` | pulsedb_vital | `structural` | clean, cross-institution (probes trained on MIMIC) — FIT SET |
| `motion_artifact_{0.1,0.3,0.6}` | pulsedb_mimic, pulsedb_vital, mimic_ext_ppg | `motion_artifact` | FIT SET |
| `lead_off_{0.1,0.3,0.6}` | pulsedb_mimic, pulsedb_vital, mimic_ext_ppg | `lead_off` | FIT SET |
| `clean` | mimic_ext_ppg | `natural_clean` / `natural_ppg_poor` / `natural_ecg_poor` (from native SQI element 0) | real ICU degradation, assigned post hoc |
| `clean` | but_ppg | `real_motion` (quality 0) / `consumer_clean` (quality 1) | real smartphone motion artifact, assigned post hoc |

The KMeans(k=3) fit uses only rows with `known_condition ∈ {motion_artifact, lead_off, structural}`; every other row is assigned to the nearest fitted centroid and reported separately (Task 11). `source_db` is kept as a feature (existing design) AND every metric is reported for the ablation without it, because with it the structural class is partly circular.

**Fault features per window** (Task 8 extends `SegmentFaultFeatures`): `sqi_value` (mean of the per-second min(ECG-SQI, PPG-SQI) trace), `sqi_drop_duration` (longest run of seconds with combined SQI < 0.5), `accel_corr` (BUT PPG only; 0 elsewhere), `source_db`, `model_disagreement` (|HR_xecg − HR_moment| / scale), `ecg_sqi_value`, `ppg_sqi_value`. SQI traces come from Task 7, computed on the exact (possibly degraded) waveform via Task 2's determinism, with one free parameter per modality (`hf_ref`) calibrated against MIMIC-III-Ext-PPG's native SQI (Task 10).

**Sizing:** extraction is 132 GPU cells (6 models × 22 dataset/condition combos, ≈2,400–3,900 windows each ≈ 1–2 min/cell) ≈ 3–4 GPU-hours, plus 22 ecg-domain CPU cells (≈5–7 min each); the fault-feature build is ≈45 min CPU (dominated by reading ~120 PulseDB subject files with mat73 once each); clustering and plots run in seconds.

## File structure

- `tests/test_degradation_calibrate.py` — modify: autouse fixture isolating the cache path.
- `trustbio/degradation/inject.py` — add `visit_rng`, `make_degraded_loader`.
- `scripts/_dataset_builders.py` — `wrap_loader_with_degradation` delegates to `make_degraded_loader`; MIMIC-ext branch reads a metadata subset from the cohort cache when present.
- `trustbio/data/mimic_ext_ppg.py` — visit id = `signal_file_name`; `parse_sqi_vector` tolerates `nan`/`np.float64(...)`.
- `tests/test_mimic_ext_ppg_adapter.py` — fixture mirrors the real metadata (integer `segment_id`).
- `scripts/sample_cohort.py` — `--max-windows-per-subject`.
- `scripts/sample_mimic_ext_segments.py` — create: stratified MIMIC-ext taxonomy sample (metadata subset + cohort CSV).
- `scripts/make_manifest.py`, `scripts/_extract_cell.sh` — condition tokens (`cond=`, `kind=`, `sev=`) and per-condition store subdirs.
- `trustbio/taxonomy/sqi.py` — create: per-second SQI traces + `hf_ref` calibration.
- `trustbio/taxonomy/features.py` — per-modality SQI features, `FEATURE_NAMES`, column selection.
- `trustbio/taxonomy/disagreement.py` — create: HR probes and disagreement scale.
- `trustbio/taxonomy/cluster.py` — `FaultClusterer` with `predict`, bootstrap accuracy, silhouette.
- `scripts/build_fault_features.py` — create: assembles `results/taxonomy/fault_features.csv` + `.npz`.
- `scripts/run_taxonomy.py` — rewrite: clustering, validation, ablations, assignments (Table 3).
- `scripts/plot_taxonomy.py` — create: Figure 3 panels.
- `scripts/run_taxonomy.sbatch` — rewrite: build → cluster → plot chain.
- Tests: `tests/test_degradation_determinism.py`, `tests/test_sample_cohort.py`, `tests/test_sample_mimic_ext_segments.py`, `tests/test_dataset_builders.py` (extend), `tests/test_extract_sbatch.py` (extend), `tests/test_make_manifest.py` (extend), `tests/test_taxonomy_sqi.py`, `tests/test_taxonomy_features.py` (extend), `tests/test_taxonomy_disagreement.py`, `tests/test_taxonomy_cluster.py` (extend), `tests/test_build_fault_features.py`, `tests/test_run_taxonomy.py`, `tests/test_plot_taxonomy.py`.

---

### Task 1: Isolate the noise-calibration cache from the tests and refit it on real BUT PPG

> **Revision (2026-09-28, during execution).** Step 5's refit hit the stop rule: the fit returned the floor `0.001` at every severity. Diagnosis on all 3,795 accelerometer-era recordings: accelerometer values are milli-g (mean ≈ 960, gravity), so `severity × max_accel` extrapolated below every observation; more fundamentally, accelerometer dynamics do not separate quality-0 from quality-1 (AUROC 0.57), PPG noise measures do not either (0.50–0.57), and noise vs accelerometer Spearman ≈ −0.11 — there is no relationship to calibrate against. (A second bug surfaced on the way: the HF-residual ratio's `np.convolve(mode="same")` zero-padding gave DC-sized edge residuals on camera PPG; fixed in both `calibrate.py` and `taxonomy/sqi.py`.) Decision (user-approved): anchor each severity to a QUANTILE of real BUT PPG noise instead — amplitudes at 0.1/0.3/0.6 are solved by bisection on clean PulseDB-MIMIC reference windows so the corrupted span's HF ratio equals the 50th/75th/95th-percentile real recording. `fit_motion_noise_amplitude(root, reference_ppg, ...)` now takes the reference windows; `scripts/inject_degradation.py` draws them from `features_cache/taxonomy/pulsedb_mimic_cohort.csv`; a `noise_amplitude_calibration.json` sidecar records targets/achieved ratios. The accelerometer cross-check is reported as a negative result in Methods.

**Files:**
- Modify: `tests/test_degradation_calibrate.py`
- Run: `scripts/inject_degradation.py --refit-calibration` as a batch job

**Interfaces:**
- Produces: a genuine `features_cache/noise_amplitude_cache.json` (`{"0.1": a1, "0.3": a2, "0.6": a3}`, monotonically non-decreasing), consumed by `inject_motion_artifact` via `load_cached_noise_amplitude()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_degradation_calibrate.py`:

```python
def test_fit_never_writes_the_production_cache(fake_but_ppg_with_accel, tmp_path):
    from trustbio.degradation import calibrate
    fit_motion_noise_amplitude(fake_but_ppg_with_accel)          # default cache=True
    assert calibrate.NOISE_AMPLITUDE_CACHE_PATH.parent == tmp_path, (
        "the autouse fixture must redirect the cache path; without it this call "
        "overwrote features_cache/noise_amplitude_cache.json with fixture numbers"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n trust-bio python -m pytest tests/test_degradation_calibrate.py::test_fit_never_writes_the_production_cache -v`
Expected: FAIL — the assertion (the path still points at the repo's `features_cache`).

- [ ] **Step 3: Add the autouse fixture**

Insert after the imports in `tests/test_degradation_calibrate.py` (before `fake_but_ppg_with_accel`):

```python
@pytest.fixture(autouse=True)
def _isolate_noise_cache(tmp_path, monkeypatch):
    """fit_motion_noise_amplitude(cache=True) writes NOISE_AMPLITUDE_CACHE_PATH.
    Left pointing at the repo, the test suite overwrote the real
    features_cache/noise_amplitude_cache.json with fixture-derived numbers
    (observed 2026-09-25). Every test in this module gets a throwaway path."""
    monkeypatch.setattr(
        "trustbio.degradation.calibrate.NOISE_AMPLITUDE_CACHE_PATH",
        tmp_path / "noise_amplitude_cache.json",
    )
```

`test_fit_writes_cache_file` keeps its own monkeypatch (harmless).

- [ ] **Step 4: Run the module**

Run: `conda run -n trust-bio python -m pytest tests/test_degradation_calibrate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Delete the polluted cache and refit on real BUT PPG as a batch job**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
rm -f features_cache/noise_amplitude_cache.json
sbatch --job-name=tb-calib --partition=short --cpus-per-task=2 --mem=8G --time=2:00:00 \
  --output=logs/calib_%j.out --error=logs/calib_%j.err \
  --wrap="cd $PWD && module load conda/miniforge3/24.11.3-0 2>/dev/null; conda run --no-capture-output -n trust-bio python scripts/inject_degradation.py --but-ppg-root /n/data1/hms/dbmi/rajpurkar/lab/datasets/but-ppg/physionet.org/files/butppg/2.0.0 --refit-calibration"
```

When it completes (reads 3,840 accelerometer records; ~10–20 min):

```bash
cat features_cache/noise_amplitude_cache.json
grep -h 'noise amplitudes' logs/calib_*.out
```

Expected: three values, non-decreasing with severity, each well below the fixture's ~0.75 (real PPG noise-to-signal ratios in poor-quality smartphone recordings are typically 0.1–0.5). If any value is ≥ 1.0 or the order is not monotone, STOP and inspect `fit_motion_noise_amplitude` — do not proceed to extraction with an implausible calibration.

- [ ] **Step 6: Commit**

```bash
git add tests/test_degradation_calibrate.py
git commit -m "fix: calibration tests no longer overwrite the production noise cache

fit_motion_noise_amplitude(cache=True) in the tests wrote fixture-derived
amplitudes to features_cache/noise_amplitude_cache.json. Autouse fixture
redirects the path; the real cache is refit on BUT PPG (3,840 accelerometer
records).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Deterministic per-window degradation

**Files:**
- Modify: `trustbio/degradation/inject.py` (add `visit_rng`, `make_degraded_loader`)
- Modify: `scripts/_dataset_builders.py` (`wrap_loader_with_degradation` delegates)
- Test: `tests/test_degradation_determinism.py` (create)

**Interfaces:**
- Produces: `visit_rng(seed: int, visit_id: str, kind: str, severity: float, modality: str) -> np.random.Generator`; `make_degraded_loader(load_signal, kind, severity, seed, noise_amplitudes=None) -> SignalLoader` (identity when `kind`/`severity` is None; raises `ValueError` for `missing_ppg` on the ppg modality, as before). `wrap_loader_with_degradation(load_signal, kind, severity, seed)` keeps its signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_degradation_determinism.py`:

```python
"""A degraded waveform must be a pure function of (seed, visit, condition,
modality): the fault-taxonomy analysis recomputes SQI traces on the SAME
waveform the model saw, in a different process, possibly in a different order."""
import numpy as np
import pytest

from trustbio.degradation.inject import make_degraded_loader, visit_rng

AMPS = {0.1: 0.2, 0.3: 0.3, 0.6: 0.4}


def _loader():
    def load(visit_id, modality):
        rng = np.random.default_rng(abs(hash((visit_id, modality))) % (2**32))
        return rng.standard_normal(1250).astype(np.float32), 125
    return load


def test_visit_rng_is_stable_and_condition_specific():
    a = visit_rng(0, "p1_w3", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    b = visit_rng(0, "p1_w3", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    c = visit_rng(0, "p1_w4", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    d = visit_rng(0, "p1_w3", "lead_off", 0.6, "ecg").integers(0, 10**9, 5)
    e = visit_rng(1, "p1_w3", "lead_off", 0.3, "ecg").integers(0, 10**9, 5)
    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist() and a.tolist() != d.tolist() and a.tolist() != e.tolist()


@pytest.mark.parametrize("kind,corrupted,untouched", [
    ("lead_off", "ecg", "ppg"), ("motion_artifact", "ppg", "ecg"),
])
def test_same_window_degrades_identically_regardless_of_call_order(kind, corrupted, untouched):
    first = make_degraded_loader(_loader(), kind, 0.3, seed=0, noise_amplitudes=AMPS)
    second = make_degraded_loader(_loader(), kind, 0.3, seed=0, noise_amplitudes=AMPS)
    x1, _ = first("v1", corrupted)
    first("v2", corrupted); first("v3", corrupted)          # advance any shared state
    second("v9", corrupted)
    x2, _ = second("v1", corrupted)
    assert np.array_equal(x1, x2)
    clean, _ = _loader()("v1", corrupted)
    assert not np.array_equal(x1, clean)
    passthrough, _ = first("v1", untouched)
    assert np.array_equal(passthrough, _loader()("v1", untouched)[0])


def test_different_windows_get_different_spans():
    loader = make_degraded_loader(_loader(), "lead_off", 0.3, seed=0, noise_amplitudes=AMPS)
    starts = []
    for v in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        x, _ = loader(v, "ecg")
        starts.append(int(np.argmax(x == 0.0)))
    assert len(set(starts)) > 1


def test_none_kind_is_identity_and_missing_ppg_raises():
    base = _loader()
    assert make_degraded_loader(base, None, None, seed=0) is base
    loader = make_degraded_loader(base, "missing_ppg", 0.3, seed=0)
    assert np.array_equal(loader("v1", "ecg")[0], base("v1", "ecg")[0])
    with pytest.raises(ValueError):
        loader("v1", "ppg")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_degradation_determinism.py -v`
Expected: FAIL — `ImportError: cannot import name 'make_degraded_loader'`.

- [ ] **Step 3: Implement in `trustbio/degradation/inject.py`**

Add `import hashlib` under `import numpy as np`, and append at the end of the file:

```python
def visit_rng(seed: int, visit_id: str, kind: str, severity: float,
              modality: str) -> np.random.Generator:
    """Generator whose stream is a pure function of (seed, window, condition,
    modality). Degradation must be reproducible per window: the fault-taxonomy
    analysis recomputes signal-quality traces on the SAME corrupted waveform a
    model was fed, in another process and another order. A generator shared
    across windows made the corrupted span depend on call order."""
    key = f"{seed}|{visit_id}|{kind}|{severity}|{modality}".encode()
    return np.random.default_rng(int(hashlib.sha256(key).hexdigest()[:16], 16))


def make_degraded_loader(load_signal, kind: str | None, severity: float | None,
                         seed: int, noise_amplitudes: dict[float, float] | None = None):
    """Wrap a SignalLoader so ECG/PPG pass through apply_degradation before any
    model preprocessing. `kind=None` (or `severity=None`) returns the loader
    unchanged -- the clean baseline."""
    if kind is None or severity is None:
        return load_signal

    def degraded_load(visit_id, modality):
        raw, sig_fs = load_signal(visit_id, modality)
        rng = visit_rng(seed, str(visit_id), kind, severity, modality)
        if modality == "ecg":
            ecg_out, _ = apply_degradation(raw, None, sig_fs, kind, severity, rng, noise_amplitudes)
            return ecg_out, sig_fs
        _, ppg_out = apply_degradation(np.zeros_like(raw), raw, sig_fs, kind, severity, rng, noise_amplitudes)
        if ppg_out is None:
            raise ValueError("missing_ppg degradation: PPG channel dropped for this visit")
        return ppg_out, sig_fs

    return degraded_load
```

- [ ] **Step 4: Delegate from `scripts/_dataset_builders.py`**

Replace the whole `wrap_loader_with_degradation` function (from `def wrap_loader_with_degradation(` through its `return degraded_load`) with:

```python
def wrap_loader_with_degradation(load_signal, kind, severity, seed):
    """Wrap a SignalLoader so ECG/PPG pairs pass through apply_degradation
    before the model's own preprocessing sees them. `kind=None` disables
    degradation entirely (the clean baseline condition). Deterministic per
    window -- see trustbio.degradation.inject.make_degraded_loader."""
    return make_degraded_loader(load_signal, kind, severity, seed)
```

and change the import line `from trustbio.degradation.inject import apply_degradation` to `from trustbio.degradation.inject import make_degraded_loader`.

- [ ] **Step 5: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_degradation_determinism.py tests/test_degradation_inject.py tests/test_dataset_builders.py -v`
Expected: all pass (4 new + 6 + 2).

- [ ] **Step 6: Commit**

```bash
git add trustbio/degradation/inject.py scripts/_dataset_builders.py tests/test_degradation_determinism.py
git commit -m "feat: deterministic per-window degradation (visit_rng, make_degraded_loader)

The corrupted span of a window is now a pure function of (seed, visit,
kind, severity, modality), so the taxonomy analysis can recompute SQI
traces on exactly the waveform each model was fed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 3: MIMIC-III-Ext-PPG adapter — unique visit ids and robust SQI parsing

**Files:**
- Modify: `trustbio/data/mimic_ext_ppg.py` (`build_mimic_ext_ppg_cohort`, `make_mimic_ext_ppg_signal_loader`, `build_mimic_ext_ppg_label_table`, `parse_sqi_vector`)
- Modify: `tests/test_mimic_ext_ppg_adapter.py` (fixture mirrors real metadata; new tests)

**Interfaces:**
- Produces: visit ids are `signal_file_name` values (unique record names such as `3000060_0002_0_2`); `parse_sqi_vector(s: str) -> np.ndarray` (floats, `nan` preserved) and `first_sqi_code(s: str) -> float` (element 0 or `nan`). Cohort columns: `visit_id, subject_id, vector_10s_pleth_sqi, vector_10s_ecg_sqi, split`.

- [ ] **Step 1: Make the fixture mirror the real metadata and write the failing tests**

In `tests/test_mimic_ext_ppg_adapter.py`, change the `rows.append({...})` block of the fixture to:

```python
        rows.append({
            # Mirror the REAL metadata.csv: segment_id is a small integer that
            # REPEATS across records (2, 4, ...); signal_file_name is the unique
            # record name; folder_path is the FULL record path (no extension).
            "segment_id": i % 3, "signal_file_name": seg_name,
            "folder_path": f"{folder}/{seg_name}",
            "subject_id": i, "event_rhythm": "SR" if i % 2 == 0 else "AF",
            "median_30s_hr": 70.0 + i,
            "vector_10s_pleth_sqi": "[1, 1, 0]" if i % 4 else "[0, 1, 1]",
            "vector_10s_ecg_sqi": "[1, -2, 1]" if i % 2 else "[1, 1, 1]",
            "strat_fold": i % 10,
        })
```

Change `test_label_table_maps_rhythm_and_hr` to build from `visit_ids=meta["signal_file_name"].tolist()`. Append:

```python
from trustbio.data.mimic_ext_ppg import first_sqi_code, parse_sqi_vector


def test_cohort_visit_ids_are_unique_record_names(fake_metadata_and_waveforms):
    root, meta = fake_metadata_and_waveforms
    cohort = build_mimic_ext_ppg_cohort(root, metadata_csv=meta)
    assert cohort.visits["visit_id"].is_unique
    assert set(cohort.visits["visit_id"]) == set(meta["signal_file_name"])


def test_parse_sqi_vector_handles_real_formats():
    assert parse_sqi_vector("[1, 1, -2]").tolist() == [1.0, 1.0, -2.0]
    v = parse_sqi_vector("[np.float64(86.21), np.float64(87.21), nan]")
    assert v[:2].tolist() == [86.21, 87.21] and np.isnan(v[2])
    assert np.isnan(parse_sqi_vector("[nan, nan, nan]")).all()
    assert first_sqi_code("[0, 1, 1]") == 0.0
    assert np.isnan(first_sqi_code("[nan, 1, 1]"))
    assert np.isnan(first_sqi_code(float("nan")))
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_mimic_ext_ppg_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name 'first_sqi_code'`; after removing that import temporarily, `test_signal_loader_reads_ecg_and_ppg` also fails (`meta.loc[visit_id]` on a repeated `segment_id` returns a DataFrame).

- [ ] **Step 3: Implement in `trustbio/data/mimic_ext_ppg.py`**

Replace the column selection in `build_mimic_ext_ppg_cohort` with:

```python
    df = meta[[
        "signal_file_name", "subject_id", "vector_10s_pleth_sqi", "vector_10s_ecg_sqi",
    ]].rename(columns={"signal_file_name": "visit_id"}).copy()
    df["visit_id"] = df["visit_id"].astype(str)
    if not df["visit_id"].is_unique:
        raise ValueError("signal_file_name is not unique -- cannot serve as visit_id")
```

In `make_mimic_ext_ppg_signal_loader` change `meta = meta.set_index("segment_id")` to `meta = meta.set_index(meta["signal_file_name"].astype(str))`. In `build_mimic_ext_ppg_label_table` change `metadata.set_index("segment_id")` to `metadata.set_index(metadata["signal_file_name"].astype(str))`. Replace `parse_sqi_vector` with:

```python
_SQI_TOKEN = re.compile(r"nan|-?\d+(?:\.\d+)?")


def parse_sqi_vector(sqi_str) -> np.ndarray:
    """Parse a stringified SQI vector into floats. Real values look like
    "[1, 1, -2]", "[nan, nan, nan]" or "[np.float64(86.21), nan]" -- the numpy
    repr breaks ast.literal_eval, so pull the numbers out with a regex."""
    if not isinstance(sqi_str, str):
        return np.array([np.nan])
    return np.asarray([float(t) for t in _SQI_TOKEN.findall(sqi_str)], dtype=float)


def first_sqi_code(sqi_str) -> float:
    """Native SQI code of the FIRST 10-s sub-window -- the one the models see
    at duration_sec=10 (encode_modality keeps the first duration_sec seconds)."""
    v = parse_sqi_vector(sqi_str)
    return float(v[0]) if len(v) else float("nan")
```

Add `import re` to the imports and delete `import ast`. Update the module docstring's first sentence to say visit ids are `signal_file_name`.

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_mimic_ext_ppg_adapter.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add trustbio/data/mimic_ext_ppg.py tests/test_mimic_ext_ppg_adapter.py
git commit -m "fix: MIMIC-ext visit ids are signal_file_name; SQI parsing handles real formats

segment_id is a small integer repeated across records in the real
metadata.csv; keying the cohort/loader on it collided. The fixture had
hidden this with unique strings. parse_sqi_vector now reads the numpy
reprs the real file contains.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Taxonomy cohorts (PulseDB subsample, stratified MIMIC-ext sample, BUT PPG)

**Files:**
- Modify: `scripts/sample_cohort.py` (`cap_windows_per_subject`, `--max-windows-per-subject`)
- Create: `scripts/sample_mimic_ext_segments.py`
- Modify: `scripts/_dataset_builders.py` (`mimic_ext_ppg` branch reads `<cohort_cache>/mimic_ext_ppg_metadata.csv` when present)
- Test: `tests/test_sample_cohort.py`, `tests/test_sample_mimic_ext_segments.py` (create); `tests/test_dataset_builders.py` (extend)

**Interfaces:**
- Produces: `cap_windows_per_subject(df, n) -> pd.DataFrame` (evenly spaced windows per subject, order preserved); `sample_mimic_ext_segments.select_segments(meta_chunks, n_per_stratum, max_per_subject, seed) -> pd.DataFrame` with a `stratum` column ∈ {clean, ppg_poor, ecg_poor} and `pleth_sqi0`, `ecg_sqi0`; files under `features_cache/taxonomy/`: `pulsedb_mimic_cohort.csv`, `pulsedb_vital_cohort.csv`, `mimic_ext_ppg_metadata.csv` (subset of the real metadata, all columns), `mimic_ext_ppg_cohort.csv` (with `split`, `stratum`, `pleth_sqi0`, `ecg_sqi0`), `but_ppg_cohort.csv` (`build_but_ppg_cohort` output incl. `quality`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sample_cohort.py`:

```python
import numpy as np
import pandas as pd

from scripts.sample_cohort import cap_windows_per_subject, sample_subjects


def _cohort():
    rows = []
    for s in ["a", "b", "c"]:
        for w in range(10):
            rows.append({"visit_id": f"{s}_w{w}", "subject_id": s, "split": "train"})
    return pd.DataFrame(rows)


def test_cap_windows_per_subject_keeps_evenly_spaced_windows_in_order():
    out = cap_windows_per_subject(_cohort(), 4)
    assert out.groupby("subject_id").size().tolist() == [4, 4, 4]
    assert out[out.subject_id == "a"]["visit_id"].tolist() == ["a_w0", "a_w3", "a_w6", "a_w9"]
    assert out["visit_id"].tolist() == sorted(out["visit_id"], key=lambda v: (v[0], int(v.split("w")[1])))


def test_cap_is_a_no_op_when_subjects_have_fewer_windows():
    df = _cohort()
    assert len(cap_windows_per_subject(df, 50)) == len(df)


def test_sample_then_cap_preserves_split_disjointness():
    df = _cohort(); df.loc[df.subject_id == "c", "split"] = "test"
    out = cap_windows_per_subject(sample_subjects(df, 2, seed=0), 3)
    assert (out.groupby("subject_id")["split"].nunique() == 1).all()
```

Create `tests/test_sample_mimic_ext_segments.py`:

```python
import numpy as np
import pandas as pd

from scripts.sample_mimic_ext_segments import select_segments


def _meta(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p0 = int(rng.random() < 0.8)
        e0 = 1 if rng.random() < 0.7 else int(rng.choice([0, -2, -3]))
        rows.append({
            "signal_file_name": f"rec_{i}", "subject_id": i % 40,
            "folder_path": f"p00/p{i%40:06d}/rec_{i}",
            "vector_10s_pleth_sqi": f"[{p0}, 1, 1]", "vector_10s_ecg_sqi": f"[{e0}, 1, 1]",
            "median_30s_hr": 70.0, "event_rhythm": "SR",
        })
    return pd.DataFrame(rows)


def test_select_segments_stratifies_and_caps_subjects():
    meta = _meta()
    out = select_segments([meta.iloc[:300], meta.iloc[300:]], n_per_stratum=30, max_per_subject=2, seed=0)
    counts = out["stratum"].value_counts().to_dict()
    assert set(counts) == {"clean", "ppg_poor", "ecg_poor"} and max(counts.values()) <= 30
    assert (out.groupby("subject_id").size() <= 2).all()
    assert (out[out.stratum == "clean"][["pleth_sqi0", "ecg_sqi0"]] == 1).all().all()
    assert (out[out.stratum == "ppg_poor"]["pleth_sqi0"] == 0).all()
    assert (out[out.stratum == "ecg_poor"]["ecg_sqi0"] <= 0).all()
    assert out["signal_file_name"].is_unique
    assert set(meta.columns) <= set(out.columns)


def test_select_segments_is_reproducible():
    meta = _meta()
    a = select_segments([meta], 20, 3, seed=1)["signal_file_name"].tolist()
    b = select_segments([meta], 20, 3, seed=1)["signal_file_name"].tolist()
    assert a == b
```

Append to `tests/test_dataset_builders.py`:

```python
def test_mimic_ext_handle_reads_metadata_subset_from_cohort_cache(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    meta = pd.DataFrame({
        "signal_file_name": ["r1", "r2", "r3"], "subject_id": [1, 2, 3],
        "folder_path": ["p00/p1/r1", "p00/p2/r2", "p00/p3/r3"],
        "vector_10s_pleth_sqi": ["[1, 1, 1]"] * 3, "vector_10s_ecg_sqi": ["[1, 1, 1]"] * 3,
        "median_30s_hr": [70.0, 71.0, 72.0], "event_rhythm": ["SR", "AF", "SR"],
    })
    meta.to_csv(cache / "mimic_ext_ppg_metadata.csv", index=False)
    args = argparse.Namespace(mimic_ext_ppg_root=tmp_path / "no-such-root", cohort_cache=cache)
    handle = build_dataset_handle("mimic_ext_ppg", args, with_labels=True)
    all_ids = sorted(v for df in handle.splits.values() for v in df["visit_id"])
    assert all_ids == ["r1", "r2", "r3"]
    assert handle.label_table[next(iter(handle.splits))].columns.tolist() == ["hr_regression", "rhythm_cls"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_sample_cohort.py tests/test_sample_mimic_ext_segments.py tests/test_dataset_builders.py -v`
Expected: FAIL — `ImportError` (`cap_windows_per_subject`; module `scripts.sample_mimic_ext_segments`); the builder test fails reading `no-such-root/metadata.csv`.

- [ ] **Step 3: Extend `scripts/sample_cohort.py`**

Add after `sample_subjects`:

```python
def cap_windows_per_subject(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep at most `n` windows per subject, evenly spaced through the
    subject's recording (windows are in chronological order in the cohort),
    preserving row order. The taxonomy needs breadth across subjects more than
    depth within one; 60 subjects x 40 windows beats 4 subjects x 600."""
    if n is None or n <= 0:
        return df
    keep = []
    for _, g in df.groupby("subject_id", sort=False):
        idx = np.unique(np.round(np.linspace(0, len(g) - 1, min(n, len(g)))).astype(int))
        keep.append(g.iloc[idx])
    return pd.concat(keep).loc[df.index.intersection(pd.concat(keep).index)]
```

Add `import numpy as np` to the imports, `ap.add_argument("--max-windows-per-subject", type=int, default=None)` to `main`, and after `out = sample_subjects(...)` add `out = cap_windows_per_subject(out, a.max_windows_per_subject)`.

- [ ] **Step 4: Create `scripts/sample_mimic_ext_segments.py`**

```python
#!/usr/bin/env python
"""Draw the MIMIC-III-Ext-PPG taxonomy sample: segments stratified by the
NATIVE signal-quality code of their first 10-s sub-window (the sub-window the
models see at duration_sec=10), spread across subjects, restricted to records
that actually contain lead II.

    clean     pleth_sqi0 == 1 and ecg_sqi0 == 1
    ppg_poor  pleth_sqi0 == 0
    ecg_poor  pleth_sqi0 == 1 and ecg_sqi0 <= 0   (0, -2, -3 ... undocumented negative codes)

Writes <out-dir>/mimic_ext_ppg_metadata.csv (the sampled rows, ALL columns,
so downstream handles never touch the 4.9 GB metadata.csv again) and
<out-dir>/mimic_ext_ppg_cohort.csv (subject-disjoint split + stratum columns).
Run as a batch job: reading metadata.csv in chunks needs ~10 GB and ~5 min.

    python scripts/sample_mimic_ext_segments.py --out-dir features_cache/taxonomy \
        --n-per-stratum 800 --max-per-subject 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from trustbio.data.mimic_ext_ppg import DEFAULT_ROOT, build_mimic_ext_ppg_cohort, first_sqi_code

STRATA = ("clean", "ppg_poor", "ecg_poor")


def _stratum(p0: float, e0: float) -> str | None:
    if np.isnan(p0) or np.isnan(e0):
        return None
    if p0 == 0:
        return "ppg_poor"
    if p0 == 1 and e0 == 1:
        return "clean"
    if p0 == 1 and e0 <= 0:
        return "ecg_poor"
    return None


def select_segments(meta_chunks, n_per_stratum: int, max_per_subject: int, seed: int) -> pd.DataFrame:
    """Reservoir-style selection over an iterable of metadata chunks: every row
    gets a random key; per stratum keep the smallest keys subject to the
    per-subject cap. Deterministic for a given seed and chunk order."""
    rng = np.random.default_rng(seed)
    pools = {s: [] for s in STRATA}
    for chunk in meta_chunks:
        chunk = chunk.copy()
        chunk["pleth_sqi0"] = chunk["vector_10s_pleth_sqi"].map(first_sqi_code)
        chunk["ecg_sqi0"] = chunk["vector_10s_ecg_sqi"].map(first_sqi_code)
        chunk["stratum"] = [_stratum(p, e) for p, e in zip(chunk["pleth_sqi0"], chunk["ecg_sqi0"])]
        chunk["_key"] = rng.random(len(chunk))
        for s in STRATA:
            part = chunk[chunk["stratum"] == s]
            if len(part):
                pools[s].append(part.nsmallest(20 * n_per_stratum, "_key"))
    out = []
    for s in STRATA:
        if not pools[s]:
            continue
        cand = pd.concat(pools[s]).sort_values("_key")
        per_subject = {}
        keep = []
        for i, row in cand.iterrows():
            c = per_subject.get(row["subject_id"], 0)
            if c >= max_per_subject:
                continue
            per_subject[row["subject_id"]] = c + 1
            keep.append(i)
            if len(keep) >= n_per_stratum:
                break
        out.append(cand.loc[keep])
    result = pd.concat(out).drop(columns=["_key"]).reset_index(drop=True)
    assert result["signal_file_name"].is_unique
    return result


def has_lead_ii(root: Path, folder_path: str) -> bool:
    try:
        return "II" in [s.upper() for s in wfdb.rdheader(str(root / folder_path)).sig_name]
    except Exception:  # noqa: BLE001 -- unreadable header == unusable record
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-per-stratum", type=int, default=800)
    ap.add_argument("--max-per-subject", type=int, default=4)
    ap.add_argument("--oversample", type=float, default=1.5,
                    help="draw this many times n-per-stratum before the lead-II check")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = pd.read_csv(args.root / "metadata.csv", chunksize=args.chunksize, low_memory=False)
    picked = select_segments(chunks, int(args.n_per_stratum * args.oversample),
                             args.max_per_subject, args.seed)
    picked["has_ecg"] = [has_lead_ii(args.root, fp) for fp in picked["folder_path"]]
    picked = picked[picked["has_ecg"]].drop(columns=["has_ecg"])
    picked = picked.groupby("stratum", group_keys=False).head(args.n_per_stratum).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    picked.to_csv(args.out_dir / "mimic_ext_ppg_metadata.csv", index=False)
    cohort = build_mimic_ext_ppg_cohort(args.root, metadata_csv=picked, seed=args.seed).visits
    extra = picked.set_index(picked["signal_file_name"].astype(str))[["stratum", "pleth_sqi0", "ecg_sqi0"]]
    cohort = cohort.join(extra, on="visit_id")
    cohort.to_csv(args.out_dir / "mimic_ext_ppg_cohort.csv", index=False)
    print(f"strata: {cohort['stratum'].value_counts().to_dict()}  subjects: {cohort['subject_id'].nunique()}")
    print(f"splits: {cohort['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Teach `build_dataset_handle` to read the metadata subset**

In `scripts/_dataset_builders.py`, replace the start of the `mimic_ext_ppg` branch (`cohort = build_mimic_ext_ppg_cohort(args.mimic_ext_ppg_root)` … `loader = make_mimic_ext_ppg_signal_loader(args.mimic_ext_ppg_root)`) with:

```python
        # A taxonomy sample ships its own metadata subset (all columns, only
        # the sampled rows) so nothing here re-reads the 4.9 GB metadata.csv.
        cache_dir = getattr(args, "cohort_cache", None)
        subset = Path(cache_dir) / "mimic_ext_ppg_metadata.csv" if cache_dir else None
        meta = pd.read_csv(subset if subset is not None and subset.exists()
                           else args.mimic_ext_ppg_root / "metadata.csv", low_memory=False)
        cohort = build_mimic_ext_ppg_cohort(args.mimic_ext_ppg_root, metadata_csv=meta)
        splits = {s: cohort.split(s) for s in ("train", "val", "test")}
        loader = make_mimic_ext_ppg_signal_loader(args.mimic_ext_ppg_root, meta)
```

and in the same branch use `meta` (not a fresh `pd.read_csv`) for `build_mimic_ext_ppg_label_table`. Add `from pathlib import Path` to the imports.

- [ ] **Step 6: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_sample_cohort.py tests/test_sample_mimic_ext_segments.py tests/test_dataset_builders.py -v`
Expected: all pass (3 + 2 + 3).

- [ ] **Step 7: Build the cohorts (PulseDB on the login node is fine; MIMIC-ext and BUT PPG as one job)**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
mkdir -p features_cache/taxonomy
for SRC in mimic vital; do
  conda run -n trust-bio python scripts/sample_cohort.py --cohort features_cache/pulsedb_${SRC}_cohort.csv \
    --out features_cache/taxonomy/pulsedb_${SRC}_cohort.csv --n-subjects 60 --max-windows-per-subject 40 --seed 0
done
sbatch --job-name=tb-tax-cohorts --partition=short --cpus-per-task=2 --mem=24G --time=3:00:00 \
  --output=logs/tax_cohorts_%j.out --error=logs/tax_cohorts_%j.err \
  --wrap="cd $PWD && module load conda/miniforge3/24.11.3-0 2>/dev/null; conda run --no-capture-output -n trust-bio python scripts/sample_mimic_ext_segments.py --out-dir features_cache/taxonomy --n-per-stratum 800 --max-per-subject 4 && conda run --no-capture-output -n trust-bio python -c \"from trustbio.data.but_ppg import build_but_ppg_cohort; build_but_ppg_cohort().visits.to_csv('features_cache/taxonomy/but_ppg_cohort.csv', index=False); print('but_ppg cohort written')\""
```

Expected: two PulseDB samples of ≤ 2,400 windows / 60 subjects each with all three splits; after the job, `mimic_ext_ppg_cohort.csv` with ≈ 800 rows per stratum (`strata: {'clean': 800, 'ppg_poor': 800, 'ecg_poor': 800}` — fewer only if the lead-II filter bit; if any stratum has < 400 rows, rerun with `--oversample 3`), and `but_ppg_cohort.csv` with 3,888 rows and a `quality` column.

- [ ] **Step 8: Commit**

```bash
git add scripts/sample_cohort.py scripts/sample_mimic_ext_segments.py scripts/_dataset_builders.py tests/test_sample_cohort.py tests/test_sample_mimic_ext_segments.py tests/test_dataset_builders.py
git commit -m "feat: taxonomy cohorts -- per-subject window cap, stratified MIMIC-ext sample, metadata subset

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Manifest and sbatch support for degradation conditions

**Files:**
- Modify: `scripts/make_manifest.py` (`--with-clean`, `--degrade KIND=SEV`, `--skip-availability-check`; `build_lines(..., conditions=None)`)
- Modify: `scripts/_extract_cell.sh` (parse `cond=`/`kind=`/`sev=` fields; per-condition store subdir; degrade flags)
- Test: `tests/test_make_manifest.py`, `tests/test_extract_sbatch.py` (extend)

**Interfaces:**
- Manifest line grammar: `model dataset duration [chunk n_chunks] [cond=NAME] [kind=KIND sev=SEV]`. With `cond=NAME` the cell's store root is `<TRUSTBIO_STORE>/NAME`; with `kind=`/`sev=` the cell runs `extract_features.py --degrade-kind KIND --degrade-severity SEV` (seed stays the CLI default 0). `build_lines(models, datasets, duration, chunks, conditions=None)` where `conditions` is a list of `(name, kind|None, severity|None)`; `None` keeps the old three-field lines.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_make_manifest.py`:

```python
def test_build_lines_emits_condition_fields():
    conds = [("clean", None, None), ("lead_off_0.3", "lead_off", 0.3)]
    lines = build_lines(["m"], ["pulsedb_vital"], 10, {}, conditions=conds)
    assert lines == ["m pulsedb_vital 10 cond=clean",
                     "m pulsedb_vital 10 cond=lead_off_0.3 kind=lead_off sev=0.3"]


def test_build_lines_condition_fields_follow_chunk_fields():
    lines = build_lines(["m"], ["pulsedb_mimic"], 10, {"pulsedb_mimic": 2},
                        conditions=[("motion_artifact_0.1", "motion_artifact", 0.1)])
    assert lines[0] == "m pulsedb_mimic 10 0 2 cond=motion_artifact_0.1 kind=motion_artifact sev=0.1"


def test_parse_degrade_specs():
    from scripts.make_manifest import parse_degrade
    assert parse_degrade(["motion_artifact=0.3", "lead_off=0.6"]) == [
        ("motion_artifact_0.3", "motion_artifact", 0.3), ("lead_off_0.6", "lead_off", 0.6)]
    with pytest.raises(ValueError):
        parse_degrade(["nope=0.3"])
    with pytest.raises(ValueError):
        parse_degrade(["lead_off=2"])
```

Append to `tests/test_extract_sbatch.py`:

```python
def test_dry_run_condition_fields_set_store_subdir_and_degrade_flags(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("papagei pulsedb_vital 10 cond=lead_off_0.3 kind=lead_off sev=0.3\n"
                        "papagei but_ppg 10 cond=clean\n")
    r = _run("scripts/extract_features.sbatch", manifest, 0)
    assert r.returncode == 0, r.stderr
    assert "--store /s/lead_off_0.3" in r.stdout
    assert "--degrade-kind lead_off --degrade-severity 0.3" in r.stdout
    r = _run("scripts/extract_features.sbatch", manifest, 1)
    assert "--store /s/clean" in r.stdout and "--degrade-kind" not in r.stdout


def test_dry_run_condition_fields_coexist_with_chunks(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("papagei pulsedb_mimic 10 3 8 cond=motion_artifact_0.6 kind=motion_artifact sev=0.6\n")
    r = _run("scripts/extract_features.sbatch", manifest, 0)
    assert "--chunk 3 --n-chunks 8" in r.stdout and "--store /s/motion_artifact_0.6" in r.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_make_manifest.py tests/test_extract_sbatch.py -v`
Expected: FAIL (`build_lines() got an unexpected keyword argument 'conditions'`; dry-run stores lack the subdir).

- [ ] **Step 3: Extend `scripts/make_manifest.py`**

Add after `parse_chunks`:

```python
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
```

Replace `build_lines` with:

```python
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
```

In `main`, add the arguments:

```python
    ap.add_argument("--with-clean", action="store_true",
                    help="emit a cond=clean line per cell (needed whenever --degrade is used)")
    ap.add_argument("--degrade", nargs="*", default=[], metavar="KIND=SEV",
                    help="also emit a degraded line per cell for each KIND=SEV")
    ap.add_argument("--skip-availability-check", action="store_true",
                    help="do not filter --models through is_model_available (the job checks anyway)")
```

build `conditions`:

```python
    conditions = ([("clean", None, None)] if args.with_clean else []) + parse_degrade(args.degrade)
    if args.degrade and not args.with_clean:
        print("[manifest] note: --degrade without --with-clean emits no clean baseline lines")
```

pass `conditions=conditions or None` to `build_lines`, and skip the availability filter when `args.skip_availability_check` is set. Import `DEGRADATION_KINDS, DEGRADATION_SEVERITIES` from `trustbio.config`.

- [ ] **Step 4: Extend `scripts/_extract_cell.sh`**

Replace the block from `read -r MODEL DATASET DURATION CHUNK NCHUNKS <<<"${LINE}"` through the `CHUNK_ARGS=()` / `[[ -n "${CHUNK:-}" ]] && ...` lines with:

```bash
read -r MODEL DATASET DURATION REST <<<"${LINE}"
if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "[extract] no manifest line $((i+1)) in ${MANIFEST}" >&2
  exit 2
fi
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"
# Optional trailing fields: positional `chunk n_chunks`, then name=value
# fields `cond=NAME` (store subdir), `kind=KIND sev=SEV` (degradation).
CHUNK=""; NCHUNKS=""; COND=""; KIND=""; SEV=""; POS=()
for field in ${REST:-}; do
  case "${field}" in
    cond=*) COND="${field#cond=}" ;;
    kind=*) KIND="${field#kind=}" ;;
    sev=*)  SEV="${field#sev=}" ;;
    *)      POS+=("${field}") ;;
  esac
done
if [[ ${#POS[@]} -ge 2 ]]; then CHUNK="${POS[0]}"; NCHUNKS="${POS[1]}"; fi
CELL_STORE="${STORE}${COND:+/${COND}}"

CHUNK_ARGS=()
[[ -n "${CHUNK}" ]] && CHUNK_ARGS=(--chunk "${CHUNK}" --n-chunks "${NCHUNKS}")
DEGRADE_ARGS=()
[[ -n "${KIND}" ]] && DEGRADE_ARGS=(--degrade-kind "${KIND}" --degrade-severity "${SEV}")
```

In the `echo "[extract] task ..."` line add ` cond=${COND:-none}` and change `store=${STORE}` to `store=${CELL_STORE}`. In `CMD=(...)` change `--store "${STORE}"` to `--store "${CELL_STORE}"` and add `"${DEGRADE_ARGS[@]}"` after `"${CHUNK_ARGS[@]}"`.

- [ ] **Step 5: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_make_manifest.py tests/test_extract_sbatch.py -v`
Expected: all pass (6 + 3 new; 6 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/make_manifest.py scripts/_extract_cell.sh tests/test_make_manifest.py tests/test_extract_sbatch.py
git commit -m "feat: manifest/sbatch condition fields (cond=, kind=, sev=) with per-condition store roots

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Extract the degraded feature store and verify it

**Files:**
- Create: `manifest_tax_gpu.txt`, `manifest_tax_cpu.txt` (generated)
- Run: extraction arrays + `scripts/verify_feature_store.py` per condition

**Interfaces:**
- Produces: `features_cache/taxonomy_store/<cond>/<dataset>/<model>/<modality>/10s/<split>.npz` for cond ∈ {clean, motion_artifact_0.1, motion_artifact_0.3, motion_artifact_0.6, lead_off_0.1, lead_off_0.3, lead_off_0.6}, datasets pulsedb_mimic / pulsedb_vital / mimic_ext_ppg (all conds) and but_ppg (clean only), all 7 models.

- [ ] **Step 1: Generate the manifests**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
GPU_MODELS="moment-base chronos-bolt-small dbeta ecgfounder xecg-10min papagei"
DEG="motion_artifact=0.1 motion_artifact=0.3 motion_artifact=0.6 lead_off=0.1 lead_off=0.3 lead_off=0.6"
conda run -n trust-bio python scripts/make_manifest.py --duration 10 --skip-availability-check --models $GPU_MODELS \
  --datasets pulsedb_mimic pulsedb_vital mimic_ext_ppg --with-clean --degrade $DEG --out manifest_tax_gpu.txt
conda run -n trust-bio python scripts/make_manifest.py --duration 10 --skip-availability-check --models $GPU_MODELS \
  --datasets but_ppg --with-clean --out /tmp/tax_gpu_butppg.txt && cat /tmp/tax_gpu_butppg.txt >> manifest_tax_gpu.txt
conda run -n trust-bio python scripts/make_manifest.py --duration 10 --skip-availability-check --models ecg-domain \
  --datasets pulsedb_mimic pulsedb_vital mimic_ext_ppg --with-clean --degrade $DEG --out manifest_tax_cpu.txt
conda run -n trust-bio python scripts/make_manifest.py --duration 10 --skip-availability-check --models ecg-domain \
  --datasets but_ppg --with-clean --out /tmp/tax_cpu_butppg.txt && cat /tmp/tax_cpu_butppg.txt >> manifest_tax_cpu.txt
wc -l manifest_tax_gpu.txt manifest_tax_cpu.txt
```

Expected: `132 manifest_tax_gpu.txt`, `22 manifest_tax_cpu.txt`.

- [ ] **Step 2: Launch**

```bash
export TRUSTBIO_REPO="$PWD" TRUSTBIO_STORE="$PWD/features_cache/taxonomy_store" TRUSTBIO_COHORT_CACHE="$PWD/features_cache/taxonomy"
mkdir -p "$TRUSTBIO_STORE"
GPU=$(sbatch --parsable --array=0-131%12 scripts/extract_features.sbatch manifest_tax_gpu.txt | cut -d';' -f1)
CPU=$(sbatch --parsable --array=0-21%14 scripts/extract_features_cpu.sbatch manifest_tax_cpu.txt | cut -d';' -f1)
echo "gpu=$GPU cpu=$CPU"
```

Note `verify_feature_store.py --datasets` requires cohort CSVs named `<dataset>_cohort.csv` in `features_cache/taxonomy/` — Task 4 wrote all four. When both arrays finish (`sacct -j $GPU,$CPU -X -n --format=State | sort | uniq -c` shows only `COMPLETED`):

```bash
sbatch --job-name=tb-tax-verify --partition=short --cpus-per-task=4 --mem=32G --time=2:00:00 \
  --output=logs/tax_verify_%j.out --error=logs/tax_verify_%j.err \
  --wrap="cd $PWD && module load conda/miniforge3/24.11.3-0 2>/dev/null; for C in clean motion_artifact_0.1 motion_artifact_0.3 motion_artifact_0.6 lead_off_0.1 lead_off_0.3 lead_off_0.6; do DS='pulsedb_mimic pulsedb_vital mimic_ext_ppg'; [ \$C = clean ] && DS=\"\$DS but_ppg\"; echo \"== \$C\"; conda run --no-capture-output -n trust-bio python scripts/verify_feature_store.py --store features_cache/taxonomy_store/\$C --cohort-cache features_cache/taxonomy --duration-sec 10 --datasets \$DS || exit 1; done"
grep -h 'checked\|BAD' logs/tax_verify_*.out
```

Expected: seven `checked N files, 0 bad` lines (N = 63 for clean incl. but_ppg, 63 otherwise... exactly: 3 datasets × 7 models × 3 modalities × 3 splits = 189 per degraded condition, 252 for clean). A `BAD ... visit_ids differ` line means a window was skipped during extraction (its `[extract]` log line will say `SKIPPED n`); find and fix the cause rather than tolerating a hole, because Task 10 joins features to SQI rows by visit id.

- [ ] **Step 3: Commit the manifests**

```bash
git add manifest_tax_gpu.txt manifest_tax_cpu.txt
git commit -m "chore: taxonomy extraction manifests (7 conditions x 4 datasets x 7 models)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 7: Per-second signal-quality traces (`trustbio/taxonomy/sqi.py`)

> **Revision (2026-09-29, after the first full analysis run; results kept in `results/taxonomy_v1_hfecg/`).** The run completed but its SQI validity check failed: ECG AUROC 0.29 vs native (inverted) and PPG 0.62 with `hf_ref_ppg` collapsing to 0.045. Two definitional errors: (1) a high-frequency residual is the wrong quality measure for ECG — QRS complexes ARE the HF content, so clean ECG scored as noisy; the ECG trace is now a pure flat-line (electrode-off/dropout) detector, which is the only ECG fault injected and the only one the native code −3 confirms. (2) The 40-ms moving-average residual is a 3-sample kernel at 30 Hz and 5 at 125 Hz, so BUT PPG's "real noise" quantiles were not comparable with PulseDB windows (every camera-PPG recording scored ≈0 and landed in the motion cluster); the PPG statistic is now the out-of-pulse-band (0.5–8 Hz) residual ratio, rate-independent, used for BOTH the SQI trace and the injection calibration (which is refit). The PPG quality-zero reference is no longer fitted to the native label (Youden on a weak label overfit): it is the 95th percentile of real BUT PPG per-second ratios, and native SQI is reported as a validation AUROC only. Per-modality drop durations (`ecg_drop_duration`, `ppg_drop_duration`) join the feature set. Motion-artifact cells are re-extracted with the refit amplitudes.

**Files:**
- Create: `trustbio/taxonomy/sqi.py`
- Test: `tests/test_taxonomy_sqi.py` (create)

**Interfaces:**
- Produces: `hf_noise_ratio(x, fs) -> float`; `mean_hf_ratio(x, fs, window_sec=1.0) -> float`; `sqi_trace(x, fs, hf_ref, window_sec=1.0) -> np.ndarray` (one value in [0, 1] per sub-window; 0 for a flat sub-window); `combined_sqi_trace(ecg_sqi, ppg_sqi) -> np.ndarray` (elementwise min); `calibrate_hf_ref(hf_good, hf_poor) -> float`. Constants `FLAT_REL_STD = 1e-4`, `SMOOTH_SEC = 0.04`, `DEFAULT_HF_REF = 0.5`.

Why this SQI: the two injected faults are a flat-lined ECG span (lead-off) and added smoothed noise on PPG (motion), and the calibration in `degradation/calibrate.py` already defines "noise" as the high-frequency residual after a short moving average. The trace therefore has exactly two ingredients — a flat-line test and that same residual ratio — with one free scale per modality (`hf_ref`, the ratio at which quality is called zero), fitted once against MIMIC-III-Ext-PPG's native SQI in Task 10. No peak detection, no template library: it must run on 10-s windows at 30–1000 Hz in microseconds.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_taxonomy_sqi.py`:

```python
import numpy as np
import pytest

from trustbio.taxonomy.sqi import (
    calibrate_hf_ref, combined_sqi_trace, hf_noise_ratio, mean_hf_ratio, sqi_trace,
)

FS = 125


def _clean(fs=FS, seconds=10, seed=0):
    t = np.arange(seconds * fs) / fs
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
            + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_trace_has_one_value_per_second_at_any_rate():
    assert len(sqi_trace(_clean(125), 125, hf_ref=0.5)) == 10
    assert len(sqi_trace(_clean(30), 30, hf_ref=0.5)) == 10
    assert len(sqi_trace(_clean(1000), 1000, hf_ref=0.5)) == 10


def test_clean_signal_scores_high_everywhere():
    s = sqi_trace(_clean(), FS, hf_ref=0.5)
    assert s.min() > 0.7 and s.max() <= 1.0


def test_flat_span_scores_zero_exactly_where_it_is_flat():
    x = _clean(); x[3 * FS:6 * FS] = 0.0                # lead-off seconds 3,4,5
    s = sqi_trace(x, FS, hf_ref=0.5)
    assert s[3:6].tolist() == [0.0, 0.0, 0.0]
    assert (s[:3] > 0.7).all() and (s[6:] > 0.7).all()


def test_noisy_span_drops_only_where_noise_is():
    x = _clean(); rng = np.random.default_rng(1)
    x[2 * FS:5 * FS] += (0.8 * np.std(x) * rng.standard_normal(3 * FS)).astype(np.float32)
    s = sqi_trace(x, FS, hf_ref=0.5)
    assert (s[2:5] < 0.5).all()
    assert (np.r_[s[:2], s[5:]] > 0.7).all()


def test_combined_is_elementwise_min():
    a, b = np.array([1.0, 0.2, 0.9]), np.array([0.5, 0.8, 0.9, 0.1])
    assert combined_sqi_trace(a, b).tolist() == [0.5, 0.2, 0.9]


def test_calibrate_hf_ref_separates_native_good_from_poor():
    rng = np.random.default_rng(0)
    good, poor = rng.normal(0.10, 0.02, 300), rng.normal(0.50, 0.05, 300)
    ref = calibrate_hf_ref(good, poor)
    # SQI < 0.5 <=> hf_ratio > ref/2, so ref/2 must sit between the two populations
    assert 0.15 < ref / 2 < 0.45
    calls_poor = np.mean(poor > ref / 2); calls_good = np.mean(good > ref / 2)
    assert calls_poor > 0.95 and calls_good < 0.05


def test_mean_hf_ratio_is_higher_for_noisier_signal():
    x = _clean(); noisy = x + (0.5 * np.std(x) * np.random.default_rng(2).standard_normal(len(x))).astype(np.float32)
    assert mean_hf_ratio(noisy, FS) > mean_hf_ratio(x, FS)
    assert hf_noise_ratio(np.zeros(100), FS) == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_sqi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trustbio.taxonomy.sqi'`.

- [ ] **Step 3: Create `trustbio/taxonomy/sqi.py`**

```python
"""Per-sub-window signal-quality traces for ECG and PPG.

The taxonomy's transient/persistent distinction is about WHEN and for HOW
LONG quality drops, so it needs a quality value per second, on the exact
(possibly degraded) waveform a model was fed. Two ingredients, deliberately
nothing more:

  * flat-line: a sub-window whose spread is a negligible fraction of the whole
    window's spread scores 0 -- an electrode off or zeroed span;
  * high-frequency residual ratio: std(x - moving_average(x, 40 ms)) / std(x),
    the same quantity degradation/calibrate.py uses to define "noise". The
    trace is 1 - ratio / hf_ref, clipped to [0, 1]; `hf_ref` is the ratio at
    which quality is called zero, one free scale per modality, fitted against
    MIMIC-III-Ext-PPG's native SQI (scripts/build_fault_features.py).
"""
from __future__ import annotations

import numpy as np

FLAT_REL_STD = 1e-4
SMOOTH_SEC = 0.04
DEFAULT_HF_REF = 0.5


def hf_noise_ratio(x: np.ndarray, fs: int) -> float:
    """High-frequency residual energy as a fraction of total spread; 0 for a
    constant signal."""
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd == 0.0:
        return 0.0
    k = max(3, int(round(SMOOTH_SEC * fs)))
    smooth = np.convolve(x, np.ones(k) / k, mode="same")
    return float(np.std(x - smooth) / sd)


def _sub_windows(x: np.ndarray, fs: int, window_sec: float) -> list[np.ndarray]:
    n = max(1, int(round(window_sec * fs)))
    n_win = max(1, len(x) // n)
    return [x[i * n:(i + 1) * n] for i in range(n_win)]


def mean_hf_ratio(x: np.ndarray, fs: int, window_sec: float = 1.0) -> float:
    """Mean sub-window residual ratio -- the per-segment statistic that gets
    compared against a native quality label when calibrating hf_ref."""
    return float(np.mean([hf_noise_ratio(w, fs) for w in _sub_windows(np.asarray(x, dtype=np.float64), fs, window_sec)]))


def sqi_trace(x: np.ndarray, fs: int, hf_ref: float, window_sec: float = 1.0) -> np.ndarray:
    """Quality in [0, 1] per sub-window of `window_sec` seconds."""
    x = np.asarray(x, dtype=np.float64)
    whole_sd = float(np.std(x))
    out = []
    for w in _sub_windows(x, fs, window_sec):
        if whole_sd == 0.0 or float(np.std(w)) < FLAT_REL_STD * whole_sd:
            out.append(0.0)
        else:
            out.append(float(np.clip(1.0 - hf_noise_ratio(w, fs) / hf_ref, 0.0, 1.0)))
    return np.asarray(out, dtype=np.float64)


def combined_sqi_trace(ecg_sqi: np.ndarray, ppg_sqi: np.ndarray) -> np.ndarray:
    n = min(len(ecg_sqi), len(ppg_sqi))
    return np.minimum(np.asarray(ecg_sqi[:n]), np.asarray(ppg_sqi[:n]))


def calibrate_hf_ref(hf_good: np.ndarray, hf_poor: np.ndarray) -> float:
    """Pick hf_ref so that the trace's 0.5 threshold (hf_ratio == hf_ref / 2)
    best separates natively-good from natively-poor segments (max Youden J
    over the pooled per-segment mean ratios)."""
    good, poor = np.asarray(hf_good, float), np.asarray(hf_poor, float)
    if len(good) == 0 or len(poor) == 0:
        return DEFAULT_HF_REF
    best_t, best_j = float(np.median(np.r_[good, poor])), -np.inf
    for t in np.unique(np.r_[good, poor]):
        j = float(np.mean(poor >= t) - np.mean(good >= t))
        if j > best_j:
            best_j, best_t = j, float(t)
    return 2.0 * best_t
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_sqi.py -v`
Expected: 7 passed. (If `test_clean_signal_scores_high_everywhere` fails because the clean residual ratio at 125 Hz exceeds 0.15, the 40-ms smoother is too short for this synthetic signal — check `hf_noise_ratio(_clean(), 125)` and adjust the synthetic signal's noise term in the test, not the constants.)

- [ ] **Step 5: Commit**

```bash
git add trustbio/taxonomy/sqi.py tests/test_taxonomy_sqi.py
git commit -m "feat: per-second SQI traces (flat-line + HF residual ratio) with hf_ref calibration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Per-modality SQI features and column selection in `trustbio/taxonomy/features.py`

**Files:**
- Modify: `trustbio/taxonomy/features.py`
- Modify: `tests/test_taxonomy_features.py` (update the names test; add two)

**Interfaces:**
- Produces: `SegmentFaultFeatures` gains `ecg_sqi_value: float = nan`, `ppg_sqi_value: float = nan`; `extract_fault_features(sqi_trace, accel_trace, fs, source_db, model_a_pred, model_b_pred, disagreement_scale, ecg_sqi_trace=None, ppg_sqi_trace=None)`; `FEATURE_NAMES = ["sqi_value", "sqi_drop_duration", "accel_corr", "source_db", "model_disagreement", "ecg_sqi_value", "ppg_sqi_value"]`; `features_to_matrix(features, columns=None) -> (X, names)` where `columns` selects/orders a subset of `FEATURE_NAMES`.

- [ ] **Step 1: Update and extend the tests**

In `tests/test_taxonomy_features.py`, change the last assertion of `test_features_to_matrix_shape_and_names` to:

```python
    assert X.shape == (2, 7)
    assert names == FEATURE_NAMES
    assert np.isnan(X[:, 5]).all() and np.isnan(X[:, 6]).all()   # no per-modality traces given
```

and add `FEATURE_NAMES` to the import. Append:

```python
def test_per_modality_sqi_values_are_recorded():
    feats = extract_fault_features(
        sqi_trace=np.array([1, 0, 1, 1.0]), accel_trace=None, fs=1, source_db="pulsedb_mimic",
        model_a_pred=70, model_b_pred=72, disagreement_scale=2.0,
        ecg_sqi_trace=np.array([1, 0, 1, 1.0]), ppg_sqi_trace=np.array([1, 1, 1, 0.5]),
    )
    assert feats.ecg_sqi_value == 0.75 and feats.ppg_sqi_value == 0.875
    assert feats.model_disagreement == 1.0


def test_features_to_matrix_column_subset_keeps_requested_order():
    feats = [extract_fault_features(np.ones(3), None, 1, "a", 70, 71, 1.0),
             extract_fault_features(np.zeros(3), None, 1, "b", 70, 75, 1.0)]
    X, names = features_to_matrix(feats, columns=["model_disagreement", "sqi_value"])
    assert names == ["model_disagreement", "sqi_value"]
    assert X.tolist() == [[1.0, 1.0], [5.0, 0.0]]
    with pytest.raises(KeyError):
        features_to_matrix(feats, columns=["not_a_feature"])
```

Add `import pytest` at the top.

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'FEATURE_NAMES'`.

- [ ] **Step 3: Implement**

In `trustbio/taxonomy/features.py`, replace the dataclass, `extract_fault_features` and `features_to_matrix` with:

```python
@dataclass
class SegmentFaultFeatures:
    sqi_value: float
    sqi_drop_duration: float
    accel_corr: float
    source_db: str
    model_disagreement: float
    ecg_sqi_value: float = float("nan")
    ppg_sqi_value: float = float("nan")


FEATURE_NAMES = ["sqi_value", "sqi_drop_duration", "accel_corr", "source_db",
                 "model_disagreement", "ecg_sqi_value", "ppg_sqi_value"]


def extract_fault_features(
    sqi_trace: np.ndarray,
    accel_trace: np.ndarray | None,
    fs: int,
    source_db: str,
    model_a_pred: float,
    model_b_pred: float,
    disagreement_scale: float,
    ecg_sqi_trace: np.ndarray | None = None,
    ppg_sqi_trace: np.ndarray | None = None,
) -> SegmentFaultFeatures:
    """`sqi_trace` is the combined (min over modalities) per-second quality;
    the optional per-modality traces add which channel lost quality -- the
    only thing that separates a flat ECG electrode from a noisy PPG at equal
    span length."""
    return SegmentFaultFeatures(
        sqi_value=float(np.mean(sqi_trace)),
        sqi_drop_duration=float(_longest_low_sqi_run(np.asarray(sqi_trace))),
        accel_corr=_accel_sqi_correlation(np.asarray(sqi_trace), accel_trace),
        source_db=source_db,
        model_disagreement=float(abs(model_a_pred - model_b_pred) / disagreement_scale),
        ecg_sqi_value=float(np.mean(ecg_sqi_trace)) if ecg_sqi_trace is not None else float("nan"),
        ppg_sqi_value=float(np.mean(ppg_sqi_trace)) if ppg_sqi_trace is not None else float("nan"),
    )


def features_to_matrix(
    features: list[SegmentFaultFeatures], columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Numeric matrix over `columns` (default FEATURE_NAMES). `source_db` is
    encoded as an integer category code, ordered by sorted name."""
    names = list(columns) if columns is not None else list(FEATURE_NAMES)
    unknown = [c for c in names if c not in FEATURE_NAMES]
    if unknown:
        raise KeyError(f"unknown feature columns {unknown}; choose from {FEATURE_NAMES}")
    source_code = {s: i for i, s in enumerate(sorted({f.source_db for f in features}))}
    rows = []
    for f in features:
        values = {
            "sqi_value": f.sqi_value, "sqi_drop_duration": f.sqi_drop_duration,
            "accel_corr": f.accel_corr, "source_db": float(source_code[f.source_db]),
            "model_disagreement": f.model_disagreement,
            "ecg_sqi_value": f.ecg_sqi_value, "ppg_sqi_value": f.ppg_sqi_value,
        }
        rows.append([values[c] for c in names])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)), names
```

Update the module docstring's bullet list with two lines: `ecg_sqi_value` / `ppg_sqi_value`: mean per-modality quality, added because lead-off (flat ECG) and motion (noisy PPG) share span lengths at equal severity and differ only in which channel dropped.

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_features.py tests/test_taxonomy_cluster.py tests/test_synthetic_demo.py -v`
Expected: all pass (the cluster tests build their own 5-column matrices and are unaffected; the demo prints cluster sizes from a 7-column matrix with two all-NaN columns — if KMeans rejects NaNs there, change the demo's `features_to_matrix(feats)` call to `features_to_matrix(feats, columns=FEATURE_NAMES[:5])`).

- [ ] **Step 5: Commit**

```bash
git add trustbio/taxonomy/features.py tests/test_taxonomy_features.py scripts/make_synthetic_demo.py
git commit -m "feat: per-modality SQI fault features and column selection for ablations

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: HR probes and model disagreement (`trustbio/taxonomy/disagreement.py`)

**Files:**
- Create: `trustbio/taxonomy/disagreement.py`
- Test: `tests/test_taxonomy_disagreement.py` (create)

**Interfaces:**
- Consumes: `trustbio.eval.probe._fit_one`, `_predict`, `select_hyperparameter`.
- Produces: `@dataclass HRProbe(scaler: StandardScaler, model, alpha: float, n_train: int)`; `fit_hr_probe(X_train, y_train, X_val, y_val, seed=0, max_train=500_000, max_val=100_000) -> HRProbe`; `predict_hr(probe, X) -> np.ndarray`; `disagreement_scale(pred_a, pred_b) -> float` (std of the difference, floored at 1e-6).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_taxonomy_disagreement.py`:

```python
import numpy as np
import pytest

from trustbio.taxonomy.disagreement import HRProbe, disagreement_scale, fit_hr_probe, predict_hr


def _linear(n, d=8, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.normal(0, 3, d)
    y = 70 + X @ w + noise * rng.standard_normal(n)
    return X, y


def test_probe_learns_a_linear_hr_relation():
    Xtr, ytr = _linear(600, seed=0); Xva, yva = _linear(150, seed=1); Xte, yte = _linear(150, seed=2)
    probe = fit_hr_probe(Xtr, ytr, Xva, yva, seed=0)
    assert isinstance(probe, HRProbe) and probe.alpha > 0
    r = np.corrcoef(predict_hr(probe, Xte), yte)[0, 1]
    assert r > 0.9


def test_probe_subsamples_train_and_ignores_nan_labels():
    Xtr, ytr = _linear(500); ytr[:50] = np.nan
    probe = fit_hr_probe(Xtr, ytr, *_linear(100, seed=3), max_train=200)
    assert probe.n_train <= 200


def test_disagreement_scale_is_std_of_difference_with_floor():
    a = np.array([70.0, 72.0, 74.0]); b = np.array([70.0, 70.0, 70.0])
    assert np.isclose(disagreement_scale(a, b), np.std(a - b))
    assert disagreement_scale(a, a) == 1e-6
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_disagreement.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trustbio.taxonomy.disagreement'`.

- [ ] **Step 3: Create `trustbio/taxonomy/disagreement.py`**

```python
"""HR probes for the `model_disagreement` fault feature.

Two linear probes -- one on the best domain FM's features, one on the best
time-series FM's -- are fit on CLEAN PulseDB-MIMIC training features with the
same ridge/alpha-selection protocol as the main evaluation. Their absolute
disagreement on a window, divided by its spread on clean in-distribution
windows, is the taxonomy's structural-shift signal: two models trained on the
same data disagreeing sharply on a window whose SQI looks fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..eval.probe import _fit_one, _predict, select_hyperparameter


@dataclass
class HRProbe:
    scaler: StandardScaler
    model: object
    alpha: float
    n_train: int


def fit_hr_probe(X_train, y_train, X_val, y_val, seed: int = 0,
                 max_train: int = 500_000, max_val: int = 100_000) -> HRProbe:
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train, dtype=float); y_val = np.asarray(y_val, dtype=float)
    tr = np.flatnonzero(np.isfinite(y_train)); va = np.flatnonzero(np.isfinite(y_val))
    if len(tr) > max_train:
        tr = rng.choice(tr, max_train, replace=False)
    if len(va) > max_val:
        va = rng.choice(va, max_val, replace=False)
    scaler = StandardScaler().fit(X_train[tr])
    Xtr, Xva = scaler.transform(X_train[tr]), scaler.transform(X_val[va])
    alpha = select_hyperparameter("regression", Xtr, y_train[tr], Xva, y_val[va], rng,
                                  train_idx=np.arange(len(tr)))
    model = _fit_one("regression", Xtr, y_train[tr], alpha)
    return HRProbe(scaler=scaler, model=model, alpha=float(alpha), n_train=int(len(tr)))


def predict_hr(probe: HRProbe, X) -> np.ndarray:
    return np.asarray(_predict("regression", probe.model, probe.scaler.transform(X)), dtype=float)


def disagreement_scale(pred_a, pred_b) -> float:
    return max(float(np.std(np.asarray(pred_a, float) - np.asarray(pred_b, float))), 1e-6)
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_disagreement.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add trustbio/taxonomy/disagreement.py tests/test_taxonomy_disagreement.py
git commit -m "feat: HR probes and disagreement scale for the fault taxonomy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 10: Assemble the fault-feature table (`scripts/build_fault_features.py`)

**Files:**
- Create: `scripts/build_fault_features.py`
- Test: `tests/test_build_fault_features.py` (create)

**Interfaces:**
- Consumes: Task 2 `visit_rng`/`apply_degradation`; Task 4 cohorts under `features_cache/taxonomy/`; Task 6 store `features_cache/taxonomy_store/<cond>/<dataset>/…`; Task 7 `sqi_trace`, `combined_sqi_trace`, `mean_hf_ratio`, `calibrate_hf_ref`; Task 8 `extract_fault_features`, `features_to_matrix`, `FEATURE_NAMES`; Task 9 `fit_hr_probe`, `predict_hr`, `disagreement_scale`; `build_dataset_handle(name, args, with_labels=False)`; `load_but_ppg_accelerometer`.
- Produces (in `--out-dir`, default `results/taxonomy/`): `fault_features.csv` — one row per (condition, dataset, window) with columns `dataset, visit_id, subject_id, condition, kind, severity, known_condition, pred_a, pred_b, disagreement_raw, model_disagreement, sqi_value, sqi_drop_duration, accel_corr, source_db, ecg_sqi_value, ppg_sqi_value, stratum, pleth_sqi0, ecg_sqi0, quality`; `fault_features.npz` — arrays `X` (fit rows × `FEATURE_NAMES`), `known_conditions`, `feature_names`, `visit_id`, `dataset`, `subject_id` (fit rows = `known_condition ∈ {motion_artifact, lead_off, structural}`); `config.json` — `hf_ref_ecg`, `hf_ref_ppg`, `sqi_auroc_ppg`, `sqi_auroc_ecg`, probe alphas / n_train, `disagreement_scale`, models, seed, row counts. Module functions: `known_condition(dataset, kind, native) -> str`, `degraded_pair(ecg, ecg_fs, ppg, ppg_fs, visit_id, kind, severity, seed, amps) -> (ecg, ppg)`, `load_condition_features(store_root, cond, dataset, model, modality, duration_sec) -> pd.DataFrame`, `raw_signals(handle, duration_sec) -> dict`, `window_rows(...) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_fault_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from scripts.build_fault_features import (
    degraded_pair, known_condition, load_condition_features, raw_signals, window_rows,
)
from trustbio.degradation.inject import make_degraded_loader
from trustbio.pipeline import DatasetHandle, extract_features_for_model
from trustbio.store import FeatureStore
from trustbio.taxonomy.disagreement import fit_hr_probe

AMPS = {0.1: 0.2, 0.3: 0.3, 0.6: 0.4}
FS = 125


def _signal(visit_id, modality):
    rng = np.random.default_rng(abs(hash((visit_id, modality))) % (2**32))
    t = np.arange(10 * FS) / FS
    return (np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32), FS


def _handle(n=12):
    ids = [f"p{i:02d}_w0" for i in range(n)]
    visits = pd.DataFrame({"visit_id": ids, "subject_id": [v.split("_")[0] for v in ids],
                           "split": ["train"] * 8 + ["val"] * 2 + ["test"] * 2})
    splits = {s: visits[visits.split == s][["visit_id"]].reset_index(drop=True) for s in ("train", "val", "test")}
    cohort = type("C", (), {"visits": visits})()
    return DatasetHandle(name="toy", cohort=cohort, splits=splits, load_signal=_signal, label_table={})


def test_degraded_pair_reproduces_what_the_loader_fed_the_model():
    loader = make_degraded_loader(_signal, "lead_off", 0.3, seed=0, noise_amplitudes=AMPS)
    ecg_l, _ = loader("p03_w0", "ecg"); ppg_l, _ = loader("p03_w0", "ppg")
    ecg, efs = _signal("p03_w0", "ecg"); ppg, pfs = _signal("p03_w0", "ppg")
    ecg_d, ppg_d = degraded_pair(ecg, efs, ppg, pfs, "p03_w0", "lead_off", 0.3, seed=0, amps=AMPS)
    assert np.array_equal(ecg_d, ecg_l) and np.array_equal(ppg_d, ppg_l)
    assert degraded_pair(ecg, efs, ppg, pfs, "p03_w0", None, None, 0, AMPS) == (ecg, ppg)


def test_known_condition_mapping():
    assert known_condition("pulsedb_mimic", "motion_artifact", {}) == "motion_artifact"
    assert known_condition("pulsedb_mimic", None, {}) == "clean"
    assert known_condition("pulsedb_vital", None, {}) == "structural"
    assert known_condition("mimic_ext_ppg", None, {"stratum": "ppg_poor"}) == "natural_ppg_poor"
    assert known_condition("but_ppg", None, {"quality": 0}) == "real_motion"
    assert known_condition("but_ppg", None, {"quality": 1}) == "consumer_clean"
    with pytest.raises(ValueError):
        known_condition("nope", None, {})


def test_window_rows_end_to_end_on_a_toy_store(tmp_path):
    handle = _handle()
    root = tmp_path / "store"
    for cond, kind, sev in [("clean", None, None), ("lead_off_0.3", "lead_off", 0.3)]:
        degraded = DatasetHandle(name=handle.name, cohort=handle.cohort, splits=handle.splits,
                                 load_signal=make_degraded_loader(_signal, kind, sev, seed=0, noise_amplitudes=AMPS),
                                 label_table={})
        for model in ("moment-base", "chronos-bolt-small"):
            extract_features_for_model(model, degraded, FeatureStore(root / cond / "toy"), duration_sec=10,
                                       device="cpu", allow_fallback=True, force_fallback=True)
    fa = load_condition_features(root, "clean", "toy", "moment-base", "ecg_ppg_mean", 10)
    fb = load_condition_features(root, "clean", "toy", "chronos-bolt-small", "ecg_ppg_mean", 10)
    assert len(fa) == 12 and fa.index.is_unique
    rng = np.random.default_rng(0); y = rng.normal(70, 10, len(fa))
    pa = fit_hr_probe(fa.to_numpy(), y, fa.to_numpy(), y); pb = fit_hr_probe(fb.to_numpy(), y, fb.to_numpy(), y)
    signals = raw_signals(handle, 10)
    native = handle.cohort.visits.set_index("visit_id").to_dict("index")
    clean = window_rows(signals, "clean", None, None, "pulsedb_mimic", fa, fb, pa, pb, 0.5, 0.5, native, 0, AMPS, 1.0)
    fa_d = load_condition_features(root, "lead_off_0.3", "toy", "moment-base", "ecg_ppg_mean", 10)
    fb_d = load_condition_features(root, "lead_off_0.3", "toy", "chronos-bolt-small", "ecg_ppg_mean", 10)
    lead = window_rows(signals, "lead_off_0.3", "lead_off", 0.3, "pulsedb_mimic", fa_d, fb_d, pa, pb, 0.5, 0.5, native, 0, AMPS, 1.0)
    assert len(clean) == 12 and len(lead) == 12
    c, l = pd.DataFrame(clean), pd.DataFrame(lead)
    assert {"known_condition", "subject_id", "ecg_sqi_value", "ppg_sqi_value", "sqi_drop_duration", "disagreement_raw"} <= set(c.columns)
    assert (c.known_condition == "clean").all() and (l.known_condition == "lead_off").all()
    assert l.ecg_sqi_value.mean() < c.ecg_sqi_value.mean()
    assert np.allclose(l.ppg_sqi_value, c.ppg_sqi_value)          # lead-off never touches PPG
    assert (l.sqi_drop_duration >= 2).all()                         # 30% of 10 s, per-second trace
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_build_fault_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_fault_features'`.

- [ ] **Step 3: Create `scripts/build_fault_features.py`**

```python
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
from trustbio.degradation.calibrate import load_cached_noise_amplitude
from trustbio.degradation.inject import apply_degradation, visit_rng
from trustbio.eval.metrics import auroc
from trustbio.store import FeatureStore
from trustbio.taxonomy.disagreement import disagreement_scale, fit_hr_probe, predict_hr
from trustbio.taxonomy.features import FEATURE_NAMES, SegmentFaultFeatures, extract_fault_features, features_to_matrix
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
    handles, signals, natives, accel = {}, {}, {}, {}
    for ds in DEGRADED_DATASETS + CLEAN_ONLY_DATASETS:
        h = build_dataset_handle(ds, args, with_labels=False)
        handles[ds] = h
        signals[ds] = raw_signals(h, args.duration_sec)
        natives[ds] = h.cohort.visits.assign(visit_id=lambda d: d["visit_id"].astype(str)).set_index("visit_id").to_dict("index")
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
                  rows_per_condition=table.groupby(["condition", "dataset"]).size().to_dict().__repr__())
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[fault-features] wrote {len(table):,} rows ({len(fit):,} in the fit set) to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_build_fault_features.py -v`
Expected: 3 passed. (`test_window_rows_end_to_end_on_a_toy_store` runs the fallback extractor for 2 models × 2 conditions × 12 windows — ~20 s.)

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/build_fault_features.py
git add scripts/build_fault_features.py tests/test_build_fault_features.py
git commit -m "feat: build_fault_features -- SQI traces, probe disagreement and known conditions per window

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---
### Task 11: Clustering with prediction, bootstrap validation, ablations — `run_taxonomy.py` v2

**Files:**
- Modify: `trustbio/taxonomy/cluster.py` (add `FaultClusterer`, `fit_fault_clusters`, `silhouette`, `condition_recall`, `bootstrap_recall`; keep `cluster_fault_segments`, `name_clusters`, `confusion_against_known_conditions`)
- Rewrite: `scripts/run_taxonomy.py`
- Test: `tests/test_taxonomy_cluster.py` (extend), `tests/test_run_taxonomy.py` (create)

**Interfaces:**
- Produces: `@dataclass FaultClusterer(scaler, kmeans, labels_: np.ndarray, names: dict[int, str])` with `.predict(X) -> np.ndarray` (cluster ids) and `.predict_names(X) -> list[str]`; `fit_fault_clusters(X, known_conditions=None, seed=0, n_clusters=3) -> FaultClusterer`; `silhouette(X, labels) -> float`; `condition_recall(assigned_names, known) -> dict[str, float]` (fraction of each known condition's rows assigned to the cluster carrying its name); `bootstrap_recall(assigned_names, known, subjects, n_boot=200, seed=0) -> dict[str, tuple[float, float, float]]` (point, 2.5%, 97.5%; subjects resampled with replacement).
- `run_taxonomy.py --features-csv results/taxonomy/fault_features.csv --out-dir results/taxonomy [--seed 0] [--n-boot 200]` writes `table3_recall.csv` (columns `feature_set, condition, dataset, severity, n, recall, ci_lo, ci_hi`), `confusion_<set>.csv`, `assignments_<set>.csv` (`visit_id, dataset, condition, severity, known_condition, cluster, in_fit`), `summary.json` (`silhouette`, `cluster_names`, `n_fit`, `dropped_nan` per set). Feature sets: `all`, `no_source_db`, `no_disagreement`, `sqi_only`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_taxonomy_cluster.py`:

```python
from trustbio.taxonomy.cluster import (
    FaultClusterer, bootstrap_recall, condition_recall, fit_fault_clusters, silhouette,
)


def test_fit_fault_clusters_names_and_predicts_new_points():
    X, conditions = _synthetic_three_group_matrix()
    fc = fit_fault_clusters(X, conditions, seed=0)
    assert isinstance(fc, FaultClusterer)
    assert set(fc.names.values()) == {"transient", "persistent", "structural"}
    assert len(fc.labels_) == len(X)
    new = np.array([[0.05, 9, 0.1, 0, 0.1], [1.0, 0, 0.0, 1, 3.0]])
    assert fc.predict_names(new) == ["persistent", "structural"]
    assert silhouette(X, fc.labels_) > 0.5


def test_condition_recall_and_subject_bootstrap():
    known = ["a"] * 10 + ["b"] * 10
    assigned = ["a"] * 9 + ["b"] + ["b"] * 8 + ["a"] * 2
    rec = condition_recall(assigned, known)
    assert rec == {"a": 0.9, "b": 0.8}
    subjects = [f"s{i % 5}" for i in range(20)]
    ci = bootstrap_recall(assigned, known, subjects, n_boot=100, seed=0)
    for cond in ("a", "b"):
        point, lo, hi = ci[cond]
        assert lo <= point <= hi and point == rec[cond]
```

Create `tests/test_run_taxonomy.py`:

```python
import json

import numpy as np
import pandas as pd

from scripts.run_taxonomy import FEATURE_SETS, main


def _table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def add(cond, dataset, n, sqi, drop, esqi, psqi, dis, source_code, severity=np.nan, in_fit=True):
        for i in range(n):
            rows.append(dict(dataset=dataset, visit_id=f"{dataset}_{cond}_{i}", subject_id=f"s{i % 7}",
                             condition=cond, kind=cond if cond in ("motion_artifact", "lead_off") else "",
                             severity=severity, known_condition=cond,
                             sqi_value=sqi + rng.normal(0, 0.02), sqi_drop_duration=drop + rng.normal(0, 0.2),
                             accel_corr=0.0, source_db=dataset, model_disagreement=dis + rng.normal(0, 0.1),
                             ecg_sqi_value=esqi + rng.normal(0, 0.02), ppg_sqi_value=psqi + rng.normal(0, 0.02),
                             pred_a=70.0, pred_b=70.0 + dis, disagreement_raw=dis))

    add("motion_artifact", "pulsedb_mimic", n, 0.7, 3, 0.95, 0.5, 0.2, 0, severity=0.3)
    add("lead_off", "pulsedb_mimic", n, 0.6, 6, 0.4, 0.95, 0.2, 0, severity=0.6)
    add("structural", "pulsedb_vital", n, 0.95, 0, 0.95, 0.95, 3.0, 1)
    add("clean", "pulsedb_mimic", 10, 0.95, 0, 0.95, 0.95, 0.2, 0)
    add("real_motion", "but_ppg", 10, 0.65, 3, 0.95, 0.45, 0.4, 2)
    return pd.DataFrame(rows)


def test_run_taxonomy_writes_tables_and_recovers_synthetic_conditions(tmp_path):
    csv = tmp_path / "fault_features.csv"
    _table().to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["--features-csv", str(csv), "--out-dir", str(out), "--n-boot", "20"]) == 0
    t3 = pd.read_csv(out / "table3_recall.csv")
    assert set(t3.feature_set) == set(FEATURE_SETS)
    overall = t3[(t3.feature_set == "all") & (t3.dataset == "all") & (t3.severity == "all")].set_index("condition")
    assert overall.loc[["motion_artifact", "lead_off", "structural"], "recall"].min() > 0.8
    assert (overall.ci_lo <= overall.recall).all() and (overall.recall <= overall.ci_hi).all()
    for s in FEATURE_SETS:
        assert (out / f"confusion_{s}.csv").exists() and (out / f"assignments_{s}.csv").exists()
    asg = pd.read_csv(out / "assignments_all.csv")
    assert set(asg.known_condition) >= {"clean", "real_motion"}         # held-out rows were assigned
    assert (asg[asg.known_condition == "real_motion"]["cluster"] == "motion_artifact").mean() > 0.8
    summary = json.loads((out / "summary.json").read_text())
    assert summary["all"]["silhouette"] > 0.3 and set(summary["all"]["cluster_names"].values()) == {"motion_artifact", "lead_off", "structural"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_cluster.py tests/test_run_taxonomy.py -v`
Expected: FAIL — `ImportError: cannot import name 'FaultClusterer'`; `ImportError: cannot import name 'FEATURE_SETS'`.

- [ ] **Step 3: Extend `trustbio/taxonomy/cluster.py`**

Add `from dataclasses import dataclass` and `from sklearn.metrics import silhouette_score` to the imports; append:

```python
@dataclass
class FaultClusterer:
    scaler: StandardScaler
    kmeans: KMeans
    labels_: np.ndarray
    names: dict[int, str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(np.asarray(X, float)))

    def predict_names(self, X: np.ndarray) -> list[str]:
        return [self.names.get(int(c), f"cluster_{c}") for c in self.predict(X)]


def fit_fault_clusters(X: np.ndarray, known_conditions: list[str] | None = None,
                       seed: int = 0, n_clusters: int = 3) -> FaultClusterer:
    """Standardize, KMeans, and (with known conditions) name the clusters by
    majority vote. Held-out rows -- clean controls, natural degradation, real
    motion -- are assigned later with `.predict_names`, never used to fit."""
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(scaler.transform(X))
    names = name_clusters(labels, list(known_conditions)) if known_conditions is not None else {}
    return FaultClusterer(scaler=scaler, kmeans=km, labels_=labels, names=names)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(StandardScaler().fit_transform(X), labels))


def condition_recall(assigned_names: list[str], known: list[str]) -> dict[str, float]:
    """Fraction of each known condition's rows that landed in the cluster
    carrying that condition's name (the diagonal of the confusion matrix,
    row-normalised)."""
    out = {}
    for cond in sorted(set(known)):
        idx = [i for i, k in enumerate(known) if k == cond]
        out[cond] = float(np.mean([assigned_names[i] == cond for i in idx])) if idx else float("nan")
    return out


def bootstrap_recall(assigned_names: list[str], known: list[str], subjects: list[str],
                     n_boot: int = 200, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Subject-clustered bootstrap CI for condition_recall: windows of one
    subject are not independent, so resample subjects, not windows."""
    rng = np.random.default_rng(seed)
    point = condition_recall(assigned_names, known)
    by_subject: dict[str, list[int]] = {}
    for i, s in enumerate(subjects):
        by_subject.setdefault(str(s), []).append(i)
    keys = list(by_subject)
    draws: dict[str, list[float]] = {c: [] for c in point}
    for _ in range(n_boot):
        idx = [i for s in rng.choice(keys, len(keys), replace=True) for i in by_subject[s]]
        rec = condition_recall([assigned_names[i] for i in idx], [known[i] for i in idx])
        for c in point:
            draws[c].append(rec.get(c, float("nan")))
    return {c: (point[c], float(np.nanpercentile(draws[c], 2.5)), float(np.nanpercentile(draws[c], 97.5)))
            for c in point}
```

- [ ] **Step 4: Rewrite `scripts/run_taxonomy.py`**

```python
#!/usr/bin/env python
"""Stage 4b-ii: fault-taxonomy clustering, validation and assignment (Table 3).

Reads results/taxonomy/fault_features.csv (scripts/build_fault_features.py).
Fits KMeans(k=3) on the rows whose known condition is motion_artifact /
lead_off / structural, names clusters by majority vote, and reports how often
each condition lands in its own cluster -- overall, by dataset, by severity,
with subject-bootstrap CIs -- for four feature sets (the full set and three
ablations, because source_db makes the structural class partly circular).
Every other row (clean controls, MIMIC-ext natural degradation, BUT PPG real
motion / consumer-clean) is assigned to the nearest fitted centroid.

    python scripts/run_taxonomy.py --features-csv results/taxonomy/fault_features.csv --out-dir results/taxonomy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trustbio.taxonomy.cluster import (
    bootstrap_recall, condition_recall, confusion_against_known_conditions, fit_fault_clusters, silhouette,
)
from trustbio.taxonomy.features import FEATURE_NAMES

FIT_CONDITIONS = ("motion_artifact", "lead_off", "structural")
FEATURE_SETS = {
    "all": list(FEATURE_NAMES),
    "no_source_db": [f for f in FEATURE_NAMES if f != "source_db"],
    "no_disagreement": [f for f in FEATURE_NAMES if f != "model_disagreement"],
    "sqi_only": ["sqi_value", "sqi_drop_duration", "ecg_sqi_value", "ppg_sqi_value"],
}


def _matrix(df: pd.DataFrame, columns: list[str], source_codes: dict[str, int]) -> np.ndarray:
    X = df[[c for c in columns]].copy()
    if "source_db" in columns:
        X["source_db"] = df["source_db"].map(source_codes).astype(float)
    return X.to_numpy(dtype=float)


def _recall_rows(feature_set, fit, assigned, n_boot, seed) -> list[dict]:
    rows = []
    known = fit["known_condition"].tolist()
    subj = fit["subject_id"].astype(str).tolist()
    groups = [("all", "all", np.ones(len(fit), bool))]
    groups += [(ds, "all", (fit["dataset"] == ds).to_numpy()) for ds in sorted(fit["dataset"].unique())]
    groups += [("all", str(sv), (fit["severity"] == sv).to_numpy()) for sv in sorted(fit["severity"].dropna().unique())]
    for ds, sev, mask in groups:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        ci = bootstrap_recall([assigned[i] for i in idx], [known[i] for i in idx], [subj[i] for i in idx],
                              n_boot=n_boot, seed=seed)
        for cond, (point, lo, hi) in ci.items():
            n = int(sum(known[i] == cond for i in idx))
            rows.append(dict(feature_set=feature_set, condition=cond, dataset=ds, severity=sev,
                             n=n, recall=point, ci_lo=lo, ci_hi=hi))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.features_csv, dtype={"visit_id": str, "subject_id": str})
    source_codes = {s: i for i, s in enumerate(sorted(table["source_db"].astype(str).unique()))}
    summary, recall_rows = {}, []
    for set_name, columns in FEATURE_SETS.items():
        usable = table.dropna(subset=[c for c in columns if c != "source_db"])
        fit = usable[usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
        rest = usable[~usable["known_condition"].isin(FIT_CONDITIONS)].reset_index(drop=True)
        fc = fit_fault_clusters(_matrix(fit, columns, source_codes), fit["known_condition"].tolist(), seed=args.seed)
        fit_names = [fc.names[int(c)] for c in fc.labels_]
        rest_names = fc.predict_names(_matrix(rest, columns, source_codes)) if len(rest) else []
        confusion = confusion_against_known_conditions(fc.labels_, fit["known_condition"].tolist(), fc.names)
        confusion.to_csv(args.out_dir / f"confusion_{set_name}.csv")
        recall_rows += _recall_rows(set_name, fit, fit_names, args.n_boot, args.seed)
        assignments = pd.concat([
            fit.assign(cluster=fit_names, in_fit=True), rest.assign(cluster=rest_names, in_fit=False),
        ])[["visit_id", "dataset", "condition", "severity", "known_condition", "cluster", "in_fit"]]
        assignments.to_csv(args.out_dir / f"assignments_{set_name}.csv", index=False)
        held_out = (assignments[~assignments.in_fit].groupby(["known_condition", "cluster"]).size()
                    .unstack(fill_value=0))
        summary[set_name] = dict(
            columns=columns, n_fit=int(len(fit)), dropped_nan=int(len(table) - len(usable)),
            silhouette=silhouette(_matrix(fit, columns, source_codes), fc.labels_),
            cluster_names={int(k): v for k, v in fc.names.items()},
            recall=condition_recall(fit_names, fit["known_condition"].tolist()),
            held_out_assignment={k: {kk: int(vv) for kk, vv in row.items()} for k, row in held_out.iterrows()},
        )
        print(f"[taxonomy] {set_name}: names={fc.names} recall={summary[set_name]['recall']} "
              f"silhouette={summary[set_name]['silhouette']:.3f}", flush=True)

    pd.DataFrame(recall_rows).to_csv(args.out_dir / "table3_recall.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[taxonomy] wrote table3_recall.csv, confusion_*.csv, assignments_*.csv, summary.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

The old `--features-npz` entry point is dropped: `fault_features.npz` remains as the compact fit-set artifact but the CSV carries the held-out rows the analysis needs. Update `scripts/run_taxonomy.sbatch` in Task 13.

- [ ] **Step 5: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_taxonomy_cluster.py tests/test_run_taxonomy.py -v`
Expected: 5 + 1 passed. (If `name_clusters` gives a synthetic cluster the fallback name `cluster_N` because two clusters share a majority, the synthetic groups in `_table` are not separable enough — widen the gaps in the test table, do not change `name_clusters`.)

- [ ] **Step 6: Commit**

```bash
git add trustbio/taxonomy/cluster.py scripts/run_taxonomy.py tests/test_taxonomy_cluster.py tests/test_run_taxonomy.py
git commit -m "feat: taxonomy clustering with held-out assignment, subject-bootstrap recall and ablations

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Figure 3 panels (`scripts/plot_taxonomy.py`)

**Files:**
- Create: `scripts/plot_taxonomy.py`
- Test: `tests/test_plot_taxonomy.py` (create)

**Interfaces:**
- `plot_taxonomy.py --features-csv F --assignments-csv A --confusion-csv C --out-dir results/taxonomy/figures [--feature-set all]` writes `fig3a_projection.png` (PCA of the standardized fit-set features, coloured by known condition and by cluster), `fig3b_confusion.png` (row-normalised heatmap), `fig3d_structural_share.png` (share of each held-out group assigned to each cluster: clean, natural_clean, natural_ppg_poor, natural_ecg_poor, real_motion, consumer_clean), `fig3e_severity_recall.png` (recall vs severity for motion and lead-off, from `table3_recall.csv` in the same directory as the features CSV). Panel (c), representative waveforms, is produced by `--waveforms` which needs dataset access and is only run in Task 13's job.
- Module functions: `plot_projection(fit_df, columns, out_png)`, `plot_confusion(confusion_df, out_png)`, `plot_heldout_shares(assignments_df, out_png)`, `plot_severity_recall(table3_df, out_png)`, `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_taxonomy.py`:

```python
import numpy as np
import pandas as pd

from scripts.plot_taxonomy import main


def test_plots_are_written_from_small_synthetic_tables(tmp_path):
    rng = np.random.default_rng(0)
    conds = ["motion_artifact"] * 20 + ["lead_off"] * 20 + ["structural"] * 20 + ["clean"] * 10 + ["real_motion"] * 10
    feats = pd.DataFrame({
        "visit_id": [f"v{i}" for i in range(80)], "dataset": ["pulsedb_mimic"] * 80, "subject_id": "s",
        "known_condition": conds, "condition": conds, "severity": [0.3] * 40 + [np.nan] * 40,
        "sqi_value": rng.random(80), "sqi_drop_duration": rng.integers(0, 10, 80), "accel_corr": 0.0,
        "source_db": "pulsedb_mimic", "model_disagreement": rng.random(80),
        "ecg_sqi_value": rng.random(80), "ppg_sqi_value": rng.random(80),
    })
    asg = feats[["visit_id", "dataset", "condition", "severity", "known_condition"]].copy()
    asg["cluster"] = [c if c in ("motion_artifact", "lead_off", "structural") else "motion_artifact" for c in conds]
    asg["in_fit"] = asg.known_condition.isin(["motion_artifact", "lead_off", "structural"])
    conf = pd.DataFrame([[18, 1, 1], [2, 17, 1], [0, 0, 20]], index=["motion_artifact", "lead_off", "structural"],
                        columns=["motion_artifact", "lead_off", "structural"])
    t3 = pd.DataFrame([dict(feature_set="all", condition=c, dataset="all", severity=s, n=10, recall=r, ci_lo=r - .1, ci_hi=r + .1)
                       for c in ("motion_artifact", "lead_off") for s, r in (("0.1", .5), ("0.3", .7), ("0.6", .9))])
    feats.to_csv(tmp_path / "fault_features.csv", index=False)
    asg.to_csv(tmp_path / "assignments_all.csv", index=False)
    conf.to_csv(tmp_path / "confusion_all.csv")
    t3.to_csv(tmp_path / "table3_recall.csv", index=False)
    out = tmp_path / "figures"
    assert main(["--features-csv", str(tmp_path / "fault_features.csv"), "--assignments-csv", str(tmp_path / "assignments_all.csv"),
                 "--confusion-csv", str(tmp_path / "confusion_all.csv"), "--out-dir", str(out)]) == 0
    for name in ("fig3a_projection.png", "fig3b_confusion.png", "fig3d_structural_share.png", "fig3e_severity_recall.png"):
        assert (out / name).stat().st_size > 1000
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n trust-bio python -m pytest tests/test_plot_taxonomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.plot_taxonomy'`.

- [ ] **Step 3: Create `scripts/plot_taxonomy.py`**

```python
#!/usr/bin/env python
"""Figure 3 panels for the fault taxonomy (a: projection, b: confusion,
d: held-out assignment shares, e: recall vs severity). Panel c (waveform
examples) needs raw-signal access and runs only with --waveforms inside the
analysis job."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from trustbio.taxonomy.features import FEATURE_NAMES

FIT = ("motion_artifact", "lead_off", "structural")
HELD_OUT_ORDER = ["clean", "natural_clean", "natural_ppg_poor", "natural_ecg_poor", "real_motion", "consumer_clean"]
COLORS = {"motion_artifact": "#d95f02", "lead_off": "#7570b3", "structural": "#1b9e77"}


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
    ap.add_argument("--feature-set", default="all", choices=["all", "no_source_db", "no_disagreement", "sqi_only"])
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(args.features_csv, dtype={"visit_id": str})
    asg = pd.read_csv(args.assignments_csv, dtype={"visit_id": str})
    columns = {"all": FEATURE_NAMES, "no_source_db": [f for f in FEATURE_NAMES if f != "source_db"],
               "no_disagreement": [f for f in FEATURE_NAMES if f != "model_disagreement"],
               "sqi_only": ["sqi_value", "sqi_drop_duration", "ecg_sqi_value", "ppg_sqi_value"]}[args.feature_set]
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
```

- [ ] **Step 4: Run the test**

Run: `conda run -n trust-bio python -m pytest tests/test_plot_taxonomy.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/plot_taxonomy.py
git add scripts/plot_taxonomy.py tests/test_plot_taxonomy.py
git commit -m "feat: Figure 3 panels for the fault taxonomy

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Run the analysis chain, record, and push

**Files:**
- Rewrite: `scripts/run_taxonomy.sbatch` (build → cluster → plot)
- Modify: `README.md` (Fault taxonomy section)

- [ ] **Step 1: Rewrite `scripts/run_taxonomy.sbatch`**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=trustbio-taxonomy
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/taxonomy_%j.out
#SBATCH --error=logs/taxonomy_%j.err
#
# Fault taxonomy, end to end: fault-feature table (needs two full-scale train
# matrices -> 64 GB) -> clustering/validation -> figures. Inputs: the taxonomy
# store and cohorts (Tasks 4-6) and the verified full-scale store.
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}/taxonomy"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"
PY=(conda run --no-capture-output -n "${ENV_NAME}" python)
"${PY[@]}" scripts/build_fault_features.py --store features_cache/taxonomy_store \
  --cohort-cache features_cache/taxonomy --full-store features_cache/full \
  --full-cohort-cache features_cache --out-dir "${OUT}"
"${PY[@]}" scripts/run_taxonomy.py --features-csv "${OUT}/fault_features.csv" --out-dir "${OUT}"
"${PY[@]}" scripts/plot_taxonomy.py --features-csv "${OUT}/fault_features.csv" \
  --assignments-csv "${OUT}/assignments_all.csv" --confusion-csv "${OUT}/confusion_all.csv" \
  --out-dir "${OUT}/figures"
```

- [ ] **Step 2: Submit and check**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
JOB=$(sbatch --parsable scripts/run_taxonomy.sbatch | cut -d';' -f1); echo "taxonomy=$JOB"
```

When `sacct -j $JOB -X -n --format=State` says `COMPLETED`:

```bash
grep -h 'probe\|hf_ref\|windows loaded\|wrote\|\[taxonomy\]' logs/taxonomy_${JOB}.out
cat results/taxonomy/config.json
conda run -n trust-bio python -c "
import pandas as pd; t=pd.read_csv('results/taxonomy/table3_recall.csv')
print(t[(t.dataset=='all')&(t.severity=='all')].pivot(index='condition',columns='feature_set',values='recall').round(3))
print(t[(t.feature_set=='all')&(t.dataset!='all')].pivot(index='condition',columns='dataset',values='recall').round(3))
print(t[(t.feature_set=='all')&(t.severity!='all')].pivot(index='condition',columns='severity',values='recall').round(3))"
ls results/taxonomy/figures
```

Expected: `sqi_auroc_ppg` clearly above 0.5 (the computed PPG quality agrees with the native PLETH SQI — if it is ≈ 0.5 the SQI definition does not capture what MIMIC-ext calls poor PPG and the taxonomy's SQI features are not credible; stop and investigate before writing anything up); recall for `lead_off` high at 0.3/0.6 and lower at 0.1 (a 1-s flat span is a small signal); `motion_artifact` recall increasing with severity; `structural` recall high with `source_db` and materially lower without it — report BOTH numbers, the ablation is the honest one; `real_motion` (BUT PPG) landing mostly in the motion cluster is the external-validity check the paper promises, and the natural MIMIC-ext groups' assignment shares are Figure 3d's content.

- [ ] **Step 3: Record and push**

Append to `README.md`:

```markdown
## Fault taxonomy (run 2026-09-xx)

Plan: `docs/superpowers/plans/2026-09-28-fault-taxonomy-analysis.md`. Taxonomy cohorts in `features_cache/taxonomy/`, degraded feature store in `features_cache/taxonomy_store/<condition>/<dataset>/`, results in `results/taxonomy/` (`fault_features.csv`, `table3_recall.csv`, `confusion_*.csv`, `assignments_*.csv`, `summary.json`, `figures/`). Jobs: calibration <id>, cohorts <id>, extraction gpu=<id> cpu=<id>, verify <id>, taxonomy <id>. Rerun: `sbatch scripts/run_taxonomy.sbatch`.
```

(fill in the ids), then:

```bash
git add README.md scripts/run_taxonomy.sbatch
git commit -m "chore: taxonomy run record + end-to-end sbatch chain

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

---

## Outcome (2026-09-29, after Task 13)

Executed end to end; final analysis job 54713469 on a store verified across all seven conditions. Result, honestly: the label-free KMeans(k=3) taxonomy recovers the two synthetic fault kinds almost perfectly when they are severe (severity 0.6: lead-off recall 0.99, motion 1.00; only 54 + 136 cross-assignments out of ~13k) and essentially not at all when mild or moderate (0.1/0.3: recall ≤ 0.02), because with severities anchored to real smartphone-PPG noise quantiles only the 95th-percentile level crosses the quality-zero threshold, and a 1–3 s fault leaves 10-s summaries near clean; the third cluster is "everything mild" and k=3 cannot split it. The `structural` class (clean cross-institution windows) has no SQI signature and only a weak disagreement signal (AUROC 0.62 vs clean MIMIC) — recall 0.0 in every feature set, i.e. not identifiable with these features on this institution pair, consistent with the small transport gap for HR. Real-world held-out groups: BUT PPG real-motion windows land in the motion cluster 14% of the time vs 8% for its clean recordings (its human quality label relates only weakly to signal noise, AUROC 0.57); MIMIC-ext's native poor-PPG/poor-ECG segments are mostly not 1-s-resolution noise bursts or flat lines (any-drop 2–5%). Native-SQI validation of our traces: PPG AUROC 0.65, ECG 0.51. Calibration used: amplitudes 0.18/0.31/1.30 × signal std. The first run's results (HF-residual ECG SQI, unfitted PPG reference) are kept in `results/taxonomy_v1_hfecg/` for the record. Natural next steps, not executed: the draft's own alternative of a supervised classifier on the same features (measures the separability ceiling — e.g. moderate lead-off IS detectable, 100% any-drop, but k=3 cannot allocate it a cluster), and clustering only SQI-detected degraded windows with k chosen by silhouette.

## Self-review notes

- **Spec coverage.** Paper Results §3 / README question 3: features = SQI (per modality, from the exact waveform) + motion (accelerometer correlation, BUT PPG) + model disagreement (Tasks 7–10); clustering label-free with post-hoc naming and confusion-matrix validation (Task 11, existing `cluster.py` design kept); "by fault class and by dataset" (Table 3) plus severity breakdown and subject-bootstrap CIs (Task 11); Figure 3 a/b/d/e (Task 12; panel c is deferred to the job via `--waveforms` and is the one item without a test); synthetic conditions injected into PulseDB AND MIMIC-III-Ext-PPG (Task 6); real degradation via BUT PPG quality labels and MIMIC-ext native SQI, used both for calibrating the SQI and as held-out validation groups (Tasks 4, 10, 11); calibration of motion noise against BUT PPG accelerometry actually run on real data (Task 1). Not covered, deliberately: PPG-DaLiA/WESAD (not in the repo's dataset set) and the degradation *performance* stress test (paper Results §2) — the degraded store from Task 6 is exactly its input, but that is a separate plan.
- **Placeholder scan.** Every code step shows the code; the only "fill in" is the README job-id record in Task 13, which cannot be known in advance.
- **Type consistency.** `window_rows(...)` signature is identical in Task 10's script and test; `FaultClusterer.predict_names`, `condition_recall`, `bootstrap_recall` names match between Task 11's module and CLI; `FEATURE_NAMES` (7) and `features_to_matrix(features, columns)` match Tasks 8, 10, 11, 12; `known_condition` strings match the design table everywhere (`clean`, `structural`, `natural_*`, `real_motion`, `consumer_clean`); manifest condition fields (`cond=`, `kind=`, `sev=`) match Tasks 5 and 6; store path `<store>/<cond>/<dataset>` matches Tasks 5, 6, 10.
- **Known risks called out in-plan:** implausible refit calibration (Task 1 stop rule), sparse MIMIC-ext strata after the lead-II filter (Task 4), skipped windows breaking the visit-id join (Task 6 stop rule), `sqi_auroc_ppg ≈ 0.5` invalidating the SQI (Task 13 stop rule), and the structural class's circularity with `source_db` (ablation reported alongside).
