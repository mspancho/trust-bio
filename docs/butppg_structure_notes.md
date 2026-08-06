# BUT PPG structure notes (confirmed by direct inspection)

Confirmed on **2026-08-06** by fetching the full dataset
(`scripts/fetch_but_ppg.py`, SLURM job 49945258: COMPLETED, 37m37s, peak RSS
88 MB, 31,016 files / 212 MB) and loading it through
`trustbio/data/but_ppg.py`.

Source: PhysioNet, open access (CC-BY 4.0), no credentialing.
<https://physionet.org/content/butppg/2.0.0/>

## On-disk layout (NOT flat)

Each six-digit recording lives in **its own subdirectory**; only the two
annotation CSVs sit at the root:

```
<root>/quality-hr-ann.csv          # ID,Quality,HR   (UTF-8 BOM!)
<root>/subject-info.csv            # ID,Gender,Age,Height,Weight,Ear/finger,
                                   #   Motion,Blood pressure,Glycaemia,SpO2
<root>/100001/100001_PPG.{dat,hea}
<root>/100001/100001_ECG.{dat,hea}
<root>/100001/100001.qrs
<root>/112001/112001_ACC.{dat,hea} # ACC only for IDs >= 112001
```

The adapter originally assumed a FLAT layout (`<root>/100001_PPG`), which
passed its tests only because the synthetic fixtures mirrored that wrong
assumption. Fixed; both fixtures now build the nested tree.

## Confirmed signal parameters

| Modality | Channels | fs | Samples | Duration |
|---|---|---|---|---|
| PPG | 3 (smartphone RGB; adapter uses ch. 0) | 30 Hz | 300 | 10 s |
| ECG | 1 | 1000 Hz | 10000 | 10 s |
| ACC | 3 (triaxial) | 100 Hz | 1000 | 10 s |

Verified uniform across a 200-record sample: 100% `(300, 30, 10000, 1000)`.

## Two release batches -- and an undocumented header defect in release 1

Per the PhysioNet Release Notes: release 1 contained signals **100001-111004**;
release 2 added **112001 onwards** (plus ACC, BP, glycaemia, SpO2).

**48 records from release 1 ship a malformed WFDB header with `nsig` and
`nsamp` TRANSPOSED.** Example -- `100001_PPG.hea` first line:

```
100001_PPG 300 30 1        <- declares 300 signals x 1 sample
```

...followed by 300 signal-spec lines (hence a 15 KB header). But the `.dat` is
600 bytes = exactly 300 int16 samples of ONE signal, so 300 samples is the
truth. Release-2 headers are correct: `112001_PPG 3 30 300`.

**Why this is dangerous:** `wfdb.rdrecord` does not raise on these. It
faithfully honours the header and returns a **single-sample** array, which
would flow silently into features and results. Confirmed: `100001` PPG/ECG
both returned `len=1` before the fix.

PhysioNet publishes **no errata or known-issues note** for v2.0.0 covering
this (checked 2026-08-06), so it appears to be an undocumented defect in the
release-1 portion -- most likely different header-writing tooling between the
two releases.

## Handling: reconstruction (NOT exclusion)

The `.dat` payload of these records is **all zeros** (verified: 48/48 PPG,
38/48 ECG), so nothing is lost by ignoring it -- the waveform is carried
entirely by the per-sample gain/baseline pairs. Applying the standard WFDB
conversion element-wise recovers it:

```
physical[i] = (digital[i] - baseline[i]) / gain[i]
```

**Validated against an independent label.** Record 100001's recovered PPG has a
dominant autocorrelation period of 0.733 s = **81.8 bpm**, against the
**83 bpm** reference in quality-hr-ann.csv -- a 1.4% error. That agreement with
a label we did not use in the reconstruction is what establishes this is the
real signal rather than decoded noise.

An earlier iteration of this adapter EXCLUDED these records. That was rejected:
it cost **24% of subjects (50 -> 38)** to remove **1.2% of records**, because
all recordings of 12 release-1 subjects are affected. Reconstruction keeps the
full **3,888 records / 50 subjects**.

### Reconstruction fidelity is annotated, not assumed

Recovery quality varies. Over the 48 records, PPG-derived HR vs. the reference
annotation gives **median error 3.9%**, but with a long tail (**29/48 within
10%**; mean ~37%). `build_but_ppg_cohort` therefore adds two columns:

| column | meaning |
|---|---|
| `reconstructed` | True for the 48 release-1 records |
| `reconstruction_hr_err_pct` | \|recovered HR - reference HR\| / reference * 100; NaN otherwise |

**Nothing is dropped.** Consumers that depend on faithful morphology should
filter on `reconstruction_hr_err_pct` themselves. This matters most for
`trustbio/degradation/calibrate.py`: BUT PPG's role in this study is calibrating
the synthetic degradation model against REAL motion artifact, so admitting
low-fidelity reconstructions there could contaminate the calibration the whole
degradation analysis rests on -- silently, since nothing would raise.

Resulting cohort: **3,888 visits / 50 subjects**; subject-disjoint splits
train 2,634 / val 602 / test 652. HR labels: n=3,888, mean 78.1 bpm,
range 38-161. Native quality labels remain imbalanced (~79% poor), as expected
for the real-motion-artifact source.

Note also: 10 of the 48 have a **non-zero ECG .dat**, i.e. a third variant. The
reader uses the .dat samples when present and non-zero, and zeros otherwise, so
both variants are handled by one code path.

## CSV column names

The real header is `ID,Quality,HR` (title case) **with a UTF-8 BOM**, so the
first column parses as `﻿ID` unless read with `encoding="utf-8-sig"`.
Earlier code looked for `signal_id`/`quality`/`hr` and survived only via
positional fallbacks (`columns[0]`, `.iloc[:, 1]`, `columns[-1]`). Now
normalised case-insensitively by name, with an explicit error if no ID/HR
column is found.
