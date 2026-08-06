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

## Handling: exclusion (report in Methods)

`build_but_ppg_cohort(..., drop_malformed_headers=True)` (the default) detects
the signature (`nsamp == 1 and nsig > 1`) and drops those records, printing the
count. Pass `False` to reproduce the unfiltered cohort.

**Cost of the exclusion -- state both numbers, they differ a lot:**

- **48 / 3,888 records excluded (1.2%)**
- but **50 -> 38 subjects (24% of subjects lost)**, because every recording
  belonging to those 12 release-1 subjects had a bad header.

Resulting cohort: **3,840 visits / 38 subjects**; subject-disjoint splits
train 2,688 / val 636 / test 516. HR labels: n=3,840, mean 78.0 bpm,
range 38-161. Native quality labels are imbalanced: **3,045 poor / 795 good
(79% poor)** -- expected, since this is the real-motion-artifact source.

## CSV column names

The real header is `ID,Quality,HR` (title case) **with a UTF-8 BOM**, so the
first column parses as `﻿ID` unless read with `encoding="utf-8-sig"`.
Earlier code looked for `signal_id`/`quality`/`hr` and survived only via
positional fallbacks (`columns[0]`, `.iloc[:, 1]`, `columns[-1]`). Now
normalised case-insensitively by name, with an explicit error if no ID/HR
column is found.
