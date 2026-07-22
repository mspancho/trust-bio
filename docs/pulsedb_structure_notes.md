# PulseDB structure notes (confirmed by direct inspection)

Confirmed by running `scripts/fetch_pulsedb.py --sample-only` and then loading
the real downloaded `.mat` files with `mat73.loadmat` (they are MATLAB v7.3 /
HDF5 format, so `mat73` works directly -- `scipy.io.loadmat` was not needed)
on **2026-07-22**.

**Source**: the official Google Drive mirror linked from
`github.com/pulselabteam/PulseDB`
(https://drive.google.com/drive/folders/10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV),
**not** the Kaggle mirror. The Kaggle listing (`pulselabteam/pulsedb`) does
not exist as the plan originally assumed; a different, third-party Kaggle
dataset exists but is a derived, non-canonical "Supplementary Subset" under
CC-BY-NC-SA-4.0. Since trust-bio is a public MIT-licensed repo, this project
uses the official Google Drive mirror instead, fetched with `gdown`.

## Google Drive folder layout (confirmed by listing 5374 entries)

```
Info_Files/
  PulseDB_Info.mat
  Train_Info.mat
  AAMI_Cal_Info.mat
  AAMI_Test_Info.mat
  CalBased_Test_Info.mat
  CalFree_Test_Info.mat
Segment_Files/
  PulseDB_MIMIC/p######.mat      (one file per subject, e.g. p000160.mat)
  PulseDB_Vital/                 (present in the folder; not directly
                                   inspected in this task -- see "What was
                                   NOT verified" below)
Subset_Files/
Supplementary_Info_Files/
  VitalDB_AAMI_Cal_Info.mat, VitalDB_AAMI_Test_Info.mat,
  VitalDB_CalBased_Test_Info.mat, VitalDB_CalFree_Test_Info.mat,
  VitalDB_Train_Info.mat
LICENSE_PulseDB_MIMIC.txt
LICENSE_PulseDB_Vital.txt
```

## Files actually downloaded and inspected in this task

- `Info_Files/AAMI_Test_Info.mat` (~4.98 MB) -- `PulseDB_Info.mat` itself
  repeatedly hit Google Drive's anonymous per-file download quota (see
  "Download reliability" below) even after retries, so `AAMI_Test_Info.mat`
  was used instead; it is the same `Info_Files/` table convention (one row
  per segment/window, subject-level metadata) and is sufficient to confirm
  the field layout.
- `Segment_Files/PulseDB_MIMIC/p000160.mat` (~9.94 MB, 82 windows)
- `Segment_Files/PulseDB_MIMIC/p000333.mat` (~1.83 MB, 15 windows)

All three are real MATLAB v7.3 (HDF5) files loaded successfully with
`mat73.loadmat`.

## Segment file structure (`Segment_Files/PulseDB_MIMIC/p*.mat`)

**This differs from the brief's assumed layout.** There is no single
`Signals` field with an ECG/PPG/ABP row-index convention. Instead, the
top-level struct (`Subj_Wins`) holds ECG, PPG, and ABP as **separate,
independent fields**, each shaped `(n_windows, 1, 1250)`:

```
top-level key: Subj_Wins
Subj_Wins keys:
  ABP_F, ABP_Lag, ABP_Raw, ABP_SPeaks, ABP_Turns,
  Age, CaseID,
  ECG_F, ECG_RPeaks, ECG_Raw, ECG_Record, ECG_Record_F,
  Gender, IncludeFlag,
  PPG_ABP_Corr, PPG_F, PPG_Raw, PPG_Record, PPG_Record_F, PPG_SPeaks, PPG_Turns,
  SegDBP, SegSBP, SegmentID, SubjectID, T, WinID, WinSeqID
```

- **No row-index convention needed**: ECG, PPG, ABP are keyed by name
  (`ECG_Raw`, `PPG_Raw`, `ABP_Raw`), not by row/index within a shared
  `Signals` array. Task 8's adapter should read these three fields directly
  rather than indexing into a combined array.
- Each of `*_Raw`, `*_F`, `ECG_Record`, `ECG_Record_F`, `PPG_Record`,
  `PPG_Record_F` is shape `(n_windows, 1, 1250)` -- one 1250-sample window
  per row. Confirmed distinct (not aliases of each other) by direct
  `np.allclose` comparison on `p000160.mat` window 0.
  - `*_Raw`: per-window raw signal (values observed in physiologic-looking
    ranges, e.g. ABP ~60-135 mmHg).
  - `*_F`: filtered/processed version of the same window (distinct values
    from `*_Raw`).
  - `ECG_Record` / `ECG_Record_F` (also `PPG_Record` / `PPG_Record_F`):
    present and distinct from `*_Raw`/`*_F`; **exact semantic
    difference (e.g. full-record-relative vs. per-window-relative
    normalization) was not conclusively determined from the data alone in
    this task -- flagged as still-uncertain, not guessed.** Task 8 should
    treat `*_Raw` as the primary raw-signal field unless/until this is
    clarified (e.g. from the PulseDB paper's methods section).
  - `ABP_SPeaks`, `ABP_Turns`, `ECG_RPeaks`, `PPG_SPeaks`, `PPG_Turns`: ragged
    (per-window variable-length) lists of detected peak/turning-point sample
    indices -- e.g. `ECG_RPeaks[0]` for `p000160.mat` window 0 is
    `[29, 154, 280, 406, 531, 655, 781, 907, 1028, 1148, ...]` (R-peak sample
    positions within the 1250-sample window).
- **Sampling rate: 125 Hz, confirmed empirically (not inferred)** via the
  `T` field, which is a real per-window timestamp vector in seconds. For
  `p000333.mat` window 0: `T[0]` runs from `370.008` to `380.0` seconds
  across 1250 samples -- a step of exactly `0.008` s = 1/125 s, and a total
  span of ~9.99 s, consistent with the documented 10-second segment
  convention. 1250 samples / 10 s = 125 Hz.
- **Subject ID**: confirmed as **both** a field (`SubjectID`, e.g.
  `'p000160'`) **and** derivable from the filename stem (`p000160.mat` ->
  subject `160`) -- the two agree in both inspected files. `CaseID` is a
  separate field holding the original MIMIC record identifier (e.g.
  `'2174-11-06-10-12'`), constant across all windows for a given subject
  file -- this is distinct from `SubjectID`/`WinID`/`SegmentID`.
- **Age / Gender**: confirmed present as per-window fields inside
  `Subj_Wins` (e.g. `Age = 50.0`, `Gender = 'F'` for all 82 windows of
  `p000160.mat`, i.e. constant per subject as expected).
- **SBP/DBP**: confirmed present, but as **per-segment** (`SegSBP`,
  `SegDBP`) rather than a single subject-level static value -- one SBP/DBP
  pair per window, derived from that window's ABP waveform (e.g.
  `SegSBP[:5] = [121.0, 133.5, 131.1, 135.1, 135.8]` for `p000160.mat`,
  varying window to window as expected for beat-to-beat blood pressure).
- **Height/Weight: NOT present** in the segment-file struct
  (`Subj_Wins`) -- confirmed absent from the key list above. This
  contradicts the original brief's assumption that Height/Weight live
  alongside Signals/SBP/Age/Gender in the same file. See the Info_Files
  section below for where they actually live.
- `IncludeFlag`: boolean per-window flag (all `True` in the first 10 windows
  inspected) -- likely PulseDB's own QC/inclusion filter; Task 8 should
  probably filter on this before use.
- `PPG_ABP_Corr`: per-window scalar, likely a PPG-ABP correlation/quality
  metric (not further verified).
- `ABP_Lag`: per-window scalar, likely a PPG-to-ABP pulse transit
  time/lag proxy (not further verified).

## Info_Files structure (`Info_Files/AAMI_Test_Info.mat`, used as the
`PulseDB_Info.mat` stand-in)

```
top-level key: AAMI_Test_Subset
AAMI_Test_Subset keys:
  Seg_DBP, Seg_SBP, Source, Subj_Age, Subj_BMI, Subj_Gender,
  Subj_Height, Subj_Name, Subj_SegIDX, Subj_Weight
```

- One row per segment/window across **all** subjects in this AAMI test
  split (1340 rows total in this file), not one row per subject.
- `Subj_Name`: e.g. `'p072634_0'` -- subject id + segment index suffix.
- `Source`: confirmed values `'MIMIC'` and `'VitalDB'` both present in the
  same Info_Files table -- this file mixes both source datasets.
- **Height/Weight/BMI: fields ARE present here**
  (`Subj_Height`, `Subj_Weight`, `Subj_BMI`), confirming the brief's
  expectation -- but **only for VitalDB-sourced rows**. Empirically: of
  1340 rows, exactly 666 have non-NaN Height/Weight/BMI and the other 674
  are NaN; the non-NaN rows are exactly the `Source == 'VitalDB'` rows.
  **MIMIC-III does not systematically record height/weight, so
  MIMIC-sourced rows in PulseDB have these fields as NaN.** Task 8 must
  handle this (e.g. drop or impute Height/Weight/BMI when working with
  PulseDB_MIMIC, or restrict Height/Weight-dependent analyses to the
  PulseDB_Vital subset).
- `Seg_SBP` / `Seg_DBP`: per-segment SBP/DBP, consistent with `SegSBP`/
  `SegDBP` in the segment files.

## What was NOT verified in this task (flagged rather than guessed)

- `Segment_Files/PulseDB_Vital/*.mat` was **not** downloaded or inspected
  directly (only confirmed to exist in the folder listing). The Info_Files
  cross-check above (VitalDB rows carry real Height/Weight) is strong
  indirect evidence that `PulseDB_Vital` segment files likely have the same
  `Subj_Wins` schema as `PulseDB_MIMIC`, since the brief states VitalDB
  signals were downsampled to match MIMIC-III's native rate for exactly this
  kind of compatibility -- but this has not been directly confirmed by
  loading a real `PulseDB_Vital/*.mat` file, and should be treated as
  probable-but-unconfirmed by Task 8.
- The exact semantic difference between `*_Raw`, `*_F`, `*_Record`, and
  `*_Record_F` signal variants (see above) was not conclusively determined
  from data inspection alone.
- `PulseDB_Info.mat` itself (the canonically-named top-level info file) was
  never successfully downloaded in this task (persistent Google Drive
  quota errors across 3 retries) -- `AAMI_Test_Info.mat` was used as a
  same-convention stand-in and is expected, but not proven, to have an
  identical schema to `PulseDB_Info.mat`.

## Download reliability note

Google Drive enforces an anonymous per-file download quota that is shared
across all downloaders of a given public file, independent of tool (gdown,
curl, or browser all hit the same "Quota exceeded" / "Too many users have
viewed or downloaded this file recently" response when tripped). During this
task, `Info_Files/PulseDB_Info.mat`, `Info_Files/AAMI_Cal_Info.mat`,
`Info_Files/CalBased_Test_Info.mat`, `Info_Files/CalFree_Test_Info.mat`,
`Info_Files/Train_Info.mat`, and `Segment_Files/PulseDB_MIMIC/p000188.mat`
all hit this quota and failed even after 3 retries with backoff, while
`Info_Files/AAMI_Test_Info.mat`, `Segment_Files/PulseDB_MIMIC/p000160.mat`,
and `Segment_Files/PulseDB_MIMIC/p000333.mat` succeeded (the latter two only
after one retry). This is transient and file-specific, not a fixed block on
this task's network path -- `scripts/fetch_pulsedb.py` retries with backoff
and reports failed files clearly; a full (non-`--sample-only`) run may need
to be re-attempted over more than one session if quota errors persist for
specific files.
