# Full-Scale Chunked Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch TRUST-BIO's full-scale feature extraction (7 models × PulseDB-MIMIC + PulseDB-Vital = 5,245,454 windows, ~260 GPU-hours + ~160 CPU-hours) as resumable, chunked SLURM arrays whose merge step can never produce a truncated feature matrix.

**Architecture:** Each (model, dataset) cell is cut into contiguous chunks of every split; one array task per chunk writes `<split>.chunkXXofNN.npz` through the existing atomic `FeatureStore.save`, and skips chunks already on disk so a resubmitted task resumes for free. A merge CLI, gated on every chunk being present, concatenates them into the final `<split>.npz` and verifies row order against the cohort. Extraction stops building label tables (a dedicated job builds them once, in parallel), the PulseDB signal loader's per-subject cache is bounded, and ecg-domain runs on CPU nodes through a second sbatch header that shares one body with the GPU header.

**Tech Stack:** Python 3.11 in conda env `trust-bio` (numpy 1.26.4, pandas, torch 2.4.0+cu121, transformers 4.43.3, neurokit2 0.2.13, mat73), pytest, bash, SLURM on HMS O2 (`gpu_quad`: 5-day limit, no per-user QOS cap; `short`: 12 h limit; `--signal=B:USR1@N` + trap for pre-timeout self-resubmit).

## Global Constraints

- Run every Python command through the pinned env: `conda run -n trust-bio python …`. The lab's `map-env-base` breaks 4 of 7 models.
- Never print the contents of `.env` (PhysioNet credentials, HF token). Jobs load it with `set -a; . ./.env; set +a` and nothing else reads it.
- Store layout is `<store>/<dataset>/<model>/<modality>/<duration>s/<split>.npz`. CLIs take the BASE store root and scope by dataset themselves (`FeatureStore(Path(args.store) / args.dataset)`); a shared root caused the pilot's write collision.
- Every `.npz` written to a store goes through `FeatureStore.save` (tmp + `os.replace`). Never write store files any other way.
- Never merge an incomplete cell; never emit a silently truncated matrix. Refuse loudly instead.
- Shared-cluster courtesy: at most 12 concurrent GPU array tasks on `gpu_quad`; ecg-domain never occupies a GPU node.
- Work on `main` (user preference). Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Repo root: `/n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio`. Run all commands from there. Scratch files go under `/tmp/claude-71839508/-n-data1-hms-dbmi-rajpurkar-lab-home-map9592/07353fa3-edaa-4daa-bf1e-35703cea810f/scratchpad/` (referred to below as `$SCRATCH`).
- **Working-tree state at plan time:** the implementation code of Tasks 1, 2 and 3 is ALREADY APPLIED but uncommitted and untested. For those tasks, verify the file contains the shown code, then write and run the tests, then commit.

### Sizing (pilot cell wall-clock × window ratio: MIMIC ×24.4, Vital ×27.6)

| Cell | Est. whole cell | Chunks | Est. per chunk |
|---|---|---|---|
| xecg-10min / pulsedb_mimic | 45 h | 8 | 5.7 h |
| ecgfounder / pulsedb_mimic | 40 h | 8 | 5.0 h |
| moment-base / pulsedb_mimic | 35 h | 8 | 4.3 h |
| papagei / pulsedb_mimic | 32 h | 8 | 4.0 h |
| chronos-bolt-small / pulsedb_mimic | 23 h | 8 | 2.8 h |
| dbeta / pulsedb_mimic | 15 h | 8 | 1.8 h |
| six GPU models / pulsedb_vital | 7–16 h | 3 | 2.4–5.4 h |
| ecg-domain / pulsedb_mimic (CPU) | 112 h | 20 | 5.6 h |
| ecg-domain / pulsedb_vital (CPU) | 45 h | 8 | 5.7 h |

GPU tasks: 24 h limit (≥4× margin), 40 GB. CPU tasks: 12 h (`short`, ≥2× margin), 16 GB. Both self-resubmit on USR1 and resume by skipping finished chunks. A MIMIC train chunk (2,691,120 / 8 ≈ 336k windows) at dim 1024 holds ≈ 4.1 GB of features in RAM (+1.4 GB transient at stack). Unbounded, the loader cache alone would reach ≈ 76 GB for full MIMIC (2,423 subjects × ~31 MB) — hence Task 1.

Full cohort caches exist: `features_cache/pulsedb_mimic_cohort.csv` (3,791,004 windows: train 2,691,120 / val 577,755 / test 522,129) and `features_cache/pulsedb_vital_cohort.csv` (1,454,450: train 1,019,397 / val 208,844 / test 226,209). Full label caches do NOT exist yet (Task 8 builds them).

---

### Task 1: Bound the PulseDB per-subject cache

**Files:**
- Modify: `trustbio/data/pulsedb.py` (imports; new class before `_load_subject_windows`; `make_pulsedb_signal_loader`; the subject loop in `build_pulsedb_label_table`) — ALREADY APPLIED
- Test: `tests/test_subject_cache.py` (create)

**Interfaces:**
- Produces: `class _BoundedSubjectCache(load_subject: Callable[[str], dict], maxsize: int = 4)` with `.get(subject_id) -> dict` and `__len__`. `make_pulsedb_signal_loader(root, source, file_ext="mat")` signature unchanged.

- [x] **Step 1: Implementation (already applied — verify it reads exactly like this)**

Imports near the top of `trustbio/data/pulsedb.py`:

```python
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
```

Class inserted immediately before `def _load_subject_windows(...)`:

```python
class _BoundedSubjectCache:
    """Most-recently-used cache of decoded subject files, bounded in size.

    Cohorts are ordered subject-by-subject, so consecutive windows hit the same
    file and a handful of entries gives an essentially perfect hit rate. The
    bound is the point: an unbounded dict keeps every subject ever touched --
    about 31 MB each for PulseDB_MIMIC (ECG+PPG as float64), ~76 GB across the
    full 2,423-subject cohort. That fits a 100-subject pilot and OOMs any
    full-scale job partway through its train split.
    """

    def __init__(self, load_subject, maxsize: int = 4):
        self._load_subject = load_subject
        self._maxsize = max(1, int(maxsize))
        self._items: OrderedDict[str, dict] = OrderedDict()

    def get(self, subject_id: str) -> dict:
        if subject_id in self._items:
            self._items.move_to_end(subject_id)
            return self._items[subject_id]
        value = self._load_subject(subject_id)
        self._items[subject_id] = value
        while len(self._items) > self._maxsize:
            self._items.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._items)
```

Body of `make_pulsedb_signal_loader` after the docstring:

```python
    paths = PulseDBPaths(Path(root))

    def _load_subject(subject_id: str) -> dict:
        path = paths.source_dir(source) / f"{subject_id}.{file_ext}"
        if not path.exists():
            raise FileNotFoundError(f"no PulseDB file for subject {subject_id} at {path}")
        return _load_subject_windows(path, file_ext)

    subjects = _BoundedSubjectCache(_load_subject)

    def load(visit_id: str, modality: str):
        subject_id, win_idx = _parse_visit_id(visit_id)
        return subjects.get(subject_id)[modality][win_idx], PULSEDB_FS

    return load
```

Subject loop in `build_pulsedb_label_table` (replaces the old `cache: dict[str, dict] = {}` block):

```python
    paths = PulseDBPaths(Path(root))
    subjects = _BoundedSubjectCache(
        lambda sid: _load_subject_windows(paths.source_dir(source) / f"{sid}.{file_ext}", file_ext)
    )
    rows = []
    for vid in visit_ids:
        subject_id, win_idx = _parse_visit_id(vid)
        windows = subjects.get(subject_id)
```

- [ ] **Step 2: Write the tests**

Create `tests/test_subject_cache.py`:

```python
"""The PulseDB adapters decode one ~287 MB .mat per subject. The per-subject
cache must be BOUNDED: unbounded it reached ~76 GB on the full MIMIC cohort."""
import numpy as np

import trustbio.data.pulsedb as pdb
from trustbio.data.pulsedb import _BoundedSubjectCache


def test_bounded_cache_evicts_least_recently_used():
    loads = []

    def load(sid):
        loads.append(sid)
        return {"sid": sid}

    cache = _BoundedSubjectCache(load, maxsize=2)
    assert cache.get("a")["sid"] == "a"
    cache.get("b")
    cache.get("a")            # refresh a -> b is now least recent
    cache.get("c")            # evicts b
    assert len(cache) == 2
    assert loads == ["a", "b", "c"]
    cache.get("a")            # still cached
    assert loads == ["a", "b", "c"]
    cache.get("b")            # evicted earlier -> reloaded
    assert loads == ["a", "b", "c", "b"]


def test_bounded_cache_never_exceeds_maxsize():
    cache = _BoundedSubjectCache(lambda sid: {}, maxsize=3)
    for i in range(50):
        cache.get(f"s{i}")
        assert len(cache) <= 3


def test_signal_loader_decodes_each_subject_once_for_contiguous_windows(monkeypatch, tmp_path):
    calls = []

    def fake_load(path, file_ext):
        calls.append(path.name)
        n = 5
        return {"ecg": np.zeros((n, 1250)), "ppg": np.ones((n, 1250)),
                "sbp": np.zeros(n), "dbp": np.zeros(n),
                "include_flag": np.ones(n, dtype=bool)}

    monkeypatch.setattr(pdb, "_load_subject_windows", fake_load)
    src_dir = pdb.PulseDBPaths(tmp_path).source_dir("mimic")
    src_dir.mkdir(parents=True)
    (src_dir / "p000001.mat").touch()
    (src_dir / "p000002.mat").touch()

    load = pdb.make_pulsedb_signal_loader(tmp_path, "mimic")
    for w in range(5):
        ecg, fs = load(f"p000001_w{w}", "ecg")
        ppg, _ = load(f"p000001_w{w}", "ppg")
        assert fs == pdb.PULSEDB_FS and ecg.shape == (1250,) and ppg[0] == 1.0
    assert calls == ["p000001.mat"]
    load("p000002_w0", "ecg")
    assert calls == ["p000001.mat", "p000002.mat"]
```

- [ ] **Step 3: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_subject_cache.py -v`
Expected: 3 passed. (If `PulseDBPaths(...).source_dir` or `_parse_visit_id` have different names, read `trustbio/data/pulsedb.py` lines 50–85 and adapt the test — do not change the adapter.)

- [ ] **Step 4: Run the existing adapter tests to confirm nothing regressed**

Run: `conda run -n trust-bio python -m pytest tests/test_pulsedb_adapter.py -q`
Expected: all pass (same count as before the change).

- [ ] **Step 5: Commit**

```bash
git add trustbio/data/pulsedb.py tests/test_subject_cache.py
git commit -m "fix: bound the PulseDB per-subject cache

The signal loader and the label builder kept every decoded subject file
forever (~31 MB each). Fine for the 100-subject pilot, ~76 GB on the full
2,423-subject MIMIC cohort -- every full-scale cell would have OOMed partway
through its train split. Cohorts are ordered by subject, so a 4-entry MRU
cache keeps the hit rate.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: FeatureStore chunk naming, discovery and merge

**Files:**
- Modify: `trustbio/store.py` (add `import re`; add `_CHUNK_RE`, `chunk_split`, `chunk_files`, `merge_chunks` after `exists`) — ALREADY APPLIED
- Test: `tests/test_store_chunks.py` (create)

**Interfaces:**
- Produces:
  - `FeatureStore.chunk_split(split: str, chunk: int, n_chunks: int) -> str` (static), e.g. `"train.chunk03of08"`; raises `ValueError` when out of range.
  - `FeatureStore.chunk_files(model, modality, duration_sec, split) -> tuple[int | None, dict[int, Path]]`.
  - `FeatureStore.merge_chunks(model, modality, duration_sec, split, expected_ids=None, remove=True) -> dict(rows:int, chunks:int, already_merged:bool, skipped:int|None)`; raises `FileNotFoundError` if any chunk is missing (and writes nothing), `ValueError` on duplicate / foreign / out-of-order ids.

- [x] **Step 1: Implementation (already applied — verify)**

`import re` sits between `import os` and `from pathlib import Path`. The following is appended to the class after `exists`:

```python
    # ------------------------------------------------------------------ #
    # Chunked extraction. A full-scale (model, dataset) cell is millions of  #
    # windows and 15-45 GPU-hours (ecg-domain on CPU: ~110 h) -- no single   #
    # wall-clock window survives that, and the extraction loop only saves at #
    # the end of a split. So a split is cut into contiguous chunks, each an  #
    # array task that saves <split>.chunkXXofNN.npz; a resubmitted task      #
    # skips chunks already on disk, and merge_chunks() assembles the final   #
    # <split>.npz only once every chunk is present.                          #
    # ------------------------------------------------------------------ #
    _CHUNK_RE = re.compile(r"^(?P<split>.+)\.chunk(?P<i>\d{2,})of(?P<n>\d{2,})\.npz$")

    @staticmethod
    def chunk_split(split: str, chunk: int, n_chunks: int) -> str:
        """Split name under which chunk `chunk` of `split` is stored, e.g.
        "train.chunk03of08"."""
        if n_chunks < 1 or not (0 <= chunk < n_chunks):
            raise ValueError(f"chunk {chunk} out of range for n_chunks={n_chunks}")
        return f"{split}.chunk{chunk:02d}of{n_chunks:02d}"

    def chunk_files(self, model, modality, duration_sec, split):
        """(n_chunks, {chunk_index: path}) for one split's chunk files;
        (None, {}) when there are none."""
        d = self._path(model, modality, duration_sec, split).parent
        found: dict[int, Path] = {}
        n_chunks = None
        if not d.exists():
            return None, found
        for p in d.iterdir():
            m = self._CHUNK_RE.match(p.name)
            if not m or m.group("split") != split:
                continue
            n = int(m.group("n"))
            if n_chunks is None:
                n_chunks = n
            elif n != n_chunks:
                raise ValueError(f"mixed chunk counts under {d}: {n_chunks} vs {n} ({p.name})")
            found[int(m.group("i"))] = p
        return n_chunks, found

    def merge_chunks(self, model, modality, duration_sec, split,
                     expected_ids=None, remove: bool = True) -> dict:
        """Concatenate a split's chunk files into the final <split>.npz.

        Refuses (FileNotFoundError) unless every chunk 0..n-1 is present, so
        an unfinished cell can never become a quietly truncated matrix. With
        `expected_ids` (the cohort's visit order for this split) the merged ids
        must be a subsequence of it -- same order, no duplicates, nothing
        foreign -- and the count of windows skipped upstream is reported.
        Chunk files are deleted only after the merged file has been re-read
        and its row count verified.
        """
        final = self._path(model, modality, duration_sec, split)
        expected = None if expected_ids is None else np.asarray(expected_ids, dtype=str)
        n_chunks, found = self.chunk_files(model, modality, duration_sec, split)
        if n_chunks is None:
            if final.exists():
                ids, _ = self.load(model, modality, duration_sec, split)
                skipped = None if expected is None else int(len(expected) - len(ids))
                return dict(rows=len(ids), chunks=0, already_merged=True, skipped=skipped)
            raise FileNotFoundError(f"no chunk files and no merged file for {final}")
        missing = sorted(set(range(n_chunks)) - set(found))
        if missing:
            raise FileNotFoundError(
                f"{final.parent}/{split}: {len(missing)} of {n_chunks} chunks missing: {missing}"
            )

        ids_parts, feat_parts = [], []
        for i in range(n_chunks):
            data = np.load(found[i], allow_pickle=False)
            ids_parts.append(np.asarray(data["visit_ids"], dtype=str))
            feat_parts.append(data["features"])
        dims = {f.shape[1] for f in feat_parts if f.ndim == 2}
        if len(dims) > 1:
            raise ValueError(f"inconsistent feature dims across chunks: {sorted(dims)}")
        ids = np.concatenate(ids_parts)
        feats = np.concatenate(feat_parts, axis=0)
        del ids_parts, feat_parts

        skipped = None
        if expected is not None:
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"duplicate visit_ids in merged {split}")
            foreign = np.setdiff1d(ids, expected)
            if len(foreign):
                raise ValueError(
                    f"{len(foreign)} merged visit_ids are not in the cohort split "
                    f"(e.g. {foreign[:3].tolist()})"
                )
            if not np.array_equal(expected[np.isin(expected, ids)], ids):
                raise ValueError(f"merged {split} rows are not in cohort order")
            skipped = int(len(expected) - len(ids))

        self.save(model, modality, duration_sec, split, ids, feats)
        check_ids, check_feats = self.load(model, modality, duration_sec, split)
        if len(check_ids) != len(ids) or check_feats.shape != feats.shape:
            raise RuntimeError(f"verification failed after merging {final}")
        if remove:
            for p in found.values():
                p.unlink()
        return dict(rows=len(ids), chunks=n_chunks, already_merged=False, skipped=skipped)
```

- [ ] **Step 2: Write the tests**

Create `tests/test_store_chunks.py`:

```python
import numpy as np
import pytest

from trustbio.store import FeatureStore


def _chunk(store, i, n, ids, dim=4, split="train"):
    X = np.full((len(ids), dim), float(i), dtype=np.float32)
    store.save("m", "ecg", 10, FeatureStore.chunk_split(split, i, n), ids, X)
    return X


def test_chunk_split_name_and_validation():
    assert FeatureStore.chunk_split("train", 3, 8) == "train.chunk03of08"
    with pytest.raises(ValueError):
        FeatureStore.chunk_split("train", 8, 8)
    with pytest.raises(ValueError):
        FeatureStore.chunk_split("train", 0, 0)


def test_chunk_files_ignores_final_and_tmp_files(tmp_path):
    store = FeatureStore(tmp_path)
    _chunk(store, 0, 2, ["a"])
    _chunk(store, 1, 2, ["b"])
    store.save("m", "ecg", 10, "train", ["z"], np.zeros((1, 4), np.float32))
    (store._path("m", "ecg", 10, "train").parent / "train.tmp-123.npz").write_bytes(b"")
    n, found = store.chunk_files("m", "ecg", 10, "train")
    assert n == 2 and sorted(found) == [0, 1]
    assert store.chunk_files("m", "ecg", 10, "val") == (None, {})


def test_merge_chunks_concatenates_in_order_and_removes_parts(tmp_path):
    store = FeatureStore(tmp_path)
    expected = ["v0", "v1", "v2", "v3", "v4"]
    _chunk(store, 0, 3, ["v0", "v1"])
    _chunk(store, 1, 3, ["v2", "v3"])
    _chunk(store, 2, 3, ["v4"])
    stats = store.merge_chunks("m", "ecg", 10, "train", expected_ids=expected)
    ids, X = store.load("m", "ecg", 10, "train")
    assert ids.tolist() == expected
    assert X[:, 0].tolist() == [0, 0, 1, 1, 2]
    assert stats == dict(rows=5, chunks=3, already_merged=False, skipped=0)
    assert store.chunk_files("m", "ecg", 10, "train") == (None, {})


def test_merge_chunks_reports_upstream_skips(tmp_path):
    store = FeatureStore(tmp_path)
    _chunk(store, 0, 2, ["v0"])
    _chunk(store, 1, 2, ["v3"])            # v1, v2 were skipped by extraction
    stats = store.merge_chunks("m", "ecg", 10, "train", expected_ids=["v0", "v1", "v2", "v3"])
    assert stats["rows"] == 2 and stats["skipped"] == 2


def test_merge_chunks_refuses_when_a_chunk_is_missing(tmp_path):
    store = FeatureStore(tmp_path)
    _chunk(store, 0, 3, ["v0"])
    _chunk(store, 2, 3, ["v2"])
    with pytest.raises(FileNotFoundError, match=r"\[1\]"):
        store.merge_chunks("m", "ecg", 10, "train")
    assert not store.exists("m", "ecg", 10, "train")
    n, found = store.chunk_files("m", "ecg", 10, "train")
    assert n == 3 and sorted(found) == [0, 2]      # nothing was deleted


def test_merge_chunks_rejects_out_of_order_and_foreign_ids(tmp_path):
    store = FeatureStore(tmp_path / "a")
    _chunk(store, 0, 2, ["v1"])
    _chunk(store, 1, 2, ["v0"])
    with pytest.raises(ValueError, match="cohort order"):
        store.merge_chunks("m", "ecg", 10, "train", expected_ids=["v0", "v1"])
    store2 = FeatureStore(tmp_path / "b")
    _chunk(store2, 0, 1, ["zzz"])
    with pytest.raises(ValueError, match="not in the cohort"):
        store2.merge_chunks("m", "ecg", 10, "train", expected_ids=["v0"])


def test_merge_chunks_already_merged_is_idempotent(tmp_path):
    store = FeatureStore(tmp_path)
    store.save("m", "ecg", 10, "train", ["v0", "v1"], np.zeros((2, 4), np.float32))
    stats = store.merge_chunks("m", "ecg", 10, "train", expected_ids=["v0", "v1", "v2"])
    assert stats == dict(rows=2, chunks=0, already_merged=True, skipped=1)
```

- [ ] **Step 3: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_store_chunks.py -v`
Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add trustbio/store.py tests/test_store_chunks.py
git commit -m "feat: chunk naming, discovery and refuse-if-incomplete merge in FeatureStore

Full-scale cells are 15-45 GPU-hours; splits are extracted as contiguous
chunks (<split>.chunkXXofNN.npz) and merged only once every chunk is on
disk, with the merged ids checked against the cohort's own order.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Chunk-aware extraction loop

**Files:**
- Modify: `trustbio/pipeline.py` (new `chunk_bounds`; `extract_features_for_model` gains `chunk`, `n_chunks`; loop uses `store_split`) — ALREADY APPLIED
- Test: `tests/test_chunked_extraction.py` (create)

**Interfaces:**
- Consumes: `FeatureStore.chunk_split`, `FeatureStore.chunk_files`, `FeatureStore.merge_chunks` (Task 2).
- Produces: `chunk_bounds(n: int, chunk: int, n_chunks: int) -> tuple[int, int]`; `extract_features_for_model(..., chunk: int | None = None, n_chunks: int | None = None) -> bool`.

- [x] **Step 1: Implementation (already applied — verify)**

Module-level helper placed before `extract_features_for_model`:

```python
def chunk_bounds(n: int, chunk: int, n_chunks: int) -> tuple[int, int]:
    """[start, stop) of contiguous chunk `chunk` when `n` items are cut into
    `n_chunks` near-equal pieces (the first n % n_chunks pieces get one extra,
    exactly like numpy.array_split). Contiguity matters: cohorts are ordered
    by subject, so a contiguous chunk touches a subset of the subject files
    and the loader's small cache keeps its hit rate."""
    if n_chunks < 1 or not (0 <= chunk < n_chunks):
        raise ValueError(f"bad chunk {chunk} for n_chunks={n_chunks}")
    base, extra = divmod(n, n_chunks)
    start = chunk * base + min(chunk, extra)
    stop = start + base + (1 if chunk < extra else 0)
    return start, stop
```

Signature gains two trailing parameters and a guard right after the docstring:

```python
    chunk: int | None = None,
    n_chunks: int | None = None,
) -> bool:
    """...existing docstring...

    With `chunk`/`n_chunks`, only contiguous chunk `chunk` of every split is
    processed and saved under the chunk split name (see
    FeatureStore.chunk_split); FeatureStore.merge_chunks assembles the final
    per-split files once all chunks exist."""
    if (chunk is None) != (n_chunks is None):
        raise ValueError("chunk and n_chunks must be given together")
```

Loop head:

```python
    for split, df in dataset.splits.items():
        store_split = split
        if chunk is not None:
            # A finished merge supersedes every chunk of this split.
            if (
                not overwrite
                and all(store.exists(model_name, m, duration_sec, split) for m in MODALITIES)
            ):
                continue
            start, stop = chunk_bounds(len(df), chunk, n_chunks)
            df = df.iloc[start:stop]
            store_split = FeatureStore.chunk_split(split, chunk, n_chunks)
        if (
            not overwrite
            and all(store.exists(model_name, m, duration_sec, store_split) for m in MODALITIES)
        ):
            continue
```

The progress message uses `{store_split}` and `store.save(model_name, m, duration_sec, store_split, kept_ids, ...)`.

- [ ] **Step 2: Write the tests**

Create `tests/test_chunked_extraction.py`:

```python
import numpy as np
import pandas as pd
import pytest

from trustbio.config import MODALITIES
from trustbio.pipeline import DatasetHandle, chunk_bounds, extract_features_for_model
from trustbio.store import FeatureStore


def _deterministic_handle(n_visits=23):
    """Like tests/test_pipeline.py's toy handle, but each visit's signal is a
    pure function of its id, so chunked and unchunked runs see identical data.
    Returns (handle, list_of_load_calls)."""
    visit_ids = [f"v{i}" for i in range(n_visits)]
    splits = {
        "train": pd.DataFrame({"visit_id": visit_ids[:15]}),
        "val": pd.DataFrame({"visit_id": visit_ids[15:19]}),
        "test": pd.DataFrame({"visit_id": visit_ids[19:]}),
    }
    calls = []

    def load_signal(visit_id, modality):
        calls.append((visit_id, modality))
        rng = np.random.default_rng(int(visit_id[1:]) * 2 + (modality == "ppg"))
        return rng.standard_normal(2500).astype(np.float32), 250

    handle = DatasetHandle(name="toy", cohort=None, splits=splits,
                           load_signal=load_signal, label_table={})
    return handle, calls


def _extract(handle, store, **kw):
    # force_fallback: the random projection is seeded from the model name, so
    # two extractor instances produce identical features for identical input.
    return extract_features_for_model(
        "moment-base", handle, store, duration_sec=10, device="cpu",
        allow_fallback=True, force_fallback=True, **kw,
    )


@pytest.mark.parametrize("n,k", [(0, 1), (1, 3), (7, 3), (23, 3), (100, 8)])
def test_chunk_bounds_matches_numpy_array_split(n, k):
    parts = np.array_split(np.arange(n), k)
    for c in range(k):
        start, stop = chunk_bounds(n, c, k)
        assert list(range(start, stop)) == parts[c].tolist()


def test_chunk_bounds_rejects_bad_indices():
    with pytest.raises(ValueError):
        chunk_bounds(10, 3, 3)
    with pytest.raises(ValueError):
        chunk_bounds(10, 0, 0)


def test_chunk_and_n_chunks_must_travel_together(tmp_path):
    handle, _ = _deterministic_handle()
    with pytest.raises(ValueError):
        _extract(handle, FeatureStore(tmp_path), chunk=0)


def test_chunked_extraction_merges_to_the_unchunked_result(tmp_path):
    handle, _ = _deterministic_handle()
    ref = FeatureStore(tmp_path / "ref")
    _extract(handle, ref)
    chunked = FeatureStore(tmp_path / "chunked")
    for c in range(3):
        _extract(handle, chunked, chunk=c, n_chunks=3)
    for split, df in handle.splits.items():
        for m in MODALITIES:
            n, found = chunked.chunk_files("moment-base", m, 10, split)
            assert n == 3 and sorted(found) == [0, 1, 2]
            stats = chunked.merge_chunks("moment-base", m, 10, split,
                                         expected_ids=df["visit_id"].tolist())
            assert stats["skipped"] == 0
            ids_ref, X_ref = ref.load("moment-base", m, 10, split)
            ids_m, X_m = chunked.load("moment-base", m, 10, split)
            assert ids_m.tolist() == ids_ref.tolist() == df["visit_id"].tolist()
            np.testing.assert_allclose(X_m, X_ref)


def test_finished_chunks_and_merged_splits_are_skipped_on_rerun(tmp_path):
    handle, calls = _deterministic_handle()
    store = FeatureStore(tmp_path)
    _extract(handle, store, chunk=1, n_chunks=3)
    n_calls = len(calls)
    assert n_calls > 0
    _extract(handle, store, chunk=1, n_chunks=3)          # chunk files present -> no work
    assert len(calls) == n_calls
    for split, df in handle.splits.items():
        for m in MODALITIES:                                # fake a completed merge
            store.save("moment-base", m, 10, split, df["visit_id"].tolist(),
                       np.zeros((len(df), 768), np.float32))
    _extract(handle, store, chunk=0, n_chunks=3)          # merged file supersedes chunks
    assert len(calls) == n_calls
```

- [ ] **Step 3: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_chunked_extraction.py -v`
Expected: 9 passed (5 parametrized + 4).

- [ ] **Step 4: Run the whole suite**

Run: `conda run -n trust-bio python -m pytest tests/ -q`
Expected: everything passes (68 before this plan + 3 + 7 + 9 new).

- [ ] **Step 5: Commit**

```bash
git add trustbio/pipeline.py tests/test_chunked_extraction.py
git commit -m "feat: chunked, resumable extraction in extract_features_for_model

--chunk i --n-chunks N processes contiguous chunk i of every split and saves
it under FeatureStore.chunk_split(); finished chunks (or a finished merge)
are skipped, so a resubmitted array task resumes where the last one died.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Extraction stops building labels; CLI grows `--chunk/--n-chunks`

**Files:**
- Modify: `scripts/_dataset_builders.py` (`_empty_labels` helper after `DATASET_CHOICES`; `build_dataset_handle(..., with_labels: bool = True)`; every branch gates its label build)
- Modify: `scripts/extract_features.py` (two new args; `with_labels=False`; pass `chunk`/`n_chunks`)
- Test: `tests/test_dataset_builders.py` (create)

**Interfaces:**
- Produces: `build_dataset_handle(name, args, degrade_kind=None, degrade_severity=None, seed=0, with_labels=True) -> DatasetHandle`; `_empty_labels(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]`.
- CLI: `python scripts/extract_features.py --model M --dataset D --duration-sec S --device DEV --store ROOT [--cohort-cache DIR] [--chunk I --n-chunks N] [--overwrite]`.

Why: `build_dataset_handle` derives `hr_regression` from every window's ECG (~4.5 ms/window measured), so without a label cache each of the 94 array tasks would spend hours building labels it never reads. Full label caches don't exist yet; building them is Task 8's separate job.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_builders.py`:

```python
import argparse

import pandas as pd
import pytest

from scripts._dataset_builders import _empty_labels, build_dataset_handle


def _fake_cohort_csv(cache_dir, source="mimic"):
    df = pd.DataFrame({
        "visit_id": ["p000001_w0", "p000001_w1", "p000002_w0"],
        "subject_id": ["p000001", "p000001", "p000002"],
        "source": [source] * 3,
        "split": ["train", "train", "test"],
    })
    df.to_csv(cache_dir / f"pulsedb_{source}_cohort.csv", index=False)
    return df


def test_empty_labels_keep_visit_index_and_have_no_columns():
    splits = {"train": pd.DataFrame({"visit_id": ["a", "b"]}),
              "test": pd.DataFrame({"visit_id": ["c"]})}
    labels = _empty_labels(splits)
    assert labels["train"].index.tolist() == ["a", "b"]
    assert labels["train"].shape == (2, 0)
    assert labels["test"].index.name == "visit_id"
    with pytest.raises(KeyError):
        labels["train"]["hr_regression"]


def test_build_dataset_handle_without_labels_never_opens_subject_files(tmp_path):
    _fake_cohort_csv(tmp_path)
    args = argparse.Namespace(pulsedb_root=tmp_path / "no-such-root", cohort_cache=tmp_path)
    handle = build_dataset_handle("pulsedb_mimic", args, with_labels=False)
    assert handle.name == "pulsedb_mimic"
    assert handle.splits["train"]["visit_id"].tolist() == ["p000001_w0", "p000001_w1"]
    assert handle.splits["test"]["visit_id"].tolist() == ["p000002_w0"]
    assert handle.label_table["train"].shape == (2, 0)
    # Labels are derived from each window's ECG, so asking for them with no
    # subject files on disk must fail -- which proves the flag is what skipped it.
    with pytest.raises(Exception):
        build_dataset_handle("pulsedb_mimic", args, with_labels=True)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_dataset_builders.py -v`
Expected: FAIL — `ImportError: cannot import name '_empty_labels'`. If instead it fails with `ModuleNotFoundError: No module named 'scripts'`, add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

(check first whether a `[tool.pytest.ini_options]` table already exists and extend it rather than duplicating).

- [ ] **Step 3: Implement `_empty_labels` and `with_labels` in `scripts/_dataset_builders.py`**

After the `DATASET_CHOICES = [...]` line add:

```python


def _empty_labels(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Label tables with the right index and no columns, for stages that never
    read labels (extraction). Any later attempt to use them fails loudly with a
    KeyError instead of silently training on nothing."""
    return {
        s: pd.DataFrame(index=pd.Index(df["visit_id"].astype(str), name="visit_id"))
        for s, df in splits.items()
    }
```

Change the signature and docstring:

```python
def build_dataset_handle(
    name: str, args, degrade_kind: str | None = None,
    degrade_severity: float | None = None, seed: int = 0,
    with_labels: bool = True,
) -> DatasetHandle:
    """Build a DatasetHandle for one of DATASET_CHOICES from parsed CLI args
    (which must include --pulsedb-root/--mimic-ext-ppg-root/--but-ppg-root, via
    add_dataset_root_args).

    `with_labels=False` skips the label tables entirely. Extraction never reads
    them, and PulseDB's hr_regression is derived from every window's ECG (~4.5
    ms/window measured), so building them inside each of ~90 array tasks would
    cost hours per task for nothing."""
```

In the `pulsedb_mimic` branch replace the label block (from the `# Build ONCE...` comment through the `labels = {...}` assignment) with:

```python
        if with_labels:
            # Build ONCE across all splits and slice, rather than three separate
            # passes: hr_regression is derived from ECG, so each call reopens the
            # subject .mat files (~16 min for the 100-subject pilot).
            _all_ids = [v for df in splits.values() for v in df["visit_id"].tolist()]
            _labels = build_pulsedb_label_table(
                args.pulsedb_root, "mimic", _all_ids,
                cache=getattr(args, "cohort_cache", None),
            )
            labels = {s: _labels.reindex(df["visit_id"].tolist())
                      for s, df in splits.items()}
        else:
            labels = _empty_labels(splits)
```

Do the same in the `pulsedb_vital` branch with `"vital"`. In the `mimic_ext_ppg` branch:

```python
        if with_labels:
            meta = pd.read_csv(args.mimic_ext_ppg_root / "metadata.csv")
            labels = {s: build_mimic_ext_ppg_label_table(meta, df["visit_id"].tolist())
                      for s, df in splits.items()}
        else:
            labels = _empty_labels(splits)
```

(`metadata.csv` is 4.92 GB — reading it for extraction was pure waste.) In the `but_ppg` branch:

```python
        if with_labels:
            qhr = pd.read_csv(args.but_ppg_root / "quality-hr-ann.csv")
            labels = {s: build_but_ppg_label_table(qhr, df["visit_id"].tolist())
                      for s, df in splits.items()}
        else:
            labels = _empty_labels(splits)
```

- [ ] **Step 4: Update `scripts/extract_features.py`**

After the `--duration-sec` argument add:

```python
    ap.add_argument("--chunk", type=int, default=None,
                    help="process only contiguous chunk I (0-based) of every split; "
                         "saved as <split>.chunkIIofNN.npz, merged by "
                         "scripts/merge_feature_chunks.py")
    ap.add_argument("--n-chunks", type=int, default=None,
                    help="number of chunks each split is cut into (with --chunk)")
```

Replace the handle construction:

```python
    # Extraction only needs splits and signals. Labels (hr_regression derived
    # from every window's ECG) are built once by scripts/build_pulsedb_labels.py
    # and read by the eval stages; building them here would cost each array
    # task hours before its first feature.
    dataset = build_dataset_handle(
        args.dataset, args, degrade_kind=args.degrade_kind,
        degrade_severity=args.degrade_severity, seed=args.seed,
        with_labels=False,
    )
    print(f"{args.dataset} cohort: {dataset.cohort.counts}")
```

And pass the chunk arguments through:

```python
    extract_features_for_model(
        args.model, dataset, FeatureStore(Path(args.store) / args.dataset),
        duration_sec=args.duration_sec, device=args.device,
        allow_fallback=args.allow_fallback, checkpoint=args.checkpoint,
        overwrite=args.overwrite, chunk=args.chunk, n_chunks=args.n_chunks,
    )
```

- [ ] **Step 5: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_dataset_builders.py tests/test_pipeline.py -v`
Expected: all pass. If `test_build_dataset_handle_without_labels_never_opens_subject_files` fails inside `build_pulsedb_cohort` because the cache-read path touches `pulsedb_root`, read that function (`trustbio/data/pulsedb.py`, the block starting `cache_file = cohort_cache_path(...)`) and, if it only needs the root to exist, use `pulsedb_root=tmp_path` in the test instead — do NOT weaken the `with_labels=True` assertion.

- [ ] **Step 6: Commit**

```bash
git add scripts/_dataset_builders.py scripts/extract_features.py tests/test_dataset_builders.py pyproject.toml
git commit -m "feat: extraction skips label building; --chunk/--n-chunks on the CLI

build_dataset_handle(with_labels=False) returns index-only label frames.
hr_regression is derived from every window's ECG (~4.5 ms/window), so with
no full-scale label cache each of ~90 array tasks would have spent hours
on labels it never reads. Labels get their own job (build_pulsedb_labels).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(Drop `pyproject.toml` from `git add` if Step 2 did not need it.)

---

### Task 5: Manifest generator emits chunk lines

**Files:**
- Modify: `scripts/make_manifest.py` (whole file)
- Test: `tests/test_make_manifest.py` (create)

**Interfaces:**
- Produces: `parse_chunks(specs: list[str]) -> dict[str, int]` (raises `ValueError`), `build_lines(models, datasets, duration, chunks) -> list[str]`. Manifest line formats: `model dataset duration` or `model dataset duration chunk n_chunks`.
- CLI: `python scripts/make_manifest.py --duration 10 --out FILE [--models M ...] [--datasets D ...] [--chunks DATASET=N ...]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_make_manifest.py`:

```python
import pytest

from scripts.make_manifest import build_lines, parse_chunks


def test_parse_chunks_accepts_dataset_eq_n():
    assert parse_chunks(["pulsedb_mimic=8", "pulsedb_vital=3"]) == {
        "pulsedb_mimic": 8, "pulsedb_vital": 3}
    assert parse_chunks([]) == {}


@pytest.mark.parametrize("bad", ["pulsedb_mimic", "pulsedb_mimic=0", "nope=3", "pulsedb_mimic=x"])
def test_parse_chunks_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_chunks([bad])


def test_build_lines_unchunked_and_chunked():
    lines = build_lines(["m1", "m2"], ["pulsedb_mimic", "pulsedb_vital"], 10, {"pulsedb_mimic": 2})
    assert lines == [
        "m1 pulsedb_mimic 10 0 2", "m1 pulsedb_mimic 10 1 2", "m1 pulsedb_vital 10",
        "m2 pulsedb_mimic 10 0 2", "m2 pulsedb_mimic 10 1 2", "m2 pulsedb_vital 10",
    ]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_make_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_lines'`.

- [ ] **Step 3: Rewrite `scripts/make_manifest.py`**

```python
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

from trustbio.config import MAIN_TEST_MODELS, is_model_available

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


def build_lines(models: list[str], datasets: list[str], duration: int,
                chunks: dict[str, int]) -> list[str]:
    lines = []
    for model in models:
        for dataset in datasets:
            n = chunks.get(dataset)
            if n is None:
                lines.append(f"{model} {dataset} {duration}")
            else:
                lines.extend(f"{model} {dataset} {duration} {c} {n}" for c in range(n))
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
    args = ap.parse_args()
    chunks = parse_chunks(args.chunks)

    models = []
    for model in (args.models or MAIN_TEST_MODELS):
        if not is_model_available(model):
            print(f"[manifest] {model}: unavailable, excluding from manifest")
            continue
        models.append(model)

    lines = build_lines(models, args.datasets, args.duration, chunks)
    Path(args.out).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} cells ({len(models)} models x {len(args.datasets)} datasets) to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_make_manifest.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_manifest.py tests/test_make_manifest.py
git commit -m "feat: make_manifest emits chunked cell lines (--chunks DATASET=N)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Merge CLI

**Files:**
- Create: `scripts/merge_feature_chunks.py`
- Test: `tests/test_merge_cli.py` (create)

**Interfaces:**
- Consumes: `FeatureStore.chunk_files`, `FeatureStore.merge_chunks` (Task 2); `build_dataset_handle(..., with_labels=False)` (Task 4); `DatasetHandle` from `trustbio.pipeline`.
- Produces: `merge_cell(store_root: Path, dataset: DatasetHandle, model: str, duration_sec: int, keep_chunks: bool = False) -> int` (0 = merged, 1 = refused).
- CLI: `python scripts/merge_feature_chunks.py --store ROOT --dataset D --model M --duration-sec S [--cohort-cache DIR] [--keep-chunks]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_merge_cli.py`:

```python
import numpy as np
import pandas as pd

from scripts.merge_feature_chunks import merge_cell
from trustbio.config import MODALITIES
from trustbio.pipeline import DatasetHandle, extract_features_for_model
from trustbio.store import FeatureStore


def _handle(n_visits=13):
    visit_ids = [f"v{i}" for i in range(n_visits)]
    splits = {"train": pd.DataFrame({"visit_id": visit_ids[:8]}),
              "val": pd.DataFrame({"visit_id": visit_ids[8:10]}),
              "test": pd.DataFrame({"visit_id": visit_ids[10:]})}

    def load_signal(visit_id, modality):
        rng = np.random.default_rng(int(visit_id[1:]) * 2 + (modality == "ppg"))
        return rng.standard_normal(2500).astype(np.float32), 250

    return DatasetHandle(name="toy", cohort=None, splits=splits,
                         load_signal=load_signal, label_table={})


def _extract_all_chunks(handle, store_root, n_chunks=2):
    store = FeatureStore(store_root / handle.name)
    for c in range(n_chunks):
        extract_features_for_model("moment-base", handle, store, duration_sec=10, device="cpu",
                                   allow_fallback=True, force_fallback=True,
                                   chunk=c, n_chunks=n_chunks)
    return store


def test_merge_cell_merges_every_modality_and_split(tmp_path, capsys):
    handle = _handle()
    store = _extract_all_chunks(handle, tmp_path)
    assert merge_cell(tmp_path, handle, "moment-base", 10) == 0
    for split, df in handle.splits.items():
        for m in MODALITIES:
            ids, X = store.load("moment-base", m, 10, split)
            assert ids.tolist() == df["visit_id"].tolist()
            assert store.chunk_files("moment-base", m, 10, split) == (None, {})
    out = capsys.readouterr().out
    assert "[merge] moment-base/toy/ecg/train: rows=8 from 2 chunks, skipped upstream 0" in out


def test_merge_cell_refuses_whole_cell_when_any_chunk_is_missing(tmp_path, capsys):
    handle = _handle()
    store = _extract_all_chunks(handle, tmp_path)
    _, found = store.chunk_files("moment-base", "ppg", 10, "val")
    found[1].unlink()                               # lose one chunk of one modality/split
    assert merge_cell(tmp_path, handle, "moment-base", 10) == 1
    for split in handle.splits:
        for m in MODALITIES:
            assert not store.exists("moment-base", m, 10, split)     # nothing partial written
    assert "REFUSED" in capsys.readouterr().out


def test_merge_cell_is_idempotent(tmp_path):
    handle = _handle()
    _extract_all_chunks(handle, tmp_path)
    assert merge_cell(tmp_path, handle, "moment-base", 10) == 0
    assert merge_cell(tmp_path, handle, "moment-base", 10) == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_merge_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.merge_feature_chunks'`.

- [ ] **Step 3: Create `scripts/merge_feature_chunks.py`**

```python
#!/usr/bin/env python
"""Stage 3b CLI: merge one cell's chunk files into final per-split matrices.

Chunked extraction (extract_features.py --chunk/--n-chunks) leaves
<store>/<dataset>/<model>/<modality>/<dur>s/<split>.chunkXXofNN.npz files.
This concatenates them into <split>.npz -- refusing, for the WHOLE cell, if
any chunk of any modality/split is missing, so an unfinished extraction can
never be merged into a quietly truncated matrix. The merged visit_ids are
checked against the cohort's own order (no duplicates, nothing foreign, same
sequence) and the number of windows skipped upstream is reported.

    python scripts/merge_feature_chunks.py --store features_cache/full \
        --dataset pulsedb_mimic --model xecg-10min --duration-sec 10 \
        --cohort-cache features_cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trustbio.config import MODALITIES
from trustbio.pipeline import DatasetHandle
from trustbio.store import FeatureStore

if __package__:
    from ._dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle
else:
    from _dataset_builders import DATASET_CHOICES, add_dataset_root_args, build_dataset_handle


def merge_cell(store_root: Path, dataset: DatasetHandle, model: str,
               duration_sec: int, keep_chunks: bool = False) -> int:
    """Merge every (modality, split) of one (model, dataset) cell. Returns 0
    when the cell is fully merged, 1 when it was refused as incomplete."""
    store = FeatureStore(Path(store_root) / dataset.name)

    # Refuse-before-write: check every (modality, split) first, so a cell is
    # merged entirely or not at all.
    problems = []
    for split in dataset.splits:
        for m in MODALITIES:
            if store.exists(model, m, duration_sec, split):
                continue
            n, found = store.chunk_files(model, m, duration_sec, split)
            if n is None:
                problems.append(f"{m}/{split}: no chunk files and no merged file")
                continue
            missing = sorted(set(range(n)) - set(found))
            if missing:
                problems.append(f"{m}/{split}: missing chunks {missing} of {n}")
    if problems:
        print(f"[merge] REFUSED {model}/{dataset.name}: cell incomplete", flush=True)
        for p in problems:
            print(f"    {p}", flush=True)
        return 1

    for split, df in dataset.splits.items():
        expected = df["visit_id"].astype(str).tolist()
        for m in MODALITIES:
            stats = store.merge_chunks(model, m, duration_sec, split,
                                       expected_ids=expected, remove=not keep_chunks)
            msg = f"[merge] {model}/{dataset.name}/{m}/{split}: rows={stats['rows']:,}"
            msg += " (already merged)" if stats["already_merged"] else f" from {stats['chunks']} chunks"
            frac = stats["skipped"] / max(len(expected), 1)
            msg += f", skipped upstream {stats['skipped']:,} ({frac:.2%})"
            if frac > 0.01:
                msg = "WARNING: " + msg
            print(msg, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    add_dataset_root_args(ap)
    ap.add_argument("--store", required=True, help="BASE store root (features live under <store>/<dataset>/)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration-sec", type=int, default=600)
    ap.add_argument("--keep-chunks", action="store_true", help="leave chunk files in place after merging")
    args = ap.parse_args()

    dataset = build_dataset_handle(args.dataset, args, with_labels=False)
    return merge_cell(Path(args.store), dataset, args.model, args.duration_sec, keep_chunks=args.keep_chunks)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n trust-bio python -m pytest tests/test_merge_cli.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/merge_feature_chunks.py
git add scripts/merge_feature_chunks.py tests/test_merge_cli.py
git commit -m "feat: merge_feature_chunks CLI -- whole-cell refuse-before-write

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: SLURM wrappers — shared body, GPU and CPU headers, merge array

**Files:**
- Create: `scripts/_extract_cell.sh` (shared body, sourced by both headers)
- Modify: `scripts/extract_features.sbatch` (GPU header only; body moves to `_extract_cell.sh`)
- Create: `scripts/extract_features_cpu.sbatch` (CPU header for ecg-domain)
- Create: `scripts/merge_feature_chunks.sbatch`
- Test: `tests/test_extract_sbatch.py` (create) — drives the scripts with `bash` and `TRUSTBIO_DRY_RUN=1`

**Interfaces:**
- Consumes: manifest line formats from Task 5; `extract_features.py` flags from Task 4; `merge_feature_chunks.py` from Task 6.
- Environment contract (all optional): `TRUSTBIO_REPO`, `TRUSTBIO_ENV` (default `trust-bio`), `TRUSTBIO_STORE` (default `<repo>/features_cache`), `TRUSTBIO_COHORT_CACHE`, `TRUSTBIO_DURATION`, `TRUSTBIO_OVERWRITE=1`, `TRUSTBIO_DEVICE` (default `cuda`; the CPU header sets `cpu`), `TRUSTBIO_DRY_RUN=1` (print the command, exit 0). Positional arg 1 (or `TRUSTBIO_MANIFEST`): manifest path.
- Behaviour: `--signal=B:USR1@600` + trap resubmits the same array index (`sbatch --array=$i <header> <manifest>`) 10 min before the wall limit and exits 0; a resubmitted task skips finished chunks (Task 3). Non-zero python exit → no resubmit (deterministic failures must not loop).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extract_sbatch.py`:

```python
"""Drive the sbatch scripts with plain bash in dry-run mode. bash ignores the
#SBATCH header lines, so this exercises exactly the argument plumbing the
cluster will run -- without a scheduler."""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST_TEXT = "moment-base pulsedb_vital 10 2 5\nxecg-10min pulsedb_mimic 10\n"


def _run(script, manifest, task_id, extra_env=None):
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(task_id), TRUSTBIO_DRY_RUN="1",
               TRUSTBIO_REPO=str(REPO), TRUSTBIO_STORE="/s", TRUSTBIO_COHORT_CACHE="/c")
    env.update(extra_env or {})
    return subprocess.run(["bash", str(REPO / script), str(manifest)], env=env, cwd=REPO,
                          capture_output=True, text=True)


@pytest.mark.parametrize("script,expect_device", [
    ("scripts/extract_features.sbatch", "--device cuda"),
    ("scripts/extract_features_cpu.sbatch", "--device cpu"),
])
def test_dry_run_builds_chunked_command(tmp_path, script, expect_device):
    manifest = tmp_path / "m.txt"
    manifest.write_text(MANIFEST_TEXT)
    r = _run(script, manifest, 0)
    assert r.returncode == 0, r.stderr
    assert "--model moment-base --dataset pulsedb_vital --duration-sec 10" in r.stdout
    assert "--chunk 2 --n-chunks 5" in r.stdout
    assert "--store /s" in r.stdout and "--cohort-cache /c" in r.stdout
    assert expect_device in r.stdout


def test_dry_run_unchunked_line_has_no_chunk_args(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text(MANIFEST_TEXT)
    r = _run("scripts/extract_features.sbatch", manifest, 1)
    assert r.returncode == 0, r.stderr
    assert "--model xecg-10min --dataset pulsedb_mimic" in r.stdout
    assert "--chunk" not in r.stdout


def test_domain_model_is_forced_to_cpu_even_on_gpu_header(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("ecg-domain pulsedb_vital 10 0 8\n")
    r = _run("scripts/extract_features.sbatch", manifest, 0)
    assert r.returncode == 0 and "--device cpu" in r.stdout


def test_missing_manifest_line_exits_2(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("moment-base pulsedb_vital 10\n")
    r = _run("scripts/extract_features.sbatch", manifest, 7)
    assert r.returncode == 2


def test_merge_sbatch_dry_run(tmp_path):
    manifest = tmp_path / "cells.txt"
    manifest.write_text("papagei pulsedb_mimic 10\n")
    r = _run("scripts/merge_feature_chunks.sbatch", manifest, 0)
    assert r.returncode == 0, r.stderr
    assert "scripts/merge_feature_chunks.py" in r.stdout
    assert "--model papagei --dataset pulsedb_mimic --duration-sec 10 --store /s --cohort-cache /c" in r.stdout
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n trust-bio python -m pytest tests/test_extract_sbatch.py -v`
Expected: FAIL (the GPU script has no dry-run mode; the CPU and merge scripts don't exist).

- [ ] **Step 3: Create `scripts/_extract_cell.sh`**

```bash
#!/usr/bin/env bash
# Shared body for scripts/extract_features.sbatch (GPU) and
# scripts/extract_features_cpu.sbatch (CPU). The header that sources this
# must set SBATCH_SCRIPT to its own path (used to resubmit itself).
#
# Manifest line: `model dataset duration [chunk n_chunks]` (see make_manifest.py).
# Env (all optional): TRUSTBIO_REPO TRUSTBIO_ENV TRUSTBIO_STORE
#   TRUSTBIO_COHORT_CACHE TRUSTBIO_DURATION TRUSTBIO_OVERWRITE=1
#   TRUSTBIO_DEVICE (default cuda) TRUSTBIO_DRY_RUN=1 (print command, exit 0)
set -uo pipefail

MANIFEST="${1:-${TRUSTBIO_MANIFEST:-manifest_extract.txt}}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
module load gcc/14.2.0 cuda/12.8 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
# HF_HOME and the gated-model token live in .env; never echo them.
[[ -f .env ]] && { set -a; . ./.env; set +a; }

i="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((i+1))p" "${MANIFEST}")
read -r MODEL DATASET DURATION CHUNK NCHUNKS <<<"${LINE}"
if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "[extract] no manifest line $((i+1)) in ${MANIFEST}" >&2
  exit 2
fi
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"

CHUNK_ARGS=()
[[ -n "${CHUNK:-}" ]] && CHUNK_ARGS=(--chunk "${CHUNK}" --n-chunks "${NCHUNKS}")
OVERWRITE_ARG=()
[[ "${TRUSTBIO_OVERWRITE:-0}" == "1" ]] && OVERWRITE_ARG=(--overwrite)
# ecg-domain is neurokit2 feature extraction -- pure CPU, no GPU kernels; it
# never gets "cuda" even if submitted through the GPU header by mistake.
DEVICE="${TRUSTBIO_DEVICE:-cuda}"
[[ "${MODEL}" == *domain* ]] && DEVICE="cpu"
CACHE_ARG=()
[[ -n "${TRUSTBIO_COHORT_CACHE:-}" ]] && CACHE_ARG=(--cohort-cache "${TRUSTBIO_COHORT_CACHE}")

echo "[extract] task ${i}: model=${MODEL} dataset=${DATASET} duration=${DURATION}s chunk=${CHUNK:-all}/${NCHUNKS:-1} device=${DEVICE} store=${STORE} host=$(hostname)"

CMD=(conda run -n "${ENV_NAME}" python scripts/extract_features.py
     --model "${MODEL}" --dataset "${DATASET}" --duration-sec "${DURATION}"
     --device "${DEVICE}" --store "${STORE}"
     "${CHUNK_ARGS[@]}" "${CACHE_ARG[@]}" "${OVERWRITE_ARG[@]}")
echo "[extract] cmd: ${CMD[*]}"
if [[ "${TRUSTBIO_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# slurm SIGKILLs on TIMEOUT, so an exit-code check alone never fires; the
# header asks for USR1 ten minutes early. Resubmit the same array index and
# exit while the shell is still alive -- finished chunks are skipped on resume.
resubmit() {
  echo "[extract] wall limit approaching; resubmitting task ${i} to resume"
  sbatch --array="${i}" "${SBATCH_SCRIPT}" "${MANIFEST}"
}
trap 'resubmit; exit 0' USR1

"${CMD[@]}" &
wait $!
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "[extract] FAILED rc=${rc}: ${MODEL} / ${DATASET} chunk=${CHUNK:-all} (not resubmitting a deterministic failure)"
  exit $rc
fi
echo "[extract] done: ${MODEL} / ${DATASET} @ ${DURATION}s chunk=${CHUNK:-all}/${NCHUNKS:-1}"
```

- [ ] **Step 4: Rewrite `scripts/extract_features.sbatch` as the GPU header**

Replace the entire file with:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=trustbio-extract
#SBATCH --partition=gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#
# GPU header. One array task per manifest line `model dataset duration
# [chunk n_chunks]`; the body lives in scripts/_extract_cell.sh.
#   sbatch --array=0-$((N-1))%12 scripts/extract_features.sbatch manifest.txt
# ecg-domain (CPU-only) belongs on scripts/extract_features_cpu.sbatch.
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
SBATCH_SCRIPT="${REPO_DIR}/scripts/extract_features.sbatch"
source "${REPO_DIR}/scripts/_extract_cell.sh" "$@"
```

- [ ] **Step 5: Create `scripts/extract_features_cpu.sbatch`**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=trustbio-extract-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#
# CPU header for ecg-domain (neurokit2; 56% of total per-window cost, no GPU
# kernels). Same manifest format and body as extract_features.sbatch.
#   sbatch --array=0-$((N-1))%14 scripts/extract_features_cpu.sbatch manifest_cpu.txt
export TRUSTBIO_DEVICE=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
SBATCH_SCRIPT="${REPO_DIR}/scripts/extract_features_cpu.sbatch"
source "${REPO_DIR}/scripts/_extract_cell.sh" "$@"
```

- [ ] **Step 6: Create `scripts/merge_feature_chunks.sbatch`**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=trustbio-merge
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/merge_%A_%a.out
#SBATCH --error=logs/merge_%A_%a.err
#
# One array task per `model dataset duration` line: merges that cell's chunk
# files into final per-split matrices, refusing incomplete cells. 64 GB
# because an xecg/ecgfounder MIMIC train split is ~11 GB of float32 and the
# merge holds the parts plus their concatenation. Submit with
#   --dependency=afterany:<gpu array id>:<cpu array id>
# and rerun for any cell it REFUSED once that cell's missing chunks exist.
set -uo pipefail
MANIFEST="${1:-${TRUSTBIO_MANIFEST:-manifest_cells.txt}}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
[[ -f .env ]] && { set -a; . ./.env; set +a; }

i="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((i+1))p" "${MANIFEST}")
read -r MODEL DATASET DURATION _ <<<"${LINE}"
if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "[merge] no manifest line $((i+1)) in ${MANIFEST}" >&2
  exit 2
fi
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"
CACHE_ARG=()
[[ -n "${TRUSTBIO_COHORT_CACHE:-}" ]] && CACHE_ARG=(--cohort-cache "${TRUSTBIO_COHORT_CACHE}")

CMD=(conda run --no-capture-output -n "${ENV_NAME}" python scripts/merge_feature_chunks.py
     --model "${MODEL}" --dataset "${DATASET}" --duration-sec "${DURATION}"
     --store "${STORE}" "${CACHE_ARG[@]}")
echo "[merge] task ${i}: ${CMD[*]}"
[[ "${TRUSTBIO_DRY_RUN:-0}" == "1" ]] && exit 0
"${CMD[@]}"
```

- [ ] **Step 7: Make them executable and run the tests**

```bash
chmod +x scripts/_extract_cell.sh scripts/extract_features.sbatch scripts/extract_features_cpu.sbatch scripts/merge_feature_chunks.sbatch
bash -n scripts/_extract_cell.sh scripts/extract_features.sbatch scripts/extract_features_cpu.sbatch scripts/merge_feature_chunks.sbatch
conda run -n trust-bio python -m pytest tests/test_extract_sbatch.py -v
```

Expected: `bash -n` silent; 6 passed.

- [ ] **Step 8: Commit**

```bash
git add scripts/_extract_cell.sh scripts/extract_features.sbatch scripts/extract_features_cpu.sbatch scripts/merge_feature_chunks.sbatch tests/test_extract_sbatch.py
git commit -m "feat: chunk-aware sbatch wrappers with USR1 self-resubmit; CPU header for ecg-domain

Shared body (_extract_cell.sh) parses optional chunk fields, sources .env for
HF_HOME/token, and resubmits its own array index ten minutes before the wall
limit; finished chunks are skipped on resume. ecg-domain runs on the short
CPU partition instead of holding a GPU for a CPU workload.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Build the full-scale label caches (parallel job, not a gate)

**Files:**
- Create: `scripts/build_pulsedb_labels.py`
- Test: `tests/test_build_pulsedb_labels.py` (create)

**Interfaces:**
- Consumes: `build_pulsedb_cohort(root, source, cache=dir)`, `build_pulsedb_label_table(root, source, visit_ids, cache=dir, rebuild=False)` from `trustbio.data.pulsedb` (bounded cache from Task 1 makes the full build fit in 16 GB).
- Produces: `features_cache/pulsedb_mimic_labels.csv`, `features_cache/pulsedb_vital_labels.csv` (columns `visit_id,hr_regression,sbp_regression,dbp_regression`), read later by `run_transport_eval.py`/`run_benchmark.py` via `--cohort-cache features_cache`.
- CLI: `python scripts/build_pulsedb_labels.py --store DIR [--root ROOT] [--source mimic|vital|both] [--rebuild]`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_pulsedb_labels.py`:

```python
import numpy as np
import pandas as pd

import scripts.build_pulsedb_labels as bl
from trustbio.data.cohort import Cohort


def test_main_builds_labels_for_each_source_into_the_store(monkeypatch, tmp_path, capsys):
    seen = []

    def fake_cohort(root, source, cache=None, rebuild=False):
        ids = [f"p1_w{i}" for i in range(4)]
        return Cohort(visits=pd.DataFrame({"visit_id": ids, "subject_id": "p1",
                                           "source": source, "split": "train"}))

    def fake_labels(root, source, visit_ids, cache=None, rebuild=False):
        seen.append((source, list(visit_ids), cache, rebuild))
        return pd.DataFrame({"hr_regression": [80.0, 82.0, np.nan, 78.0],
                             "sbp_regression": [120.0] * 4, "dbp_regression": [80.0] * 4},
                            index=pd.Index(visit_ids, name="visit_id"))

    monkeypatch.setattr(bl, "build_pulsedb_cohort", fake_cohort)
    monkeypatch.setattr(bl, "build_pulsedb_label_table", fake_labels)
    assert bl.main(["--store", str(tmp_path), "--source", "both"]) == 0
    assert [s[0] for s in seen] == ["mimic", "vital"]
    assert all(s[2] == tmp_path and s[1] == [f"p1_w{i}" for i in range(4)] for s in seen)
    out = capsys.readouterr().out
    assert "median hr 80.0" in out and "hr NaN 1" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `conda run -n trust-bio python -m pytest tests/test_build_pulsedb_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_pulsedb_labels'`.

- [ ] **Step 3: Create `scripts/build_pulsedb_labels.py`**

```python
#!/usr/bin/env python
"""Stage 2b: build and cache the PulseDB label tables ONCE.

hr_regression is derived from each window's ECG, so this opens every subject
file: ~4.5 ms/window measured on the pilot, i.e. ~4.7 h for the full 3.79M-
window MIMIC cohort and ~2 h for Vital. Every eval stage then reads the cached
CSV in seconds. Extraction does not need labels (extract_features.py builds
its handle with with_labels=False), so this runs alongside the extraction
arrays rather than gating them.

    python scripts/build_pulsedb_labels.py --store features_cache --source mimic
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from trustbio.data.pulsedb import (
    DEFAULT_ROOT, build_pulsedb_cohort, build_pulsedb_label_table, label_cache_path,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--store", type=Path, required=True,
                    help="directory holding the cohort CSVs; label CSVs are written beside them")
    ap.add_argument("--source", choices=["mimic", "vital", "both"], default="both")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    for src in (["mimic", "vital"] if args.source == "both" else [args.source]):
        cohort = build_pulsedb_cohort(args.root, source=src, cache=args.store)
        ids = cohort.visits["visit_id"].astype(str).tolist()
        t0 = time.time()
        table = build_pulsedb_label_table(args.root, src, ids, cache=args.store, rebuild=args.rebuild)
        hr = table["hr_regression"].to_numpy(dtype=float)
        print(f"\n=== pulsedb_{src} labels ===")
        print(f"  windows      {len(table):,} (cohort {len(ids):,})")
        print(f"  built in     {(time.time() - t0) / 60:.1f} min")
        print(f"  median hr {np.nanmedian(hr):.1f} bpm, hr NaN {int(np.isnan(hr).sum()):,}")
        print(f"  median sbp {np.nanmedian(table['sbp_regression']):.1f} / "
              f"dbp {np.nanmedian(table['dbp_regression']):.1f} mmHg")
        print(f"  cache        {label_cache_path(args.store, src)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test**

Run: `conda run -n trust-bio python -m pytest tests/test_build_pulsedb_labels.py -v`
Expected: 1 passed. (If `DEFAULT_ROOT` or `label_cache_path` import fails, check their names in `trustbio/data/pulsedb.py` — both exist at plan time.)

- [ ] **Step 5: Submit the two label-build jobs**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
for SRC in mimic vital; do
  sbatch --job-name=tb-labels-${SRC} --partition=short --cpus-per-task=2 --mem=16G \
    --time=11:30:00 --output=logs/labels_${SRC}_%j.out --error=logs/labels_${SRC}_%j.err \
    --wrap="cd $PWD && module load conda/miniforge3/24.11.3-0 2>/dev/null; conda run --no-capture-output -n trust-bio python scripts/build_pulsedb_labels.py --store features_cache --source ${SRC}"
done
squeue -u $USER -h -o '%i %j %T' | grep tb-labels
```

Expected: two job ids; both `PENDING`/`RUNNING`. Acceptance (hours later): `features_cache/pulsedb_mimic_labels.csv` with 3,791,004 rows and `pulsedb_vital_labels.csv` with 1,454,450 rows; log medians near the pilot's 87 bpm (MIMIC) / 78 bpm (Vital).

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/build_pulsedb_labels.py
git add scripts/build_pulsedb_labels.py tests/test_build_pulsedb_labels.py
git commit -m "feat: build_pulsedb_labels -- one-off full-scale label cache build

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Real-data smoke test, manifests, launch, record

**Files:**
- Create: `manifest_full_gpu.txt`, `manifest_full_cpu.txt`, `manifest_full_cells.txt`
- Modify: `README.md` (append a "Full-scale extraction" section with the exact commands and the job ids)

**Interfaces:**
- Consumes: everything above. Store root for the full run: `features_cache/full` (features land in `features_cache/full/pulsedb_{mimic,vital}/<model>/...`). Cohort cache: `features_cache` (holds the full cohort CSVs; label CSVs arrive from Task 8's jobs).

- [ ] **Step 1: Smoke-test the chunked CLI on real PulseDB data (pilot cohort, tiny chunk)**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
SMOKE=$SCRATCH/smoke_store
rm -rf "$SMOKE"
( set -a; . ./.env; set +a; conda run --no-capture-output -n trust-bio python scripts/extract_features.py \
    --model papagei --dataset pulsedb_vital --duration-sec 10 --device cpu \
    --store "$SMOKE" --cohort-cache features_cache/pilot100 --chunk 0 --n-chunks 200 )
find "$SMOKE" -name '*.npz' | sort
```

Expected: three lines like `[extract] papagei/pulsedb_vital/train.chunk00of200: kept 154/154 windows`, `.../val.chunk00of200: kept 62/62`, `.../test.chunk00of200: kept 50/50` (30,622/200 → 154 for the first chunk; 12,232/200 → 62; 9,877/200 → 50), and 9 files named `<split>.chunk00of200.npz` under `$SMOKE/pulsedb_vital/papagei/{ecg,ppg,ecg_ppg_mean}/10s/`. Runtime: a few minutes on the login node. Then confirm the merge refuses:

```bash
( set -a; . ./.env; set +a; conda run --no-capture-output -n trust-bio python scripts/merge_feature_chunks.py \
    --model papagei --dataset pulsedb_vital --duration-sec 10 --store "$SMOKE" --cohort-cache features_cache/pilot100; echo "rc=$?" )
```

Expected: `[merge] REFUSED papagei/pulsedb_vital: cell incomplete`, nine `missing chunks [1, 2, ...] of 200` lines, `rc=1`, and no `train.npz` anywhere under `$SMOKE`.

- [ ] **Step 2: Build the three manifests**

`.env` must be loaded because `dbeta` is an HF-gated model and `is_model_available` checks for the token; the subshell keeps the variables out of the session and nothing prints them.

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
( set -a; . ./.env; set +a
  conda run -n trust-bio python scripts/make_manifest.py --duration 10 --out manifest_full_gpu.txt \
    --models moment-base chronos-bolt-small dbeta ecgfounder xecg-10min papagei \
    --datasets pulsedb_mimic pulsedb_vital --chunks pulsedb_mimic=8 pulsedb_vital=3
  conda run -n trust-bio python scripts/make_manifest.py --duration 10 --out manifest_full_cpu.txt \
    --models ecg-domain --datasets pulsedb_mimic pulsedb_vital --chunks pulsedb_mimic=20 pulsedb_vital=8
  conda run -n trust-bio python scripts/make_manifest.py --duration 10 --out manifest_full_cells.txt \
    --models moment-base chronos-bolt-small dbeta ecgfounder xecg-10min papagei ecg-domain \
    --datasets pulsedb_mimic pulsedb_vital )
wc -l manifest_full_gpu.txt manifest_full_cpu.txt manifest_full_cells.txt
grep -c dbeta manifest_full_gpu.txt
```

Expected: `66 manifest_full_gpu.txt`, `28 manifest_full_cpu.txt`, `14 manifest_full_cells.txt`; `11` dbeta lines (8 mimic + 3 vital). If any model prints `unavailable, excluding`, STOP — the environment is missing weights/token and the manifest is wrong.

- [ ] **Step 3: Launch**

```bash
cd /n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
export TRUSTBIO_STORE="$PWD/features_cache/full"
export TRUSTBIO_COHORT_CACHE="$PWD/features_cache"
mkdir -p "$TRUSTBIO_STORE" logs
GPU=$(sbatch --parsable --array=0-65%12 scripts/extract_features.sbatch manifest_full_gpu.txt | cut -d';' -f1)
CPU=$(sbatch --parsable --array=0-27%14 scripts/extract_features_cpu.sbatch manifest_full_cpu.txt | cut -d';' -f1)
MERGE=$(sbatch --parsable --array=0-13 --dependency=afterany:${GPU}:${CPU} scripts/merge_feature_chunks.sbatch manifest_full_cells.txt | cut -d';' -f1)
echo "gpu=${GPU} cpu=${CPU} merge=${MERGE}"
squeue -u $USER -h -o '%i %j %T %M' | grep trustbio | head
```

Expected: three job ids; `squeue` shows up to 12 `trustbio-extract` and 14 `trustbio-extract-cpu` tasks `RUNNING`/`PENDING`, and the merge array `PENDING (Dependency)`.

- [ ] **Step 4: First health check (≈20 minutes after tasks start)**

```bash
grep -h '\[extract\] task\|kept\|WARNING\|FAILED' logs/extract_${GPU}_*.out logs/extract_${CPU}_*.out 2>/dev/null | head -40
tail -q -n 3 logs/extract_${GPU}_*.err 2>/dev/null | grep -iv 'warn\|^$' | head
```

Expected: each running task printed its `[extract] task N: ...` line with the right chunk; no `FAILED`, no tracebacks. (Progress lines only appear per finished split because `conda run` buffers stdout; a quiet log is not a stalled task — check `sacct -j $GPU --format=JobID,State,Elapsed` instead.)

- [ ] **Step 5: Record the run in `README.md` and commit everything**

Append to `README.md`:

```markdown
## Full-scale extraction (launched 2026-09-21)

Chunked, resumable SLURM arrays; see `docs/superpowers/plans/2026-09-21-full-scale-chunked-extraction.md`.

- Store: `features_cache/full/<dataset>/<model>/<modality>/10s/<split>.npz` (chunks: `<split>.chunkXXofNN.npz` until merged)
- Manifests: `manifest_full_gpu.txt` (66 GPU chunk tasks), `manifest_full_cpu.txt` (28 ecg-domain CPU tasks), `manifest_full_cells.txt` (14 cells for the merge)
- Jobs: gpu=<GPU id> cpu=<CPU id> merge=<MERGE id> (merge waits on both arrays; it REFUSES any cell with missing chunks -- redo those chunks with `sbatch --array=<idx> scripts/extract_features[_cpu].sbatch <manifest>` then rerun the merge task)
- Labels: `scripts/build_pulsedb_labels.py --store features_cache` (jobs tb-labels-mimic / tb-labels-vital)
- Watch: `squeue -u $USER | grep trustbio`; `grep -h kept logs/extract_<id>_*.out`
```

Then:

```bash
git add README.md manifest_full_gpu.txt manifest_full_cpu.txt manifest_full_cells.txt docs/superpowers/plans/2026-09-21-full-scale-chunked-extraction.md
git commit -m "chore: full-scale extraction manifests + run record

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin main
```

Expected: push succeeds to `github.com/mspancho/trust-bio` `main`.

---

## Self-review notes

- **Spec coverage:** bounded loader memory (T1); chunk naming/merge safety (T2); resumable chunked loop (T3); extraction independent of labels + CLI flags (T4); manifest chunk lines (T5); merge CLI refuse-before-write (T6); GPU/CPU headers, USR1 self-resubmit, dry-run tests (T7); label cache built in parallel (T8); real-data smoke, manifests, launch with courtesy limits, record + push (T9). Post-launch verification of the merged full-scale store and the eval stages are deliberately out of scope for this plan — they need the ~2-day run to finish first.
- **Type consistency:** `FeatureStore.chunk_split/chunk_files/merge_chunks` names and return shapes match between T2, T3, T6; `build_dataset_handle(..., with_labels=...)` matches T4/T6/T8; manifest line format matches T5/T7; env var names match T7/T9.
- **Known risk to watch in T4 Step 5:** whether `build_pulsedb_cohort`'s cache-read path needs `pulsedb_root` to exist; the step says how to adapt without weakening the test.
