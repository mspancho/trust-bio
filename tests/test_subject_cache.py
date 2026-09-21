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
