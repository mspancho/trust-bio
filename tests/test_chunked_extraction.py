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
