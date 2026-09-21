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
