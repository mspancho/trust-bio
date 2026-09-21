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
