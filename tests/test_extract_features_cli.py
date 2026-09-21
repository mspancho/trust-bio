"""extract_features.py exit codes: 0 when the cell ran (or the model is
legitimately unavailable per the registry), 3 when an available model failed
to load -- an array task must not report success with nothing extracted."""
from types import SimpleNamespace

import scripts.extract_features as ef

ARGS = ["--model", "papagei", "--dataset", "pulsedb_vital", "--duration-sec", "10",
        "--device", "cpu", "--store", "/nonexistent-store", "--chunk", "0", "--n-chunks", "4"]


def _stub_dataset(*a, **kw):
    return SimpleNamespace(cohort=SimpleNamespace(counts={"train": 1}))


def test_exit_3_when_available_model_fails_to_load(monkeypatch):
    monkeypatch.setattr(ef, "is_model_available", lambda *a, **kw: True)
    monkeypatch.setattr(ef, "build_dataset_handle", _stub_dataset)
    seen = {}

    def fake_extract(model, dataset, store, **kw):
        seen.update(kw)
        return False

    monkeypatch.setattr(ef, "extract_features_for_model", fake_extract)
    assert ef.main(ARGS) == 3
    assert seen["chunk"] == 0 and seen["n_chunks"] == 4


def test_exit_0_when_extraction_ran(monkeypatch):
    monkeypatch.setattr(ef, "is_model_available", lambda *a, **kw: True)
    monkeypatch.setattr(ef, "build_dataset_handle", _stub_dataset)
    monkeypatch.setattr(ef, "extract_features_for_model", lambda *a, **kw: True)
    assert ef.main(ARGS) == 0


def test_exit_0_when_model_unavailable_per_registry(monkeypatch):
    monkeypatch.setattr(ef, "is_model_available", lambda *a, **kw: False)
    monkeypatch.setattr(ef, "build_dataset_handle",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")))
    assert ef.main(ARGS) == 0
