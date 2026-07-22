# tests/test_config.py
from trustbio.config import (
    ALL_TASKS, CATEGORIES, TASKS_BY_NAME, FM_REGISTRY, MAIN_TEST_MODELS,
    DEGRADATION_SEVERITIES, DEGRADATION_KINDS, is_model_available,
    resolve_checkpoint, available_models,
)


def test_task_registry_has_expected_tasks():
    names = {t.name for t in ALL_TASKS}
    assert names == {"hr_regression", "sbp_regression", "dbp_regression", "rhythm_cls"}


def test_task_categories():
    assert set(CATEGORIES) == {"hr", "bp", "rhythm"}
    assert TASKS_BY_NAME["hr_regression"].category == "hr"
    assert TASKS_BY_NAME["sbp_regression"].category == "bp"
    assert TASKS_BY_NAME["dbp_regression"].category == "bp"
    assert TASKS_BY_NAME["rhythm_cls"].category == "rhythm"


def test_task_dataset_scoping():
    # hr_regression is available in all three datasets; sbp/dbp only in PulseDB;
    # rhythm_cls only in MIMIC-III-Ext-PPG.
    assert TASKS_BY_NAME["hr_regression"].dataset == "all"
    assert TASKS_BY_NAME["sbp_regression"].dataset == "pulsedb"
    assert TASKS_BY_NAME["dbp_regression"].dataset == "pulsedb"
    assert TASKS_BY_NAME["rhythm_cls"].dataset == "mimic_ext_ppg"


def test_fm_registry_has_seven_main_models_plus_csfm():
    # ECGFounder, xECG-10min, PaPaGei, MOMENT-base, Chronos-Bolt-small,
    # ecg-domain are always-available with no external gate; D-BETA is
    # HF-gated (only available with a valid token/login — see the two tests
    # below); csfm-base is optional (local checkpoint, restricted access).
    assert "csfm-base" in FM_REGISTRY
    assert set(MAIN_TEST_MODELS) - {"csfm-base"} <= set(FM_REGISTRY)
    ungated_always_available = set(MAIN_TEST_MODELS) - {"csfm-base", "dbeta"}
    for name in ungated_always_available:
        assert is_model_available(name), f"{name} should be available without CSFM or an HF token"


def test_dbeta_unavailable_without_hf_token(monkeypatch):
    # D-BETA's HF repo (Manhph2211/D-BETA) is gated: reachable in name, but
    # the download 401s without an accepted access request. Model this
    # honestly — is_model_available must return False with no token present,
    # not optimistically assume the gate is already open.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "trustbio.config._has_huggingface_access", lambda: False,
    )
    assert not is_model_available("dbeta")


def test_dbeta_available_with_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    assert is_model_available("dbeta")


def test_csfm_unavailable_without_checkpoint(monkeypatch):
    monkeypatch.delenv("SIGNALMCMED_CKPT_CSFM_BASE", raising=False)
    monkeypatch.delenv("TRUSTBIO_CKPT_CSFM_BASE", raising=False)
    assert resolve_checkpoint("csfm-base") is None
    assert not is_model_available("csfm-base")


def test_available_models_excludes_csfm_by_default():
    avail = available_models()
    assert "csfm-base" not in avail
    assert "ecgfounder" in avail


def test_degradation_constants():
    assert DEGRADATION_SEVERITIES == [0.1, 0.3, 0.6]
    assert set(DEGRADATION_KINDS) == {"motion_artifact", "lead_off", "missing_ppg"}
