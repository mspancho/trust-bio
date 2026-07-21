"""Verify that vendored model repo structure allows expected imports."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Test fixtures to verify each vendor repo has the right structure for imports


def _get_repo_root() -> Path:
    """Get the signal-mcmed-msp root directory."""
    return Path(__file__).resolve().parents[1]


def test_ecgfounder_has_finetune_model():
    """ECGFounder should have finetune_model.py with ft_1lead_ECGFounder."""
    repo = _get_repo_root() / "model_repos" / "ECGFounder"
    if not repo.exists():
        pytest.skip("ECGFounder repo not vendored")

    finetune = repo / "finetune_model.py"
    assert finetune.exists(), "finetune_model.py not found"

    # Verify the function is defined
    content = finetune.read_text()
    assert "def ft_1lead_ECGFounder" in content, "ft_1lead_ECGFounder not defined"


def test_xecg_has_xecg_module():
    """xECG repo should have xecg/xECG.py with xECG class."""
    repo = _get_repo_root() / "model_repos" / "xecg"
    if not repo.exists():
        pytest.skip("xECG repo not vendored")

    xecg_module = repo / "xecg" / "xECG.py"
    assert xecg_module.exists(), "xecg/xECG.py not found"

    content = xecg_module.read_text()
    assert "class xECG" in content, "xECG class not defined"


def test_papagei_has_required_modules():
    """PaPaGei should have linearprobing/utils.py and models/resnet.py."""
    repo = _get_repo_root() / "model_repos" / "papagei-foundation-model"
    if not repo.exists():
        pytest.skip("PaPaGei repo not vendored")

    utils = repo / "linearprobing" / "utils.py"
    assert utils.exists(), "linearprobing/utils.py not found"

    content = utils.read_text()
    assert "def load_model_without_module_prefix" in content, \
        "load_model_without_module_prefix not defined"

    resnet = repo / "models" / "resnet.py"
    assert resnet.exists(), "models/resnet.py not found"

    content = resnet.read_text()
    assert "class ResNet1DMoE" in content, "ResNet1DMoE class not defined"


def test_dbeta_has_processor():
    """D-BETA should have models/processor.py with get_model and get_ecg_feats."""
    repo = _get_repo_root() / "model_repos" / "D-BETA"
    if not repo.exists():
        pytest.skip("D-BETA repo not vendored")

    processor = repo / "models" / "processor.py"
    assert processor.exists(), "models/processor.py not found"

    content = processor.read_text()
    assert "def get_model" in content, "get_model not defined"
    assert "def get_ecg_feats" in content, "get_ecg_feats not defined"


def test_all_repos_present():
    """All four required repos should be cloned."""
    base = _get_repo_root() / "model_repos"
    repos = ["ECGFounder", "xecg", "papagei-foundation-model", "D-BETA"]
    for repo_name in repos:
        repo_path = base / repo_name
        # D-BETA is optional (HF fallback), others are required
        if repo_name != "D-BETA":
            assert repo_path.is_dir(), f"{repo_name} not found in {base}"
