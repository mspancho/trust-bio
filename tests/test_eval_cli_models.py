"""The eval CLIs must accept an explicit model list: evaluation reads cached
features from disk and must not depend on weights or HF tokens being present
(available_models() would silently drop a gated model in a job without them)."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["scripts/run_transport_eval.py", "scripts/run_benchmark.py"])
def test_eval_cli_exposes_models_option(script):
    r = subprocess.run([sys.executable, str(REPO / script), "--help"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--models" in r.stdout
