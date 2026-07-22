import subprocess
import sys


def test_synthetic_demo_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, "scripts/make_synthetic_demo.py", "--n-visits", "40"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "SUCCESS" in result.stdout
