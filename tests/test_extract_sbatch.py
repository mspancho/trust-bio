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


def test_dry_run_condition_fields_set_store_subdir_and_degrade_flags(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("papagei pulsedb_vital 10 cond=lead_off_0.3 kind=lead_off sev=0.3\n"
                        "papagei but_ppg 10 cond=clean\n")
    r = _run("scripts/extract_features.sbatch", manifest, 0)
    assert r.returncode == 0, r.stderr
    assert "--store /s/lead_off_0.3" in r.stdout
    assert "--degrade-kind lead_off --degrade-severity 0.3" in r.stdout
    r = _run("scripts/extract_features.sbatch", manifest, 1)
    assert "--store /s/clean" in r.stdout and "--degrade-kind" not in r.stdout


def test_dry_run_condition_fields_coexist_with_chunks(tmp_path):
    manifest = tmp_path / "m.txt"
    manifest.write_text("papagei pulsedb_mimic 10 3 8 cond=motion_artifact_0.6 kind=motion_artifact sev=0.6\n")
    r = _run("scripts/extract_features.sbatch", manifest, 0)
    assert "--chunk 3 --n-chunks 8" in r.stdout and "--store /s/motion_artifact_0.6" in r.stdout
