#!/usr/bin/env bash
#SBATCH --job-name=trustbio-fetch
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fetch_%j.out
#SBATCH --error=logs/fetch_%j.err
#
# Fetches all three datasets. MIMIC-III-Ext-PPG needs PHYSIONET_USERNAME/
# PHYSIONET_PASSWORD in the environment (credentialed access, already granted);
# BUT PPG and PulseDB need no credentials.
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

conda run -n "${ENV_NAME}" python scripts/fetch_pulsedb.py
conda run -n "${ENV_NAME}" python scripts/fetch_but_ppg.py
conda run -n "${ENV_NAME}" python scripts/fetch_mimic_ext_ppg.py
echo "[fetch] all datasets fetched"
