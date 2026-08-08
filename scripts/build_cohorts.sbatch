#!/usr/bin/env bash
#SBATCH --job-name=trustbio-cohort
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/cohort_%j.out
#SBATCH --error=logs/cohort_%j.err
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

conda run -n "${ENV_NAME}" python scripts/build_cohorts.py --store "${STORE}"
