#!/usr/bin/env bash
#SBATCH --job-name=trustbio-transport
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/transport_%j.out
#SBATCH --error=logs/transport_%j.err
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-map-env-base}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"

conda run -n "${ENV_NAME}" python scripts/run_transport_eval.py \
  --pulsedb-root /n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb \
  --store "${STORE}" --out "${OUT}/transport_eval.csv"
