#!/usr/bin/env bash
#SBATCH --job-name=trustbio-taxonomy
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/taxonomy_%j.out
#SBATCH --error=logs/taxonomy_%j.err
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}"
FEATURES_NPZ="${TRUSTBIO_TAXONOMY_FEATURES:-${OUT}/fault_features.npz}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"

conda run -n "${ENV_NAME}" python scripts/run_taxonomy.py \
  --features-npz "${FEATURES_NPZ}" --out "${OUT}/fault_confusion_matrix.csv"
