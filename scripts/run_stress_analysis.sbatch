#!/usr/bin/env bash
#SBATCH --job-name=trustbio-stress-analysis
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/stress_analysis_%j.out
#SBATCH --error=logs/stress_analysis_%j.err
#
# Pools the per-model prediction files (~4M rows) into the Results §2 tables and
# figures. Submit with --dependency=afterok:<array job id> of run_stress.sbatch.
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}/stress"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"
conda run --no-capture-output -n "${ENV_NAME}" python scripts/run_stress_analysis.py \
  --pred-dir "${OUT}" --fault-features results/taxonomy/fault_features.csv --out-dir "${OUT}"
