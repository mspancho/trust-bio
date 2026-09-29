#!/usr/bin/env bash
#SBATCH --job-name=trustbio-stress
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/stress_%A_%a.out
#SBATCH --error=logs/stress_%A_%a.err
#
# Degradation stress test, prediction pass: one array task per model. Each task
# fits 6 probe sets (3 modalities x 2 source institutions) on the clean
# full-scale store and predicts the taxonomy store's cells. The full-scale
# train matrix is up to 11 GB (2.69M x 1024 float32) and ridge copies it, so
# 64 GB; the transport job (all 42 fit-sets in one process) took 8.5 h / 101 GB.
set -euo pipefail
MODELS=(xecg-10min moment-base chronos-bolt-small ecgfounder papagei dbeta ecg-domain)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}/stress"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
conda run --no-capture-output -n "${ENV_NAME}" python scripts/run_stress_predict.py \
  --model "${MODEL}" --out-dir "${OUT}"
