#!/usr/bin/env bash
#SBATCH --job-name=trustbio-benchmark
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"

# run_benchmark.py builds its DatasetHandle directly from --dataset (see
# scripts/_dataset_builders.py), so this just loops over the four dataset
# choices rather than depending on any intermediate artifact.
for dataset in pulsedb_mimic pulsedb_vital mimic_ext_ppg but_ppg; do
  conda run -n "${ENV_NAME}" python scripts/run_benchmark.py \
    --dataset "${dataset}" --store "${STORE}" \
    --out "${OUT}/benchmark_${dataset}.csv"
done
