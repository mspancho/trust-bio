#!/usr/bin/env bash
#SBATCH --job-name=trustbio-extract
#SBATCH --partition=gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#
# One array task per (model, dataset) manifest line:
#   sbatch --array=0-$((N-1))%3 scripts/extract_features.sbatch manifest.txt
set -euo pipefail

MANIFEST="${1:-${TRUSTBIO_MANIFEST:-manifest_extract.txt}}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
module load gcc/14.2.0 cuda/12.8 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

i="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((i+1))p" "${MANIFEST}")
MODEL=$(echo "${LINE}" | awk '{print $1}')
DATASET=$(echo "${LINE}" | awk '{print $2}')
DURATION=$(echo "${LINE}" | awk '{print $3}')
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"

echo "[extract] task ${i}: model=${MODEL} dataset=${DATASET} duration=${DURATION}s store=${STORE}"

OVERWRITE_ARG=()
[[ "${TRUSTBIO_OVERWRITE:-0}" == "1" ]] && OVERWRITE_ARG=(--overwrite)

# ecg-domain is neurokit2 feature extraction -- pure CPU, no GPU kernels. It is
# also 56% of total per-window cost (68.2 of 122.8 ms, measured on an A40), so
# pinning it to "cuda" would occupy a scarce GPU to run a CPU workload.
DEVICE="cuda"
[[ "${MODEL}" == *domain* ]] && DEVICE="cpu"

# Pass the cohort/label cache: without it every array task re-derives the label
# table, which costs ~16 min per (dataset) handle because hr_regression is
# computed from the ECG. Across an array that is hours of duplicated work.
CACHE_ARG=()
[[ -n "${TRUSTBIO_COHORT_CACHE:-}" ]] && CACHE_ARG=(--cohort-cache "${TRUSTBIO_COHORT_CACHE}")

conda run -n "${ENV_NAME}" python scripts/extract_features.py \
  --model "${MODEL}" --dataset "${DATASET}" --duration-sec "${DURATION}" \
  --device "${DEVICE}" --store "${STORE}" \
  "${CACHE_ARG[@]}" "${OVERWRITE_ARG[@]}"

echo "[extract] done: ${MODEL} / ${DATASET} @ ${DURATION}s"
