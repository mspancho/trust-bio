#!/usr/bin/env bash
#SBATCH --job-name=trustbio-taxonomy
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/taxonomy_%j.out
#SBATCH --error=logs/taxonomy_%j.err
#
# Fault taxonomy, end to end: fault-feature table (needs two full-scale train
# matrices -> 64 GB) -> clustering/validation -> figures. Inputs: the taxonomy
# store and cohorts (plan Tasks 4-6) and the verified full-scale store.
set -euo pipefail
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
OUT="${TRUSTBIO_OUT:-${REPO_DIR}/results}/taxonomy"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs" "${OUT}"
cd "${REPO_DIR}"
PY=(conda run --no-capture-output -n "${ENV_NAME}" python)
"${PY[@]}" scripts/build_fault_features.py --store features_cache/taxonomy_store \
  --cohort-cache features_cache/taxonomy --full-store features_cache/full \
  --full-cohort-cache features_cache --out-dir "${OUT}"
"${PY[@]}" scripts/run_taxonomy.py --features-csv "${OUT}/fault_features.csv" --out-dir "${OUT}"
"${PY[@]}" scripts/plot_taxonomy.py --features-csv "${OUT}/fault_features.csv" \
  --assignments-csv "${OUT}/assignments_all.csv" --confusion-csv "${OUT}/confusion_all.csv" \
  --out-dir "${OUT}/figures"
