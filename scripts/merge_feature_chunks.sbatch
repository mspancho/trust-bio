#!/usr/bin/env bash
#SBATCH --job-name=trustbio-merge
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/merge_%A_%a.out
#SBATCH --error=logs/merge_%A_%a.err
#
# One array task per `model dataset duration` line: merges that cell's chunk
# files into final per-split matrices, refusing incomplete cells. 64 GB
# because an xecg/ecgfounder MIMIC train split is ~11 GB of float32 and the
# merge holds the parts plus their concatenation. Submit with
#   --dependency=afterany:<gpu array id>:<cpu array id>
# and rerun for any cell it REFUSED once that cell's missing chunks exist.
set -uo pipefail
MANIFEST="${1:-${TRUSTBIO_MANIFEST:-manifest_cells.txt}}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
[[ -f .env ]] && { set -a; . ./.env; set +a; }

i="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((i+1))p" "${MANIFEST}")
read -r MODEL DATASET DURATION _ <<<"${LINE}"
if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "[merge] no manifest line $((i+1)) in ${MANIFEST}" >&2
  exit 2
fi
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"
CACHE_ARG=()
[[ -n "${TRUSTBIO_COHORT_CACHE:-}" ]] && CACHE_ARG=(--cohort-cache "${TRUSTBIO_COHORT_CACHE}")

CMD=(conda run --no-capture-output -n "${ENV_NAME}" python scripts/merge_feature_chunks.py
     --model "${MODEL}" --dataset "${DATASET}" --duration-sec "${DURATION}"
     --store "${STORE}" "${CACHE_ARG[@]}")
echo "[merge] task ${i}: ${CMD[*]}"
[[ "${TRUSTBIO_DRY_RUN:-0}" == "1" ]] && exit 0
"${CMD[@]}"
