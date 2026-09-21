#!/usr/bin/env bash
# Shared body for scripts/extract_features.sbatch (GPU) and
# scripts/extract_features_cpu.sbatch (CPU). The header that sources this
# must set SBATCH_SCRIPT to its own path (used to resubmit itself).
#
# Manifest line: `model dataset duration [chunk n_chunks]` (see make_manifest.py).
# Env (all optional): TRUSTBIO_REPO TRUSTBIO_ENV TRUSTBIO_STORE
#   TRUSTBIO_COHORT_CACHE TRUSTBIO_DURATION TRUSTBIO_OVERWRITE=1
#   TRUSTBIO_DEVICE (default cuda) TRUSTBIO_DRY_RUN=1 (print command, exit 0)
set -uo pipefail

MANIFEST="${1:-${TRUSTBIO_MANIFEST:-manifest_extract.txt}}"
ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
STORE="${TRUSTBIO_STORE:-${REPO_DIR}/features_cache}"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
module load gcc/14.2.0 cuda/12.8 2>/dev/null || true
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
# HF_HOME and the gated-model token live in .env; never echo them.
[[ -f .env ]] && { set -a; . ./.env; set +a; }

i="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((i+1))p" "${MANIFEST}")
read -r MODEL DATASET DURATION CHUNK NCHUNKS <<<"${LINE}"
if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "[extract] no manifest line $((i+1)) in ${MANIFEST}" >&2
  exit 2
fi
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"

CHUNK_ARGS=()
[[ -n "${CHUNK:-}" ]] && CHUNK_ARGS=(--chunk "${CHUNK}" --n-chunks "${NCHUNKS}")
OVERWRITE_ARG=()
[[ "${TRUSTBIO_OVERWRITE:-0}" == "1" ]] && OVERWRITE_ARG=(--overwrite)
# ecg-domain is neurokit2 feature extraction -- pure CPU, no GPU kernels; it
# never gets "cuda" even if submitted through the GPU header by mistake.
DEVICE="${TRUSTBIO_DEVICE:-cuda}"
[[ "${MODEL}" == *domain* ]] && DEVICE="cpu"
CACHE_ARG=()
[[ -n "${TRUSTBIO_COHORT_CACHE:-}" ]] && CACHE_ARG=(--cohort-cache "${TRUSTBIO_COHORT_CACHE}")

echo "[extract] task ${i}: model=${MODEL} dataset=${DATASET} duration=${DURATION}s chunk=${CHUNK:-all}/${NCHUNKS:-1} device=${DEVICE} store=${STORE} host=$(hostname)"

CMD=(conda run -n "${ENV_NAME}" python scripts/extract_features.py
     --model "${MODEL}" --dataset "${DATASET}" --duration-sec "${DURATION}"
     --device "${DEVICE}" --store "${STORE}"
     "${CHUNK_ARGS[@]}" "${CACHE_ARG[@]}" "${OVERWRITE_ARG[@]}")
echo "[extract] cmd: ${CMD[*]}"
if [[ "${TRUSTBIO_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# slurm SIGKILLs on TIMEOUT, so an exit-code check alone never fires; the
# header asks for USR1 ten minutes early. Resubmit the same array index and
# exit while the shell is still alive -- finished chunks are skipped on resume.
resubmit() {
  echo "[extract] wall limit approaching; resubmitting task ${i} to resume"
  sbatch --array="${i}" "${SBATCH_SCRIPT}" "${MANIFEST}"
}
trap 'resubmit; exit 0' USR1

"${CMD[@]}" &
wait $!
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "[extract] FAILED rc=${rc}: ${MODEL} / ${DATASET} chunk=${CHUNK:-all} (not resubmitting a deterministic failure)"
  exit $rc
fi
echo "[extract] done: ${MODEL} / ${DATASET} @ ${DURATION}s chunk=${CHUNK:-all}/${NCHUNKS:-1}"
