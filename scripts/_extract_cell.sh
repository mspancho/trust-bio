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

# JIT-compiled CUDA extensions (xECG's xLSTM kernels) are cached per Python/
# CUDA version only. gpu_quad mixes L40S/A100/A40/V100 nodes, so a kernel
# built on one architecture was loaded on another and every window died with
# "no kernel image is available for execution on the device". Key the cache
# by the GPU's compute capability so each architecture builds its own.
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -1 | tr -d '. ')
  if [[ -n "${GPU_CC}" ]]; then
    export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/torch_extensions/py311_cu121_sm${GPU_CC}"
    mkdir -p "${TORCH_EXTENSIONS_DIR}"
    echo "[extract] torch extensions dir: ${TORCH_EXTENSIONS_DIR}"
  fi
fi

i="${SLURM_ARRAY_TASK_ID:-0}"
LINE=$(sed -n "$((i+1))p" "${MANIFEST}")
read -r MODEL DATASET DURATION REST <<<"${LINE}"
if [[ -z "${MODEL:-}" || -z "${DATASET:-}" ]]; then
  echo "[extract] no manifest line $((i+1)) in ${MANIFEST}" >&2
  exit 2
fi
DURATION="${DURATION:-${TRUSTBIO_DURATION:-600}}"
# Optional trailing fields: positional `chunk n_chunks`, then name=value
# fields `cond=NAME` (store subdir), `kind=KIND sev=SEV` (degradation).
CHUNK=""; NCHUNKS=""; COND=""; KIND=""; SEV=""; POS=()
for field in ${REST:-}; do
  case "${field}" in
    cond=*) COND="${field#cond=}" ;;
    kind=*) KIND="${field#kind=}" ;;
    sev=*)  SEV="${field#sev=}" ;;
    *)      POS+=("${field}") ;;
  esac
done
if [[ ${#POS[@]} -ge 2 ]]; then CHUNK="${POS[0]}"; NCHUNKS="${POS[1]}"; fi
CELL_STORE="${STORE}${COND:+/${COND}}"

CHUNK_ARGS=()
[[ -n "${CHUNK}" ]] && CHUNK_ARGS=(--chunk "${CHUNK}" --n-chunks "${NCHUNKS}")
DEGRADE_ARGS=()
[[ -n "${KIND}" ]] && DEGRADE_ARGS=(--degrade-kind "${KIND}" --degrade-severity "${SEV}")
OVERWRITE_ARG=()
[[ "${TRUSTBIO_OVERWRITE:-0}" == "1" ]] && OVERWRITE_ARG=(--overwrite)
# ecg-domain is neurokit2 feature extraction -- pure CPU, no GPU kernels; it
# never gets "cuda" even if submitted through the GPU header by mistake.
DEVICE="${TRUSTBIO_DEVICE:-cuda}"
[[ "${MODEL}" == *domain* ]] && DEVICE="cpu"
CACHE_ARG=()
[[ -n "${TRUSTBIO_COHORT_CACHE:-}" ]] && CACHE_ARG=(--cohort-cache "${TRUSTBIO_COHORT_CACHE}")

echo "[extract] task ${i}: model=${MODEL} dataset=${DATASET} duration=${DURATION}s chunk=${CHUNK:-all}/${NCHUNKS:-1} cond=${COND:-none} device=${DEVICE} store=${CELL_STORE} host=$(hostname)"

CMD=(conda run -n "${ENV_NAME}" python scripts/extract_features.py
     --model "${MODEL}" --dataset "${DATASET}" --duration-sec "${DURATION}"
     --device "${DEVICE}" --store "${CELL_STORE}"
     "${CHUNK_ARGS[@]}" "${DEGRADE_ARGS[@]}" "${CACHE_ARG[@]}" "${OVERWRITE_ARG[@]}")
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
