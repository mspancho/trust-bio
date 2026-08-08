#!/usr/bin/env bash
# Submit the entire TRUST-BIO pipeline as a SLURM job DAG.
#
#   bash scripts/run_all.sh
#
# Order: fetch (once) -> cohort (once, depends on fetch) -> extract (GPU
# array, depends on cohort) -> {transport eval, taxonomy, benchmark} (CPU,
# depend on extract; run in parallel with each other).
set -euo pipefail

export TRUSTBIO_REPO="${TRUSTBIO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRUSTBIO_ENV="${TRUSTBIO_ENV:-trust-bio}"
export TRUSTBIO_STORE="${TRUSTBIO_STORE:-${TRUSTBIO_REPO}/features_cache}"
export TRUSTBIO_OUT="${TRUSTBIO_OUT:-${TRUSTBIO_REPO}/results}"
export TRUSTBIO_DURATION="${TRUSTBIO_DURATION:-600}"
MAX_PARALLEL="${TRUSTBIO_MAX_PARALLEL:-3}"

cd "${TRUSTBIO_REPO}"
mkdir -p logs

echo "repo=${TRUSTBIO_REPO} env=${TRUSTBIO_ENV} store=${TRUSTBIO_STORE}"

MANIFEST="${TRUSTBIO_REPO}/manifest_extract.txt"
conda run -n "${TRUSTBIO_ENV}" python scripts/make_manifest.py \
  --duration "${TRUSTBIO_DURATION}" --out "${MANIFEST}"
N=$(wc -l < "${MANIFEST}")
if [[ "${N}" -eq 0 ]]; then
  echo "ERROR: no available (model, dataset) cells. Download checkpoints into model_weights/." >&2
  exit 1
fi
echo "extraction manifest: ${N} cells, max ${MAX_PARALLEL} concurrent"

# --- 0. Fetch datasets (skip if already present) --------------------------- #
if [[ -d "/n/data1/hms/dbmi/rajpurkar/lab/datasets/pulsedb/Segment_Files/PulseDB_MIMIC" ]]; then
  echo "datasets already present, skipping fetch"
  FETCH_JOB="0"
else
  FETCH_JOB=$(sbatch --parsable scripts/fetch_datasets.sbatch)
  echo "submitted fetch: ${FETCH_JOB}"
fi

# --- 1. Cohort build --------------------------------------------------------- #
COHORT_FILE="${TRUSTBIO_STORE}/pulsedb_mimic_cohort.csv"
if [[ -f "${COHORT_FILE}" ]]; then
  echo "cohorts already built, skipping"
  COHORT_JOB="0"
elif [[ "${FETCH_JOB}" == "0" ]]; then
  COHORT_JOB=$(sbatch --parsable scripts/build_cohorts.sbatch)
else
  COHORT_JOB=$(sbatch --parsable --dependency=afterok:${FETCH_JOB} scripts/build_cohorts.sbatch)
fi
[[ "${COHORT_JOB}" != "0" ]] && echo "submitted cohort build: ${COHORT_JOB}"

# --- 2. Feature extraction (GPU array), after cohort ------------------------ #
DEP_ARG=()
[[ "${COHORT_JOB}" != "0" ]] && DEP_ARG=(--dependency=afterok:${COHORT_JOB})
EXTRACT_JOB=$(sbatch --parsable "${DEP_ARG[@]}" \
  --array=0-$((N-1))%${MAX_PARALLEL} scripts/extract_features.sbatch "${MANIFEST}")
echo "submitted extraction array: ${EXTRACT_JOB} (0-$((N-1))%${MAX_PARALLEL})"

# --- 3. Transport eval, taxonomy, benchmark (CPU), after extraction, parallel #
TRANSPORT_JOB=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} scripts/run_transport_eval.sbatch)
TAXONOMY_JOB=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} scripts/run_taxonomy.sbatch)
BENCHMARK_JOB=$(sbatch --parsable --dependency=afterok:${EXTRACT_JOB} scripts/run_benchmark.sbatch)
echo "submitted transport eval: ${TRANSPORT_JOB}"
echo "submitted taxonomy: ${TAXONOMY_JOB}"
echo "submitted benchmark: ${BENCHMARK_JOB}"

echo
echo "DAG submitted. Watch with: squeue -u \$USER"
echo "Final results will appear in: ${TRUSTBIO_OUT}"
