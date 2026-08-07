#!/usr/bin/env bash
#SBATCH --job-name=tb-mimicext
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=11:30:00
#SBATCH --output=logs/mimicext_%j.out
#SBATCH --error=logs/mimicext_%j.err
#
# MIMIC-III-Ext-PPG fetch, resumable AND self-resubmitting.
#
# Why self-resubmit: metadata.csv alone is 4.92 GB and PhysioNet serves it at
# only ~5-11 MB/min (measured), so a single wall-clock window may not suffice.
# The fetcher resumes via HTTP Range, so relaunching costs nothing already done.
# scrontab is disabled on O2, so chain with `sbatch --begin` instead.
#
#   sbatch scripts/fetch_mimic_ext_ppg.sbatch            # metadata only
#   FETCH_ARGS="--max-patients 20" sbatch scripts/...    # + some waveforms
set -uo pipefail
REPO=/n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
cd "${REPO}"
mkdir -p logs
module load conda/miniforge3/24.11.3-0 2>/dev/null || true

set -a; . ./.env; set +a
ARGS="${FETCH_ARGS:---metadata-only}"

conda run -n map-env-base python scripts/fetch_mimic_ext_ppg.py ${ARGS}
rc=$?

if [[ $rc -ne 0 ]]; then
  echo "[sbatch] fetch exited ${rc}; resubmitting in 2 min to resume"
  FETCH_ARGS="${ARGS}" sbatch --begin=now+2minutes "${REPO}/scripts/fetch_mimic_ext_ppg.sbatch"
else
  echo "[sbatch] fetch completed cleanly; no resubmission"
fi
exit $rc
