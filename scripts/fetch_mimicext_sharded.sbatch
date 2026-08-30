#!/usr/bin/env bash
#SBATCH --job-name=tb-mxshard
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=11:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/mxshard_%A_%a.out
#SBATCH --error=logs/mxshard_%A_%a.err
#
# Sharded MIMIC-III-Ext-PPG waveform fetch.
#
# The dataset is ~6.4M files averaging ~23 KB, so throughput is bound by HTTP
# request round-trips, not bandwidth: a single stream managed only ~6 GB in
# 11.5 h (~15 days for all 149.5 GB). Sharding parallelises the round-trips.
#
# BE A POLITE CLIENT. physionet.org is shared infrastructure used by other
# researchers (including others on this cluster), there is no S3 mirror for this
# credentialed dataset, and robots.txt sets no Crawl-delay -- absence of a stated
# limit is not licence to saturate it. Hence: start at low concurrency, keep a
# per-request throttle, and only scale after measuring that the server is happy.
#
#   N_SHARDS=8 sbatch --array=0-7 scripts/fetch_mimicext_sharded.sbatch
set -uo pipefail
REPO=/n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
cd "${REPO}"; mkdir -p logs
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
set -a; . ./.env; set +a

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-8}"
THROTTLE_MS="${THROTTLE_MS:-50}"

resubmit() {
  echo "[shard ${SHARD}] resubmitting to resume"
  N_SHARDS="${N_SHARDS}" THROTTLE_MS="${THROTTLE_MS}" \
    sbatch --begin=now+2minutes --array="${SHARD}" \
      "${REPO}/scripts/fetch_mimicext_sharded.sbatch"
}
# slurm SIGKILLs on TIMEOUT, so an exit-code check alone never fires; catch the
# pre-timeout USR1 and resubmit while the shell is still alive.
trap 'echo "[shard ${SHARD}] wall limit approaching"; resubmit; exit 0' USR1

echo "[shard ${SHARD}/${N_SHARDS}] throttle=${THROTTLE_MS}ms"
conda run -n trust-bio python scripts/fetch_mimic_ext_ppg.py \
  --record-list "${REPO}/features_cache/mimicext_records.txt" \
  --shard "${SHARD}" --n-shards "${N_SHARDS}" --throttle-ms "${THROTTLE_MS}" &
wait $!
rc=$?
[[ $rc -ne 0 ]] && resubmit
exit $rc
