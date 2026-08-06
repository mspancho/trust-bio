#!/usr/bin/env bash
#SBATCH --job-name=tb-fetch-pulsedb
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fetch_pulsedb_%j.out
#SBATCH --error=logs/fetch_pulsedb_%j.err
#
# PulseDB via the Box mirror (~18GB in 26 resumable multipart archives).
# Resumable: if this job is killed or times out, just resubmit -- curl -C -
# skips completed parts and resumes partial ones.
set -euo pipefail
REPO=/n/data1/hms/dbmi/rajpurkar/lab/home/map9592/trust-bio
mkdir -p "${REPO}/logs"
cd "${REPO}"
module load conda/miniforge3/24.11.3-0 2>/dev/null || true
conda run -n map-env-base python scripts/fetch_pulsedb_box.py
