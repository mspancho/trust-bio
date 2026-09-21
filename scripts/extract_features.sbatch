#!/usr/bin/env bash
#SBATCH --job-name=trustbio-extract
#SBATCH --partition=gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#
# GPU header. One array task per manifest line `model dataset duration
# [chunk n_chunks]`; the body lives in scripts/_extract_cell.sh.
#   sbatch --array=0-$((N-1))%12 scripts/extract_features.sbatch manifest.txt
# ecg-domain (CPU-only) belongs on scripts/extract_features_cpu.sbatch.
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
SBATCH_SCRIPT="${REPO_DIR}/scripts/extract_features.sbatch"
source "${REPO_DIR}/scripts/_extract_cell.sh" "$@"
