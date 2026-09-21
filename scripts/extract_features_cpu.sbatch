#!/usr/bin/env bash
#SBATCH --job-name=trustbio-extract-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err
#
# CPU header for ecg-domain (neurokit2; 56% of total per-window cost, no GPU
# kernels). Same manifest format and body as extract_features.sbatch.
#   sbatch --array=0-$((N-1))%14 scripts/extract_features_cpu.sbatch manifest_cpu.txt
export TRUSTBIO_DEVICE=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
REPO_DIR="${TRUSTBIO_REPO:-$(pwd)}"
SBATCH_SCRIPT="${REPO_DIR}/scripts/extract_features_cpu.sbatch"
source "${REPO_DIR}/scripts/_extract_cell.sh" "$@"
