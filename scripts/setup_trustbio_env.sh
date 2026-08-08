#!/usr/bin/env bash
# Build the DEDICATED trust-bio conda env.
#
# Why a separate env rather than reusing map-env-base:
#   map-env-base drifted well ahead of what the vendored foundation models
#   support -- transformers 5.10.2 / torch 2.11 / Python 3.14, against
#   D-BETA's own requirements.txt pin of transformers==4.43.3 and torch==2.4.0.
#   That mismatch broke FOUR of seven models:
#     dbeta              -> ImportError: apply_chunking_to_forward (removed in v5)
#     chronos-bolt-small -> ValueError on the device_map API change
#     xecg-10min         -> HFValidationError (local path vs HF repo id)
#     papagei            -> No module named 'models.resnet'
#   and _FMBase SILENTLY substitutes a random-projection fallback when a model
#   fails to load, so those runs look green while measuring nothing real.
#
# Pinning here keeps map-env-base untouched for other work and makes the
# study's environment reproducible for the paper.
#
#   bash scripts/setup_trustbio_env.sh
#   conda run -n trust-bio python scripts/check_models_load.py   # verify
set -euo pipefail

ENV_NAME="${TRUSTBIO_ENV:-trust-bio}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
module load gcc/14.2.0 cuda/12.8 2>/dev/null || true

# ~/.cache is a symlink to ./scratch/.cache -> /n/scratch/users/m/map9592, which
# O2 has purged, so it is a DANGLING symlink: anything defaulting to ~/.cache
# dies with "Permission denied". That already cost us the HuggingFace token and
# blocked conda env creation outright. Point every cache at durable lab storage.
CACHE_ROOT="${TRUSTBIO_CACHE_ROOT:-/n/data1/hms/dbmi/rajpurkar/lab/home/map9592/.caches}"
mkdir -p "${CACHE_ROOT}"/{conda,pip,hf,mpl,xdg}
export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export HF_HOME="${CACHE_ROOT}/hf"
export MPLCONFIGDIR="${CACHE_ROOT}/mpl"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
echo "[setup] caches -> ${CACHE_ROOT} (NOT ~/.cache, which dangles into purged scratch)"

# Python 3.11: xecg requires >=3.11, and the pinned torch/transformers stack
# below has no wheels for 3.13+.
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[setup] creating conda env ${ENV_NAME} (python 3.11)"
  conda create -y -n "${ENV_NAME}" python=3.11
else
  echo "[setup] reusing existing env ${ENV_NAME}"
fi

run() { conda run -n "${ENV_NAME}" "$@"; }

run python -m pip install --upgrade pip

# Core stack, pinned to what the vendored models actually expect.
# numpy <2 because D-BETA/ECGFounder pin 1.26.x and several vendored repos
# still use APIs numpy 2 removed.
run python -m pip install "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.2" \
  "scikit-learn==1.7.2" "h5py==3.11.0" "wfdb>=4.1" "mat73"

# torch 2.4.0 + cu121 wheels: D-BETA's pin, and new enough for the others.
run python -m pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121 || \
  run python -m pip install "torch==2.4.0"

# transformers 4.43.3 is D-BETA's pin and still supports chronos-bolt's
# device_map usage. tokenizers must match transformers' expected range.
run python -m pip install "transformers==4.43.3" "tokenizers==0.19.1" \
  "huggingface_hub<0.25"

# FM backends. --no-deps keeps them from re-resolving the torch/transformers
# pins above; their real runtime deps are installed explicitly on the next line
# (omitting these is exactly what silently broke 5 of 7 models before).
run python -m pip install --no-deps momentfm chronos-forecasting
run python -m pip install einops omegaconf xlstm

# Signal-processing extras (the domain-feature baselines).
# --no-deps is REQUIRED here: neurokit2 pulls numpy>=2, which would silently
# break the numpy==1.26.4 pin the vendored FMs depend on (and a plain install
# also tries to rebuild scipy from source, which fails). PyWavelets is
# neurokit2's one genuinely needed extra.
run python -m pip install --no-deps neurokit2 PyWavelets || \
  echo "[setup] WARNING: neurokit2 failed; ecg-domain baseline will skip"
run python -m pip install --no-deps pyPPG || \
  echo "[setup] WARNING: pyPPG failed; ppg-domain baseline will skip"

run python -m pip install pytest

# Install trust-bio itself last so nothing re-resolves the pins.
run python -m pip install --no-deps -e "${REPO_DIR}"

echo
echo "[setup] verifying imports:"
run python - <<'PY'
import importlib
for m in ["numpy","pandas","sklearn","wfdb","mat73","h5py","torch",
          "transformers","momentfm","chronos","einops","omegaconf","xlstm",
          "neurokit2","trustbio"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {m:14} MISSING ({type(e).__name__})")
PY

echo
echo "[setup] NOW VERIFY THE MODELS ACTUALLY LOAD -- an env that imports cleanly"
echo "[setup] can still fall back to random projections at model-load time:"
echo "[setup]   sbatch --partition=gpu_quad --qos=gpuquad_qos --gres=gpu:1 \\"
echo "[setup]     --wrap='conda run -n ${ENV_NAME} python scripts/check_models_load.py'"
echo
echo "[setup] Set HF_TOKEN (D-BETA is a gated HF repo) and HF_HOME (keep the"
echo "[setup] cache OFF /n/scratch, which O2 purges) -- see .env."
