#!/usr/bin/env bash
# One-time environment setup for TRUST-BIO on the HMS O2 cluster.
set -euo pipefail

ENV_NAME="${TRUSTBIO_ENV:-map-env-base}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
module load gcc/14.2.0 cuda/12.8 2>/dev/null || true

echo "[setup] using conda env: ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_DIR}"
conda run -n "${ENV_NAME}" python -m pip install neurokit2 pyPPG mat73 || \
  echo "[setup] WARNING: neurokit2/pyPPG/mat73 install failed"
conda run -n "${ENV_NAME}" python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || \
  conda run -n "${ENV_NAME}" python -m pip install torch
conda run -n "${ENV_NAME}" python -m pip install --no-deps \
  momentfm chronos-forecasting umap-learn pynndescent numba llvmlite || \
  echo "[setup] WARNING: some FM backends failed"
# --no-deps above keeps those backends from dragging in their own (often
# conflicting) torch pins, but it also skips dependencies the models genuinely
# need at import time. Installing them explicitly, because without these FIVE of
# the seven models silently fail to load and the pipeline substitutes a random-
# projection fallback -- a run that looks green while measuring nothing real:
#   einops     -> chronos-bolt
#   omegaconf  -> D-BETA
#   xlstm      -> xECG
#   (einops is also required by papagei's models.resnet)
conda run -n "${ENV_NAME}" python -m pip install einops omegaconf xlstm || \
  echo "[setup] WARNING: FM runtime deps failed; chronos/D-BETA/xECG/papagei will fall back"
conda run -n "${ENV_NAME}" python -m pip install huggingface_hub || \
  echo "[setup] WARNING: huggingface_hub install failed; D-BETA's gate check will fall back to env-var-only detection"

bash "${REPO_DIR}/scripts/vendor_model_repos.sh" || \
  echo "[setup] WARNING: model-repo vendoring incomplete; those models will skip cleanly"

echo "[setup] done. Verifying imports:"
conda run -n "${ENV_NAME}" python - <<'PY'
import importlib
for m in ["numpy","pandas","sklearn","wfdb","mat73","torch","momentfm","chronos","neurokit2","pyPPG","huggingface_hub"]:
    try:
        importlib.import_module(m); print(f"  {m:14} OK")
    except Exception as e:
        print(f"  {m:14} MISSING ({type(e).__name__})")
PY

echo "[setup] NOTE: CSFM (Cardiac-Sensing-FM) is access-restricted; skipped automatically"
echo "[setup]       until its checkpoint is placed in model_weights/."
echo "[setup] NOTE: D-BETA's HuggingFace repo (Manhph2211/D-BETA) is GATED — even"
echo "[setup]       with an accepted access request, this shell/job needs credentials"
echo "[setup]       to actually use it. Do ONE of the following before running any"
echo "[setup]       extraction stage that includes dbeta, or it will report as"
echo "[setup]       unavailable and skip cleanly (config.is_model_available):"
echo "[setup]         (a) export HF_TOKEN=<your-huggingface-token>   (simplest for SLURM jobs)"
echo "[setup]         (b) conda run -n ${ENV_NAME} huggingface-cli login   (interactive, caches a token)"
echo "[setup]       Verify it took effect with:"
echo "[setup]         conda run -n ${ENV_NAME} python -c \"from trustbio.config import is_model_available; print(is_model_available('dbeta'))\""
