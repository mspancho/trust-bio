#!/usr/bin/env bash
# Clone the external model source repos that the local-repo feature extractors
# import (ECGFounder, xECG, PaPaGei, CSFM). MOMENT, Chronos-Bolt, and D-BETA
# load from HuggingFace and need NO repo here.
#
# Directory names MUST match those imported in signalmcmed/features/fms.py:
#   ECGFounder, xecg, papagei-foundation-model, Cardiac-Sensing-FM, D-BETA(opt).
#
#   bash scripts/vendor_model_repos.sh
#
# Override the target dir with SIGNALMCMED_MODEL_REPOS. CSFM (Cardiac-Sensing-FM)
# is access-restricted: clone it manually once granted; until then it is skipped.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${SIGNALMCMED_MODEL_REPOS:-${REPO_DIR}/model_repos}"
mkdir -p "${DEST}"
cd "${DEST}"

clone() {  # name url
  local name="$1" url="$2"
  if [[ -d "${name}" ]]; then
    echo "[vendor] ${name} already present — skipping"
    return 0
  fi
  echo "[vendor] cloning ${name} <- ${url}"
  if git clone --depth 1 "${url}" "${name}"; then
    echo "[vendor] ${name} OK"
  else
    echo "[vendor] WARNING: failed to clone ${name}; that model will skip cleanly"
  fi
}

# Public repos (best-effort; URLs may need adjusting if upstream moves).
clone "ECGFounder"              "https://github.com/PKUDigitalHealth/ECGFounder.git"
clone "xecg"                    "https://github.com/dlaskalab/bench-xecg.git"
clone "papagei-foundation-model" "https://github.com/nokia-bell-labs/papagei-foundation-model.git"
# D-BETA: optional local repo (otherwise the HF AutoModel fallback is used).
clone "D-BETA"                  "https://github.com/manhph2211/D-BETA.git"

# CSFM — access-restricted (Oxford). Clone manually into Cardiac-Sensing-FM/
# once access is granted, e.g.:
#   git clone <oxford-csfm-url> "${DEST}/Cardiac-Sensing-FM"
if [[ ! -d "Cardiac-Sensing-FM" ]]; then
  echo "[vendor] NOTE: Cardiac-Sensing-FM (CSFM) not cloned — access-restricted."
  echo "[vendor]       Place it at ${DEST}/Cardiac-Sensing-FM when available."
fi

echo "[vendor] done. Vendored repos in ${DEST}:"
ls -1 "${DEST}" 2>/dev/null | sed 's/^/  /'
