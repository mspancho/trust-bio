"""Central configuration for TRUST-BIO.

Task registry, FM registry, signal/segmentation parameters, and degradation
constants. The FM registry (models, checkpoint resolution, availability
checks) is carried over unchanged from signal-mcmed-msp/signalmcmed/config.py
since none of it is MC-MED-specific — it describes model weights, not tasks.
The task registry below replaces SignalMC-MED's 20 ED-specific tasks with the
task overlap that actually exists across PulseDB, MIMIC-III-Ext-PPG, and BUT
PPG: heart rate regression (all three), systolic/diastolic blood pressure
regression (PulseDB only, from its arterial blood pressure channel), and
rhythm classification (MIMIC-III-Ext-PPG only, from its native rhythm labels).
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass, field
from pathlib import Path as _Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Signal / segmentation parameters (matches signal-mcmed-msp exactly)         #
# --------------------------------------------------------------------------- #
FULL_DURATION_SEC = 600
SEGMENT_SEC = 10
N_SEGMENTS_FULL = FULL_DURATION_SEC // SEGMENT_SEC
DEFAULT_FS = 250

SIGNAL_DURATIONS_SEC = [10, 30, 60, 120, 300, 600]

ECG_DOMAIN_SEGMENT_SEC = 10
PPG_DOMAIN_SEGMENT_SECS = [20, 60, 120]
PPG_DOMAIN_DEFAULT_SEC = 60

ECG_DOMAIN_DIM = 54
PPG_DOMAIN_DIM = 306

# --------------------------------------------------------------------------- #
# Evaluation protocol (matches signal-mcmed-msp exactly)                      #
# --------------------------------------------------------------------------- #
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
N_RESAMPLE_REPEATS = 5
RESAMPLE_WITH_REPLACEMENT = True

MODALITIES = ["ecg", "ppg", "ecg_ppg_mean"]

RIDGE_ALPHAS = np.logspace(-6, 6, 31)
LOGREG_CS = np.logspace(-4, 4, 25)

HP_SELECTION_REG_TASK = "hr_regression"
HP_SELECTION_CLS_TASK = "rhythm_cls"

# --------------------------------------------------------------------------- #
# Degradation constants                                                       #
# --------------------------------------------------------------------------- #
DEGRADATION_SEVERITIES = [0.1, 0.3, 0.6]   # fraction of segment corrupted
DEGRADATION_KINDS = ["motion_artifact", "lead_off", "missing_ppg"]

# --------------------------------------------------------------------------- #
# Task definitions                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    name: str
    kind: str          # "regression" | "classification"
    category: str       # "hr" | "bp" | "rhythm"
    dataset: str         # "all" | "pulsedb" | "mimic_ext_ppg" | "but_ppg"
    description: str = ""


CATEGORY_HR = "hr"
CATEGORY_BP = "bp"
CATEGORY_RHYTHM = "rhythm"
CATEGORIES = [CATEGORY_HR, CATEGORY_BP, CATEGORY_RHYTHM]

ALL_TASKS = [
    Task("hr_regression", "regression", CATEGORY_HR, "all",
         "Heart rate (beats per minute), derived from ECG/PPG"),
    Task("sbp_regression", "regression", CATEGORY_BP, "pulsedb",
         "Systolic blood pressure (mmHg), from PulseDB's synchronized ABP channel"),
    Task("dbp_regression", "regression", CATEGORY_BP, "pulsedb",
         "Diastolic blood pressure (mmHg), from PulseDB's synchronized ABP channel"),
    Task("rhythm_cls", "classification", CATEGORY_RHYTHM, "mimic_ext_ppg",
         "Sinus rhythm vs. atrial fibrillation, from MIMIC-III-Ext-PPG's native rhythm labels"),
]
TASKS_BY_NAME = {t.name: t for t in ALL_TASKS}

assert len(ALL_TASKS) == 4, "TRUST-BIO defines exactly 4 tasks"


def tasks_for_dataset(dataset: str) -> list[Task]:
    """Tasks scoped to `dataset` (its own tasks plus any "all"-scoped tasks).

    NOTE: `dataset` here means a Task.dataset value ("pulsedb", "mimic_ext_ppg",
    "but_ppg", or "all") -- NOT one of scripts/_dataset_builders.py's
    DATASET_CHOICES ("pulsedb_mimic"/"pulsedb_vital" both collapse to
    "pulsedb" here, since PulseDB's task scoping doesn't distinguish its two
    source institutions). Passing "pulsedb_mimic"/"pulsedb_vital" directly
    would silently return only the "all"-scoped tasks (hr_regression),
    dropping sbp/dbp_regression -- always pass "pulsedb", not a DATASET_CHOICES
    value, when calling this function for PulseDB.
    """
    return [t for t in ALL_TASKS if t.dataset in ("all", dataset)]


# --------------------------------------------------------------------------- #
# Foundation-model registry (identical to signal-mcmed-msp; carried over      #
# unchanged since it describes model weights, not MC-MED-specific tasks)      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FMSpec:
    name: str
    family: str
    kind: str
    feature_dim: int
    sampling_freq: int
    size: str = ""
    long_input: bool = False
    notes: str = ""


FM_REGISTRY: dict[str, FMSpec] = {
    "moment-small": FMSpec("moment-small", "moment", "general_ts", 512, DEFAULT_FS, "35M"),
    "moment-base":  FMSpec("moment-base", "moment", "general_ts", 768, DEFAULT_FS, "110M"),
    "moment-large": FMSpec("moment-large", "moment", "general_ts", 1024, DEFAULT_FS, "341M"),
    "chronos-bolt-tiny":  FMSpec("chronos-bolt-tiny", "chronos", "general_ts", 256, DEFAULT_FS, "9M"),
    "chronos-bolt-mini":  FMSpec("chronos-bolt-mini", "chronos", "general_ts", 384, DEFAULT_FS, "21M"),
    "chronos-bolt-small": FMSpec("chronos-bolt-small", "chronos", "general_ts", 512, DEFAULT_FS, "48M"),
    "chronos-bolt-base":  FMSpec("chronos-bolt-base", "chronos", "general_ts", 768, DEFAULT_FS, "205M"),
    "papagei":     FMSpec("papagei", "papagei", "ppg", 512, 125, "6M",
                          notes="PaPaGei-S, ResNet1DMoE, single input channel"),
    "dbeta":       FMSpec("dbeta", "dbeta", "ecg", 768, 500, "62M",
                          notes="get_ecg_feats, signal in 2nd of 12 channels"),
    "ecgfounder":  FMSpec("ecgfounder", "ecgfounder", "ecg", 1024, 500, "31M",
                          notes="single-lead model, ft_1lead_ECGFounder"),
    "xecg":        FMSpec("xecg", "xecg", "ecg", 1024, 100, "57M",
                          notes="xLSTM, signal in 2nd channel"),
    "xecg-10min":  FMSpec("xecg-10min", "xecg", "ecg", 1024, 100, "57M",
                          long_input=True,
                          notes="xLSTM long-input variant, full 10-min signal"),
    "csfm-tiny":  FMSpec("csfm-tiny", "csfm", "ecg_ppg", 768, 250, "43M"),
    "csfm-base":  FMSpec("csfm-base", "csfm", "ecg_ppg", 768, 250, "109M"),
    "csfm-large": FMSpec("csfm-large", "csfm", "ecg_ppg", 1024, 250, "334M"),
    "ecg-domain": FMSpec("ecg-domain", "domain", "domain", ECG_DOMAIN_DIM, DEFAULT_FS, "-",
                         notes="NeuroKit2 ECG features, 10-s segments"),
    "ppg-domain": FMSpec("ppg-domain", "domain", "domain", PPG_DOMAIN_DIM, DEFAULT_FS, "-",
                         notes="pyPPG features, >=20-s segments (60 s for main)"),
}

MAIN_TEST_MODELS = [
    "moment-base", "chronos-bolt-small", "dbeta", "ecgfounder",
    "xecg-10min", "papagei", "csfm-base", "ecg-domain",
]

MODEL_WEIGHTS_DIR = _Path(
    _os.environ.get(
        "TRUSTBIO_WEIGHTS",
        _Path(__file__).resolve().parent.parent / "model_weights",
    )
)

WEIGHT_SOURCE = {
    "moment": "hf",           # public, ungated — no login needed
    "chronos": "hf",          # public, ungated — no login needed
    "dbeta": "hf_gated",      # HF repo Manhph2211/D-BETA requires an accepted
                              # access request; do NOT assume it always
                              # succeeds like a plain "hf" source (see below)
    "ecgfounder": "local",
    "xecg": "local",
    "papagei": "local",
    "csfm": "local",
    "domain": "none",
}

CHECKPOINT_FILES = {
    "ecgfounder": ["1_lead_ECGFounder.pth"],
    "xecg": ["xECG_base_model_v1.safetensors"],
    "papagei": ["papagei_s.pt"],
    "csfm": ["csfm_tiny.pt", "csfm_base.pt", "csfm_large.pt"],
}


def resolve_checkpoint(name: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    spec = FM_REGISTRY[name]
    source = WEIGHT_SOURCE.get(spec.family, "local")
    if source in ("hf", "hf_gated", "none"):
        return None
    env_key = "TRUSTBIO_CKPT_" + name.upper().replace("-", "_")
    if env_key in _os.environ:
        return _os.environ[env_key]
    for fname in CHECKPOINT_FILES.get(spec.family, []):
        cand = MODEL_WEIGHTS_DIR / fname
        if cand.exists():
            return str(cand)
    return None


def _has_huggingface_access() -> bool:
    """True if a cached HF login/token exists, so a gated model's download is
    expected to succeed rather than fail with an opaque 401 at extraction
    time. Checks HF_TOKEN/HUGGING_FACE_HUB_TOKEN first (fast, no I/O), then
    falls back to huggingface_hub's cached-token file if the package is
    importable."""
    if _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    # Cached-token fallback (i.e. `huggingface-cli login` with no env var set).
    # `get_token()` is the current API; `HfFolder.get_token()` is the legacy one
    # and is GONE in recent huggingface_hub (>=1.0), so try the modern name
    # first -- relying on HfFolder alone silently disabled this whole fallback.
    # Catch Exception, not only ImportError: reading the cached token touches the
    # filesystem, and on this cluster ~/.cache is a DANGLING symlink into purged
    # /n/scratch, so get_token() raises PermissionError instead of returning
    # None. A narrow `except ImportError` let that propagate and crash the
    # caller. Availability detection must degrade to "no token available" --
    # an unreadable token cache simply means we have no token.
    try:
        from huggingface_hub import get_token
        return get_token() is not None
    except ImportError:
        pass
    except Exception:
        return False
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token() is not None
    except Exception:
        return False


def is_model_available(name: str, explicit_ckpt: str | None = None) -> bool:
    if name not in FM_REGISTRY:
        return False
    spec = FM_REGISTRY[name]
    source = WEIGHT_SOURCE.get(spec.family, "local")
    if source == "hf":
        return True
    if source == "hf_gated":
        # Unlike plain "hf", a gated repo's download can fail with a 401 even
        # though the package/model name is valid — only report available when
        # a token/login is actually present, so an ungranted access request
        # produces a clean upstream skip instead of a confusing crash.
        return _has_huggingface_access()
    if source == "none":
        return True
    return resolve_checkpoint(name, explicit_ckpt) is not None


def available_models(candidates: list[str] | None = None) -> list[str]:
    candidates = candidates if candidates is not None else MAIN_TEST_MODELS
    return [m for m in candidates if is_model_available(m)]
