"""Factory mapping model names to their feature-extractor wrappers."""
from __future__ import annotations

from ..config import FM_REGISTRY, PPG_DOMAIN_DEFAULT_SEC, resolve_checkpoint
from .base import FeatureExtractor
from .domain import ECGDomainFeatures, PPGDomainFeatures
from .fms import (
    ChronosBoltExtractor,
    CSFMExtractor,
    DBetaExtractor,
    ECGFounderExtractor,
    MomentExtractor,
    PaPaGeiExtractor,
    XECGExtractor,
)

_FAMILY_TO_CLASS = {
    "moment": MomentExtractor,
    "chronos": ChronosBoltExtractor,
    "dbeta": DBetaExtractor,
    "ecgfounder": ECGFounderExtractor,
    "xecg": XECGExtractor,
    "papagei": PaPaGeiExtractor,
    "csfm": CSFMExtractor,
}


def get_extractor(
    name: str,
    device: str = "cpu",
    allow_fallback: bool = False,
    checkpoint: str | None = None,
    ppg_domain_segment_sec: int = PPG_DOMAIN_DEFAULT_SEC,
    force_fallback: bool = False,
) -> FeatureExtractor:
    """Instantiate (but do not yet `load()`) the extractor for `name`.

    `force_fallback=True` bypasses the real model even when its package/weights
    are present — used by the synthetic demo / CI to exercise plumbing cheaply.
    Domain-feature baselines have no fallback (they are pure compute).
    """
    if name not in FM_REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(FM_REGISTRY)}")
    spec = FM_REGISTRY[name]

    if name == "ecg-domain":
        return ECGDomainFeatures(spec, device)
    if name == "ppg-domain":
        return PPGDomainFeatures(spec, device, segment_sec=ppg_domain_segment_sec)

    # Auto-resolve the checkpoint from model_weights/ when not given explicitly.
    checkpoint = resolve_checkpoint(name, checkpoint)
    cls = _FAMILY_TO_CLASS[spec.family]
    return cls(spec, device=device, allow_fallback=allow_fallback,
               checkpoint=checkpoint, force_fallback=force_fallback)
