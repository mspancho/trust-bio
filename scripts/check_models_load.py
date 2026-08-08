#!/usr/bin/env python
"""Confirm every available FM actually LOADS and ENCODES (no silent fallback).

_FMBase falls back to a deterministic random projection when a real model can't
load, so a green pipeline run does NOT imply real weights were used. Assert the
fallback was not taken.
"""
import numpy as np, torch
from trustbio.config import available_models
from trustbio.features.registry import get_extractor

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}\n")
sig = np.random.default_rng(0).standard_normal(1250).astype(np.float32)
ok, bad = [], []
for m in available_models():
    try:
        ex = get_extractor(m, device=dev, allow_fallback=False).load()
        v = ex.encode_modality(sig, 125, "ecg", 10)
        used_fallback = getattr(ex, "_fallback", None) is not None
        if used_fallback:
            bad.append((m, "SILENTLY FELL BACK to random projection"))
        else:
            ok.append((m, v.shape))
            print(f"  {m:20s} OK   dim={v.shape}")
        del ex
        if dev == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        bad.append((m, f"{type(e).__name__}: {str(e)[:80]}"))
        print(f"  {m:20s} FAIL {type(e).__name__}: {str(e)[:80]}")
print(f"\nreal: {len(ok)}   failed/fallback: {len(bad)}")
for m, why in bad: print(f"  !! {m}: {why}")
