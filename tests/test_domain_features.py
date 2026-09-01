"""Regression tests for the ecg-domain feature extractor.

These exist because of a three-layer silent failure: neurokit2 0.2.13 calls
np.trapezoid (numpy >= 2.0 only; this env pins numpy 1.26.4), so ecg_quality
raised AttributeError on every call; the vendored except then set `sqi = 0`
(0-dim), making np.concatenate raise ValueError on every window with good
R-peaks; and ECGDomainFeatures.encode_segments swallowed that and zero-filled.
The pilot produced 155k x 54 matrices of ALL ZEROS with green logs. Each test
below pins one layer of that chain open.
"""
import numpy as np
import pytest

nk = pytest.importorskip("neurokit2")

from trustbio.config import ECG_DOMAIN_DIM, FM_REGISTRY
from trustbio.features.domain import ECGDomainFeatures
from trustbio.vendored_utils import extract_ecg_feature


def _sim_ecg(seconds=10, fs=250, seed=0):
    return nk.ecg_simulate(
        duration=seconds, sampling_rate=fs, heart_rate=75, random_state=seed
    ).astype(np.float64)


def test_extract_ecg_feature_nonzero_on_clean_ecg():
    """A clean simulated ECG must yield real (non-zero) features, not the
    insufficient-R-peaks zero vector and not an exception."""
    f = extract_ecg_feature(_sim_ecg(), fs=250)
    f = np.asarray(f, dtype=float).ravel()
    assert len(f) == ECG_DOMAIN_DIM
    assert np.count_nonzero(np.nan_to_num(f)) > ECG_DOMAIN_DIM // 2


def test_encode_segments_distinct_across_windows():
    """Different windows must produce different feature rows; an all-constant
    output means every segment silently fell back."""
    extractor = ECGDomainFeatures(FM_REGISTRY["ecg-domain"], device="cpu").load()
    segs = np.stack([_sim_ecg(seed=s) for s in range(3)])
    X = extractor.encode_segments(segs, "ecg")
    assert X.shape == (3, ECG_DOMAIN_DIM)
    assert np.isfinite(X).all()
    assert len({row.tobytes() for row in X}) == 3, "feature rows are not distinct"
    assert np.count_nonzero(X) > 0, "all-zero features: extractor is silently failing"


def test_numpy_trapezoid_alias_present():
    """importing the vendored module must leave np.trapezoid usable, since
    neurokit2 0.2.8+ calls it unconditionally on numpy 1.x."""
    import trustbio.vendored_utils.extract_ecg_feature  # noqa: F401
    assert hasattr(np, "trapezoid")
