"""encode_modality must only ever look at the analysed duration. MIMIC-III-
Ext-PPG records are 30 s; a NaN gap after the 10-s window used to poison the
whole preprocessed signal."""
import numpy as np

from trustbio.features.registry import get_extractor


def test_encode_modality_ignores_samples_beyond_duration():
    ex = get_extractor("moment-base", device="cpu", allow_fallback=True, force_fallback=True).load()
    fs = 125
    rng = np.random.default_rng(0)
    ten = rng.standard_normal(10 * fs).astype(np.float32)
    thirty = np.concatenate([ten, np.full(20 * fs, np.nan, dtype=np.float32)])
    a = ex.encode_modality(ten, fs, "ecg", 10)
    b = ex.encode_modality(thirty, fs, "ecg", 10)
    assert np.isfinite(b).all()
    np.testing.assert_allclose(a, b)
