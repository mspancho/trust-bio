import numpy as np
import pytest

from trustbio.taxonomy.disagreement import HRProbe, disagreement_scale, fit_hr_probe, predict_hr


def _linear(n, d=8, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = np.random.default_rng(123).normal(0, 3, d)      # same weights for every split
    y = 70 + X @ w + noise * rng.standard_normal(n)
    return X, y


def test_probe_learns_a_linear_hr_relation():
    Xtr, ytr = _linear(600, seed=0); Xva, yva = _linear(150, seed=1); Xte, yte = _linear(150, seed=2)
    probe = fit_hr_probe(Xtr, ytr, Xva, yva, seed=0)
    assert isinstance(probe, HRProbe) and probe.alpha > 0
    r = np.corrcoef(predict_hr(probe, Xte), yte)[0, 1]
    assert r > 0.9


def test_probe_subsamples_train_and_ignores_nan_labels():
    Xtr, ytr = _linear(500); ytr[:50] = np.nan
    probe = fit_hr_probe(Xtr, ytr, *_linear(100, seed=3), max_train=200)
    assert probe.n_train <= 200


def test_disagreement_scale_is_std_of_difference_with_floor():
    a = np.array([70.0, 72.0, 74.0]); b = np.array([70.0, 70.0, 70.0])
    assert np.isclose(disagreement_scale(a, b), np.std(a - b))
    assert disagreement_scale(a, a) == 1e-6
