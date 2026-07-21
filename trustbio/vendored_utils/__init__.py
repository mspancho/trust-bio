"""Vendored, verbatim copies of the original SignalMC-MED `utils/` modules.

These are copied unchanged from the official repo
(https://github.com/fregu856/SignalMC-MED) so that preprocessing, ECG domain
features, and PPG domain features are byte-for-byte the paper's reference
implementation rather than a re-derivation. Do not edit the algorithmic bodies;
update them only by re-copying from upstream.

Each `preprocess_*` variant resamples to a different target frequency
(250 / 100 / 125 / 500 Hz) matching the model that consumes it.
"""
from .preprocess_no_truncate import (
    preprocess_ecg_no_truncate,
    preprocess_ppg_no_truncate,
)
from .preprocess_no_truncate_100hz import (
    preprocess_ecg_no_truncate_100hz,
    preprocess_ppg_no_truncate_100hz,
)
from .preprocess_no_truncate_125hz import (
    preprocess_ecg_no_truncate_125hz,
    preprocess_ppg_no_truncate_125hz,
)
from .preprocess_no_truncate_500hz import (
    preprocess_ecg_no_truncate_500hz,
    preprocess_ppg_no_truncate_500hz,
)
from .extract_ecg_feature import extract_ecg_feature

__all__ = [
    "preprocess_ecg_no_truncate", "preprocess_ppg_no_truncate",
    "preprocess_ecg_no_truncate_100hz", "preprocess_ppg_no_truncate_100hz",
    "preprocess_ecg_no_truncate_125hz", "preprocess_ppg_no_truncate_125hz",
    "preprocess_ecg_no_truncate_500hz", "preprocess_ppg_no_truncate_500hz",
    "extract_ecg_feature",
]


def get_preprocess_fns(fs_target: int):
    """Return (preprocess_ecg, preprocess_ppg) for the given target frequency."""
    table = {
        250: (preprocess_ecg_no_truncate, preprocess_ppg_no_truncate),
        100: (preprocess_ecg_no_truncate_100hz, preprocess_ppg_no_truncate_100hz),
        125: (preprocess_ecg_no_truncate_125hz, preprocess_ppg_no_truncate_125hz),
        500: (preprocess_ecg_no_truncate_500hz, preprocess_ppg_no_truncate_500hz),
    }
    if fs_target not in table:
        raise ValueError(
            f"no vendored preprocess variant for fs_target={fs_target}; "
            f"have {sorted(table)}"
        )
    return table[fs_target]


def extract_ppg_features(signal, fs):
    """Lazy wrapper around the vendored pyPPG extractor.

    pyPPG (and its matplotlib import) is heavy and only needed for PPG domain
    features, so it is imported on first call rather than at package import.
    """
    from .extract_ppg_feature_stdout_suppression import (
        extract_ppg_features as _impl,
    )
    return _impl(signal, fs)
