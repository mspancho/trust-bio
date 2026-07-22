import numpy as np

from trustbio.taxonomy.features import (
    SegmentFaultFeatures, extract_fault_features, features_to_matrix,
)


def test_extract_fault_features_transient_pattern():
    # SQI dips briefly then recovers -> short sqi_drop_duration.
    sqi = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
    accel = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
    feats = extract_fault_features(
        sqi_trace=sqi, accel_trace=accel, fs=1, source_db="pulsedb_mimic",
        model_a_pred=70.0, model_b_pred=70.5, disagreement_scale=5.0,
    )
    assert isinstance(feats, SegmentFaultFeatures)
    assert feats.sqi_value == np.mean(sqi)
    assert feats.sqi_drop_duration == 2   # two consecutive low-SQI samples
    assert feats.accel_corr > 0.5   # accel spikes align with the SQI dip
    assert feats.source_db == "pulsedb_mimic"


def test_extract_fault_features_persistent_pattern_no_accel():
    # SQI stays low the whole segment, no accelerometer available (e.g.
    # MIMIC-III-Ext-PPG / lead-off condition) -> accel_corr defaults to 0.
    sqi = np.zeros(10, dtype=float)
    feats = extract_fault_features(
        sqi_trace=sqi, accel_trace=None, fs=1, source_db="mimic_ext_ppg",
        model_a_pred=70.0, model_b_pred=70.0, disagreement_scale=5.0,
    )
    assert feats.sqi_drop_duration == 10
    assert feats.accel_corr == 0.0


def test_extract_fault_features_structural_shift_pattern():
    # SQI is perfect (clean signal) but models disagree a lot -> high
    # model_disagreement despite sqi_value == 1.0, signaling structural shift.
    sqi = np.ones(10, dtype=float)
    feats = extract_fault_features(
        sqi_trace=sqi, accel_trace=None, fs=1, source_db="pulsedb_vital",
        model_a_pred=70.0, model_b_pred=95.0, disagreement_scale=5.0,
    )
    assert feats.sqi_value == 1.0
    assert feats.model_disagreement == 5.0   # |70-95| / disagreement_scale


def test_features_to_matrix_shape_and_names():
    feats = [
        extract_fault_features(np.ones(5), None, 1, "pulsedb_mimic", 70, 71, 5.0),
        extract_fault_features(np.zeros(5), None, 1, "pulsedb_vital", 70, 90, 5.0),
    ]
    X, names = features_to_matrix(feats)
    assert X.shape == (2, 5)
    assert names == ["sqi_value", "sqi_drop_duration", "accel_corr", "source_db", "model_disagreement"]
