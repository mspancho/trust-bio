import numpy as np
import pandas as pd

from trustbio.taxonomy.cluster import (
    cluster_fault_segments, name_clusters, confusion_against_known_conditions,
)


def _synthetic_three_group_matrix(seed=0):
    """Three well-separated synthetic groups in feature space, each tagged
    with its true generating condition, so clustering has an unambiguous
    right answer to recover."""
    rng = np.random.default_rng(seed)
    n_per_group = 30
    transient = rng.normal(loc=[0.9, 2, 0.8, 0, 0.1], scale=0.05, size=(n_per_group, 5))
    persistent = rng.normal(loc=[0.05, 9, 0.1, 0, 0.1], scale=0.05, size=(n_per_group, 5))
    structural = rng.normal(loc=[1.0, 0, 0.0, 1, 3.0], scale=0.05, size=(n_per_group, 5))
    X = np.concatenate([transient, persistent, structural], axis=0)
    conditions = (
        ["transient"] * n_per_group + ["persistent"] * n_per_group + ["structural"] * n_per_group
    )
    return X, conditions


def test_cluster_fault_segments_recovers_three_groups():
    X, conditions = _synthetic_three_group_matrix()
    labels = cluster_fault_segments(X, seed=0)
    assert len(np.unique(labels)) == 3
    assert len(labels) == len(conditions)


def test_name_clusters_majority_vote_matches_known_conditions():
    X, conditions = _synthetic_three_group_matrix()
    labels = cluster_fault_segments(X, seed=0)
    names = name_clusters(labels, conditions)
    assert set(names.values()) == {"transient", "persistent", "structural"}
    # Each cluster's assigned name must equal its majority true condition.
    for cluster_idx, name in names.items():
        mask = labels == cluster_idx
        majority = pd.Series(conditions)[mask].mode()[0]
        assert name == majority


def test_confusion_against_known_conditions_is_mostly_diagonal():
    X, conditions = _synthetic_three_group_matrix()
    labels = cluster_fault_segments(X, seed=0)
    names = name_clusters(labels, conditions)
    confusion = confusion_against_known_conditions(labels, conditions, names)
    assert set(confusion.index) == {"transient", "persistent", "structural"}
    assert set(confusion.columns) == {"transient", "persistent", "structural"}
    # With well-separated synthetic groups, off-diagonal counts should be
    # small relative to the diagonal.
    diag_total = sum(confusion.loc[c, c] for c in confusion.index)
    total = confusion.values.sum()
    assert diag_total / total > 0.8
