"""KMeans(k=3) clustering of fault-taxonomy features, with post-hoc cluster
naming against known synthetic-condition labels (majority vote) — the paper
draft's Results subsection "Degraded segments decompose into distinguishable
transient, persistent, and structural fault classes."

Clustering is unsupervised (no labels used to fit KMeans); the known synthetic
condition (motion_artifact / lead_off / clean-but-cross-source) is used only
AFTER clustering, to name each discovered cluster and to build the confusion-
matrix validation artifact — preserving the "label-free" framing while still
allowing quantitative validation against ground truth.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_fault_segments(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Standardize features and run KMeans(k=3). Returns cluster label per row."""
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, random_state=seed, n_init=10)
    return km.fit_predict(X_scaled)


def name_clusters(cluster_labels: np.ndarray, known_conditions: list[str]) -> dict[int, str]:
    """Majority-vote each cluster's name from its members' known conditions.

    If two clusters would receive the same majority name (a degenerate case
    for a poorly-separated real fit), the second-place cluster keeps its raw
    integer label as a string rather than silently colliding names.
    """
    names: dict[int, str] = {}
    used_names: set[str] = set()
    cluster_ids = sorted(set(cluster_labels))
    # Process clusters in order of "most confident majority" first, so
    # genuine majorities claim their name before any collision fallback.
    majority_strength = []
    for cid in cluster_ids:
        members = [c for c, lbl in zip(known_conditions, cluster_labels) if lbl == cid]
        counts = Counter(members)
        top_name, top_count = counts.most_common(1)[0]
        majority_strength.append((top_count / len(members), cid, top_name))
    for _, cid, top_name in sorted(majority_strength, reverse=True):
        if top_name not in used_names:
            names[cid] = top_name
            used_names.add(top_name)
        else:
            names[cid] = f"cluster_{cid}"
    return names


def confusion_against_known_conditions(
    cluster_labels: np.ndarray,
    known_conditions: list[str],
    cluster_names: dict[int, str],
) -> pd.DataFrame:
    """Rows = named cluster, columns = known condition, values = counts."""
    named = [cluster_names[lbl] for lbl in cluster_labels]
    df = pd.DataFrame({"cluster": named, "condition": known_conditions})
    conditions = sorted(set(known_conditions))
    table = (
        df.groupby(["cluster", "condition"]).size().unstack(fill_value=0)
        .reindex(index=sorted(set(named)), columns=conditions, fill_value=0)
    )
    return table
