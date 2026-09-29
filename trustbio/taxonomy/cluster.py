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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


@dataclass
class FaultClusterer:
    scaler: StandardScaler
    kmeans: KMeans
    labels_: np.ndarray
    names: dict[int, str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(np.asarray(X, float)))

    def predict_names(self, X: np.ndarray) -> list[str]:
        return [self.names.get(int(c), f"cluster_{c}") for c in self.predict(X)]


def fit_fault_clusters(X: np.ndarray, known_conditions: list[str] | None = None,
                       seed: int = 0, n_clusters: int = 3) -> FaultClusterer:
    """Standardize, KMeans, and (with known conditions) name the clusters by
    majority vote. Held-out rows -- clean controls, natural degradation, real
    motion -- are assigned later with `.predict_names`, never used to fit."""
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(scaler.transform(X))
    names = name_clusters(labels, list(known_conditions)) if known_conditions is not None else {}
    return FaultClusterer(scaler=scaler, kmeans=km, labels_=labels, names=names)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(StandardScaler().fit_transform(X), labels))


def condition_recall(assigned_names: list[str], known: list[str]) -> dict[str, float]:
    """Fraction of each known condition's rows that landed in the cluster
    carrying that condition's name (the diagonal of the confusion matrix,
    row-normalised)."""
    out = {}
    for cond in sorted(set(known)):
        idx = [i for i, k in enumerate(known) if k == cond]
        out[cond] = float(np.mean([assigned_names[i] == cond for i in idx])) if idx else float("nan")
    return out


def bootstrap_recall(assigned_names: list[str], known: list[str], subjects: list[str],
                     n_boot: int = 200, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Subject-clustered bootstrap CI for condition_recall: windows of one
    subject are not independent, so resample subjects, not windows."""
    rng = np.random.default_rng(seed)
    point = condition_recall(assigned_names, known)
    by_subject: dict[str, list[int]] = {}
    for i, s in enumerate(subjects):
        by_subject.setdefault(str(s), []).append(i)
    keys = list(by_subject)
    draws: dict[str, list[float]] = {c: [] for c in point}
    for _ in range(n_boot):
        idx = [i for s in rng.choice(keys, len(keys), replace=True) for i in by_subject[s]]
        rec = condition_recall([assigned_names[i] for i in idx], [known[i] for i in idx])
        for c in point:
            draws[c].append(rec.get(c, float("nan")))
    return {c: (point[c], float(np.nanpercentile(draws[c], 2.5)), float(np.nanpercentile(draws[c], 97.5)))
            for c in point}
