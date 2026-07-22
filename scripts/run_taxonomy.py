#!/usr/bin/env python
"""Stage 4b CLI: fault taxonomy clustering + validation, writing the confusion
matrix (Figure 3b in the paper draft) to disk."""
from __future__ import annotations

import argparse
from pathlib import Path

from trustbio.taxonomy.cluster import (
    cluster_fault_segments, confusion_against_known_conditions, name_clusters,
)
from trustbio.taxonomy.features import features_to_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-npz", required=True,
                     help="path to a .npz with arrays 'X' (n_segments, 5) and "
                          "'known_conditions' (n_segments,) produced by a "
                          "prior feature-extraction pass over degraded segments")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    data = np.load(args.features_npz, allow_pickle=True)
    X, known_conditions = data["X"], data["known_conditions"].tolist()

    labels = cluster_fault_segments(X, seed=args.seed)
    names = name_clusters(labels, known_conditions)
    confusion = confusion_against_known_conditions(labels, known_conditions, names)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(out_path)
    print(f"cluster names: {names}")
    print(f"wrote confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
