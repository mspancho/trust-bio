import numpy as np
import pandas as pd

from scripts.sample_mimic_ext_segments import select_segments


def _meta(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p0 = int(rng.random() < 0.8)
        e0 = 1 if rng.random() < 0.7 else int(rng.choice([0, -2, -3]))
        rows.append({
            "signal_file_name": f"rec_{i}", "subject_id": i % 40,
            "folder_path": f"p00/p{i%40:06d}/rec_{i}",
            "vector_10s_pleth_sqi": f"[{p0}, 1, 1]", "vector_10s_ecg_sqi": f"[{e0}, 1, 1]",
            "median_30s_hr": 70.0, "event_rhythm": "SR",
        })
    return pd.DataFrame(rows)


def test_select_segments_stratifies_and_caps_subjects():
    meta = _meta()
    out = select_segments([meta.iloc[:300], meta.iloc[300:]], n_per_stratum=30, max_per_subject=2, seed=0)
    counts = out["stratum"].value_counts().to_dict()
    assert set(counts) == {"clean", "ppg_poor", "ecg_poor"} and max(counts.values()) <= 30
    assert (out.groupby("subject_id").size() <= 2).all()
    assert (out[out.stratum == "clean"][["pleth_sqi0", "ecg_sqi0"]] == 1).all().all()
    assert (out[out.stratum == "ppg_poor"]["pleth_sqi0"] == 0).all()
    assert (out[out.stratum == "ecg_poor"]["ecg_sqi0"] <= 0).all()
    assert out["signal_file_name"].is_unique
    assert set(meta.columns) <= set(out.columns)


def test_select_segments_is_reproducible():
    meta = _meta()
    a = select_segments([meta], 20, 3, seed=1)["signal_file_name"].tolist()
    b = select_segments([meta], 20, 3, seed=1)["signal_file_name"].tolist()
    assert a == b
