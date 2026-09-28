import numpy as np
import pandas as pd

from scripts.sample_cohort import cap_windows_per_subject, sample_subjects


def _cohort():
    rows = []
    for s in ["a", "b", "c"]:
        for w in range(10):
            rows.append({"visit_id": f"{s}_w{w}", "subject_id": s, "split": "train"})
    return pd.DataFrame(rows)


def test_cap_windows_per_subject_keeps_evenly_spaced_windows_in_order():
    out = cap_windows_per_subject(_cohort(), 4)
    assert out.groupby("subject_id").size().tolist() == [4, 4, 4]
    assert out[out.subject_id == "a"]["visit_id"].tolist() == ["a_w0", "a_w3", "a_w6", "a_w9"]
    assert out["visit_id"].tolist() == sorted(out["visit_id"], key=lambda v: (v[0], int(v.split("w")[1])))


def test_cap_is_a_no_op_when_subjects_have_fewer_windows():
    df = _cohort()
    assert len(cap_windows_per_subject(df, 50)) == len(df)


def test_sample_then_cap_preserves_split_disjointness():
    df = _cohort(); df.loc[df.subject_id == "c", "split"] = "test"
    out = cap_windows_per_subject(sample_subjects(df, 2, seed=0), 3)
    assert (out.groupby("subject_id")["split"].nunique() == 1).all()
