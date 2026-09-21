import numpy as np
import pandas as pd

import scripts.build_pulsedb_labels as bl
from trustbio.data.cohort import Cohort


def test_main_builds_labels_for_each_source_into_the_store(monkeypatch, tmp_path, capsys):
    seen = []

    def fake_cohort(root, source, cache=None, rebuild=False):
        ids = [f"p1_w{i}" for i in range(4)]
        return Cohort(visits=pd.DataFrame({"visit_id": ids, "subject_id": "p1",
                                           "source": source, "split": "train"}))

    def fake_labels(root, source, visit_ids, cache=None, rebuild=False):
        seen.append((source, list(visit_ids), cache, rebuild))
        return pd.DataFrame({"hr_regression": [80.0, 82.0, np.nan, 78.0],
                             "sbp_regression": [120.0] * 4, "dbp_regression": [80.0] * 4},
                            index=pd.Index(visit_ids, name="visit_id"))

    monkeypatch.setattr(bl, "build_pulsedb_cohort", fake_cohort)
    monkeypatch.setattr(bl, "build_pulsedb_label_table", fake_labels)
    assert bl.main(["--store", str(tmp_path), "--source", "both"]) == 0
    assert [s[0] for s in seen] == ["mimic", "vital"]
    assert all(s[2] == tmp_path and s[1] == [f"p1_w{i}" for i in range(4)] for s in seen)
    out = capsys.readouterr().out
    assert "median hr 80.0" in out and "hr NaN 1" in out
