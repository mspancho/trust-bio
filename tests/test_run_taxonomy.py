import json

import numpy as np
import pandas as pd

from scripts.run_taxonomy import FEATURE_SETS, main


def _table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def add(cond, dataset, n, sqi, drop, esqi, psqi, dis, severity=np.nan):
        ecg_drop = drop if cond == "lead_off" else 0.0
        ppg_drop = drop if cond in ("motion_artifact", "real_motion") else 0.0
        for i in range(n):
            rows.append(dict(dataset=dataset, visit_id=f"{dataset}_{cond}_{i}", subject_id=f"s{i % 7}",
                             condition=cond, kind=cond if cond in ("motion_artifact", "lead_off") else "",
                             severity=severity, known_condition=cond,
                             sqi_value=sqi + rng.normal(0, 0.02), sqi_drop_duration=drop + rng.normal(0, 0.2),
                             accel_corr=0.0, source_db=dataset, model_disagreement=dis + rng.normal(0, 0.1),
                             ecg_sqi_value=esqi + rng.normal(0, 0.02), ppg_sqi_value=psqi + rng.normal(0, 0.02),
                             ecg_drop_duration=ecg_drop + rng.normal(0, 0.2), ppg_drop_duration=ppg_drop + rng.normal(0, 0.2),
                             pred_a=70.0, pred_b=70.0 + dis, disagreement_raw=dis))

    add("motion_artifact", "pulsedb_mimic", n, 0.7, 3, 0.95, 0.5, 0.2, severity=0.3)
    add("lead_off", "pulsedb_mimic", n, 0.6, 6, 0.4, 0.95, 0.2, severity=0.6)
    add("structural", "pulsedb_vital", n, 0.95, 0, 0.95, 0.95, 3.0)
    add("clean", "pulsedb_mimic", 10, 0.95, 0, 0.95, 0.95, 0.2)
    add("real_motion", "but_ppg", 10, 0.65, 3, 0.95, 0.45, 0.4)
    return pd.DataFrame(rows)


def test_run_taxonomy_writes_tables_and_recovers_synthetic_conditions(tmp_path):
    csv = tmp_path / "fault_features.csv"
    _table().to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["--features-csv", str(csv), "--out-dir", str(out), "--n-boot", "20"]) == 0
    t3 = pd.read_csv(out / "table3_recall.csv")
    assert set(t3.feature_set) == set(FEATURE_SETS)
    overall = t3[(t3.feature_set == "all") & (t3.dataset == "all") & (t3.severity == "all")].set_index("condition")
    assert overall.loc[["motion_artifact", "lead_off", "structural"], "recall"].min() > 0.8
    assert (overall.ci_lo <= overall.recall).all() and (overall.recall <= overall.ci_hi).all()
    for s in FEATURE_SETS:
        assert (out / f"confusion_{s}.csv").exists() and (out / f"assignments_{s}.csv").exists()
    asg = pd.read_csv(out / "assignments_all.csv")
    assert set(asg.known_condition) >= {"clean", "real_motion"}         # held-out rows were assigned
    assert (asg[asg.known_condition == "real_motion"]["cluster"] == "motion_artifact").mean() > 0.8
    summary = json.loads((out / "summary.json").read_text())
    assert summary["all"]["silhouette"] > 0.3 and set(summary["all"]["cluster_names"].values()) == {"motion_artifact", "lead_off", "structural"}
