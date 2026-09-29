import json

import numpy as np
import pandas as pd

from scripts.run_taxonomy_supervised import main


def _table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def add(cond, dataset, n, esqi, psqi, edrop, pdrop, dis, severity=np.nan):
        for i in range(n):
            rows.append(dict(dataset=dataset, visit_id=f"{dataset}_{cond}_{i}", subject_id=f"s{i % 9}",
                             condition=cond, kind=cond if cond in ("motion_artifact", "lead_off") else "",
                             severity=severity, known_condition=cond,
                             sqi_value=min(esqi, psqi) + rng.normal(0, 0.02),
                             sqi_drop_duration=max(edrop, pdrop) + rng.normal(0, 0.2),
                             accel_corr=0.0, source_db=dataset, model_disagreement=dis + rng.normal(0, 0.1),
                             ecg_sqi_value=esqi + rng.normal(0, 0.02), ppg_sqi_value=psqi + rng.normal(0, 0.02),
                             ecg_drop_duration=edrop + rng.normal(0, 0.2), ppg_drop_duration=pdrop + rng.normal(0, 0.2),
                             pred_a=70.0, pred_b=70.0 + dis, disagreement_raw=dis))

    add("motion_artifact", "pulsedb_mimic", n, 1.0, 0.5, 0, 3, 0.7, severity=0.3)
    add("motion_artifact", "pulsedb_vital", n, 1.0, 0.2, 0, 6, 0.8, severity=0.6)
    add("lead_off", "pulsedb_mimic", n, 0.7, 0.95, 3, 0, 0.7, severity=0.3)
    add("lead_off", "pulsedb_vital", n, 0.4, 0.95, 6, 0, 0.9, severity=0.6)
    add("structural", "pulsedb_vital", n, 1.0, 0.95, 0, 0, 3.0)
    add("clean", "pulsedb_mimic", 20, 1.0, 0.95, 0, 0, 0.7)
    add("real_motion", "but_ppg", 20, 1.0, 0.45, 0, 3, 0.9)
    return pd.DataFrame(rows)


def test_supervised_ceiling_recovers_separable_synthetic_conditions(tmp_path):
    csv = tmp_path / "fault_features.csv"
    _table().to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["--features-csv", str(csv), "--out-dir", str(out), "--n-boot", "20", "--models", "logreg"]) == 0
    rec = pd.read_csv(out / "supervised_recall.csv")
    overall = rec[(rec.model == "logreg") & (rec.feature_set == "all") & (rec.dataset == "all") & (rec.severity == "all")]
    assert set(overall.condition) == {"motion_artifact", "lead_off", "structural"}
    assert overall.recall.min() > 0.9
    assert (overall.ci_lo <= overall.recall).all() and (overall.recall <= overall.ci_hi).all()
    assert (out / "supervised_confusion_logreg_all.csv").exists()
    held = pd.read_csv(out / "supervised_heldout_logreg_all.csv", index_col=0)
    assert {"clean", "real_motion"} <= set(held.index)
    assert held.loc["real_motion", "motion_artifact"] > 0.9
    summary = json.loads((out / "supervised_summary.json").read_text())
    assert summary["logreg"]["all"]["macro_f1"] > 0.9
    assert summary["structural_vs_clean_auroc"]["all"] > 0.9
