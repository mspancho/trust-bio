import json

import numpy as np
import pandas as pd

from scripts.run_taxonomy_degraded_only import main


def _table(n=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    def add(cond, dataset, n, esqi, psqi, edrop, pdrop, dis, severity=np.nan):
        for i in range(n):
            rows.append(dict(dataset=dataset, visit_id=f"{dataset}_{cond}_{severity}_{i}", subject_id=f"s{i % 9}",
                             condition=cond, kind=cond if cond in ("motion_artifact", "lead_off") else "",
                             severity=severity, known_condition=cond,
                             sqi_value=min(esqi, psqi) + rng.normal(0, 0.01),
                             sqi_drop_duration=max(0.0, max(edrop, pdrop) + (rng.normal(0, 0.3) if max(edrop, pdrop) else 0.0)),
                             accel_corr=0.0, source_db=dataset, model_disagreement=dis + rng.normal(0, 0.05),
                             ecg_sqi_value=esqi + rng.normal(0, 0.01), ppg_sqi_value=psqi + rng.normal(0, 0.01),
                             ecg_drop_duration=max(0.0, edrop + (rng.normal(0, 0.3) if edrop else 0.0)),
                             ppg_drop_duration=max(0.0, pdrop + (rng.normal(0, 0.3) if pdrop else 0.0)),
                             pred_a=70.0, pred_b=70.0 + dis, disagreement_raw=dis))

    add("motion_artifact", "pulsedb_mimic", n, 1.0, 0.93, 0, 0, 0.7, severity=0.1)    # undetectable
    add("motion_artifact", "pulsedb_mimic", n, 1.0, 0.30, 0, 6, 0.8, severity=0.6)    # detected
    add("lead_off", "pulsedb_mimic", n, 1.0, 0.93, 0, 0, 0.7, severity=0.1)           # undetectable
    add("lead_off", "pulsedb_mimic", n, 0.5, 0.93, 5, 0, 0.9, severity=0.6)           # detected
    add("structural", "pulsedb_vital", n, 1.0, 0.93, 0, 0, 0.8)
    add("clean", "pulsedb_mimic", 40, 1.0, 0.93, 0, 0, 0.7)
    add("real_motion", "but_ppg", 20, 1.0, 0.35, 0, 5, 0.9)
    return pd.DataFrame(rows)


def test_degraded_only_two_stage_recall(tmp_path):
    csv = tmp_path / "fault_features.csv"
    _table().to_csv(csv, index=False)
    out = tmp_path / "out"
    assert main(["--features-csv", str(csv), "--out-dir", str(out)]) == 0
    rec = pd.read_csv(out / "degraded_only_recall.csv")
    drop = rec[(rec.rule == "drop") & (rec.feature_set == "sqi_only") & (rec.severity != "all")]
    piv = drop.pivot(index="condition", columns="severity", values="detection_rate")
    assert piv.loc["motion_artifact", "0.6"] > 0.95 and piv.loc["motion_artifact", "0.1"] < 0.05
    assert piv.loc["lead_off", "0.6"] > 0.95 and piv.loc["lead_off", "0.1"] < 0.05
    typed = drop.set_index(["condition", "severity"])
    assert typed.loc[("motion_artifact", "0.6"), "typing_recall"] > 0.95
    assert typed.loc[("lead_off", "0.6"), "typing_recall"] > 0.95
    detected = rec.n_detected > 0
    assert np.allclose(rec.overall_recall[detected], (rec.detection_rate * rec.typing_recall)[detected], atol=1e-6)
    assert (rec.overall_recall[~detected] == 0).all() and rec.typing_recall[~detected].isna().all()
    summary = json.loads((out / "degraded_only_summary.json").read_text())
    assert summary["drop"]["sqi_only"]["k"] in (2, 3)
    assert set(summary["drop"]["sqi_only"]["silhouette_by_k"]) == {"2", "3", "4", "5", "6"}
    held = pd.read_csv(out / "degraded_only_heldout_drop_sqi_only.csv", index_col=0)
    assert held.loc["clean", "detection_rate"] < 0.05 and held.loc["real_motion", "detection_rate"] > 0.95
    assert held.loc["real_motion", "motion_artifact"] > 0.95
    assert (out / "degraded_only_confusion_drop_sqi_only.csv").exists()
