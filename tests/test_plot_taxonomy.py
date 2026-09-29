import numpy as np
import pandas as pd

from scripts.plot_taxonomy import main


def test_plots_are_written_from_small_synthetic_tables(tmp_path):
    rng = np.random.default_rng(0)
    conds = ["motion_artifact"] * 20 + ["lead_off"] * 20 + ["structural"] * 20 + ["clean"] * 10 + ["real_motion"] * 10
    feats = pd.DataFrame({
        "visit_id": [f"v{i}" for i in range(80)], "dataset": ["pulsedb_mimic"] * 80, "subject_id": "s",
        "known_condition": conds, "condition": conds, "severity": [0.3] * 40 + [np.nan] * 40,
        "sqi_value": rng.random(80), "sqi_drop_duration": rng.integers(0, 10, 80), "accel_corr": 0.0,
        "source_db": "pulsedb_mimic", "model_disagreement": rng.random(80),
        "ecg_sqi_value": rng.random(80), "ppg_sqi_value": rng.random(80),
        "ecg_drop_duration": rng.integers(0, 10, 80), "ppg_drop_duration": rng.integers(0, 10, 80),
    })
    asg = feats[["visit_id", "dataset", "condition", "severity", "known_condition"]].copy()
    asg["cluster"] = [c if c in ("motion_artifact", "lead_off", "structural") else "motion_artifact" for c in conds]
    asg["in_fit"] = asg.known_condition.isin(["motion_artifact", "lead_off", "structural"])
    conf = pd.DataFrame([[18, 1, 1], [2, 17, 1], [0, 0, 20]], index=["motion_artifact", "lead_off", "structural"],
                        columns=["motion_artifact", "lead_off", "structural"])
    t3 = pd.DataFrame([dict(feature_set="all", condition=c, dataset="all", severity=s, n=10, recall=r, ci_lo=r - .1, ci_hi=r + .1)
                       for c in ("motion_artifact", "lead_off") for s, r in (("0.1", .5), ("0.3", .7), ("0.6", .9))])
    feats.to_csv(tmp_path / "fault_features.csv", index=False)
    asg.to_csv(tmp_path / "assignments_all.csv", index=False)
    conf.to_csv(tmp_path / "confusion_all.csv")
    t3.to_csv(tmp_path / "table3_recall.csv", index=False)
    out = tmp_path / "figures"
    assert main(["--features-csv", str(tmp_path / "fault_features.csv"), "--assignments-csv", str(tmp_path / "assignments_all.csv"),
                 "--confusion-csv", str(tmp_path / "confusion_all.csv"), "--out-dir", str(out)]) == 0
    for name in ("fig3a_projection.png", "fig3b_confusion.png", "fig3d_structural_share.png", "fig3e_severity_recall.png"):
        assert (out / name).stat().st_size > 1000
