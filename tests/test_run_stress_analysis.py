import json

import numpy as np
import pandas as pd

from test_stress_analysis import _pred


def _fault_features(pred, rng):
    """One taxonomy row per (dataset, condition, visit_id) of the synthetic
    predictions, with the label-free features; severe rows carry drops."""
    keys = pred[pred.condition != "missing_ppg"][["target", "condition", "kind", "severity", "visit_id"]].drop_duplicates()
    n = len(keys)
    sev = keys.severity.fillna(0).to_numpy()
    drop = np.where(sev >= 0.6, 5.0 + rng.normal(0, 0.5, n), 0.0)
    return pd.DataFrame(dict(
        dataset=keys.target.to_numpy(), condition=keys.condition.to_numpy(), visit_id=keys.visit_id.to_numpy(),
        kind=keys.kind.where(keys.kind != "clean", np.nan).to_numpy(),
        severity=keys.severity.where(keys.severity > 0, np.nan).to_numpy(),
        known_condition=np.where(keys.kind == "clean", "clean", keys.kind).astype(object),
        subject_id=[v.split("_w")[0] for v in keys.visit_id], source_db=keys.target.to_numpy(),
        sqi_value=1 - drop / 10 + rng.normal(0, 0.01, n), sqi_drop_duration=drop, accel_corr=0.0,
        model_disagreement=1.0 + sev + rng.normal(0, 0.1, n), ecg_sqi_value=1.0 - drop / 10,
        ppg_sqi_value=0.93 - sev / 5 + rng.normal(0, 0.01, n), ecg_drop_duration=drop, ppg_drop_duration=0.0))


def test_analysis_cli_end_to_end(tmp_path):
    from scripts.run_stress_analysis import main
    rng = np.random.default_rng(0)
    pred = _pred()
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for m, g in pred.groupby("model"):
        g.to_csv(pred_dir / f"predictions_{m}.csv.gz", index=False)
    ff = tmp_path / "fault_features.csv"
    _fault_features(pred, rng).to_csv(ff, index=False)
    out = tmp_path / "out"
    assert main(["--pred-dir", str(pred_dir), "--fault-features", str(ff), "--out-dir", str(out),
                 "--domain", "A", "--ts", "C"]) == 0
    for name in ("stress_scores.csv", "stress_rank_stability.csv", "stress_gap.csv", "stress_fusion.csv",
                 "stress_harm_coverage.csv", "stress_detection_flags.csv", "stress_summary.json"):
        assert (out / name).exists(), name
    assert (out / "figures" / "fig2a_severity_curves_hr_regression.png").exists()
    assert (out / "figures" / "fig2d_harm_coverage_hr_regression.png").exists()
    cov = pd.read_csv(out / "stress_harm_coverage.csv")
    lead = cov[(cov.condition == "lead_off_0.6") & (cov.modality == "ecg") & (cov.rule == "drop")]
    assert len(lead) == 3 and (lead.detection_rate == 1.0).all()          # severe rows carry drops
    summary = json.loads((out / "stress_summary.json").read_text())
    assert summary["models"] == ["A", "B", "C"] and summary["n_predictions"] == len(pred)
    flags = pd.read_csv(out / "stress_detection_flags.csv")
    assert {"det_drop", "det_outlier", "det_supervised"} <= set(flags.columns)
