import json

import numpy as np
import pandas as pd
import pytest

from trustbio.store import FeatureStore


def _ids(subjects, n_windows):
    return [f"{s}_w{w}" for s in subjects for w in range(n_windows)]


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(0)
    dim, model = 6, "m"
    tr_subj = [f"s{i:03d}" for i in range(50)]
    va_subj = [f"s{i:03d}" for i in range(50, 60)]
    tax_subj = tr_subj[:5]                       # taxonomy cohort = first 5 train subjects
    full = FeatureStore(tmp_path / "full" / "pulsedb_mimic")
    X_by_split = {}
    for split, subj in (("train", tr_subj), ("val", va_subj)):
        ids = _ids(subj, 10)
        X_by_split[split] = (ids, rng.normal(size=(len(ids), dim)).astype(np.float32))
    w = rng.normal(size=dim)                     # one label rule for every split
    frames = []
    for split, (ids, X) in X_by_split.items():
        hr = 70 + 10 * X @ w + rng.normal(0, 0.5, len(ids))
        frames.append(pd.DataFrame(dict(visit_id=ids, hr_regression=hr, sbp_regression=120 + hr / 2,
                                        dbp_regression=60 + hr / 4)))
        for modality in ("ecg", "ppg", "ecg_ppg_mean"):
            full.save(model, modality, 10, split, ids, X)
    pd.concat(frames).to_csv(tmp_path / "pulsedb_mimic_labels.csv", index=False)
    (tmp_path / "tax").mkdir()
    pd.DataFrame(dict(visit_id=_ids(tax_subj, 10), subject_id=[s for s in tax_subj for _ in range(10)],
                      source="mimic", split="train")).to_csv(tmp_path / "tax" / "pulsedb_mimic_cohort.csv", index=False)
    # taxonomy store: the 5 taxonomy subjects' clean train vectors, then degraded copies
    tax_ids = _ids(tax_subj, 10)
    tr_ids, X_tr = X_by_split["train"]
    X_tax = X_tr[[tr_ids.index(v) for v in tax_ids]]
    noisy = (X_tax + rng.normal(0, 3, X_tax.shape)).astype(np.float32)
    cells = {
        ("clean", "ecg"): X_tax, ("clean", "ppg"): X_tax, ("clean", "ecg_ppg_mean"): X_tax,
        ("lead_off_0.6", "ecg"): noisy, ("lead_off_0.6", "ppg"): X_tax, ("lead_off_0.6", "ecg_ppg_mean"): noisy,
        ("motion_artifact_0.3", "ecg"): X_tax, ("motion_artifact_0.3", "ppg"): noisy,
        ("motion_artifact_0.3", "ecg_ppg_mean"): noisy,
    }
    for (cond, modality), X in cells.items():
        FeatureStore(tmp_path / "taxonomy_store" / cond / "pulsedb_mimic").save(model, modality, 10, "test", tax_ids, X)
    return tmp_path


def test_parse_condition_and_subject():
    from trustbio.eval.stress import parse_condition, subject_of
    assert parse_condition("clean") == ("clean", 0.0)
    assert parse_condition("motion_artifact_0.3") == ("motion_artifact", 0.3)
    assert parse_condition("lead_off_0.6") == ("lead_off", 0.6)
    assert parse_condition("missing_ppg") == ("missing_ppg", 1.0)
    with pytest.raises(ValueError):
        parse_condition("weird")
    assert subject_of("p001049_w12") == "p001049"


def test_load_source_excludes_taxonomy_subjects(synthetic):
    from trustbio.eval.stress import load_pulsedb_labels, load_source
    labels = load_pulsedb_labels(synthetic, "mimic")
    src = load_source(synthetic / "full", "mimic", "m", "ecg", 10, labels,
                      exclude_subjects={"s000", "s001", "s002", "s003", "s004"})
    assert len(src["train"]["ids"]) == 450 and src["train"]["n_excluded"] == 50
    assert not any(v.startswith("s000_") for v in src["train"]["ids"])
    assert len(src["val"]["ids"]) == 100 and src["val"]["n_excluded"] == 0
    assert src["train"]["y"].index.tolist() == src["train"]["ids"].tolist()
    sub = load_source(synthetic / "full", "mimic", "m", "ecg", 10, labels, max_rows=100, seed=1)
    assert len(sub["train"]["ids"]) == 100 and len(sub["val"]["ids"]) == 100


def test_predict_cli_end_to_end(synthetic):
    from scripts.run_stress_predict import main
    out = synthetic / "out"
    rc = main(["--model", "m", "--full-store", str(synthetic / "full"), "--label-cache", str(synthetic),
               "--taxonomy-store", str(synthetic / "taxonomy_store"), "--taxonomy-cohort-cache", str(synthetic / "tax"),
               "--out-dir", str(out), "--sources", "mimic", "--targets", "pulsedb_mimic"])
    assert rc == 0
    pred = pd.read_csv(out / "predictions_m.csv.gz", dtype={"visit_id": str})
    assert set(pred.columns) >= {"model", "modality", "source", "target", "condition", "kind", "severity",
                                 "visit_id", "task", "y_true", "y_pred"}
    conds = pred.groupby("modality")["condition"].apply(set)
    assert conds["ecg"] == {"clean", "lead_off_0.6", "motion_artifact_0.3"}
    assert conds["ecg_ppg_mean"] == {"clean", "lead_off_0.6", "motion_artifact_0.3", "missing_ppg"}
    hr = pred[pred.task == "hr_regression"]

    def r(modality, cond):
        g = hr[(hr.modality == modality) & (hr.condition == cond)]
        return np.corrcoef(g.y_true, g.y_pred)[0, 1]

    assert r("ecg", "clean") > 0.9
    assert r("ecg", "lead_off_0.6") < r("ecg", "clean") - 0.2
    # PPG is untouched by lead-off: identical embeddings give identical predictions
    a = hr[(hr.modality == "ppg") & (hr.condition == "clean")].set_index("visit_id").y_pred
    b = hr[(hr.modality == "ppg") & (hr.condition == "lead_off_0.6")].set_index("visit_id").y_pred
    assert np.allclose(a, b.reindex(a.index))
    assert (pred[pred.condition == "missing_ppg"].severity == 1.0).all()
    assert len(pred[pred.condition == "missing_ppg"]) == 50 * 3
    alphas = json.loads((out / "alphas_m.json").read_text())
    assert alphas["ecg/mimic"]["n_train"] == 450 and alphas["ecg/mimic"]["n_excluded_train"] == 50
