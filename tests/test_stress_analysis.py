import numpy as np
import pandas as pd


def _pred(seed=0):
    """3 models of decreasing quality x 3 modalities, one source/target/task.
    lead_off_0.6 harms ecg + fusion (noisier prediction), motion_artifact_0.3
    harms ppg + fusion; the untouched modality is IDENTICAL to clean."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i:02d}_w{w}" for i in range(20) for w in range(5)]
    y = 70 + rng.normal(0, 10, len(ids))
    rows = []
    for m, noise in (("A", 1.0), ("B", 10.0), ("C", 25.0)):   # gaps wide enough that a fault cannot reorder them
        base = {mod: y + rng.normal(0, noise, len(ids)) for mod in ("ecg", "ppg", "ecg_ppg_mean")}
        for mod in base:
            for cond, kind, sev, hit in (("clean", "clean", 0.0, ()),
                                         ("lead_off_0.6", "lead_off", 0.6, ("ecg", "ecg_ppg_mean")),
                                         ("motion_artifact_0.3", "motion_artifact", 0.3, ("ppg", "ecg_ppg_mean"))):
                pred = base[mod] + (rng.normal(0, 8, len(ids)) if mod in hit else 0.0)
                rows += [dict(model=m, modality=mod, source="mimic", target="pulsedb_mimic", condition=cond,
                              kind=kind, severity=sev, visit_id=v, task="hr_regression", y_true=yy, y_pred=p)
                         for v, yy, p in zip(ids, y, pred)]
        # missing_ppg for fusion only: the fusion probe fed the ECG-only vector
        for v, yy, p in zip(ids, y, base["ecg"] + 1.0):
            rows.append(dict(model=m, modality="ecg_ppg_mean", source="mimic", target="pulsedb_mimic",
                             condition="missing_ppg", kind="missing_ppg", severity=1.0, visit_id=v,
                             task="hr_regression", y_true=yy, y_pred=p))
    return pd.DataFrame(rows)


def test_score_table_and_deltas():
    from trustbio.eval.stress_analysis import score_table
    s = score_table(_pred())
    a = s[(s.model == "A") & (s.modality == "ecg")].set_index("condition")
    assert a.loc["clean", "r"] > 0.95 and a.loc["clean", "delta_r"] == 0.0
    assert a.loc["lead_off_0.6", "delta_r"] < -0.05 and a.loc["lead_off_0.6", "delta_mae"] > 1.0
    assert abs(a.loc["motion_artifact_0.3", "delta_r"]) < 1e-12          # ecg untouched by motion
    assert {"n", "r", "mae", "r_clean", "mae_clean", "delta_r", "delta_mae"} <= set(s.columns)


def test_paired_harm_is_zero_for_untouched_modality():
    from trustbio.eval.stress_analysis import paired_harm
    h = paired_harm(_pred())
    assert "clean" not in set(h.condition)
    ecg_motion = h[(h.modality == "ecg") & (h.condition == "motion_artifact_0.3")]
    assert np.allclose(ecg_motion.harm, 0.0)
    ecg_lead = h[(h.modality == "ecg") & (h.condition == "lead_off_0.6") & (h.model == "A")]
    assert ecg_lead.harm.mean() > 1.0 and len(ecg_lead) == 100


def test_rank_stability_gap_and_fusion():
    from trustbio.eval.stress_analysis import fusion_table, gap_table, rank_stability, score_table
    s = score_table(_pred())
    st = rank_stability(s)
    row = st[(st.modality == "ecg") & (st.condition == "lead_off_0.6")].iloc[0]
    assert row.n_models == 3 and row.spearman == 1.0 and row.top_clean == "A"
    g = gap_table(s, domain="A", ts="C")
    ge = g[g.modality == "ecg"].set_index("condition")
    assert ge.loc["clean", "gap"] > 0 and (ge.gap_clean == ge.loc["clean", "gap"]).all()
    f = fusion_table(s)
    fa = f[f.model == "A"].set_index("condition")
    assert fa.loc["lead_off_0.6", "best_unimodal"] == "ppg"           # ppg untouched by lead-off
    assert fa.loc["motion_artifact_0.3", "best_unimodal"] == "ecg"
    assert fa.loc["missing_ppg", "best_unimodal"] == "ecg"
    ecg_clean_r = s[(s.model == "A") & (s.modality == "ecg") & (s.condition == "clean")].r.iloc[0]
    assert np.isclose(fa.loc["missing_ppg", "fusion_minus_best"], fa.loc["missing_ppg", "r_fusion"] - ecg_clean_r)


def test_harm_coverage_math():
    from trustbio.eval.stress_analysis import harm_coverage, paired_harm
    h = paired_harm(_pred())
    key = h[(h.model == "A") & (h.modality == "ecg") & (h.condition == "lead_off_0.6")]
    flags = pd.DataFrame(dict(dataset="pulsedb_mimic", condition="lead_off_0.6", visit_id=key.visit_id.to_numpy(),
                              det_drop=(key.harm > 0).to_numpy(), det_outlier=False,
                              det_supervised=(key.harm > 5.0).to_numpy()))
    cov = harm_coverage(h, flags).set_index(["model", "modality", "condition", "rule"])
    drop = cov.loc[("A", "ecg", "lead_off_0.6", "drop")]
    assert np.isclose(drop.harm_share_caught, 1.0) and drop.material_recall == 1.0 and drop.n == 100
    outl = cov.loc[("A", "ecg", "lead_off_0.6", "outlier")]
    assert outl.harm_share_caught == 0.0 and outl.material_recall == 0.0 and outl.detection_rate == 0.0
    sup = cov.loc[("A", "ecg", "lead_off_0.6", "supervised")]
    assert sup.material_recall == 1.0 and 0.0 < sup.harm_share_caught < 1.0
    assert ("A", "ecg", "motion_artifact_0.3", "drop") not in cov.index     # no flags for that condition
