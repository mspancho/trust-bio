import pytest

from scripts.make_manifest import build_lines, parse_chunks


def test_parse_chunks_accepts_dataset_eq_n():
    assert parse_chunks(["pulsedb_mimic=8", "pulsedb_vital=3"]) == {
        "pulsedb_mimic": 8, "pulsedb_vital": 3}
    assert parse_chunks([]) == {}


@pytest.mark.parametrize("bad", ["pulsedb_mimic", "pulsedb_mimic=0", "nope=3", "pulsedb_mimic=x"])
def test_parse_chunks_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_chunks([bad])


def test_build_lines_unchunked_and_chunked():
    lines = build_lines(["m1", "m2"], ["pulsedb_mimic", "pulsedb_vital"], 10, {"pulsedb_mimic": 2})
    assert lines == [
        "m1 pulsedb_mimic 10 0 2", "m1 pulsedb_mimic 10 1 2", "m1 pulsedb_vital 10",
        "m2 pulsedb_mimic 10 0 2", "m2 pulsedb_mimic 10 1 2", "m2 pulsedb_vital 10",
    ]


def test_build_lines_emits_condition_fields():
    conds = [("clean", None, None), ("lead_off_0.3", "lead_off", 0.3)]
    lines = build_lines(["m"], ["pulsedb_vital"], 10, {}, conditions=conds)
    assert lines == ["m pulsedb_vital 10 cond=clean",
                     "m pulsedb_vital 10 cond=lead_off_0.3 kind=lead_off sev=0.3"]


def test_build_lines_condition_fields_follow_chunk_fields():
    lines = build_lines(["m"], ["pulsedb_mimic"], 10, {"pulsedb_mimic": 2},
                        conditions=[("motion_artifact_0.1", "motion_artifact", 0.1)])
    assert lines[0] == "m pulsedb_mimic 10 0 2 cond=motion_artifact_0.1 kind=motion_artifact sev=0.1"


def test_parse_degrade_specs():
    from scripts.make_manifest import parse_degrade
    assert parse_degrade(["motion_artifact=0.3", "lead_off=0.6"]) == [
        ("motion_artifact_0.3", "motion_artifact", 0.3), ("lead_off_0.6", "lead_off", 0.6)]
    with pytest.raises(ValueError):
        parse_degrade(["nope=0.3"])
    with pytest.raises(ValueError):
        parse_degrade(["lead_off=2"])
