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
