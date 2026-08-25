import json

import pytest

from gate_status import list_passed_gates, record_gate, require_gate_passed


def test_record_preserves_earlier_attempt(tmp_path):
    path = tmp_path / "gate_status.json"
    record_gate("G0", "failed", {"x": 1}, ["a.txt"], path=path)
    record_gate("G0", "passed", {"x": 2}, ["b.txt"], path=path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["gates"]) == 2
    assert data["gates"][0]["status"] == "failed"
    assert data["gates"][1]["status"] == "passed"
    assert data["gates"][0]["metrics"] == {"x": 1}
    assert data["gates"][1]["evidence_paths"] == ["b.txt"]


def test_record_rejects_invalid_gate():
    with pytest.raises(ValueError):
        record_gate("G15", "passed")
    with pytest.raises(ValueError):
        record_gate("G0", "pending")


def test_require_gate_passed(tmp_path):
    path = tmp_path / "gate_status.json"
    with pytest.raises(RuntimeError):
        require_gate_passed("G0", path=path)
    record_gate("G0", "passed", path=path)
    require_gate_passed("G0", path=path)  # must not raise


def test_list_passed_gates(tmp_path):
    path = tmp_path / "gate_status.json"
    record_gate("G0", "passed", path=path)
    record_gate("G1", "passed", path=path)
    record_gate("G2", "failed", path=path)
    assert list_passed_gates(path=path) == ["G0", "G1"]
