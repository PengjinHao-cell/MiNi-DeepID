"""Append-only G0-G14 gate evidence ledger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIG

VALID_GATES = {f"G{i}" for i in range(15)}  # G0..G14
VALID_STATUS = {"passed", "failed"}


def default_path() -> Path:
    return CONFIG.outputs_dir / "gate_status.json"


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"gates": []}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_gate(gate: str, status: str, metrics=None, evidence_paths=None, path=None) -> dict:
    """Append a timestamped attempt. Earlier attempts are never removed."""
    if gate not in VALID_GATES:
        raise ValueError(f"invalid gate: {gate}")
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status: {status}")
    path = Path(path) if path is not None else default_path()
    data = _load(path)
    entry = {
        "gate": gate,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
        "evidence_paths": evidence_paths or [],
    }
    data.setdefault("gates", []).append(entry)
    _write(path, data)
    return entry


def require_gate_passed(gate: str, path=None) -> None:
    """Raise unless the latest recorded status for ``gate`` is ``passed``."""
    if gate not in VALID_GATES:
        raise ValueError(f"invalid gate: {gate}")
    path = Path(path) if path is not None else default_path()
    data = _load(path)
    for entry in data.get("gates", []):
        if entry["gate"] == gate and entry["status"] == "passed":
            return
    raise RuntimeError(f"gate {gate} has not passed")


def list_passed_gates(path=None) -> list[str]:
    """Return the sorted list of gate ids that have at least one ``passed`` entry."""
    path = Path(path) if path is not None else default_path()
    data = _load(path)
    passed: list[str] = []
    for entry in data.get("gates", []):
        if entry["status"] == "passed" and entry["gate"] not in passed:
            passed.append(entry["gate"])
    return sorted(passed, key=lambda gate: int(gate[1:]))
