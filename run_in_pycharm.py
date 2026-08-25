"""No-argument PyCharm launcher for the user-owned formal training run (Gate G10).

Preflight checks print the interpreter, GPU, data count, and passed gates, then
``PYCHARM_TRAIN_READY``. With ``--check-only`` it stops there; otherwise it
prints ``PYCHARM_FORMAL_TRAIN_START`` and runs the same training function as
``train.py``. It never imports or calls final-evaluation code.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import CONFIG
from dataset import load_manifest
from gate_status import list_passed_gates, record_gate, require_gate_passed


def _check_interpreter() -> None:
    actual = str(Path(sys.executable).resolve()).lower()
    expected = str((CONFIG.project_root / ".venv" / "Scripts" / "python.exe").resolve()).lower()
    if actual != expected:
        raise RuntimeError(
            f"Wrong interpreter: {sys.executable}. Expected {expected}. "
            "Select the project .venv interpreter in PyCharm."
        )


def _check_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to fall back to CPU.")
    name = torch.cuda.get_device_name(0)
    if "RTX 5060" not in name:
        raise RuntimeError(f"Unexpected GPU: {name}")
    return name


def _check_no_final_receipt() -> None:
    receipt = CONFIG.outputs_dir / "final_test_receipt.json"
    if receipt.exists():
        raise RuntimeError("final_test_receipt.json already exists; refusing to train again.")


def preflight() -> dict:
    _check_interpreter()
    gpu = _check_cuda()
    frame = load_manifest(CONFIG)
    require_gate_passed("G8")  # tiny overfit
    require_gate_passed("G9")  # smoke train
    _check_no_final_receipt()
    return {
        "executable": sys.executable,
        "gpu": gpu,
        "samples": int(len(frame)),
        "passed_gates": list_passed_gates(),
    }


def main() -> int:
    info = preflight()
    print(f"sys.executable={info['executable']}")
    print(f"gpu={info['gpu']}")
    print(f"data_samples={info['samples']}")
    print(f"passed_gates={','.join(info['passed_gates'])}")
    print("PYCHARM_TRAIN_READY")

    if "--check-only" in sys.argv:
        return 0

    from train import train_model

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = CONFIG.outputs_dir / run_id

    print("PYCHARM_FORMAL_TRAIN_START")
    result = train_model(CONFIG, max_epochs=None, run_dir=run_dir)
    record_gate(
        "G10",
        "passed",
        {
            "best_epoch": result["best_epoch"],
            "best_val_accuracy": result["best_val_accuracy"],
            "epochs": len(result["history"]),
        },
        [f"outputs/{run_id}/history.json", "checkpoints/mini_deepid_best.pth"],
    )
    print(
        f"MINI_DEEPID_TRAIN_OK best_epoch={result['best_epoch']} "
        f"best_val_accuracy={result['best_val_accuracy']:.4f} run_dir={run_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
