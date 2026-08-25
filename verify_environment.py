"""Strict CUDA environment verifier for Mini-DeepID (Gate G2).

Proves real CUDA computation on ``cuda:0`` rather than just checking
``torch.cuda.is_available()``:

* reads the GPU name (must contain "RTX 5060");
* records the CUDA runtime and compute capability;
* runs a 2000x2000 matrix multiply with backward() and checks finiteness;
* runs a model forward pass, computes cross-entropy loss, calls backward(),
  and checks every parameter receives a finite gradient.

Writes ``environment.txt`` and ``outputs/environment_report.json``. Raises
``RuntimeError`` and never falls back to CPU when CUDA is unavailable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SEED = 42


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None
    return None


def _cudnn_version() -> str | None:
    try:
        version = torch.backends.cudnn.version()
        return str(version) if version else None
    except Exception:
        return None


def collect_versions() -> dict:
    import numpy
    import pandas
    import PIL
    import sklearn
    import matplotlib
    import seaborn
    import torchvision

    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
    }


def run_matmul_check(device: str) -> dict:
    torch.manual_seed(SEED)
    a = torch.randn(2000, 2000, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(2000, 2000, device=device, dtype=torch.float32, requires_grad=True)
    s = (a @ b).sum()
    s.backward()
    result_finite = bool(torch.isfinite(s).item())
    grad_a_finite = bool(torch.isfinite(a.grad).all().item())
    grad_b_finite = bool(torch.isfinite(b.grad).all().item())
    passed = result_finite and grad_a_finite and grad_b_finite
    return {
        "passed": passed,
        "result_finite": result_finite,
        "grad_a_finite": grad_a_finite,
        "grad_b_finite": grad_b_finite,
    }


def run_model_check(device: str) -> dict:
    import torch.nn.functional as functional

    from model import MiniDeepID

    torch.manual_seed(SEED)
    model = MiniDeepID(num_classes=10).to(device)
    model.train()
    x = torch.randn(2, 1, 64, 64, device=device)
    y = torch.tensor([3, 7], device=device)
    _, logits = model(x)
    loss = functional.cross_entropy(logits, y)
    loss.backward()
    grads_finite = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all().item())
        for p in model.parameters()
    )
    num_params = sum(p.numel() for p in model.parameters())
    loss_finite = bool(torch.isfinite(loss).item())
    return {
        "passed": loss_finite and grads_finite,
        "loss_finite": loss_finite,
        "gradients_finite": grads_finite,
        "num_params": num_params,
    }


def build_report(matmul: dict, model: dict) -> dict:
    capability = tuple(torch.cuda.get_device_capability(0))
    device_name = torch.cuda.get_device_name(0)
    return {
        "gate": "G2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": {"random_seed": SEED, "device": "cuda:0"},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": collect_versions(),
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "device_index": 0,
            "device_name": device_name,
            "device_count": torch.cuda.device_count(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": _cudnn_version(),
            "driver_version": _driver_version(),
            "compute_capability": list(capability),
        },
        "checks": {
            "matmul_2000x2000": matmul,
            "model_forward_backward": model,
        },
    }


def write_environment_txt(report: dict) -> None:
    c = report["cuda"]
    p = report["packages"]
    py = report["python"]
    checks = report["checks"]
    lines = [
        "Mini-DeepID environment report (Gate G2)",
        f"timestamp={report['timestamp']}",
        f"random_seed={report['protocol']['random_seed']}",
        f"device={report['protocol']['device']}",
        f"python={py['version']}",
        f"python_executable={py['executable']}",
        f"torch={p['torch']}",
        f"torchvision={p['torchvision']}",
        f"numpy={p['numpy']}",
        f"pandas={p['pandas']}",
        f"pillow={p['pillow']}",
        f"scikit_learn={p['scikit_learn']}",
        f"matplotlib={p['matplotlib']}",
        f"seaborn={p['seaborn']}",
        f"cuda_available={c['available']}",
        f"cuda_runtime={c['cuda_runtime']}",
        f"cudnn_version={c['cudnn_version']}",
        f"driver_version={c['driver_version']}",
        f"device_name={c['device_name']}",
        f"compute_capability={c['compute_capability']}",
        f"device_count={c['device_count']}",
        f"matmul_2000x2000_passed={checks['matmul_2000x2000']['passed']}",
        f"model_forward_backward_passed={checks['model_forward_backward']['passed']}",
        f"model_gradients_finite={checks['model_forward_backward']['gradients_finite']}",
    ]
    (ROOT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to fall back to CPU.")

    device = "cuda:0"
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5060" not in device_name:
        raise RuntimeError(
            f"Unexpected GPU name {device_name!r}; expected one containing 'RTX 5060'."
        )

    print(f"device_name={device_name}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"compute_capability={torch.cuda.get_device_capability(0)}")

    matmul = run_matmul_check(device)
    if not matmul["passed"]:
        raise RuntimeError("2000x2000 matmul/backward check failed on cuda:0.")
    print("MINI_DEEPID_CUDA_TENSOR_OK device=cuda:0")

    model = run_model_check(device)
    if not model["passed"]:
        raise RuntimeError("Model forward/backward check failed on cuda:0.")

    report = build_report(matmul, model)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "environment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_environment_txt(report)

    print("MINI_DEEPID_CUDA_SMOKE_OK device=cuda:0")


if __name__ == "__main__":
    main()
