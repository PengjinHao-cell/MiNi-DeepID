"""Reproducibility and training-loop utilities for Mini-DeepID."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_reproducible(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and all CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch.

    With an optimizer the model trains (zero-grad, backward, step); without one
    it evaluates under ``no_grad``. Returns ``(average_loss, accuracy)``.
    """
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        if optimizer is not None:
            optimizer.zero_grad()
            _, logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss during training")
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _, logits = model(images)
                loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))

    if total_samples == 0:
        raise RuntimeError("empty data loader")
    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint_atomic(path, payload) -> None:
    """Write a sibling ``.tmp`` file and atomically ``replace`` the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, expected_identities=None):
    """Load a checkpoint, optionally verify the identity mapping, and restore
    model/optimizer state with strict loading."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if expected_identities is not None:
        stored = payload.get("identities")
        if stored != expected_identities:
            raise ValueError("checkpoint identity mapping mismatch")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def write_history_json(history, path) -> None:
    """Write one JSON record per epoch, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
