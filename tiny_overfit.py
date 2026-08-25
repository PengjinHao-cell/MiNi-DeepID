"""Tiny-set overfit gate (G8).

Selects 32 train images deterministically (as evenly as possible across the ten
identities), disables random augmentation, sets dropout and weight decay to 0,
and trains for at most 300 epochs. Requires at least 95% train accuracy and
targets 100%. Writes ``outputs/tiny_overfit.json`` and ``outputs/tiny_overfit_curve.png``.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import LFWManifestDataset, build_eval_transform, load_manifest
from model import MiniDeepID
from training import run_epoch, set_reproducible

TINY_N = 32
MAX_EPOCHS = 300
PASS_ACCURACY = 0.95
TARGET_ACCURACY = 1.0


def select_tiny_frame(frame: pd.DataFrame, n: int = TINY_N) -> pd.DataFrame:
    """Deterministically pick ``n`` train rows, balanced across identities."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    base = n // CONFIG.num_classes
    extra = n % CONFIG.num_classes
    parts = []
    for label in range(CONFIG.num_classes):
        sub = train[train["model_label"] == label].sort_values("source_index")
        count = base + (1 if label < extra else 0)
        parts.append(sub.head(count))
    return pd.concat(parts, ignore_index=True)


def _save_curve(history: list[dict]) -> None:
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    accuracies = [h["train_accuracy"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, color="#4C72B0", label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracies, color="#C44E52", label="train accuracy")
    ax2.set_ylabel("accuracy", color="#C44E52")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    fig.suptitle("Tiny-set overfit (32 train images, no augmentation, dropout=0)")
    fig.tight_layout()
    fig.savefig(CONFIG.outputs_dir / "tiny_overfit_curve.png", dpi=150)
    plt.close(fig)


def main() -> int:
    set_reproducible(CONFIG.seed)
    frame = load_manifest(CONFIG)
    tiny = select_tiny_frame(frame, TINY_N)
    if len(tiny) != TINY_N:
        raise RuntimeError(f"expected {TINY_N} tiny images, got {len(tiny)}")

    transform = build_eval_transform(CONFIG)  # no random augmentation
    dataset = LFWManifestDataset(tiny, split="train", transform=transform)
    loader = DataLoader(dataset, batch_size=TINY_N, shuffle=False)

    model = MiniDeepID(num_classes=CONFIG.num_classes, dropout=0.0).to(CONFIG.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=0.0)

    history: list[dict] = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        loss, acc = run_epoch(model, loader, criterion, CONFIG.device, optimizer=optimizer)
        history.append({"epoch": epoch, "loss": loss, "train_accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        if acc >= TARGET_ACCURACY:
            break

    passed = best_acc >= PASS_ACCURACY
    report = {
        "gate": "G8",
        "num_images": TINY_N,
        "max_epochs": MAX_EPOCHS,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_train_accuracy": best_acc,
        "pass_accuracy": PASS_ACCURACY,
        "target_accuracy": TARGET_ACCURACY,
        "passed": passed,
        "device": CONFIG.device,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "augmentation": False,
        "seed": CONFIG.seed,
    }
    CONFIG.outputs_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.outputs_dir / "tiny_overfit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _save_curve(history)

    if passed:
        print(
            f"MINI_DEEPID_TINY_OVERFIT_OK best_train_accuracy={best_acc:.4f} "
            f"epochs={len(history)} target={TARGET_ACCURACY:.2f}"
        )
        return 0
    print(f"MINI_DEEPID_TINY_OVERFIT_FAIL best_train_accuracy={best_acc:.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
