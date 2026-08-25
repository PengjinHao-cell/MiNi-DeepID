"""Training CLI for Mini-DeepID (formal training and smoke testing)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import (
    LFWManifestDataset,
    build_eval_transform,
    build_train_transform,
    load_identities,
    load_manifest,
)
from gate_status import record_gate
from model import MiniDeepID
from training import run_epoch, save_checkpoint_atomic, set_reproducible, write_history_json


def train_model(config=CONFIG, max_epochs: int | None = None, run_dir: Path | None = None) -> dict:
    set_reproducible(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to fall back to CPU.")

    frame = load_manifest(config)
    identities = load_identities(config)
    train_ds = LFWManifestDataset(frame, "train", build_train_transform(config))
    val_ds = LFWManifestDataset(frame, "val", build_eval_transform(config))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = MiniDeepID(
        num_classes=config.num_classes,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    epochs = min(max_epochs if max_epochs is not None else config.max_epochs, config.max_epochs)
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoints_dir / "mini_deepid_best.pth"

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, config.device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, config.device, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience = 0
            payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val_acc,
                "identities": identities,
                "seed": config.seed,
                "config": {
                    "num_classes": config.num_classes,
                    "embedding_dim": config.embedding_dim,
                    "image_size": config.image_size,
                    "dropout": config.dropout,
                },
            }
            save_checkpoint_atomic(checkpoint_path, payload)
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"early stopping at epoch {epoch} (patience={patience})")
                break

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_history_json(history, run_dir / "history.json")
        _save_curves(history, run_dir)

    return {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }


def _save_curves(history, run_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Loss curve")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_accuracy"] for h in history], label="train accuracy")
    ax.plot(epochs, [h["val_accuracy"] for h in history], label="val accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.set_title("Accuracy curve")
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Mini-DeepID.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum epochs (capped at 80; use a small value for smoke testing).",
    )
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = CONFIG.outputs_dir / run_id

    result = train_model(CONFIG, max_epochs=args.epochs, run_dir=run_dir)
    is_smoke = args.epochs is not None and args.epochs < CONFIG.max_epochs
    record_gate(
        "G9" if is_smoke else "G10",
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
