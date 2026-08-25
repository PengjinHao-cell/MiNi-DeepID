"""One-shot closed-set evaluation for the frozen best checkpoint (Gate G12).

Runs exactly one inference pass over the frozen test split and writes
``final_test_receipt.json`` plus metrics, confusion matrix, predictions grid, and
``test_embeddings.npz``. Refuses to run if a receipt already exists.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import LFWManifestDataset, build_eval_transform, load_identities, load_manifest
from gate_status import record_gate
from model import MiniDeepID
from training import load_checkpoint

CLOSED_SET_WARNING = "WARNING: closed-set result; unknown identities are not supported."


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_predictions(model, loader, device) -> dict:
    """Run inference and collect labels, predictions, probabilities, embeddings,
    source indices, and image paths."""
    model.eval()
    labels, preds, probs, embs, srcs, paths = [], [], [], [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_srcs, batch_paths in loader:
            images = images.to(device)
            embedding, logits = model(images)
            labels.append(batch_labels.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            embs.append(embedding.cpu().numpy())
            srcs.append(batch_srcs.cpu().numpy())
            paths.extend(batch_paths)
    return {
        "labels": np.concatenate(labels),
        "predictions": np.concatenate(preds),
        "probabilities": np.concatenate(probs),
        "embeddings": np.concatenate(embs),
        "source_indices": np.concatenate(srcs),
        "image_paths": list(paths),
    }


def compute_metrics(labels, predictions, num_classes, identity_names) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    accuracy = float(accuracy_score(labels, predictions))
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    per_class = [
        {
            "label": int(i),
            "identity": identity_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    ]
    return {
        "random_guess_accuracy": 0.10,
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "closed_set_warning": CLOSED_SET_WARNING,
    }


def _save_confusion_matrix(cm, identity_names, out_path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(identity_names)))
    ax.set_yticks(range(len(identity_names)))
    ax.set_xticklabels(identity_names, rotation=45, ha="right")
    ax.set_yticklabels(identity_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("LFW closed-set confusion matrix")
    fig.colorbar(image, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_predictions_grid(results, identity_names, config, out_path, n: int = 12) -> None:
    labels = results["labels"]
    preds = results["predictions"]
    probs = results["probabilities"]
    paths = results["image_paths"]
    n = min(n, len(labels))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for k in range(n):
        ax = axes[k // cols, k % cols]
        img = np.asarray(Image.open(config.project_root / paths[k]).convert("L"))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        truth = identity_names[labels[k]]
        pred = identity_names[preds[k]]
        conf = float(probs[k][preds[k]])
        color = "green" if labels[k] == preds[k] else "red"
        ax.set_title(f"T:{truth}\nP:{pred} ({conf:.2f})", fontsize=6, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle("LFW closed-set test predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    receipt_path = CONFIG.outputs_dir / "final_test_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(
            "final_test_receipt.json already exists; refusing a second final evaluation."
        )

    frame = load_manifest(CONFIG)
    identities = load_identities(CONFIG)
    test_ds = LFWManifestDataset(frame, "test", build_eval_transform(CONFIG))
    test_loader = DataLoader(test_ds, batch_size=CONFIG.batch_size, shuffle=False)

    model = MiniDeepID(
        num_classes=CONFIG.num_classes,
        embedding_dim=CONFIG.embedding_dim,
        dropout=CONFIG.dropout,
    ).to(CONFIG.device)
    checkpoint_path = CONFIG.checkpoints_dir / "mini_deepid_best.pth"
    load_checkpoint(checkpoint_path, model, expected_identities=identities)

    results = collect_predictions(model, test_loader, CONFIG.device)
    n = len(results["labels"])
    expected_n = CONFIG.num_classes * CONFIG.test_per_class
    if n != expected_n:
        raise RuntimeError(f"expected {expected_n} test samples, got {n}")

    metrics = compute_metrics(
        results["labels"], results["predictions"], CONFIG.num_classes, identities
    )
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["identities"] = identities

    CONFIG.outputs_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.outputs_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]), identities, CONFIG.outputs_dir / "confusion_matrix.png"
    )
    _save_predictions_grid(results, identities, CONFIG, CONFIG.outputs_dir / "predictions.png")
    np.savez(
        CONFIG.outputs_dir / "test_embeddings.npz",
        embeddings=results["embeddings"],
        labels=results["labels"],
    )

    receipt = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "split_manifest_sha256": sha256(CONFIG.split_manifest_path),
        "protocol": (
            "Mini-DeepID closed-set identification; 10 identities x 50; seed 42; "
            "train 350 / val 70 / test 80; grayscale 1x64x64; 160D embedding"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "random_guess_accuracy": metrics["random_guess_accuracy"],
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    record_gate(
        "G12",
        "passed",
        {
            "test_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "samples": n,
        },
        ["outputs/final_test_receipt.json", "outputs/metrics.json"],
    )

    print(CLOSED_SET_WARNING)
    print(f"MINI_DEEPID_EVAL_OK samples={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
