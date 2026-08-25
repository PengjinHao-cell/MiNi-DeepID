"""PCA embedding visualization (Gate G13).

Fits PCA(n_components=2) on train embeddings ONLY, then transforms the saved
test embeddings from the one final test pass. Never instantiates or iterates a
test DataLoader and never fits PCA on test embeddings.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import LFWManifestDataset, build_eval_transform, load_identities, load_manifest
from gate_status import record_gate
from model import MiniDeepID
from training import load_checkpoint


def collect_embeddings(model, loader, device):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for images, batch_labels, *_ in loader:
            images = images.to(device)
            embedding, _ = model(images)
            embs.append(embedding.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    return np.concatenate(embs), np.concatenate(labels)


def main() -> int:
    frame = load_manifest(CONFIG)
    identities = load_identities(CONFIG)

    model = MiniDeepID(
        num_classes=CONFIG.num_classes,
        embedding_dim=CONFIG.embedding_dim,
        dropout=CONFIG.dropout,
    ).to(CONFIG.device)
    load_checkpoint(
        CONFIG.checkpoints_dir / "mini_deepid_best.pth", model, expected_identities=identities
    )

    train_ds = LFWManifestDataset(frame, "train", build_eval_transform(CONFIG))
    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=False)
    train_emb, train_labels = collect_embeddings(model, train_loader, CONFIG.device)

    # Fit PCA on train embeddings only.
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_emb)

    # Transform the saved test embeddings (no test DataLoader, no test fit).
    test = np.load(CONFIG.outputs_dir / "test_embeddings.npz")
    test_emb = test["embeddings"]
    test_labels = test["labels"]

    train_2d = pca.transform(train_emb)
    test_2d = pca.transform(test_emb)
    explained = pca.explained_variance_ratio_

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 8))
    for label in range(CONFIG.num_classes):
        ax.scatter(
            train_2d[train_labels == label, 0],
            train_2d[train_labels == label, 1],
            marker="o",
            s=16,
            color=cmap(label),
            alpha=0.35,
        )
        ax.scatter(
            test_2d[test_labels == label, 0],
            test_2d[test_labels == label, 1],
            marker="x",
            s=48,
            color=cmap(label),
        )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=8,
            label=identities[i],
        )
        for i in range(CONFIG.num_classes)
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", title="identities")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title("DeepID 160D embeddings PCA (fit=train, transform=test; o=train, x=test)")
    fig.tight_layout()
    fig.savefig(CONFIG.outputs_dir / "embeddings_pca.png", dpi=150)
    plt.close(fig)

    record_gate(
        "G13",
        "passed",
        {"pca": "fit=train transform=test", "samples": int(len(test_labels))},
        ["outputs/embeddings_pca.png"],
    )

    print(
        f"MINI_DEEPID_PCA_OK fit=train transform=test dimensions=160->2 samples={len(test_labels)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
