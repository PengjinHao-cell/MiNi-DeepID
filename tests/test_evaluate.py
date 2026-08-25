import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate import collect_predictions, compute_metrics
from model import MiniDeepID


class _TupleDataset(Dataset):
    def __init__(self, n: int = 16):
        torch.manual_seed(0)
        self.images = torch.randn(n, 1, 64, 64)
        self.labels = torch.tensor([i % 10 for i in range(n)])
        self.srcs = torch.arange(n)
        self.paths = [f"img_{i}.png" for i in range(n)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i], self.srcs[i], self.paths[i]


def test_collect_predictions_sample_counts():
    model = MiniDeepID(num_classes=10).to("cpu")
    model.eval()
    loader = DataLoader(_TupleDataset(16), batch_size=8)

    res = collect_predictions(model, loader, "cpu")

    n = 16
    assert len(res["labels"]) == n
    assert len(res["predictions"]) == n
    assert res["probabilities"].shape == (n, 10)
    assert res["embeddings"].shape == (n, 160)
    assert len(res["source_indices"]) == n
    assert len(res["image_paths"]) == n


def test_compute_metrics_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    preds = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    names = [f"ID{i}" for i in range(10)]

    m = compute_metrics(labels, preds, 10, names)

    assert m["random_guess_accuracy"] == 0.10
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert len(m["confusion_matrix"]) == 10
    assert len(m["per_class"]) == 10
    assert m["per_class"][0]["identity"] == "ID0"
    assert "closed_set_warning" in m
