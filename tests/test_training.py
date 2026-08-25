import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import MiniDeepID
from training import run_epoch, set_reproducible


def _make_loader(n: int = 32, batch_size: int = 16, seed: int = 42):
    torch.manual_seed(seed)
    images = torch.randn(n, 1, 64, 64)
    labels = torch.tensor([i % 10 for i in range(n)])
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_set_reproducible_repeats_random_values():
    set_reproducible(42)
    torch_a = torch.randn(5)
    np_a = np.random.rand(5)
    set_reproducible(42)
    torch_b = torch.randn(5)
    np_b = np.random.rand(5)
    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(np_a, np_b)


def test_run_epoch_train_returns_finite_and_updates_params():
    model = MiniDeepID(num_classes=10).to("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _make_loader()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    loss, acc = run_epoch(model, loader, criterion, "cpu", optimizer=optimizer)

    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    changed = any(not torch.equal(before[name], p) for name, p in model.named_parameters())
    assert changed


def test_run_epoch_eval_leaves_params_unchanged():
    model = MiniDeepID(num_classes=10).to("cpu")
    criterion = nn.CrossEntropyLoss()
    loader = _make_loader()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    loss, acc = run_epoch(model, loader, criterion, "cpu", optimizer=None)

    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    unchanged = all(torch.equal(before[name], p) for name, p in model.named_parameters())
    assert unchanged
