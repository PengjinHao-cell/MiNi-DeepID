import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import MiniDeepID
from training import load_checkpoint, run_epoch, save_checkpoint_atomic, set_reproducible


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


def _identities() -> list[str]:
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def test_checkpoint_round_trip(tmp_path):
    model = MiniDeepID(num_classes=10).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _, logits = model(torch.randn(2, 1, 64, 64))
    F.cross_entropy(logits, torch.tensor([1, 2])).backward()
    optimizer.step()

    identities = _identities()
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": 1,
        "best_val_accuracy": 0.5,
        "identities": identities,
        "seed": 42,
    }
    path = tmp_path / "ckpt.pth"
    save_checkpoint_atomic(path, payload)

    reloaded = MiniDeepID(num_classes=10).to("cpu")
    reloaded_opt = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, reloaded, optimizer=reloaded_opt, expected_identities=identities)
    assert loaded["epoch"] == 1
    assert loaded["identities"] == identities

    model.eval()
    reloaded.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        _, logits_a = model(x)
        _, logits_b = reloaded(x)
    assert torch.equal(logits_a, logits_b)


def test_checkpoint_identity_mismatch_raises(tmp_path):
    model = MiniDeepID(num_classes=10).to("cpu")
    payload = {"model_state": model.state_dict(), "identities": _identities()}
    path = tmp_path / "ckpt.pth"
    save_checkpoint_atomic(path, payload)

    other = MiniDeepID(num_classes=10).to("cpu")
    wrong = ["A"] * 10
    with pytest.raises(ValueError):
        load_checkpoint(path, other, expected_identities=wrong)
