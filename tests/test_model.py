import torch
import torch.nn.functional as F

from config import CONFIG
from model import MiniDeepID


def test_forward_shapes():
    model = MiniDeepID(num_classes=10, embedding_dim=160)
    model.eval()
    embedding, logits = model(torch.randn(4, 1, 64, 64))
    assert embedding.shape == (4, 160)
    assert logits.shape == (4, 10)


def test_forward_finite():
    model = MiniDeepID(num_classes=10, embedding_dim=160)
    model.eval()
    embedding, logits = model(torch.randn(4, 1, 64, 64))
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(logits).all()


def test_save_load_deterministic():
    model = MiniDeepID(num_classes=10, embedding_dim=160)
    model.eval()
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        emb_a, logits_a = model(x)

    reloaded = MiniDeepID(num_classes=10, embedding_dim=160)
    reloaded.load_state_dict(model.state_dict())
    reloaded.eval()
    with torch.no_grad():
        emb_b, logits_b = reloaded(x)

    assert torch.equal(emb_a, emb_b)
    assert torch.equal(logits_a, logits_b)


def test_gradients_finite():
    model = MiniDeepID(num_classes=10, embedding_dim=160).to(CONFIG.device)
    model.train()
    x = torch.randn(4, 1, 64, 64, device=CONFIG.device)
    y = torch.randint(0, 10, (4,), device=CONFIG.device)
    _, logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    assert torch.isfinite(loss).all()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == sum(1 for _ in model.parameters())  # every param trainable
    assert len(trainable) > 0
    for p in trainable:
        assert p.grad is not None, "a trainable parameter has no gradient"
        assert torch.isfinite(p.grad).all(), "a trainable parameter has non-finite gradient"
