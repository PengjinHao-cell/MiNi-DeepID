import torch

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
