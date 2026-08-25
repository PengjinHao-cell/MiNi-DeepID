import numpy as np
import pandas as pd
import pytest
import torch

from dataset import (
    LFWManifestDataset,
    assert_splits_disjoint,
    build_eval_transform,
    build_train_transform,
    load_manifest,
    sample_and_split_indices,
    select_top_labels,
    validate_exported_images,
    validate_manifest,
)


def _synthetic_targets() -> np.ndarray:
    # Counts: 70, 65, 60, 58, 57, 56, 55, 54, 53, 52, 51 (labels 0..10).
    counts = [70, 65, 60, 58, 57, 56, 55, 54, 53, 52, 51]
    parts = [np.full(count, label, dtype=np.int64) for label, count in enumerate(counts)]
    return np.concatenate(parts)


def test_select_top_labels_selects_ten_largest():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    assert len(selected) == 10
    assert selected == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_sample_and_split_is_deterministic_and_balanced():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    rows_a = sample_and_split_indices(targets, selected, seed=42)
    rows_b = sample_and_split_indices(targets, selected, seed=42)
    assert rows_a == rows_b

    frame = pd.DataFrame(rows_a)
    assert len(frame) == 10 * 50
    for label in range(10):
        sub = frame[frame["model_label"] == label]
        assert len(sub) == 50
        assert (sub["split"] == "train").sum() == 35
        assert (sub["split"] == "val").sum() == 7
        assert (sub["split"] == "test").sum() == 8
    assert not frame["source_index"].duplicated().any()


def test_validate_manifest_accepts_valid():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    validate_manifest(frame, num_classes=10)  # must not raise


def test_validate_manifest_rejects_duplicate_source_index():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    frame.loc[0, "source_index"] = frame.loc[1, "source_index"]
    with pytest.raises(ValueError):
        validate_manifest(frame, num_classes=10)


def test_validate_manifest_rejects_non_contiguous_labels():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    frame.loc[frame["model_label"] == 9, "model_label"] = 99
    with pytest.raises(ValueError):
        validate_manifest(frame, num_classes=10)


def test_validate_manifest_rejects_bad_split_name():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    frame.loc[0, "split"] = "oops"
    with pytest.raises(ValueError):
        validate_manifest(frame, num_classes=10)


def test_validate_manifest_rejects_wrong_per_class_count():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    with pytest.raises(ValueError):
        validate_manifest(frame.iloc[:-1].copy(), num_classes=10)


def test_validate_manifest_rejects_wrong_split_count():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    idx = frame.index[(frame["model_label"] == 0) & (frame["split"] == "val")][0]
    frame.loc[idx, "split"] = "test"
    with pytest.raises(ValueError):
        validate_manifest(frame, num_classes=10)


def test_assert_splits_disjoint_rejects_overlap():
    targets = _synthetic_targets()
    selected = select_top_labels(targets, num_classes=10, samples_per_class=50)
    frame = pd.DataFrame(sample_and_split_indices(targets, selected, seed=42))
    assert_splits_disjoint(frame)  # valid: no raise

    # inject an overlap: move one val source_index into train
    moved = int(frame.loc[frame["split"] == "val", "source_index"].iloc[0])
    frame.loc[frame["source_index"] == moved, "split"] = "train"
    # keep a duplicate row so the source index appears in both splits
    dup = frame[frame["source_index"] == moved].iloc[[0]].copy()
    dup["split"] = "val"
    overlap = pd.concat([frame, dup], ignore_index=True)
    with pytest.raises(ValueError):
        assert_splits_disjoint(overlap)


def test_load_manifest_reads_frozen_split():
    frame = load_manifest()
    assert len(frame) == 500
    assert (frame["split"] == "train").sum() == 350
    assert (frame["split"] == "val").sum() == 70
    assert (frame["split"] == "test").sum() == 80
    assert frame["model_label"].nunique() == 10

    train_idx = set(frame.loc[frame["split"] == "train", "source_index"])
    val_idx = set(frame.loc[frame["split"] == "val", "source_index"])
    test_idx = set(frame.loc[frame["split"] == "test", "source_index"])
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_validate_exported_images(tmp_path):
    from PIL import Image

    from config import CONFIG

    img_dir = tmp_path / "data" / "processed" / "X"
    img_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L").save(img_dir / "1.png")

    frame = pd.DataFrame([{"image_path": "data/processed/X/1.png"}])
    assert validate_exported_images(CONFIG, frame, project_root=tmp_path) == 1

    missing = pd.DataFrame([{"image_path": "data/processed/X/nope.png"}])
    with pytest.raises(ValueError):
        validate_exported_images(CONFIG, missing, project_root=tmp_path)

    Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(img_dir / "2.png")
    wrong_size = pd.DataFrame([{"image_path": "data/processed/X/2.png"}])
    with pytest.raises(ValueError):
        validate_exported_images(CONFIG, wrong_size, project_root=tmp_path)

    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(img_dir / "3.png")
    not_gray = pd.DataFrame([{"image_path": "data/processed/X/3.png"}])
    with pytest.raises(ValueError):
        validate_exported_images(CONFIG, not_gray, project_root=tmp_path)


def test_transforms_and_dataset(tmp_path):
    from PIL import Image

    from config import CONFIG

    img_dir = tmp_path / "data" / "processed" / "X"
    img_dir.mkdir(parents=True)
    arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(img_dir / "1.png")

    # train transform returns finite float32 [1,64,64]
    train_tf = build_train_transform(CONFIG)
    out = train_tf(Image.open(img_dir / "1.png"))
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (1, 64, 64)
    assert torch.isfinite(out).all()

    # eval transform is deterministic and normalized to [-1, 1]
    eval_tf = build_eval_transform(CONFIG)
    a = eval_tf(Image.open(img_dir / "1.png"))
    b = eval_tf(Image.open(img_dir / "1.png"))
    assert torch.equal(a, b)
    assert a.shape == (1, 64, 64)
    assert float(a.min()) >= -1.0
    assert float(a.max()) <= 1.0

    # dataset returns (image, model_label, source_index, image_path)
    frame = pd.DataFrame(
        [
            {
                "model_label": 0,
                "source_index": 1,
                "split": "train",
                "image_path": "data/processed/X/1.png",
            }
        ]
    )
    ds = LFWManifestDataset(frame, split="train", transform=eval_tf, root=tmp_path)
    img, label, src, path = ds[0]
    assert img.shape == (1, 64, 64)
    assert label == 0
    assert src == 1
    assert path == "data/processed/X/1.png"
