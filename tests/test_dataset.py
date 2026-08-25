import numpy as np
import pandas as pd
import pytest

from dataset import (
    assert_splits_disjoint,
    load_manifest,
    sample_and_split_indices,
    select_top_labels,
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
