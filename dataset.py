"""Deterministic balanced LFW manifest construction for Mini-DeepID.

Pure selection/splitting helpers are unit-tested with synthetic labels; the
real LFW download and PNG export live in :func:`prepare_lfw_dataset`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG

VALID_SPLITS = {"train", "val", "test"}


def select_top_labels(targets: np.ndarray, num_classes: int, samples_per_class: int) -> list[int]:
    """Return the ``num_classes`` most frequent labels that each have at least
    ``samples_per_class`` samples (descending count, then ascending label)."""
    counts = Counter(int(t) for t in np.asarray(targets))
    candidates = sorted(counts.keys(), key=lambda label: (-counts[label], label))
    selected = [label for label in candidates if counts[label] >= samples_per_class][:num_classes]
    if len(selected) < num_classes:
        raise ValueError(
            f"Only {len(selected)} identities have >= {samples_per_class} samples; "
            f"need {num_classes}."
        )
    return selected


def sample_and_split_indices(
    targets: np.ndarray, selected_labels: list[int], seed: int
) -> list[dict]:
    """Deterministically pick ``samples_per_class`` indices per identity and
    split them into train/val/test using a single seeded RNG."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    per = CONFIG.samples_per_class
    n_train = CONFIG.train_per_class
    n_val = CONFIG.val_per_class
    rows: list[dict] = []
    for rank, label in enumerate(selected_labels):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        chosen = indices[:per]
        if len(chosen) < per:
            raise ValueError(f"identity {label} has fewer than {per} samples")
        for pos, idx in enumerate(chosen):
            if pos < n_train:
                split = "train"
            elif pos < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rows.append(
                {
                    "source_index": int(idx),
                    "source_label": int(label),
                    "model_label": int(rank),
                    "split": split,
                }
            )
    return rows


def validate_manifest(frame: pd.DataFrame, num_classes: int) -> None:
    """Raise ``ValueError`` when the manifest violates the frozen split protocol."""
    required = {"source_index", "source_label", "model_label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    if frame["source_index"].duplicated().any():
        raise ValueError("duplicate source_index in manifest")

    labels = sorted(int(x) for x in frame["model_label"].unique())
    if labels != list(range(num_classes)):
        raise ValueError(f"model labels {labels} are not contiguous 0..{num_classes - 1}")

    per_class = frame.groupby("model_label").size()
    if (per_class != CONFIG.samples_per_class).any():
        raise ValueError("per-class sample count is not samples_per_class")

    unexpected = set(frame["split"].unique()) - VALID_SPLITS
    if unexpected:
        raise ValueError(f"unexpected split names: {sorted(unexpected)}")

    expected = {
        "train": CONFIG.train_per_class,
        "val": CONFIG.val_per_class,
        "test": CONFIG.test_per_class,
    }
    for split, count in expected.items():
        per_split = frame[frame["split"] == split].groupby("model_label").size()
        if len(per_split) != num_classes or (per_split != count).any():
            raise ValueError(f"incorrect {split} split count")


def sanitize_identity_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def prepare_lfw_dataset(config=CONFIG, force: bool = False) -> pd.DataFrame:
    """Download LFW, select top identities, split deterministically, export
    64x64 grayscale PNGs, and write the frozen manifest plus ``identities.json``,
    ``data_summary.json``, and a class-distribution figure."""
    from PIL import Image
    from sklearn.datasets import fetch_lfw_people

    config.data_cache_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.manifests_dir.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = config.split_manifest_path

    if not force and manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        try:
            validate_manifest(existing, config.num_classes)
            return existing
        except ValueError:
            pass  # rebuild below

    lfw = fetch_lfw_people(
        min_faces_per_person=config.min_faces_per_person,
        resize=config.lfw_resize,
        color=False,
        funneled=True,
        data_home=str(config.data_cache_dir),
    )
    targets = np.asarray(lfw.target)
    images = np.asarray(lfw.images)
    names = np.asarray(lfw.target_names)

    selected = select_top_labels(targets, config.num_classes, config.samples_per_class)
    rows = sample_and_split_indices(targets, selected, config.seed)

    frame = pd.DataFrame(rows)
    name_of_source = {label: sanitize_identity_name(str(names[label])) for label in selected}
    frame["identity_name"] = frame["source_label"].map(name_of_source)
    frame["image_path"] = frame.apply(
        lambda r: f"data/processed/{r['identity_name']}/{int(r['source_index'])}.png",
        axis=1,
    )

    # Export lossless grayscale PNGs. fetch_lfw_people returns float images in
    # [0, 1]; scale to [0, 255] before the uint8 conversion and 64x64 resize.
    for _, row in frame.iterrows():
        arr = (np.clip(images[int(row["source_index"])], 0.0, 1.0) * 255.0).round().astype(np.uint8)
        image = Image.fromarray(arr, mode="L").resize(
            (config.image_size, config.image_size), Image.BILINEAR
        )
        out_dir = config.processed_dir / row["identity_name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / f"{int(row['source_index'])}.png", format="PNG")

    validate_manifest(frame, config.num_classes)

    identities = [
        {
            "model_label": rank,
            "source_label": int(selected[rank]),
            "identity_name": sanitize_identity_name(str(names[selected[rank]])),
            "display_name": str(names[selected[rank]]),
        }
        for rank in range(config.num_classes)
    ]
    _atomic_write_text(
        config.identities_path,
        json.dumps({"identities": identities}, indent=2, ensure_ascii=False) + "\n",
    )

    tmp_csv = manifest_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp_csv, index=False)
    tmp_csv.replace(manifest_path)

    summary = {
        "source": "lfw funneled grayscale",
        "seed": config.seed,
        "num_classes": config.num_classes,
        "samples_per_class": config.samples_per_class,
        "total_samples": int(len(frame)),
        "train": int((frame["split"] == "train").sum()),
        "val": int((frame["split"] == "val").sum()),
        "test": int((frame["split"] == "test").sum()),
        "image_size": config.image_size,
        "channels": 1,
    }
    _atomic_write_text(
        config.outputs_dir / "data_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )

    _write_class_distribution(frame, config)
    return frame


def _write_class_distribution(frame: pd.DataFrame, config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = frame.groupby("model_label")["identity_name"].first().tolist()
    order = ["train", "val", "test"]
    counts = {split: [] for split in order}
    for label in range(config.num_classes):
        sub = frame[frame["model_label"] == label]
        for split in order:
            counts[split].append(int((sub["split"] == split).sum()))

    x = list(range(config.num_classes))
    bottom = [0] * config.num_classes
    colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    fig, ax = plt.subplots(figsize=(12, 5))
    for split in order:
        ax.bar(x, counts[split], bottom=bottom, label=split, color=colors[split])
        bottom = [b + c for b, c in zip(bottom, counts[split])]
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("samples")
    ax.set_title("LFW closed-set class distribution (10 identities x 50)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.outputs_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
