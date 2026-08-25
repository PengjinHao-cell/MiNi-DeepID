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
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

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


def assert_splits_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any source index appears in more than one split."""
    by_split = {
        split: set(frame.loc[frame["split"] == split, "source_index"])
        for split in VALID_SPLITS
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by_split[left] & by_split[right]
        if overlap:
            raise ValueError(
                f"split overlap between {left} and {right}: {sorted(overlap)[:10]}"
            )


def load_manifest(config=CONFIG) -> pd.DataFrame:
    """Read the frozen split manifest and validate it.

    All later stages must load the split through this function and must never
    call ``random_split`` or re-derive the split from raw data.
    """
    path = config.split_manifest_path
    if not path.exists():
        raise FileNotFoundError(f"frozen manifest not found: {path}")
    frame = pd.read_csv(path)
    validate_manifest(frame, config.num_classes)
    assert_splits_disjoint(frame)
    return frame


def load_identities(config=CONFIG) -> list[str]:
    """Return identity names in model-label order (checkpoint identity mapping)."""
    data = json.loads(config.identities_path.read_text(encoding="utf-8"))
    entries = sorted(data["identities"], key=lambda entry: int(entry["model_label"]))
    return [str(entry["identity_name"]) for entry in entries]


def validate_exported_images(config, frame: pd.DataFrame, project_root=None) -> int:
    """Verify every manifest row points to an existing 64x64 finite grayscale
    (mode ``L``) uint8 PNG. Returns the number of images checked."""
    from PIL import Image

    root = Path(project_root) if project_root is not None else config.project_root
    count = 0
    for image_path in frame["image_path"]:
        path = root / image_path
        if not path.exists():
            raise ValueError(f"missing image: {path}")
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(f"not grayscale (mode={image.mode}): {path}")
            if image.size != (config.image_size, config.image_size):
                raise ValueError(f"wrong size {image.size}: {path}")
            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                raise ValueError(f"not uint8 (dtype={arr.dtype}): {path}")
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite pixel in {path}")
        count += 1
    return count


def generate_sample_grid(config, frame: pd.DataFrame, samples_per_class: int = 5) -> Path:
    """Generate a ten-identity sample grid PNG and return its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_rows = config.num_classes
    n_cols = samples_per_class
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = np.atleast_2d(axes)
    for rank in range(config.num_classes):
        sub = frame[frame["model_label"] == rank].sort_values("source_index")
        name = str(sub["identity_name"].iloc[0])
        for col, (_, row) in enumerate(sub.head(samples_per_class).iterrows()):
            path = config.project_root / row["image_path"]
            arr = np.asarray(Image.open(path))
            ax = axes[rank, col]
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=8)
            if rank == 0:
                ax.set_title(f"sample {col + 1}", fontsize=8)
    fig.suptitle("LFW closed-set sample grid (10 identities)")
    fig.tight_layout()
    out_path = config.outputs_dir / "sample_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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


def build_train_transform(config=CONFIG):
    """Training transform: grayscale, resize, random flip/affine, normalize.

    Random augmentation is applied only to manifest rows whose split is ``train``.
    """
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((config.image_size, config.image_size)),
            T.RandomHorizontalFlip(p=config.horizontal_flip_probability),
            T.RandomAffine(
                degrees=config.rotation_degrees,
                translate=(config.translation_fraction, config.translation_fraction),
                scale=config.scale_range,
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def build_eval_transform(config=CONFIG):
    """Evaluation transform: grayscale, resize, tensor, normalize (no augmentation)."""
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class LFWManifestDataset(Dataset):
    """Read-only dataset that filters the frozen manifest by split.

    Returns ``(image, model_label, source_index, image_path)`` and never mutates
    the underlying manifest frame.
    """

    def __init__(self, manifest: pd.DataFrame, split: str, transform=None, root=None, config=CONFIG):
        self.config = config
        self.root = Path(root) if root is not None else config.project_root
        self.frame = manifest[manifest["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.root / row["image_path"]
        from PIL import Image

        with Image.open(path) as image:
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["model_label"]), int(row["source_index"]), str(row["image_path"])
