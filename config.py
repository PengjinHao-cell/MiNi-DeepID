"""Frozen experiment configuration for Mini-DeepID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    image_size: int = 64
    num_classes: int = 10
    samples_per_class: int = 50
    train_per_class: int = 35
    val_per_class: int = 7
    test_per_class: int = 8
    embedding_dim: int = 160
    batch_size: int = 32
    max_epochs: int = 80
    early_stopping_patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda:0"
    min_faces_per_person: int = 20
    lfw_resize: float = 0.5
    dropout: float = 0.4
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 8.0
    translation_fraction: float = 0.05
    scale_range: tuple = (0.95, 1.05)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_cache_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "cache"

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def manifests_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "manifests"

    @property
    def outputs_dir(self) -> Path:
        return PROJECT_ROOT / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints"

    @property
    def split_manifest_path(self) -> Path:
        return self.manifests_dir / "split_manifest.csv"

    @property
    def identities_path(self) -> Path:
        return self.manifests_dir / "identities.json"


CONFIG = ExperimentConfig()
