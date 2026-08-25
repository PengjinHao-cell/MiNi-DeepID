"""Data acceptance check (Gate G5): overlap, labels, image specs, sample grid."""

from __future__ import annotations

from config import CONFIG
from dataset import generate_sample_grid, load_manifest, validate_exported_images


def main() -> None:
    frame = load_manifest(CONFIG)  # validates labels, counts, and split disjointness
    images = validate_exported_images(CONFIG, frame)
    grid = generate_sample_grid(CONFIG, frame)
    print(
        f"MINI_DEEPID_DATA_ACCEPT_OK classes={CONFIG.num_classes} samples={len(frame)} "
        f"images={images} grid={grid.name}"
    )


if __name__ == "__main__":
    main()
