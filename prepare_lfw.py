"""CLI entry point that builds the balanced 10x50 LFW subset."""

from __future__ import annotations

import argparse

from config import CONFIG
from dataset import prepare_lfw_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced Mini-DeepID LFW subset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild generated data even if the manifest already exists (never deletes the LFW cache).",
    )
    args = parser.parse_args()

    frame = prepare_lfw_dataset(config=CONFIG, force=args.force)
    train = int((frame["split"] == "train").sum())
    val = int((frame["split"] == "val").sum())
    test = int((frame["split"] == "test").sum())
    print(
        f"LFW_PREPARE_OK classes={CONFIG.num_classes} samples={len(frame)} "
        f"train={train} val={val} test={test}"
    )


if __name__ == "__main__":
    main()
