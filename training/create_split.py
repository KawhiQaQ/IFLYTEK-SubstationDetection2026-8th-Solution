#!/usr/bin/env python3
"""Create deterministic 1,183/290 train-validation lists from official images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--valid-images", type=int, default=290)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.images.resolve()
    paths = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(paths) != 1473:
        raise RuntimeError(f"Expected 1,473 official images, found {len(paths)}")
    if len({path.stem for path in paths}) != len(paths):
        raise RuntimeError("Image stems must be unique")
    shuffled = paths.copy()
    random.Random(args.seed).shuffle(shuffled)
    valid = set(shuffled[: args.valid_images])
    train = [path for path in paths if path not in valid]
    valid_sorted = [path for path in paths if path in valid]
    if set(train) & set(valid_sorted):
        raise RuntimeError("Train-validation leakage")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("fold0_train.txt", train),
        ("fold0_val.txt", valid_sorted),
        ("full_train.txt", paths),
    ):
        (args.output / name).write_text(
            "\n".join(str(path) for path in rows) + "\n", encoding="utf-8"
        )
    print(
        f"train={len(train)} valid={len(valid_sorted)} overlap=0 seed={args.seed}"
    )


if __name__ == "__main__":
    main()
