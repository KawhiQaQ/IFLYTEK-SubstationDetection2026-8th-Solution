#!/usr/bin/env python3
"""Build a YOLO-OBB view without duplicating source imagery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Build a fixed-epoch full-training view; validation is disabled.",
    )
    return parser.parse_args()


def read_images(path: Path) -> list[Path]:
    images = [Path(line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]
    if len(images) != len(set(images)):
        raise RuntimeError(f"Duplicate images in {path}")
    return images


def source_label(image: Path) -> Path:
    if image.parent.name != "images":
        raise RuntimeError(f"Unexpected image layout: {image}")
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def convert_label(source: Path) -> str:
    rows: list[str] = []
    if not source.exists():
        return ""
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or int(float(fields[0])) != 0:
            raise RuntimeError(f"Invalid label {source}:{line_number}")
        _, xc, yc, width, height = map(float, fields)
        x1, x2 = xc - width / 2.0, xc + width / 2.0
        y1, y2 = yc - height / 2.0, yc + height / 2.0
        coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        if min(coords) < -1e-6 or max(coords) > 1.0 + 1e-6:
            raise RuntimeError(f"Out-of-range label {source}:{line_number}")
        rows.append("0 " + " ".join(f"{min(max(value, 0.0), 1.0):.10f}" for value in coords))
    return "\n".join(rows) + ("\n" if rows else "")


def build_split(images: list[Path], root: Path, split: str) -> tuple[int, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)
        destination = image_dir / image.name
        if destination.is_symlink():
            if destination.resolve() != image:
                raise RuntimeError(f"Symlink drift: {destination}")
        elif destination.exists():
            raise RuntimeError(f"Refusing to replace: {destination}")
        else:
            os.symlink(image, destination)
        converted = convert_label(source_label(image))
        (label_dir / f"{image.stem}.txt").write_text(converted, encoding="utf-8")
        boxes += converted.count("\n")
    return len(images), boxes


def main() -> None:
    args = parse_args()
    train = read_images(args.train_list)
    valid = [] if args.train_only else read_images(args.val_list)
    overlap = set(train) & set(valid)
    if overlap:
        raise RuntimeError(f"Train/valid overlap: {len(overlap)}")
    root = args.output.resolve()
    train_count, train_boxes = build_split(train, root, "train")
    if args.train_only:
        valid_count, valid_boxes = 0, 0
        validation_entry = "images/train"
    else:
        valid_count, valid_boxes = build_split(valid, root, "val")
        validation_entry = "images/val"
    yaml = root / "dataset.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: {validation_entry}\nnames:\n  0: substation\n",
        encoding="utf-8",
    )
    manifest = {
        "component": "dota_obb_specialist",
        "task": "obb",
        "train_images": train_count,
        "train_boxes": train_boxes,
        "valid_images": valid_count,
        "valid_boxes": valid_boxes,
        "train_valid_overlap": 0,
        "round1_test_images_read": 0,
        "round2_test_images_read": 0,
        "train_only_fixed_epoch": args.train_only,
        "conversion": "horizontal YOLO boxes -> four normalized rectangle corners",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
