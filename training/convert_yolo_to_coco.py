#!/usr/bin/env python3
"""Convert fixed YOLO split lists to lossless COCO detection annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=1183)
    parser.add_argument("--expected-val", type=int, default=290)
    return parser.parse_args()


def read_paths(path: Path) -> list[Path]:
    paths = [
        Path(line.strip()).resolve()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate paths in {path}")
    return paths


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def convert_split(paths: list[Path]) -> dict:
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for image_id, image_path in enumerate(sorted(paths), start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (1024, 1024):
            raise ValueError(f"Expected 1024x1024 image: {image_path}")
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        rows = [
            line.split()
            for line in label_path(image_path)
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        ]
        for row in rows:
            if len(row) != 5 or int(row[0]) != 0:
                raise ValueError(f"Invalid YOLO row for {image_path}: {row}")
            center_x, center_y, box_width, box_height = map(float, row[1:])
            x = max(0.0, (center_x - box_width / 2) * width)
            y = max(0.0, (center_y - box_height / 2) * height)
            box_width = min(box_width * width, width - x)
            box_height = min(box_height * height, height - y)
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"Degenerate box for {image_path}: {row}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "substation", "supercategory": "infrastructure"}
        ],
    }


def main() -> None:
    args = parse_args()
    train = read_paths(args.train_list)
    valid = read_paths(args.val_list)
    if set(train) & set(valid):
        raise ValueError("Train-validation image overlap")
    if len(train) != args.expected_train or len(valid) != args.expected_val:
        raise ValueError(
            f"Unexpected split sizes: train={len(train)}, valid={len(valid)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, paths in (("fold0_train", train), ("fold0_val", valid)):
        payload = convert_split(paths)
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: images={len(payload['images'])} "
            f"boxes={len(payload['annotations'])}"
        )
    full_payload = convert_split(train + valid)
    (args.output_dir / "full_train.json").write_text(
        json.dumps(full_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"full_train: images={len(full_payload['images'])} "
        f"boxes={len(full_payload['annotations'])}"
    )


if __name__ == "__main__":
    main()
