#!/usr/bin/env python3
"""Evaluate one predeclared protected-shadow addition to a fixed OOF parent."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster


IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=float, required=True)
    parser.add_argument(
        "--allow-short-parent",
        action="store_true",
        help="Permit 1..20 parent rows per image for transformed pressure suites.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError(f"No predictions in {path}")
    return [dict(row) for row in rows]


def grouped(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for order, row in enumerate(rows):
        item = dict(row)
        item["_order"] = order
        result[int(item["image_id"])].append(item)
    for image_rows in result.values():
        image_rows.sort(key=lambda row: (-float(row["score"]), int(row["_order"])))
    return result


def xyxy(row: dict[str, Any]) -> np.ndarray:
    x, y, width, height = map(float, row["bbox"])
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    wh = np.maximum(np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]), 0.0)
    intersection = float(np.prod(wh))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    return intersection / max(left_area + right_area - intersection, 1e-12)


def clean(row: dict[str, Any], row_id: int) -> dict[str, Any]:
    item = copy.deepcopy(row)
    for key in ("_order", "segmentation", "area"):
        item.pop(key, None)
    item["id"] = row_id
    return item


def evaluate(coco_gt: COCO, rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluator = COCOeval_faster(
        coco_gt,
        coco_gt.loadRes(rows),
        iouType="bbox",
        print_function=lambda *_args, **_kwargs: None,
        separate_eval=True,
    )
    evaluator.params.maxDets = [1, 10, 20]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
    values = []
    for index in range(len(IOU_THRESHOLDS)):
        valid = precision[index, :, :, 0, -1]
        valid = valid[valid > -1]
        values.append(float(valid.mean()))
    return {
        "map_50_95": float(np.mean(values)),
        "ap_95": values[-1],
        "ap_by_iou": {
            f"{threshold:.2f}": value
            for threshold, value in zip(IOU_THRESHOLDS, values)
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite fixed shadow audit")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_ids = sorted(int(row["id"]) for row in annotations["images"])
    parent_rows = load_rows(args.parent)
    specialist_rows = load_rows(args.specialist)
    parent = grouped(parent_rows)
    specialist = grouped(specialist_rows)
    if set(parent) != set(image_ids):
        raise RuntimeError("Parent does not cover exact fold0 images")
    if args.allow_short_parent:
        if any(not 1 <= len(parent[i]) <= 20 for i in image_ids):
            raise RuntimeError("Pressure parent is not within 1..20 rows per image")
    elif any(len(parent[i]) != 20 for i in image_ids):
        raise RuntimeError("Parent is not exact fold0 top-20")
    if set(specialist) - set(image_ids):
        raise RuntimeError("Specialist contains images outside fold0")

    output_rows: list[dict[str, Any]] = []
    inserted = 0
    below_limit = 0
    for image_id in image_ids:
        protected = parent[image_id]
        candidates: list[tuple[float, int, int, dict[str, Any]]] = [
            (float(row["score"]), 0, rank, row) for rank, row in enumerate(protected)
        ]
        if image_id in specialist and specialist[image_id]:
            candidate = specialist[image_id][0]
            if box_iou(xyxy(protected[0]), xyxy(candidate)) < 0.95:
                below_limit += 1
                shadow = copy.deepcopy(candidate)
                shadow["score"] = float(protected[1]["score"]) * min(
                    float(protected[0]["score"]), float(candidate["score"])
                )
                candidates.append((float(shadow["score"]), 1, 0, shadow))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        retained = candidates[:20]
        inserted += int(any(source == 1 for _, source, _, _ in retained))
        for _, _, _, row in retained:
            output_rows.append(clean(row, len(output_rows) + 1))

    coco_gt = COCO(str(args.annotations))
    parent_metrics = evaluate(coco_gt, parent_rows)
    specialist_metrics = evaluate(coco_gt, specialist_rows)
    fused_metrics = evaluate(coco_gt, output_rows)
    report = {
        "evaluation": "fixed_parent_protected_shadow",
        "validation_parameter_search": False,
        "rule": "fixed IoU<0.95 protected shadow",
        "test_images_read": 0,
        "metrics": {
            "parent": parent_metrics,
            "specialist": specialist_metrics,
            "fused": fused_metrics,
        },
        "diagnostics": {
            "specialist_images_with_predictions": len(specialist),
            "mutual_iou_below_0_95": below_limit,
            "inserted_shadow_candidates": inserted,
            "rank1_parent_preserved": all(
                np.array_equal(xyxy(grouped(output_rows)[i][0]), xyxy(parent[i][0]))
                for i in image_ids
            ),
        },
        "reference": args.reference,
        "fused_minus_reference": fused_metrics["map_50_95"] - args.reference,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions.json").write_text(json.dumps(output_rows), encoding="utf-8")
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
