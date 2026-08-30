#!/usr/bin/env python3
"""Final protected ensemble with a DOTA-pretrained OBB specialist."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import infer_parent_ensemble as parent_components
import infer_protected_ensemble as base_system


OBB_SPECIALIST_FILENAME = "yolo26x_obb_dota_specialist.pt"
MUTUAL_IOU_LIMIT = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/work/output"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--expected-images", type=int)
    return parser.parse_args()


def resolve_weights(root: Path) -> tuple[dict[str, Path], Path, int]:
    base = {
        name: root / filename
        for name, filename in base_system.WEIGHT_FILENAMES.items()
    }
    specialist = root / OBB_SPECIALIST_FILENAME
    missing = [str(path) for path in [*base.values(), specialist] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model weights: " + ", ".join(missing))
    observed = sum(path.stat().st_size for path in [*base.values(), specialist])
    if observed >= 600_000_000:
        raise RuntimeError(f"Final ensemble exceeds 600MB: {observed}")
    return base, specialist, observed


def infer_obb_specialist(
    paths: list[Path], checkpoint: Path, device: torch.device,
) -> dict[str, dict[str, Any]]:
    from ultralytics import YOLO

    detector = YOLO(str(checkpoint))
    records: dict[str, dict[str, Any]] = {}
    batch_size = 2
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        predictions = detector.predict(
            source=[str(path) for path in batch_paths],
            imgsz=parent_components.IMAGE_SIZE,
            batch=len(batch_paths),
            device=str(device),
            half=True,
            conf=0.001,
            iou=0.7,
            max_det=parent_components.MAX_DETECTIONS,
            single_cls=True,
            augment=False,
            stream=False,
            verbose=False,
        )
        if len(predictions) != len(batch_paths):
            raise RuntimeError("Incomplete OBB-specialist prediction batch")
        for path, result in zip(batch_paths, predictions):
            if result.obb is None or len(result.obb) == 0:
                records[path.name] = {
                    "boxes": np.empty((0, 4), dtype=np.float64),
                    "scores": np.empty((0,), dtype=np.float64),
                }
                continue
            corners = result.obb.xyxyxyxy.detach().float().cpu().numpy()
            scores = result.obb.conf.detach().float().cpu().numpy()
            order = np.argsort(-scores, kind="stable")[: parent_components.MAX_DETECTIONS]
            boxes = []
            for index in order:
                points = corners[index] / parent_components.IMAGE_SIZE
                boxes.append(
                    [
                        float(points[:, 0].min()),
                        float(points[:, 1].min()),
                        float(points[:, 0].max()),
                        float(points[:, 1].max()),
                    ]
                )
            records[path.name] = {
                "boxes": np.clip(np.asarray(boxes, dtype=np.float64), 0.0, 1.0),
                "scores": scores[order].astype(np.float64),
            }
        completed = min(start + len(batch_paths), len(paths))
        if completed % 100 < len(batch_paths) or completed == len(paths):
            print(f"obb_specialist={completed}/{len(paths)}", flush=True)
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return records


def append_obb_shadow(
    paths: list[Path], parent_output: Path,
    specialist: dict[str, dict[str, Any]], output: Path,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.txt")):
        raise FileExistsError("Refusing to overwrite existing result files")
    inserted = 0
    below_limit = 0
    total_rows = 0
    for path in paths:
        parent_rows = [line.split() for line in (parent_output / f"{path.stem}.txt").read_text().splitlines() if line.strip()]
        if len(parent_rows) < 2:
            raise RuntimeError(f"Parent produced fewer than two rows: {path.name}")
        candidates: list[tuple[float, int, int, list[str]]] = [
            (float(row[5]), 0, rank, row) for rank, row in enumerate(parent_rows)
        ]
        candidate_boxes = specialist[path.name]["boxes"]
        candidate_scores = specialist[path.name]["scores"]
        if len(candidate_boxes):
            candidate_box = candidate_boxes[0]
            candidate_score = float(candidate_scores[0])
            mutual_iou = parent_components.box_iou(
                base_system.yolo_to_xyxy(parent_rows[0]), candidate_box
            )
            if mutual_iou < MUTUAL_IOU_LIMIT:
                below_limit += 1
                shadow_score = float(parent_rows[1][5]) * min(
                    float(parent_rows[0][5]), candidate_score
                )
                yolo = base_system.xyxy_to_yolo(candidate_box)
                shadow = [
                    "0",
                    *[f"{value:.8f}" for value in yolo],
                    f"{shadow_score:.8f}",
                ]
                candidates.append((shadow_score, 1, 0, shadow))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        retained = candidates[: parent_components.MAX_DETECTIONS]
        inserted += int(any(source == 1 for _, source, _, _ in retained))
        rows = [row for _, _, _, row in retained]
        total_rows += len(rows)
        (output / f"{path.stem}.txt").write_text(
            "\n".join(" ".join(row) for row in rows) + "\n", encoding="utf-8"
        )
    return {
        "images": len(paths),
        "prediction_rows": total_rows,
        "mutual_iou_below_0_95": below_limit,
        "inserted_shadow_candidates": inserted,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Inference requires an NVIDIA CUDA GPU")
    paths = parent_components.image_paths(args.images.resolve(), args.expected_images)
    base_weights, specialist_weight, total_weight_bytes = resolve_weights(
        args.weights.resolve()
    )

    semantic_main = parent_components.infer_semantic_main(
        paths, base_weights, device, args.batch_size
    )
    gc.collect(); torch.cuda.empty_cache()
    boundary_candidates = parent_components.infer_boundary_refiner(
        paths, base_weights, device
    )
    gc.collect(); torch.cuda.empty_cache()
    coordinate_specialist = parent_components.infer_coordinate_specialist(
        paths, base_weights, device, args.batch_size
    )
    gc.collect(); torch.cuda.empty_cache()
    domain_specialist = base_system.infer_domain_specialist(
        paths,
        base_weights["domain_specialist"],
        device,
        args.batch_size,
    )
    gc.collect(); torch.cuda.empty_cache()
    obb_specialist = infer_obb_specialist(paths, specialist_weight, device)

    with tempfile.TemporaryDirectory(prefix="protected-parent-") as temporary:
        parent_output = Path(temporary)
        parent_stats = base_system.write_submission(
            paths,
            semantic_main,
            boundary_candidates,
            coordinate_specialist,
            domain_specialist,
            parent_output,
        )
        result = append_obb_shadow(
            paths, parent_output, obb_specialist, args.output.resolve()
        )
    result["parent"] = parent_stats
    result["weight_bytes"] = total_weight_bytes
    result["under_600mb"] = total_weight_bytes < 600_000_000
    result["test_used_for_training_or_selection"] = False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
