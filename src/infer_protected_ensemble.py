#!/usr/bin/env python3
"""Protected ensemble with an additional domain-generalized specialist."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import infer_parent_ensemble as parent_components


WEIGHT_FILENAMES = {
    "semantic_main": "rfdetr_semantic_main.pth",
    "yolo_hbb": "yolo26x_hbb_detector.pt",
    "boundary_refiner": "yolo_boundary_refiner.pt",
    "coordinate_specialist": "deimv2_coordinate_specialist_fp16.pth",
    "domain_specialist": "deimv2_domain_generalized_specialist_int8.pth",
}
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


def resolve_weights(root: Path) -> dict[str, Path]:
    weights = {
        name: root / filename for name, filename in WEIGHT_FILENAMES.items()
    }
    missing = [str(path) for path in weights.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model weights: " + ", ".join(missing))
    observed = sum(path.stat().st_size for path in weights.values())
    if observed != 442_678_946:
        raise RuntimeError(
            "Protected-ensemble model-size identity drift: {}".format(observed)
        )
    if observed >= 600_000_000:
        raise RuntimeError("Protected ensemble exceeds the 600MB model limit")
    return weights


def inflate_domain_specialist(source: Path, output: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("model")
    scales = payload.get("scales")
    metadata = payload.get("hybrid_int8")
    if (
        not isinstance(state, dict)
        or not isinstance(scales, dict)
        or not isinstance(metadata, dict)
        or metadata.get("scheme") != "per_output_row_symmetric_int8"
        or int(metadata.get("quantized_tensor_count", -1)) != 60
        or len(scales) != 60
        or metadata.get("all_other_tensors_unchanged") is not True
    ):
        raise RuntimeError("Domain-specialist INT8 provenance contract failed")

    inflated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key not in scales:
            inflated[key] = value
            continue
        scale = scales[key].float()
        if (
            value.dtype != torch.int8
            or value.ndim != 2
            or scale.ndim != 1
            or scale.shape[0] != value.shape[0]
        ):
            raise RuntimeError(
                "Invalid domain-specialist quantized tensor: {}".format(key)
            )
        inflated[key] = (value.float() * scale[:, None]).half()
    torch.save({"model": inflated}, output)
    del payload, state, scales, inflated
    gc.collect()


def infer_domain_specialist(
    paths: list[Path],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    package_root = parent_components.PACKAGE_ROOT
    deim_root = package_root / "external/DEIMv2"
    sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(deim_root))
    parent_components.install_inference_tensorboard_stub()

    # Register the domain-generalized detector before
    # YAMLConfig resolves its model name.
    from workspace.models.deimv2_domain_generalized_boundary import (  # noqa: F401
        DomainGeneralizedBoundaryDEIM,
    )
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS

    with tempfile.TemporaryDirectory(prefix="domain-specialist-") as temporary:
        runtime = Path(temporary)
        index_path = runtime / "images.json"
        index = {
            "images": [
                {
                    "id": image_id,
                    "file_name": path.name,
                    "width": parent_components.IMAGE_SIZE,
                    "height": parent_components.IMAGE_SIZE,
                }
                for image_id, path in enumerate(paths, 1)
            ],
            "annotations": [],
            "categories": [{"id": 0, "name": "substation"}],
        }
        index_path.write_text(json.dumps(index), encoding="utf-8")
        inflated_path = runtime / "domain_specialist_runtime_fp16.pth"
        inflate_domain_specialist(checkpoint, inflated_path)

        config_path = runtime / "domain_specialist_inference.yml"
        recipe = {
            "__include__": [
                str(
                    (
                        package_root
                        / "workspace/configs/domain_generalized_specialist_full.yml"
                    ).resolve()
                )
            ],
            "val_dataloader": {
                "total_batch_size": batch_size,
                "num_workers": 2,
                "dataset": {
                    "img_folder": str(paths[0].parent),
                    "ann_file": str(index_path),
                    "transforms": {
                        "ops": [
                            {"type": "Resize", "size": [1024, 1024]},
                            {
                                "type": "ConvertPILImage",
                                "dtype": "float32",
                                "scale": True,
                            },
                            {
                                "type": "Normalize",
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        ]
                    },
                },
            },
        }
        config_path.write_text(
            yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8"
        )

        dist_utils.setup_distributed(
            print_rank=0, print_method="builtin", seed=3407
        )
        cfg = YAMLConfig(
            str(config_path),
            resume=str(inflated_path),
            output_dir=str(runtime),
            device=str(device),
            seed=3407,
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        model = (solver.ema.module if solver.ema else solver.model).eval()
        results_by_name: dict[str, dict[str, Any]] = {}
        id_to_path = {image_id: path for image_id, path in enumerate(paths, 1)}
        try:
            with torch.inference_mode():
                for samples, targets in solver.val_dataloader:
                    samples = samples.to(device)
                    output = model(samples)
                    sizes = torch.stack(
                        [target["orig_size"] for target in targets]
                    ).to(device)
                    results = solver.postprocessor(output, sizes)
                    for target, prediction in zip(targets, results):
                        image_id = int(target["image_id"].item())
                        order = prediction["scores"].argsort(
                            descending=True
                        )[: parent_components.MAX_DETECTIONS]
                        boxes = (
                            prediction["boxes"][order].float().cpu().numpy()
                            / parent_components.IMAGE_SIZE
                        )
                        scores = (
                            prediction["scores"][order].float().cpu().numpy()
                        )
                        results_by_name[id_to_path[image_id].name] = {
                            "boxes": boxes.astype(np.float64),
                            "scores": scores.astype(np.float64),
                        }
            if len(results_by_name) != len(paths):
                raise RuntimeError("Incomplete domain-specialist inference coverage")
            return results_by_name
        finally:
            dist_utils.cleanup()


def format_parent_rows(
    pure_boxes: np.ndarray,
    scores: np.ndarray,
    boundary_candidates: np.ndarray,
    coordinate_boxes: np.ndarray,
) -> list[list[str]]:
    if (
        len(pure_boxes) != parent_components.MAX_DETECTIONS
        or len(coordinate_boxes) == 0
    ):
        raise RuntimeError("Incomplete parent detector output")
    pure = pure_boxes[0]
    refined = (
        parent_components.match_boundary_candidate(
            pure * parent_components.IMAGE_SIZE, boundary_candidates
        )
        / parent_components.IMAGE_SIZE
    )
    protected = pure + parent_components.BOUNDARY_BLEND_WEIGHT * (
        refined - pure
    )
    specialist = coordinate_boxes[0]
    rank1 = np.median(np.stack((protected, refined, specialist)), axis=0)
    rank1 = np.clip(rank1, 0.0, 1.0)
    if (
        rank1[2] < rank1[0] + 1.0 / parent_components.IMAGE_SIZE
        or rank1[3] < rank1[1] + 1.0 / parent_components.IMAGE_SIZE
    ):
        rank1 = protected.copy()
    boxes = pure_boxes.copy()
    boxes[0] = rank1

    rows: list[list[str]] = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = np.clip(box, 0.0, 1.0)
        width, height = x2 - x1, y2 - y1
        if width <= 0.0 or height <= 0.0:
            continue
        values = (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            width,
            height,
            float(score),
        )
        rows.append(["0", *["{:.8f}".format(value) for value in values]])
    if len(rows) < 2:
        raise RuntimeError("Parent ensemble produced fewer than two valid candidates")
    return rows


def yolo_to_xyxy(row: list[str]) -> np.ndarray:
    center_x, center_y, width, height = map(float, row[1:5])
    return np.asarray(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        dtype=np.float64,
    )


def xyxy_to_yolo(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.tolist()
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    )


def write_submission(
    paths: list[Path],
    semantic_main: dict[str, dict[str, Any]],
    boundary_candidates: dict[str, np.ndarray],
    coordinate_specialist: dict[str, dict[str, Any]],
    domain_specialist: dict[str, dict[str, Any]],
    output: Path,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*.txt"))
    if existing:
        raise FileExistsError(
            "Refusing to overwrite {} result files".format(len(existing))
        )

    total_rows = 0
    inserted = 0
    below_limit = 0
    for path in paths:
        parent_rows = format_parent_rows(
            semantic_main[path.name]["boxes"],
            semantic_main[path.name]["scores"],
            boundary_candidates[path.name],
            coordinate_specialist[path.name]["boxes"],
        )
        domain_box = np.clip(
            domain_specialist[path.name]["boxes"][0], 0.0, 1.0
        )
        domain_score = float(domain_specialist[path.name]["scores"][0])
        mutual_iou = parent_components.box_iou(
            yolo_to_xyxy(parent_rows[0]), domain_box
        )
        candidates: list[tuple[float, int, int, list[str]]] = [
            (float(row[5]), 0, rank, row)
            for rank, row in enumerate(parent_rows)
        ]
        if mutual_iou < MUTUAL_IOU_LIMIT:
            below_limit += 1
            shadow_score = float(parent_rows[1][5]) * min(
                float(parent_rows[0][5]), domain_score
            )
            yolo = xyxy_to_yolo(domain_box)
            shadow = [
                "0",
                *["{:.8f}".format(value) for value in yolo],
                "{:.8f}".format(shadow_score),
            ]
            candidates.append((shadow_score, 1, 0, shadow))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        retained = candidates[: parent_components.MAX_DETECTIONS]
        inserted += int(any(source == 1 for _, source, _, _ in retained))
        rows = [row for _, _, _, row in retained]
        total_rows += len(rows)
        (output / "{}.txt".format(path.stem)).write_text(
            "\n".join(" ".join(row) for row in rows) + "\n",
            encoding="utf-8",
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
    paths = parent_components.image_paths(
        args.images.resolve(), args.expected_images
    )
    weights = resolve_weights(args.weights.resolve())

    semantic_main = parent_components.infer_semantic_main(
        paths, weights, device, args.batch_size
    )
    gc.collect()
    torch.cuda.empty_cache()
    boundary_candidates = parent_components.infer_boundary_refiner(
        paths, weights, device
    )
    gc.collect()
    torch.cuda.empty_cache()
    coordinate_specialist = parent_components.infer_coordinate_specialist(
        paths, weights, device, args.batch_size
    )
    gc.collect()
    torch.cuda.empty_cache()
    domain_specialist = infer_domain_specialist(
        paths, weights["domain_specialist"], device, args.batch_size
    )
    result = write_submission(
        paths,
        semantic_main,
        boundary_candidates,
        coordinate_specialist,
        domain_specialist,
        args.output.resolve(),
    )
    result["weights"] = {
        name: {"bytes": path.stat().st_size}
        for name, path in weights.items()
    }
    result["test_used_for_training_or_selection"] = False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
