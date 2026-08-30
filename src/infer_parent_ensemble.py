#!/usr/bin/env python3
"""Semantic-main inference, protected boundary refinement, and triangulation.

The entry point runs three trained detectors, refines the leading boundary,
and triangulates its four coordinates with an independent specialist. It
performs no fitting and never reads labels.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT
IMAGE_SIZE = 1024
MAX_DETECTIONS = 20
BOUNDARY_BLEND_WEIGHT = 0.24456708133220673


def install_inference_tensorboard_stub() -> None:
    """Keep DEIMv2 inference independent of training-only TensorBoard.

    DEIMv2 imports ``SummaryWriter`` while constructing its generic config,
    even though evaluation never writes TensorBoard events.  On the fixed
    final-round image that import loads TensorFlow 2.11 and conflicts with the
    vendored protobuf runtime.  A false-y no-op writer preserves DEIMv2's
    configuration interface without importing TensorBoard or TensorFlow.
    """
    module_name = "torch.utils.tensorboard"
    if module_name in sys.modules:
        return

    tensorboard_stub = types.ModuleType(module_name)

    class SummaryWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __bool__(self) -> bool:
            return False

        def add_text(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    tensorboard_stub.SummaryWriter = SummaryWriter
    sys.modules[module_name] = tensorboard_stub

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=PACKAGE_ROOT / "weights")
    parser.add_argument("--output", type=Path, default=Path("/work/output"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-images", type=int)
    return parser.parse_args()


def resolve_weights(root: Path) -> dict[str, Path]:
    paths = {
        "semantic_main": root / "rfdetr_semantic_main.pth",
        "yolo_hbb": root / "yolo26x_hbb_detector.pt",
        "boundary_refiner": root / "yolo_boundary_refiner.pt",
        "coordinate_specialist": root / "deimv2_coordinate_specialist_fp16.pth",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} weight: {path}")
    return paths


def image_paths(root: Path, expected: int | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    suffixes = {".jpg", ".jpeg", ".png"}
    candidates = (root, root / "images", root / "test" / "images", root / "test")
    paths = []
    for candidate in candidates:
        if candidate.is_dir():
            paths = sorted(path for path in candidate.iterdir() if path.suffix.lower() in suffixes)
        if paths:
            break
    if not paths:
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    if not paths or len({path.stem for path in paths}) != len(paths):
        raise RuntimeError("Input directory is empty or contains duplicate stems")
    if expected is not None and len(paths) != expected:
        raise RuntimeError(f"Expected {expected} images, observed {len(paths)}")
    for path in paths:
        with Image.open(path) as image:
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"Expected a 1024x1024 image: {path}")
    return paths


def query_config(recipe: dict[str, Any]) -> Any:
    sys.path.insert(0, str(PACKAGE_ROOT))
    from workspace.models.rfdetr_boundary_evidence import BoundaryEvidenceConfig
    from workspace.models.rfdetr_query_consensus import QueryConsensusConfig

    source = recipe["model"]["query_consensus"]
    return QueryConsensusConfig(
        boundary=BoundaryEvidenceConfig(
            feature_channels=int(source["feature_channels"]),
            query_channels=int(source["boundary_query_channels"]),
            profile_channels=int(source["profile_channels"]),
            normal_samples=int(source["normal_samples"]),
            tangent_samples=int(source["tangent_samples"]),
            normal_range=float(source["normal_range"]),
            max_edge_shift=float(source["boundary_max_edge_shift"]),
            edge_loss_weight=float(source["boundary_edge_loss_weight"]),
            smooth_l1_beta=float(source["boundary_smooth_l1_beta"]),
        ),
        topk=int(source["topk"]),
        overlap_threshold=float(source["overlap_threshold"]),
        overlap_rank_bonus=float(source["overlap_rank_bonus"]),
        query_channels=int(source["consensus_query_channels"]),
        pair_channels=int(source["pair_channels"]),
        value_channels=int(source["value_channels"]),
        geometry_channels=int(source["geometry_channels"]),
        max_edge_shift=float(source["consensus_max_edge_shift"]),
        edge_loss_weight=float(source["consensus_edge_loss_weight"]),
        smooth_l1_beta=float(source["consensus_smooth_l1_beta"]),
    )


def infer_semantic_main(
    paths: list[Path], weights: dict[str, Path], device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    sys.path.insert(0, str(PACKAGE_ROOT))
    from rfdetr import RFDETRLarge
    from workspace.models.rfdetr_query_consensus import install_query_consensus
    from workspace.scripts.query_feature_cache import build_cache

    recipe = yaml.safe_load(
        (
            PACKAGE_ROOT
            / "workspace/configs/semantic_main.yaml"
        ).read_text(encoding="utf-8")
    )
    consensus = query_config(recipe)
    detector = RFDETRLarge(
        pretrain_weights=None,
        resolution=IMAGE_SIZE,
        num_classes=1,
    )
    installed = install_query_consensus(detector.model.model, consensus)
    checkpoint = torch.load(
        weights["semantic_main"], map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model")
    if not isinstance(state, dict) or len(state) != 575:
        raise RuntimeError("Semantic-main checkpoint provenance failed")
    incompatible = installed.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Semantic-main checkpoint did not load strictly")
    detector.model.model = installed.to(device).eval()
    empty = {path.name: np.empty((0, 4), dtype=np.float32) for path in paths}
    cache = build_cache(
        detector=detector,
        paths=paths,
        views=["identity"],
        image_id_by_name={path.name: i for i, path in enumerate(paths, 1)},
        boxes_by_name=empty,
        config={
            "detector": {"batch_size": batch_size, "top_k": MAX_DETECTIONS,
                         "class_index": 0, "decoder_layer": -1,
                         "resolution": IMAGE_SIZE},
            "target": {"iou_thresholds": [0.5 + 0.05 * i for i in range(10)]},
            "model": {"query_dim": 256},
        },
        device=device,
    )
    scores = cache["base_logits"].float().sigmoid()
    return {
        name: {
            "boxes": boxes.float().numpy().astype(np.float64),
            "scores": image_scores.numpy().astype(np.float64),
        }
        for name, boxes, image_scores in zip(
            cache["file_names"], cache["boxes_xyxy"], scores
        )
    }


def infer_boundary_refiner(
    paths: list[Path], weights: dict[str, Path], device: torch.device,
) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(PACKAGE_ROOT))
    sys.path.insert(0, str(PACKAGE_ROOT / "workspace"))
    from ultralytics import YOLO
    from workspace.models.boundary_refiner import (
        BoundaryDistributionRefiner, FrozenYOLOShallowEncoder,
    )
    from workspace.scripts.boundary_refinement import refine_yolo_cache

    # Ultralytics treats a Python list of paths as one in-memory source batch,
    # even with ``stream=True``.  On all 1473 final-round images that attempted
    # a 69-GiB allocation.  Bound the source itself, not just the result mode.
    detector = YOLO(str(weights["yolo_hbb"]))
    records = []
    detection_batch_size = 2
    for start in range(0, len(paths), detection_batch_size):
        batch_paths = paths[start : start + detection_batch_size]
        predictions = detector.predict(
            source=[str(path) for path in batch_paths],
            imgsz=IMAGE_SIZE,
            batch=detection_batch_size,
            device=str(device),
            half=True,
            conf=0.001,
            iou=0.7,
            max_det=MAX_DETECTIONS,
            single_cls=True,
            end2end=False,
            augment=False,
            stream=False,
            verbose=False,
        )
        if len(predictions) != len(batch_paths):
            raise RuntimeError("Incomplete batched YOLO predictions")
        for path, result in zip(batch_paths, predictions):
            boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().tolist()
            scores = [] if result.boxes is None else result.boxes.conf.cpu().tolist()
            records.append({"image_path": str(path.resolve()), "boxes_xyxy": boxes,
                            "confidences": scores})
        completed = min(start + len(batch_paths), len(paths))
        if completed % 100 < len(batch_paths) or completed == len(paths):
            print("yolo_detect={}/{}".format(completed, len(paths)), flush=True)
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    payload = torch.load(
        weights["boundary_refiner"], map_location="cpu", weights_only=False
    )
    parent = YOLO(str(weights["yolo_hbb"])).model.to(device).eval()
    encoder = FrozenYOLOShallowEncoder(list(parent.model[:5]))
    cfg = payload["config"]["model"]
    model = BoundaryDistributionRefiner(
        encoder=encoder, bins=int(cfg["bins"]),
        residual_range=float(cfg["residual_range"]),
        grid_beta=float(cfg["grid_beta"]),
    ).to(device)
    model.load_trainable_state_dict(payload["head_state"])
    model.eval()
    refined, _ = refine_yolo_cache(
        {"records": records}, {path.resolve() for path in paths}, model, cfg, device
    )
    return refined


def infer_coordinate_specialist(
    paths: list[Path], weights: dict[str, Path], device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    deim_root = PACKAGE_ROOT / "external/DEIMv2"
    sys.path.insert(0, str(deim_root))
    install_inference_tensorboard_stub()
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS

    config = (
        PACKAGE_ROOT
        / "workspace/configs/coordinate_specialist_inference.yml"
    )
    with tempfile.TemporaryDirectory(prefix="coordinate-specialist-") as temp:
        index_path = Path(temp) / "images.json"
        index = {
            "images": [{"id": i, "file_name": path.name, "width": IMAGE_SIZE,
                        "height": IMAGE_SIZE} for i, path in enumerate(paths, 1)],
            "annotations": [], "categories": [{"id": 0, "name": "substation"}],
        }
        index_path.write_text(json.dumps(index), encoding="utf-8")
        recipe = yaml.safe_load(config.read_text(encoding="utf-8"))
        recipe["val_dataloader"]["total_batch_size"] = batch_size
        recipe["val_dataloader"]["dataset"]["img_folder"] = str(paths[0].parent)
        recipe["val_dataloader"]["dataset"]["ann_file"] = str(index_path)
        config_path = Path(temp) / "coordinate_specialist_inference.yml"
        # Preserve the complete production recipe by making the first
        # include absolute.  The temporary file then overrides only the
        # inference-only loader paths.
        recipe["__include__"] = [
            str(
                (
                    PACKAGE_ROOT
                    / "workspace/configs/coordinate_specialist_full.yml"
                ).resolve()
            )
        ]
        config_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
        dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=3407)
        cfg = YAMLConfig(
            str(config_path), resume=str(weights["coordinate_specialist"]), output_dir=temp,
            device=str(device), seed=3407,
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        model = (solver.ema.module if solver.ema else solver.model).eval()
        loader = solver.val_dataloader
        postprocessor = solver.postprocessor
        result_by_name: dict[str, dict[str, Any]] = {}
        id_to_path = {i: path for i, path in enumerate(paths, 1)}
        try:
            with torch.inference_mode():
                for samples, targets in loader:
                    samples = samples.to(device)
                    output = model(samples)
                    sizes = torch.stack(
                        [target["orig_size"] for target in targets]
                    ).to(device)
                    results = postprocessor(output, sizes)
                    for target, prediction in zip(targets, results):
                        image_id = int(target["image_id"].item())
                        order = prediction["scores"].argsort(
                            descending=True
                        )[:MAX_DETECTIONS]
                        boxes = (
                            prediction["boxes"][order].float().cpu().numpy()
                            / IMAGE_SIZE
                        )
                        scores = prediction["scores"][order].float().cpu().numpy()
                        result_by_name[id_to_path[image_id].name] = {
                            "boxes": boxes.astype(np.float64),
                            "scores": scores.astype(np.float64),
                        }
            return result_by_name
        finally:
            dist_utils.cleanup()


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    tl = np.maximum(left[:2], right[:2])
    br = np.minimum(left[2:], right[2:])
    intersection = float(np.clip(br - tl, 0.0, None).prod())
    la = float(np.clip(left[2:] - left[:2], 0.0, None).prod())
    ra = float(np.clip(right[2:] - right[:2], 0.0, None).prod())
    return intersection / max(la + ra - intersection, 1e-12)


def match_boundary_candidate(
    pure_pixels: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    if not len(candidates):
        return pure_pixels.copy()
    overlaps = np.asarray([box_iou(pure_pixels, candidate) for candidate in candidates])
    best = int(overlaps.argmax())
    return candidates[best].astype(np.float64) if overlaps[best] >= 0.5 else pure_pixels.copy()


def write_submission(
    paths: list[Path], semantic_main: dict[str, dict[str, Any]],
    boundary_candidates: dict[str, np.ndarray],
    coordinate_specialist: dict[str, dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*.txt"))
    if existing:
        raise FileExistsError(f"Refusing to overwrite {len(existing)} result files in {output}")
    total_rows = 0
    for path in paths:
        pure_boxes = semantic_main[path.name]["boxes"]
        scores = semantic_main[path.name]["scores"].copy()
        if (
            len(pure_boxes) != MAX_DETECTIONS
            or len(coordinate_specialist[path.name]["boxes"]) == 0
        ):
            raise RuntimeError(f"Incomplete detector output for {path.name}")
        pure = pure_boxes[0]
        refined = (
            match_boundary_candidate(
                pure * IMAGE_SIZE, boundary_candidates[path.name]
            )
            / IMAGE_SIZE
        )
        protected = pure + BOUNDARY_BLEND_WEIGHT * (refined - pure)
        specialist = coordinate_specialist[path.name]["boxes"][0]
        triangulated = np.median(
            np.stack((protected, refined, specialist)), axis=0
        )
        triangulated = np.clip(triangulated, 0.0, 1.0)
        if (
            triangulated[2] < triangulated[0] + 1.0 / IMAGE_SIZE
            or triangulated[3] < triangulated[1] + 1.0 / IMAGE_SIZE
        ):
            triangulated = protected.copy()
        boxes = pure_boxes.copy()
        boxes[0] = triangulated
        lines = []  # final-round sample requires no header
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = np.clip(box, 0.0, 1.0)
            width, height = x2 - x1, y2 - y1
            if width <= 0.0 or height <= 0.0:
                continue
            lines.append(
                "0 " + " ".join(f"{value:.8f}" for value in (
                    (x1 + x2) / 2, (y1 + y2) / 2, width, height, float(score)
                ))
            )
        total_rows += len(lines)
        (output / f"{path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"images": len(paths), "prediction_rows": total_rows}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Inference requires an NVIDIA CUDA GPU")
    paths = image_paths(args.images.resolve(), args.expected_images)
    weights = resolve_weights(args.weights.resolve())
    semantic_main = infer_semantic_main(paths, weights, device, args.batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    boundary_candidates = infer_boundary_refiner(paths, weights, device)
    gc.collect()
    torch.cuda.empty_cache()
    coordinate_specialist = infer_coordinate_specialist(
        paths, weights, device, args.batch_size
    )
    result = write_submission(
        paths,
        semantic_main,
        boundary_candidates,
        coordinate_specialist,
        args.output.resolve(),
    )
    result["weights"] = {name: {"bytes": path.stat().st_size}
                         for name, path in weights.items()}
    result["test_used_for_training_or_selection"] = False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
