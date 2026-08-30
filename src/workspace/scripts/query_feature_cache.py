"""Capture RF-DETR decoder queries and boxes during inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def transform_image(image: Image.Image, view: str) -> Image.Image:
    if view == "identity":
        return image
    if view == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if view == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if view == "rot90_cw":
        return image.transpose(Image.Transpose.ROTATE_270)
    if view == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if view == "rot270_cw":
        return image.transpose(Image.Transpose.ROTATE_90)
    if view == "transpose":
        return image.transpose(Image.Transpose.TRANSPOSE)
    if view == "transverse":
        return image.transpose(Image.Transpose.TRANSVERSE)
    raise ValueError(f"Unsupported geometric view: {view}")


def transform_boxes(boxes: np.ndarray, view: str) -> np.ndarray:
    if not len(boxes) or view == "identity":
        return boxes.copy()
    x1, y1, x2, y2 = boxes.T
    if view == "hflip":
        transformed = np.stack((1.0 - x2, y1, 1.0 - x1, y2), axis=-1)
    elif view == "vflip":
        transformed = np.stack((x1, 1.0 - y2, x2, 1.0 - y1), axis=-1)
    elif view == "rot90_cw":
        transformed = np.stack((1.0 - y2, x1, 1.0 - y1, x2), axis=-1)
    elif view == "rot180":
        transformed = np.stack(
            (1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1),
            axis=-1,
        )
    elif view == "rot270_cw":
        transformed = np.stack((y1, 1.0 - x2, y2, 1.0 - x1), axis=-1)
    elif view == "transpose":
        transformed = np.stack((y1, x1, y2, x2), axis=-1)
    elif view == "transverse":
        transformed = np.stack(
            (1.0 - y2, 1.0 - x2, 1.0 - y1, 1.0 - x1),
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported geometric view: {view}")
    return transformed.astype(np.float32, copy=False)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    centers = boxes[..., :2]
    half_sizes = boxes[..., 2:] / 2.0
    return torch.cat((centers - half_sizes, centers + half_sizes), dim=-1).clamp(
        0.0, 1.0
    )


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    first_area = np.clip(first[:, 2:] - first[:, :2], 0.0, None).prod(axis=1)
    second_area = np.clip(second[:, 2:] - second[:, :2], 0.0, None).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def build_cache(
    *,
    detector: Any,
    paths: list[Path],
    views: list[str],
    image_id_by_name: dict[str, int],
    boxes_by_name: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    detector_config = config["detector"]
    target_config = config["target"]
    batch_size = int(detector_config["batch_size"])
    top_k = int(detector_config["top_k"])
    class_index = int(detector_config["class_index"])
    decoder_layer = int(detector_config["decoder_layer"])
    thresholds = np.asarray(target_config["iou_thresholds"], dtype=np.float32)

    items = [
        (path, view)
        for path in paths
        for view in views
    ]
    captured: dict[str, Any] = {}
    forward_calls = 0

    def transformer_hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: tuple[Any, ...],
    ) -> None:
        captured["hs"] = output[0]

    def model_hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: dict[str, Any],
    ) -> None:
        nonlocal forward_calls
        forward_calls += 1
        captured["raw"] = output

    transformer_handle = detector.model.model.transformer.register_forward_hook(
        transformer_hook
    )
    model_handle = detector.model.model.register_forward_hook(model_hook)
    stored: dict[str, list[torch.Tensor]] = {
        "query_features": [],
        "base_logits": [],
        "boxes_cxcywh": [],
        "boxes_xyxy": [],
        "query_indices": [],
        "target_iou": [],
        "target_quality": [],
        "image_ids": [],
        "view_indices": [],
    }
    file_names: list[str] = []
    expected_forward_calls = 0
    try:
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            images: list[Image.Image] = []
            for image_path, view in batch_items:
                with Image.open(image_path) as source:
                    rgb = source.convert("RGB")
                    transformed = transform_image(rgb, view)
                    images.append(transformed.copy())

            captured.clear()
            before_calls = forward_calls
            autocast_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            ):
                detector.predict(
                    images,
                    threshold=0.0,
                    shape=(
                        int(detector_config["resolution"]),
                        int(detector_config["resolution"]),
                    ),
                    include_source_image=False,
                )
            expected_forward_calls += 1
            if forward_calls != before_calls + 1:
                raise RuntimeError("RF-DETR predict did not perform exactly one forward")
            if set(captured) != {"hs", "raw"}:
                raise RuntimeError(f"Incomplete RF-DETR hook capture: {set(captured)}")

            hidden = captured["hs"][decoder_layer]
            raw = captured["raw"]
            logits = raw["pred_logits"][..., class_index].float()
            raw_boxes = raw["pred_boxes"].float()
            if (
                hidden.shape[:2] != logits.shape
                or raw_boxes.shape[:2] != logits.shape
                or hidden.shape[-1] != int(config["model"]["query_dim"])
            ):
                raise RuntimeError("Unexpected RF-DETR query tensor shapes")
            top_logits, top_indices = torch.topk(
                logits,
                k=top_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            feature_indices = top_indices.unsqueeze(-1).expand(
                -1, -1, hidden.shape[-1]
            )
            box_indices = top_indices.unsqueeze(-1).expand(-1, -1, 4)
            top_features = torch.gather(hidden.float(), 1, feature_indices)
            top_boxes = torch.gather(raw_boxes, 1, box_indices)
            top_xyxy = cxcywh_to_xyxy(top_boxes)

            feature_cpu = top_features.detach().to(
                device="cpu", dtype=torch.float16
            )
            logits_cpu = top_logits.detach().cpu()
            boxes_cpu = top_boxes.detach().cpu()
            xyxy_cpu = top_xyxy.detach().cpu()
            indices_cpu = top_indices.detach().to(
                device="cpu", dtype=torch.int16
            )
            target_ious = []
            target_qualities = []
            image_ids = []
            view_indices = []
            for batch_index, (image_path, view) in enumerate(batch_items):
                ground_truth = transform_boxes(boxes_by_name[image_path.name], view)
                proposal_boxes = xyxy_cpu[batch_index].numpy()
                overlaps = box_iou(proposal_boxes, ground_truth)
                maximum_iou = (
                    overlaps.max(axis=1)
                    if overlaps.shape[1]
                    else np.zeros(top_k, dtype=np.float32)
                )
                metric_quality = (
                    maximum_iou[:, None] >= thresholds[None, :]
                ).mean(axis=1, dtype=np.float32)
                target_ious.append(torch.from_numpy(maximum_iou))
                target_qualities.append(torch.from_numpy(metric_quality))
                image_ids.append(image_id_by_name[image_path.name])
                view_indices.append(views.index(view))
                file_names.append(image_path.name)

            stored["query_features"].append(feature_cpu)
            stored["base_logits"].append(logits_cpu)
            stored["boxes_cxcywh"].append(boxes_cpu)
            stored["boxes_xyxy"].append(xyxy_cpu)
            stored["query_indices"].append(indices_cpu)
            stored["target_iou"].append(torch.stack(target_ious))
            stored["target_quality"].append(torch.stack(target_qualities))
            stored["image_ids"].append(torch.tensor(image_ids, dtype=torch.int64))
            stored["view_indices"].append(
                torch.tensor(view_indices, dtype=torch.int8)
            )
            completed = min(start + len(batch_items), len(items))
            if completed % 100 < len(batch_items) or completed == len(items):
                print(f"cached={completed}/{len(items)}", flush=True)
    finally:
        transformer_handle.remove()
        model_handle.remove()

    if forward_calls != expected_forward_calls:
        raise RuntimeError(
            f"RF-DETR forward count mismatch: {forward_calls} != "
            f"{expected_forward_calls}"
        )
    result = {
        key: torch.cat(chunks, dim=0)
        for key, chunks in stored.items()
    }
    result["file_names"] = file_names
    result["views"] = views
    result["schema_version"] = 1
    return result
