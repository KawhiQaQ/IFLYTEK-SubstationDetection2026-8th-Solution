"""Apply the trained four-edge boundary refiner to YOLO proposals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def refine_yolo_cache(
    payload: dict[str, Any],
    expected_paths: set[Path],
    model: Any,
    model_config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Refine every cached YOLO proposal without changing its confidence."""
    from workspace.models.boundary_refiner import apply_residuals_to_boxes
    from workspace.models.refinement_data import crop_context_tensor

    records = payload["records"]
    record_paths = {
        Path(record["image_path"]).expanduser().resolve()
        for record in records
    }
    if record_paths != expected_paths or len(records) != len(expected_paths):
        raise RuntimeError("YOLO proposal cache does not match its image split")

    refined_by_name: dict[str, np.ndarray] = {}
    scores_by_name: dict[str, list[float]] = {}
    for record_index, record in enumerate(records):
        image_path = Path(record["image_path"]).expanduser().resolve()
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.shape[:2] != (1024, 1024):
            raise RuntimeError(f"Invalid 1024x1024 image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
            .to(device)
        )
        boxes = torch.tensor(
            record["boxes_xyxy"], dtype=torch.float32, device=device
        ).reshape(-1, 4)
        if len(boxes):
            tight = crop_context_tensor(
                image,
                boxes,
                float(model_config["tight_scale"]),
                int(model_config["crop_size"]),
            )
            wide = crop_context_tensor(
                image,
                boxes,
                float(model_config["wide_scale"]),
                int(model_config["crop_size"]),
            )
            widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(2.0)
            heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(2.0)
            geometry = torch.stack(
                (
                    torch.log2(widths / 1024.0),
                    torch.log2(heights / 1024.0),
                    torch.log(widths / heights),
                ),
                dim=1,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                residuals = model.decode(model(tight, wide, geometry))
            refined = apply_residuals_to_boxes(
                boxes, residuals, 1024, 1024
            ).cpu().numpy()
        else:
            refined = np.empty((0, 4), dtype=np.float32)
        refined_by_name[image_path.name] = refined
        scores_by_name[image_path.name] = [
            float(value) for value in record["confidences"]
        ]
        if (
            (record_index + 1) % 200 == 0
            or record_index + 1 == len(records)
        ):
            print(
                f"boundary_refine={record_index + 1}/{len(records)}",
                flush=True,
            )
    return refined_by_name, scores_by_name
