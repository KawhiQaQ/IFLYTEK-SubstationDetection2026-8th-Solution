"""Data and prediction-cache utilities for boundary refinement."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_path_list(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def image_to_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"No images component in {image_path}") from exc
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_yolo_gt_xyxy(
    label_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO fields in {label_path}: {line}")
        class_id, center_x, center_y, box_width, box_height = fields
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class {class_id} in {label_path}")
        x, y, w, h = map(float, (center_x, center_y, box_width, box_height))
        boxes.append(
            [
                (x - w / 2.0) * width,
                (y - h / 2.0) * height,
                (x + w / 2.0) * width,
                (y + h / 2.0) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_iou_numpy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
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


def crop_context_rgb(
    image_rgb: np.ndarray,
    proposal_xyxy: np.ndarray,
    scale: float,
    output_size: int,
) -> torch.Tensor:
    """Sample a floating-point proposal crop with reflected image borders."""
    x1, y1, x2, y2 = map(float, proposal_xyxy)
    width = max(x2 - x1, 2.0) * scale
    height = max(y2 - y1, 2.0) * scale
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = center_x - width / 2.0
    crop_y1 = center_y - height / 2.0
    denominator = max(output_size - 1, 1)
    transform = np.asarray(
        [
            [width / denominator, 0.0, crop_x1],
            [0.0, height / denominator, crop_y1],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        image_rgb,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return torch.from_numpy(crop).permute(2, 0, 1).contiguous().float().div_(255.0)


def crop_context_tensor(
    image: torch.Tensor,
    proposals_xyxy: torch.Tensor,
    scale: float,
    output_size: int,
) -> torch.Tensor:
    """Differentiably sample N proposal crops from one CHW RGB image."""
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {image.shape}")
    if proposals_xyxy.ndim != 2 or proposals_xyxy.shape[1] != 4:
        raise ValueError(f"Expected Nx4 proposals, got {proposals_xyxy.shape}")
    count = proposals_xyxy.shape[0]
    if count == 0:
        return image.new_empty((0, image.shape[0], output_size, output_size))
    _, image_height, image_width = image.shape
    boxes = proposals_xyxy.float()
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(2.0) * scale
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(2.0) * scale
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    unit = torch.linspace(
        -0.5,
        0.5,
        output_size,
        device=image.device,
        dtype=torch.float32,
    )
    sample_x = centers_x[:, None] + widths[:, None] * unit[None, :]
    sample_y = centers_y[:, None] + heights[:, None] * unit[None, :]
    normalized_x = 2.0 * sample_x / max(image_width - 1, 1) - 1.0
    normalized_y = 2.0 * sample_y / max(image_height - 1, 1) - 1.0
    grid_x = normalized_x[:, None, :].expand(-1, output_size, -1)
    grid_y = normalized_y[:, :, None].expand(-1, -1, output_size)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    expanded = image.float().unsqueeze(0).expand(count, -1, -1, -1)
    return F.grid_sample(
        expanded,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )


def proposal_geometry(
    proposal_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    width = max(float(proposal_xyxy[2] - proposal_xyxy[0]), 2.0)
    height = max(float(proposal_xyxy[3] - proposal_xyxy[1]), 2.0)
    return torch.tensor(
        [
            math.log2(width / image_width),
            math.log2(height / image_height),
            math.log(width / height),
        ],
        dtype=torch.float32,
    )


def flip_spatial_relation(
    relation: torch.Tensor,
    grid_size: int,
    horizontal: bool,
    vertical: bool,
) -> torch.Tensor:
    """Apply image flips as matching row/column token permutations."""
    if relation.shape != (grid_size * grid_size, grid_size * grid_size):
        raise ValueError(
            f"Unexpected relation shape {tuple(relation.shape)} for grid {grid_size}"
        )
    indices = torch.arange(grid_size * grid_size).view(grid_size, grid_size)
    if horizontal:
        indices = indices.flip(1)
    if vertical:
        indices = indices.flip(0)
    permutation = indices.flatten()
    return relation.index_select(0, permutation).index_select(1, permutation)


def target_residuals(
    proposal_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
) -> np.ndarray:
    width = max(float(proposal_xyxy[2] - proposal_xyxy[0]), 2.0)
    height = max(float(proposal_xyxy[3] - proposal_xyxy[1]), 2.0)
    return np.asarray(
        [
            (gt_xyxy[0] - proposal_xyxy[0]) / width,
            (gt_xyxy[1] - proposal_xyxy[1]) / height,
            (gt_xyxy[2] - proposal_xyxy[2]) / width,
            (gt_xyxy[3] - proposal_xyxy[3]) / height,
        ],
        dtype=np.float32,
    )


def jitter_box(
    reference_xyxy: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Independently perturb all four boundaries relative to box size."""
    box = reference_xyxy.astype(np.float32).copy()
    width = max(float(box[2] - box[0]), 2.0)
    height = max(float(box[3] - box[1]), 2.0)
    noise = np.clip(rng.normal(0.0, sigma, 4), -3.0 * sigma, 3.0 * sigma)
    box += noise * np.asarray([width, height, width, height], dtype=np.float32)
    box[0] = np.clip(box[0], 0.0, image_width - 2.0)
    box[1] = np.clip(box[1], 0.0, image_height - 2.0)
    box[2] = np.clip(box[2], box[0] + 2.0, float(image_width))
    box[3] = np.clip(box[3], box[1] + 2.0, float(image_height))
    return box


def load_prediction_cache(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported prediction cache schema in {path}")
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"Missing records in {path}")
    return payload


@dataclass(frozen=True)
class RefinementRecord:
    image_path: Path
    image_width: int
    image_height: int
    gt_xyxy: np.ndarray
    baseline_xyxy: np.ndarray | None
    baseline_iou: float | None


class BoundaryRefinementDataset(Dataset):
    """One deterministic proposal/target pair per training object and epoch."""

    def __init__(
        self,
        train_list: Path,
        val_list: Path,
        prediction_cache: Path,
        crop_size: int = 160,
        tight_scale: float = 1.5,
        wide_scale: float = 2.0,
        residual_range: float = 0.25,
        base_seed: int = 3407,
        baseline_probability: float = 0.45,
        moderate_probability: float = 0.40,
        baseline_jitter_sigma: float = 0.008,
        moderate_jitter_sigma: float = 0.040,
        tiny_jitter_sigma: float = 0.004,
    ) -> None:
        super().__init__()
        self.train_list = train_list.resolve()
        self.val_list = val_list.resolve()
        self.prediction_cache = prediction_cache.resolve()
        self.crop_size = crop_size
        self.tight_scale = tight_scale
        self.wide_scale = wide_scale
        self.residual_range = residual_range
        self.base_seed = base_seed
        self.epoch = 0
        self.baseline_probability = baseline_probability
        self.moderate_probability = moderate_probability
        self.baseline_jitter_sigma = baseline_jitter_sigma
        self.moderate_jitter_sigma = moderate_jitter_sigma
        self.tiny_jitter_sigma = tiny_jitter_sigma

        if baseline_probability + moderate_probability > 1.0:
            raise ValueError("Proposal source probabilities exceed one")

        train_paths = read_path_list(self.train_list)
        val_paths = read_path_list(self.val_list)
        train_set = set(train_paths)
        overlap = train_set.intersection(val_paths)
        if overlap:
            raise RuntimeError(
                f"Train/validation leakage detected: {len(overlap)} paths overlap"
            )

        cache = load_prediction_cache(self.prediction_cache)
        cache_paths = {
            Path(record["image_path"]).resolve() for record in cache["records"]
        }
        if cache_paths != train_set:
            missing = train_set.difference(cache_paths)
            extra = cache_paths.difference(train_set)
            raise RuntimeError(
                f"Training cache does not exactly match fold train list: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        by_path = {
            Path(record["image_path"]).resolve(): record
            for record in cache["records"]
        }

        records: list[RefinementRecord] = []
        for image_path in train_paths:
            cached = by_path[image_path]
            width = int(cached["width"])
            height = int(cached["height"])
            gt_boxes = load_yolo_gt_xyxy(
                image_to_label(image_path),
                width=width,
                height=height,
            )
            predicted = np.asarray(cached["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
            ious = box_iou_numpy(gt_boxes, predicted)
            for gt_index, gt_box in enumerate(gt_boxes):
                baseline_box = None
                baseline_iou = None
                if predicted.size:
                    pred_index = int(np.argmax(ious[gt_index]))
                    candidate_iou = float(ious[gt_index, pred_index])
                    if candidate_iou >= 0.50:
                        baseline_box = predicted[pred_index].copy()
                        baseline_iou = candidate_iou
                records.append(
                    RefinementRecord(
                        image_path=image_path,
                        image_width=width,
                        image_height=height,
                        gt_xyxy=gt_box.copy(),
                        baseline_xyxy=baseline_box,
                        baseline_iou=baseline_iou,
                    )
                )
        if not records:
            raise RuntimeError("No training objects were loaded")
        self.records = records
        self.audit = {
            "train_images": len(train_paths),
            "validation_images": len(val_paths),
            "training_objects": len(records),
            "train_validation_overlap": 0,
            "cache_images": len(cache_paths),
            "cache_weights_sha256": cache.get("weights_sha256"),
            "baseline_matched_objects": sum(
                record.baseline_xyxy is not None for record in records
            ),
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _make_proposal(
        self,
        record: RefinementRecord,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str]:
        draw = float(rng.random())
        if (
            draw < self.baseline_probability
            and record.baseline_xyxy is not None
        ):
            proposal = jitter_box(
                record.baseline_xyxy,
                self.baseline_jitter_sigma,
                rng,
                record.image_width,
                record.image_height,
            )
            source = "baseline"
        elif draw < self.baseline_probability + self.moderate_probability:
            proposal = jitter_box(
                record.gt_xyxy,
                self.moderate_jitter_sigma,
                rng,
                record.image_width,
                record.image_height,
            )
            source = "moderate_jitter"
        else:
            proposal = jitter_box(
                record.gt_xyxy,
                self.tiny_jitter_sigma,
                rng,
                record.image_width,
                record.image_height,
            )
            source = "tiny_jitter"

        residual = target_residuals(proposal, record.gt_xyxy)
        if np.max(np.abs(residual)) > self.residual_range * 0.98:
            proposal = jitter_box(
                record.gt_xyxy,
                self.moderate_jitter_sigma,
                rng,
                record.image_width,
                record.image_height,
            )
            source = "range_fallback"
        return proposal, source

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        seed = self.base_seed + self.epoch * 1_000_003 + index * 9_176
        rng = np.random.default_rng(seed)
        proposal, source = self._make_proposal(record, rng)
        residual = target_residuals(proposal, record.gt_xyxy)

        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(record.image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if image_rgb.shape[1] != record.image_width or image_rgb.shape[0] != record.image_height:
            raise RuntimeError(
                f"Cached shape mismatch for {record.image_path}: "
                f"cache={(record.image_height, record.image_width)}, "
                f"actual={image_rgb.shape[:2]}"
            )
        image_tensor = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
        )
        proposal_tensor = torch.from_numpy(proposal).view(1, 4)
        tight = crop_context_tensor(
            image_tensor,
            proposal_tensor,
            self.tight_scale,
            self.crop_size,
        )[0]
        wide = crop_context_tensor(
            image_tensor,
            proposal_tensor,
            self.wide_scale,
            self.crop_size,
        )[0]

        horizontal_flip = rng.random() < 0.5
        vertical_flip = rng.random() < 0.5
        if horizontal_flip:
            tight = tight.flip(-1)
            wide = wide.flip(-1)
            residual = np.asarray(
                [-residual[2], residual[1], -residual[0], residual[3]],
                dtype=np.float32,
            )
        if vertical_flip:
            tight = tight.flip(-2)
            wide = wide.flip(-2)
            residual = np.asarray(
                [residual[0], -residual[3], residual[2], -residual[1]],
                dtype=np.float32,
            )

        payload = {
            "tight": tight,
            "wide": wide,
            "geometry": proposal_geometry(
                proposal,
                record.image_width,
                record.image_height,
            ),
            "target": torch.from_numpy(residual.copy()),
            "source": source,
        }
        satellite_relations = getattr(self, "satellite_relations", None)
        if satellite_relations is not None:
            relations = satellite_relations[record.image_path.resolve()]
            for grid_size, relation in relations.items():
                payload[f"sat_relation_{grid_size}"] = flip_spatial_relation(
                    relation,
                    int(grid_size),
                    horizontal_flip,
                    vertical_flip,
                )
        return payload
