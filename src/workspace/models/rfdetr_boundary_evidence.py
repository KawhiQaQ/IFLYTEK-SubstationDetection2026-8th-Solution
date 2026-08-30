"""High-resolution boundary-evidence alignment for RF-DETR.

RF-DETR-L uses a strong semantic transformer feature at stride 16.  This module
adds a lightweight stride-4 localization-only path.  For each decoder box it
samples learned RGB features and fixed Sobel cues along the four predicted
boundaries, conditions those profiles on the decoder query, and predicts small
per-edge corrections.

The module is trained inside the detector: refined boxes participate in
Hungarian matching and RF-DETR's L1/GIoU losses, while an additional normalized
edge loss directly supervises the localization branch.  The final correction
layer is zero-initialized, making installation an exact identity operation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities import box_ops
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list


@dataclass(frozen=True)
class BoundaryEvidenceConfig:
    """Fixed architecture and loss settings for the boundary module."""

    feature_channels: int = 24
    query_channels: int = 32
    profile_channels: int = 16
    normal_samples: int = 5
    tangent_samples: int = 5
    normal_range: float = 0.12
    max_edge_shift: float = 0.15
    edge_loss_weight: float = 1.0
    smooth_l1_beta: float = 0.02

    def validate(self) -> None:
        if min(
            self.feature_channels,
            self.query_channels,
            self.profile_channels,
        ) <= 0:
            raise ValueError("Boundary feature dimensions must be positive")
        if self.normal_samples < 3 or self.normal_samples % 2 == 0:
            raise ValueError("normal_samples must be odd and >= 3")
        if self.tangent_samples < 3:
            raise ValueError("tangent_samples must be >= 3")
        if not 0 < self.normal_range <= 0.25:
            raise ValueError("normal_range must be in (0, 0.25]")
        if not 0 < self.max_edge_shift < 0.5:
            raise ValueError("max_edge_shift must be in (0, 0.5)")
        if self.edge_loss_weight <= 0 or self.smooth_l1_beta <= 0:
            raise ValueError("Boundary loss settings must be positive")


class ConvGNAct(nn.Sequential):
    """Convolution, group normalization, and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True),
        )


class HighResolutionBoundaryEncoder(nn.Module):
    """Low-cost stride-4 encoder that preserves spatial edge evidence."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        stem_channels = max(16, channels)
        self.stem = ConvGNAct(3, stem_channels, 7, stride=4)
        self.depthwise = nn.Sequential(
            ConvGNAct(
                stem_channels,
                stem_channels,
                3,
                groups=stem_channels,
            ),
            ConvGNAct(stem_channels, channels, 1),
            ConvGNAct(channels, channels, 3, groups=channels),
            ConvGNAct(channels, channels, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.depthwise(self.stem(images))


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    centers = boxes[..., :2]
    half_size = boxes[..., 2:] * 0.5
    return torch.cat((centers - half_size, centers + half_size), dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    top_left, bottom_right = boxes[..., :2], boxes[..., 2:]
    return torch.cat(
        ((top_left + bottom_right) * 0.5, bottom_right - top_left),
        dim=-1,
    )


class BoundaryEvidenceHead(nn.Module):
    """Sample four directional profiles and regress normalized edge shifts."""

    def __init__(self, query_dim: int, config: BoundaryEvidenceConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = HighResolutionBoundaryEncoder(config.feature_channels)
        cue_channels = config.feature_channels + 3
        self.profile_projection = nn.Sequential(
            nn.LayerNorm(cue_channels * 2),
            nn.Linear(cue_channels * 2, config.profile_channels),
            nn.SiLU(inplace=True),
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, config.query_channels),
            nn.SiLU(inplace=True),
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.edge_embedding = nn.Embedding(4, 8)
        input_dim = (
            config.normal_samples * config.profile_channels
            + config.query_channels
            + 16
            + 8
        )
        self.edge_regressor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.edge_regressor[-1].weight)
        nn.init.zeros_(self.edge_regressor[-1].bias)

        normal = torch.linspace(
            -config.normal_range,
            config.normal_range,
            config.normal_samples,
        )
        tangent = torch.linspace(0.10, 0.90, config.tangent_samples)
        self.register_buffer("normal_positions", normal, persistent=True)
        self.register_buffer("tangent_positions", tangent, persistent=True)

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ) / 8.0
        sobel_y = sobel_x.transpose(0, 1).contiguous()
        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _boundary_features(self, normalized_images: torch.Tensor) -> torch.Tensor:
        images = (
            normalized_images.float() * self.image_std + self.image_mean
        ).clamp(0.0, 1.0)
        learned = self.encoder(images)
        gray = (
            images[:, 0:1] * 0.2989
            + images[:, 1:2] * 0.5870
            + images[:, 2:3] * 0.1140
        )
        padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, self.sobel_x).abs()
        grad_y = F.conv2d(padded, self.sobel_y).abs()
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
        sobel = F.interpolate(
            torch.cat((grad_x, grad_y, magnitude), dim=1),
            size=learned.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return torch.cat((learned, sobel.to(learned.dtype)), dim=1)

    def _sample_edge(
        self,
        features: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        edge_index: int,
    ) -> torch.Tensor:
        """Return B x Q x N x 2C mean/max directional profiles."""

        batch_size, query_count = boxes_xyxy.shape[:2]
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
        width = (x2 - x1).clamp_min(1e-4)
        height = (y2 - y1).clamp_min(1e-4)
        normal = self.normal_positions.to(boxes_xyxy.dtype)
        tangent = self.tangent_positions.to(boxes_xyxy.dtype)

        if edge_index in (0, 2):
            base_x = x1 if edge_index == 0 else x2
            sample_x = base_x[..., None] + width[..., None] * normal
            sample_y = y1[..., None] + height[..., None] * tangent
            grid_x = sample_x[..., :, None].expand(
                -1,
                -1,
                -1,
                self.config.tangent_samples,
            )
            grid_y = sample_y[..., None, :].expand(
                -1,
                -1,
                self.config.normal_samples,
                -1,
            )
        else:
            base_y = y1 if edge_index == 1 else y2
            sample_y = base_y[..., None] + height[..., None] * normal
            sample_x = x1[..., None] + width[..., None] * tangent
            grid_y = sample_y[..., :, None].expand(
                -1,
                -1,
                -1,
                self.config.tangent_samples,
            )
            grid_x = sample_x[..., None, :].expand(
                -1,
                -1,
                self.config.normal_samples,
                -1,
            )

        grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
        sampled = F.grid_sample(
            features,
            grid.reshape(
                batch_size,
                query_count * self.config.normal_samples,
                self.config.tangent_samples,
                2,
            ),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.reshape(
            batch_size,
            features.shape[1],
            query_count,
            self.config.normal_samples,
            self.config.tangent_samples,
        )
        mean = sampled.mean(dim=-1)
        maximum = sampled.amax(dim=-1)
        return torch.cat((mean, maximum), dim=1).permute(0, 2, 3, 1)

    def forward(
        self,
        normalized_images: torch.Tensor,
        query_features: torch.Tensor,
        reference_boxes: torch.Tensor,
    ) -> torch.Tensor:
        boundary_features = self._boundary_features(normalized_images)
        sample_boxes = cxcywh_to_xyxy(reference_boxes.detach())
        profiles = []
        for edge_index in range(4):
            edge_profile = self._sample_edge(
                boundary_features,
                sample_boxes,
                edge_index,
            )
            profiles.append(self.profile_projection(edge_profile))
        profile = torch.stack(profiles, dim=2).flatten(start_dim=3)

        query = self.query_projection(query_features).unsqueeze(2).expand(
            -1,
            -1,
            4,
            -1,
        )
        width = reference_boxes[..., 2].detach().clamp_min(1e-4)
        height = reference_boxes[..., 3].detach().clamp_min(1e-4)
        geometry = torch.stack(
            (
                reference_boxes[..., 0].detach(),
                reference_boxes[..., 1].detach(),
                width,
                height,
                width.log2(),
                height.log2(),
                (width / height).log(),
            ),
            dim=-1,
        )
        geometry = self.geometry_projection(geometry).unsqueeze(2).expand(
            -1,
            -1,
            4,
            -1,
        )
        edge_ids = torch.arange(4, device=reference_boxes.device)
        edge = self.edge_embedding(edge_ids).view(1, 1, 4, -1).expand(
            reference_boxes.shape[0],
            reference_boxes.shape[1],
            -1,
            -1,
        )
        raw_shift = self.edge_regressor(
            torch.cat((profile, query, geometry, edge), dim=-1)
        ).squeeze(-1)
        return raw_shift.tanh() * self.config.max_edge_shift


def apply_edge_shift(
    reference_boxes: torch.Tensor,
    normalized_shift: torch.Tensor,
) -> torch.Tensor:
    """Apply dx1/dy1/dx2/dy2 shifts relative to reference width/height."""

    reference_xyxy = cxcywh_to_xyxy(reference_boxes)
    width = reference_boxes[..., 2].clamp_min(1e-4)
    height = reference_boxes[..., 3].clamp_min(1e-4)
    scale = torch.stack((width, height, width, height), dim=-1)
    return xyxy_to_cxcywh(reference_xyxy + normalized_shift * scale)


class BoundaryEvidenceLWDETR(LWDETR):
    """LWDETR forward augmented with final-layer boundary evidence alignment."""

    boundary_evidence_head: BoundaryEvidenceHead

    def forward(self, samples: Any, targets: Any = None) -> dict[str, Any]:
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        captured: dict[str, torch.Tensor] = {}

        def capture_decoder_states(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            transformer_outputs: tuple[Any, ...],
        ) -> None:
            captured["hs"] = transformer_outputs[0]

        handle = self.transformer.register_forward_hook(capture_decoder_states)
        try:
            outputs = super().forward(samples, targets)
        finally:
            handle.remove()
        hidden_states = captured.get("hs")
        if hidden_states is None:
            raise RuntimeError("RF-DETR transformer did not expose decoder states")

        reference_boxes = outputs["pred_boxes"]
        normalized_shift = self.boundary_evidence_head(
            samples.tensors,
            hidden_states[-1],
            reference_boxes,
        )
        outputs["pred_boxes"] = apply_edge_shift(
            reference_boxes,
            normalized_shift,
        )
        outputs["pred_edge_shift"] = normalized_shift
        outputs["pred_edge_ref_boxes"] = reference_boxes
        return outputs


def install_boundary_evidence(
    model: LWDETR,
    config: BoundaryEvidenceConfig,
) -> BoundaryEvidenceLWDETR:
    """Install the zero-initialized boundary module on a loaded RF-DETR model."""

    config.validate()
    if not isinstance(model, LWDETR):
        raise TypeError(f"Expected LWDETR, received {type(model)!r}")
    if isinstance(model, BoundaryEvidenceLWDETR):
        return model
    model.boundary_evidence_head = BoundaryEvidenceHead(
        int(model.transformer.d_model),
        config,
    )
    model.__class__ = BoundaryEvidenceLWDETR
    return model


class BoundaryEvidenceCriterion(nn.Module):
    """RF-DETR criterion plus direct normalized four-edge supervision."""

    supports_loss_normalizer_override: bool = True

    def __init__(
        self,
        base_criterion: nn.Module,
        config: BoundaryEvidenceConfig,
    ) -> None:
        super().__init__()
        self.base = base_criterion
        self.config = config
        self.weight_dict = copy.copy(base_criterion.weight_dict)
        self.weight_dict["loss_edge"] = float(config.edge_loss_weight)

    def num_boxes_for_targets(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        return self.base.num_boxes_for_targets(outputs, targets)

    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]],
        num_boxes: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = self.base(outputs, targets, num_boxes=num_boxes)
        if "pred_edge_shift" not in outputs:
            return losses

        group_detr = self.base.group_detr if self.training else 1
        match_outputs = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        indices = self.base.matcher(
            match_outputs,
            targets,
            group_detr=group_detr,
        )
        batch_indices, query_indices = self.base._get_src_permutation_idx(indices)
        if query_indices.numel() == 0:
            losses["loss_edge"] = outputs["pred_edge_shift"].sum() * 0.0
            return losses

        predicted_shift = outputs["pred_edge_shift"][
            batch_indices,
            query_indices,
        ]
        reference = outputs["pred_edge_ref_boxes"][
            batch_indices,
            query_indices,
        ].detach()
        target_boxes = torch.cat(
            [target["boxes"][target_index] for target, (_, target_index) in zip(targets, indices)],
            dim=0,
        )
        reference_xyxy = box_ops.box_cxcywh_to_xyxy(reference)
        target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        width = reference[:, 2].clamp_min(1e-4)
        height = reference[:, 3].clamp_min(1e-4)
        edge_scale = torch.stack((width, height, width, height), dim=-1)
        target_shift = (
            (target_xyxy - reference_xyxy) / edge_scale
        ).clamp(
            -self.config.max_edge_shift,
            self.config.max_edge_shift,
        )

        if num_boxes is None:
            normalizer = self.base.num_boxes_for_targets(outputs, targets)
        elif torch.is_tensor(num_boxes):
            normalizer = num_boxes.to(
                device=predicted_shift.device,
                dtype=torch.float32,
            )
        else:
            normalizer = torch.as_tensor(
                num_boxes,
                device=predicted_shift.device,
                dtype=torch.float32,
            )
        edge_loss = F.smooth_l1_loss(
            predicted_shift,
            target_shift,
            beta=self.config.smooth_l1_beta,
            reduction="none",
        )
        losses["loss_edge"] = edge_loss.sum() / normalizer
        return losses


_TRAINING_CONFIG: BoundaryEvidenceConfig | None = None
_TRAINING_ENCODER_STATE: dict[str, torch.Tensor] | None = None
_TRAINING_BBOX_ONLY_EVALUATION = False


class BBoxOnlyPostProcess(nn.Module):
    """Drop auxiliary mask logits only at validation post-processing time."""

    def __init__(self, base_postprocess: nn.Module) -> None:
        super().__init__()
        self.base = base_postprocess

    def forward(
        self,
        outputs: dict[str, Any],
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        bbox_outputs = {
            key: value
            for key, value in outputs.items()
            if key != "pred_masks"
        }
        return self.base(bbox_outputs, target_sizes)


def configure_training_patch(
    config: BoundaryEvidenceConfig,
    encoder_state_dict: dict[str, torch.Tensor] | None = None,
    bbox_only_evaluation: bool = False,
) -> None:
    """Patch RF-DETR's high-level training factory for the boundary module."""

    config.validate()
    global _TRAINING_CONFIG
    global _TRAINING_ENCODER_STATE
    global _TRAINING_BBOX_ONLY_EVALUATION
    _TRAINING_CONFIG = config
    _TRAINING_ENCODER_STATE = encoder_state_dict
    _TRAINING_BBOX_ONLY_EVALUATION = bool(bbox_only_evaluation)

    import rfdetr.training as training_package
    from rfdetr.training.module_model import RFDETRModelModule

    class BoundaryEvidenceRFDETRModelModule(RFDETRModelModule):
        def __init__(self, model_config: Any, train_config: Any) -> None:
            super().__init__(model_config, train_config)
            if _TRAINING_CONFIG is None:
                raise RuntimeError("Boundary evidence config was not installed")
            self.model = install_boundary_evidence(
                self.model,
                _TRAINING_CONFIG,
            )
            if _TRAINING_ENCODER_STATE is not None:
                incompatible = (
                    self.model.boundary_evidence_head.encoder.load_state_dict(
                        _TRAINING_ENCODER_STATE,
                        strict=True,
                    )
                )
                if (
                    incompatible.missing_keys
                    or incompatible.unexpected_keys
                ):
                    raise RuntimeError(
                        "Boundary encoder pretraining mismatch: "
                        f"{incompatible.missing_keys}, "
                        f"{incompatible.unexpected_keys}"
                    )
            self.criterion = BoundaryEvidenceCriterion(
                self.criterion,
                _TRAINING_CONFIG,
            )
            if _TRAINING_BBOX_ONLY_EVALUATION:
                self.postprocess = BBoxOnlyPostProcess(self.postprocess)

    training_package.RFDETRModelModule = BoundaryEvidenceRFDETRModelModule
    if _TRAINING_BBOX_ONLY_EVALUATION:
        original_build_trainer = training_package.build_trainer

        def build_bbox_evaluation_trainer(
            train_config: Any,
            model_config: Any,
            **trainer_kwargs: Any,
        ) -> Any:
            # Trainer construction only uses this flag to select COCO callback
            # IoU types and checkpoint monitors. Restore it before fit so the
            # data module still decodes masks and the model retains its head.
            original_flag = bool(model_config.segmentation_head)
            model_config.segmentation_head = False
            try:
                return original_build_trainer(
                    train_config,
                    model_config,
                    **trainer_kwargs,
                )
            finally:
                model_config.segmentation_head = original_flag

        training_package.build_trainer = build_bbox_evaluation_trainer


def load_boundary_evidence_state_dict(
    model: LWDETR,
    checkpoint: dict[str, Any],
    config: BoundaryEvidenceConfig,
) -> BoundaryEvidenceLWDETR:
    """Install the module and restore a boundary-aware checkpoint exactly."""

    refined_model = install_boundary_evidence(model, config)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = refined_model.load_state_dict(state_dict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Boundary-evidence checkpoint mismatch: "
            f"{incompatible.missing_keys}, {incompatible.unexpected_keys}"
        )
    return refined_model
