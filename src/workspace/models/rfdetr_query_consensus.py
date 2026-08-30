"""Edge-wise query consensus for high-IoU RF-DETR localization.

The boundary-evidence module refines every detection query independently from high-resolution image
evidence.  The fold0 error audit nevertheless shows complementary proposals:
for some objects a lower-ranked query has a more accurate left, top, right, or
bottom edge than the winning query.

This module adds one final collective refinement stage.  Within each RF-DETR
training group, every seed query gathers the five highest-confidence proposals
that overlap it.  Four learned attention distributions aggregate those
neighbours independently for the four box edges, then predict a small residual
around the boundary-aware box. The design is inspired by Union-over-Intersections
(ICLR 2025), while retaining RF-DETR's fixed set of predictions and end-to-end
Hungarian training.

Neighbour selection is detached and restricted to the detector's independent
Group-DETR partitions.  The final regressor is zero-initialized, so installing
the module is an exact identity before optimization.
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

from workspace.models.rfdetr_boundary_evidence import (
    BBoxOnlyPostProcess,
    BoundaryEvidenceConfig,
    BoundaryEvidenceCriterion,
    BoundaryEvidenceHead,
    apply_edge_shift,
)


@dataclass(frozen=True)
class QueryConsensusConfig:
    """Architecture and loss settings for collective edge refinement."""

    boundary: BoundaryEvidenceConfig
    topk: int = 5
    overlap_threshold: float = 0.5
    overlap_rank_bonus: float = 0.05
    query_channels: int = 32
    pair_channels: int = 48
    value_channels: int = 32
    geometry_channels: int = 16
    max_edge_shift: float = 0.08
    edge_loss_weight: float = 1.0
    smooth_l1_beta: float = 0.01

    def validate(self) -> None:
        self.boundary.validate()
        if not 2 <= self.topk <= 16:
            raise ValueError("topk must be in [2, 16]")
        if not 0.0 < self.overlap_threshold < 1.0:
            raise ValueError("overlap_threshold must be in (0, 1)")
        if not 0.0 <= self.overlap_rank_bonus <= 0.25:
            raise ValueError("overlap_rank_bonus must be in [0, 0.25]")
        if min(
            self.query_channels,
            self.pair_channels,
            self.value_channels,
            self.geometry_channels,
        ) <= 0:
            raise ValueError("Consensus feature dimensions must be positive")
        if not 0.0 < self.max_edge_shift < 0.25:
            raise ValueError("max_edge_shift must be in (0, 0.25)")
        if self.edge_loss_weight <= 0.0 or self.smooth_l1_beta <= 0.0:
            raise ValueError("Consensus loss settings must be positive")


def _batched_pairwise_iou(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for a B x Q x 4 tensor without a Python batch loop."""

    top_left = torch.maximum(
        boxes_xyxy[:, :, None, :2],
        boxes_xyxy[:, None, :, :2],
    )
    bottom_right = torch.minimum(
        boxes_xyxy[:, :, None, 2:],
        boxes_xyxy[:, None, :, 2:],
    )
    intersection_wh = (bottom_right - top_left).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    wh = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp_min(0.0)
    area = wh[..., 0] * wh[..., 1]
    union = area[:, :, None] + area[:, None, :] - intersection
    return intersection / union.clamp_min(1e-7)


class EdgeWiseQueryConsensusHead(nn.Module):
    """Aggregate overlapping query evidence independently for four edges."""

    def __init__(
        self,
        query_dim: int,
        config: QueryConsensusConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, config.query_channels),
            nn.SiLU(inplace=True),
        )
        # Pair geometry contains relative x1/y1/x2/y2, log width/height
        # ratios, pair IoU, neighbour confidence, and seed confidence.
        pair_input_channels = config.query_channels * 2 + 9
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_input_channels),
            nn.Linear(pair_input_channels, config.pair_channels),
            nn.SiLU(inplace=True),
            nn.Linear(config.pair_channels, config.pair_channels),
            nn.SiLU(inplace=True),
        )
        self.edge_attention = nn.Linear(config.pair_channels, 4)
        self.value_projection = nn.Linear(
            config.pair_channels,
            config.value_channels,
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, config.geometry_channels),
            nn.SiLU(inplace=True),
            nn.Linear(config.geometry_channels, config.geometry_channels),
            nn.SiLU(inplace=True),
        )
        self.edge_embedding = nn.Embedding(4, 8)
        edge_input_channels = (
            config.query_channels
            + config.value_channels
            + config.geometry_channels
            + 8
            + 1
        )
        self.edge_regressor = nn.Sequential(
            nn.LayerNorm(edge_input_channels),
            nn.Linear(edge_input_channels, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.edge_regressor[-1].weight)
        nn.init.zeros_(self.edge_regressor[-1].bias)

    def forward(
        self,
        query_features: torch.Tensor,
        reference_boxes: torch.Tensor,
        logits: torch.Tensor,
        group_count: int,
        *,
        detach_neighbour_features: bool = False,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, query_count = reference_boxes.shape[:2]
        if group_count <= 0 or query_count % group_count:
            raise ValueError(
                "Query count must divide into complete Group-DETR partitions"
            )
        queries_per_group = query_count // group_count
        if queries_per_group < self.config.topk:
            raise ValueError("A query group is smaller than consensus topk")

        grouped_queries = query_features.reshape(
            batch_size * group_count,
            queries_per_group,
            query_features.shape[-1],
        )
        grouped_boxes = reference_boxes.reshape(
            batch_size * group_count,
            queries_per_group,
            4,
        )
        grouped_logits = logits.reshape(
            batch_size * group_count,
            queries_per_group,
            logits.shape[-1],
        )

        projected_queries = self.query_projection(grouped_queries)
        detached_boxes = grouped_boxes.detach()
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detached_boxes)
        pair_iou = _batched_pairwise_iou(boxes_xyxy)
        confidence = grouped_logits.detach().sigmoid().amax(dim=-1)

        valid_pairs = pair_iou >= self.config.overlap_threshold
        diagonal = torch.eye(
            queries_per_group,
            dtype=torch.bool,
            device=reference_boxes.device,
        ).unsqueeze(0)
        valid_pairs = valid_pairs | diagonal
        neighbour_rank = (
            confidence[:, None, :]
            + self.config.overlap_rank_bonus * pair_iou
        ).masked_fill(~valid_pairs, -1e4)
        _, neighbour_indices = neighbour_rank.topk(
            self.config.topk,
            dim=-1,
            sorted=True,
        )

        batch_indices = torch.arange(
            batch_size * group_count,
            device=reference_boxes.device,
        )[:, None, None]
        neighbour_query_bank = (
            projected_queries.detach()
            if detach_neighbour_features
            else projected_queries
        )
        neighbour_queries = neighbour_query_bank[
            batch_indices,
            neighbour_indices,
        ]
        neighbour_boxes = boxes_xyxy[batch_indices, neighbour_indices]
        neighbour_confidence = confidence[
            batch_indices,
            neighbour_indices,
        ]
        selected_iou = pair_iou.gather(dim=-1, index=neighbour_indices)
        selected_valid = valid_pairs.gather(
            dim=-1,
            index=neighbour_indices,
        )

        seed_xyxy = boxes_xyxy[:, :, None, :]
        seed_width = detached_boxes[..., 2].clamp_min(1e-4)
        seed_height = detached_boxes[..., 3].clamp_min(1e-4)
        edge_scale = torch.stack(
            (seed_width, seed_height, seed_width, seed_height),
            dim=-1,
        )
        relative_edges = (
            neighbour_boxes - seed_xyxy
        ) / edge_scale[:, :, None, :]
        neighbour_wh = (
            neighbour_boxes[..., 2:] - neighbour_boxes[..., :2]
        ).clamp_min(1e-4)
        size_ratio = torch.stack(
            (
                (
                    neighbour_wh[..., 0]
                    / seed_width[:, :, None]
                ).log(),
                (
                    neighbour_wh[..., 1]
                    / seed_height[:, :, None]
                ).log(),
            ),
            dim=-1,
        )
        pair_geometry = torch.cat(
            (
                relative_edges,
                size_ratio,
                selected_iou.unsqueeze(-1),
                neighbour_confidence.unsqueeze(-1),
                confidence[:, :, None, None].expand(
                    -1,
                    -1,
                    self.config.topk,
                    -1,
                ),
            ),
            dim=-1,
        )
        seed_queries = projected_queries[:, :, None, :].expand(
            -1,
            -1,
            self.config.topk,
            -1,
        )
        encoded_pairs = self.pair_encoder(
            torch.cat(
                (seed_queries, neighbour_queries, pair_geometry),
                dim=-1,
            )
        )
        attention_logits = self.edge_attention(encoded_pairs)
        attention_logits = attention_logits.masked_fill(
            ~selected_valid.unsqueeze(-1),
            -1e4,
        )
        attention = attention_logits.softmax(dim=2)
        pair_values = self.value_projection(encoded_pairs)
        aggregated_values = torch.einsum(
            "bqke,bqkv->bqev",
            attention,
            pair_values,
        )
        aggregated_edge_delta = (
            attention * relative_edges
        ).sum(dim=2)

        width = detached_boxes[..., 2].clamp_min(1e-4)
        height = detached_boxes[..., 3].clamp_min(1e-4)
        geometry = self.geometry_projection(
            torch.stack(
                (
                    detached_boxes[..., 0],
                    detached_boxes[..., 1],
                    width,
                    height,
                    width.log2(),
                    height.log2(),
                    (width / height).log(),
                ),
                dim=-1,
            )
        )
        edge_ids = torch.arange(4, device=reference_boxes.device)
        edge_embedding = self.edge_embedding(edge_ids).view(1, 1, 4, -1)
        edge_embedding = edge_embedding.expand(
            batch_size * group_count,
            queries_per_group,
            -1,
            -1,
        )
        edge_input = torch.cat(
            (
                projected_queries.unsqueeze(2).expand(-1, -1, 4, -1),
                aggregated_values,
                geometry.unsqueeze(2).expand(-1, -1, 4, -1),
                edge_embedding,
                aggregated_edge_delta.unsqueeze(-1),
            ),
            dim=-1,
        )
        normalized_shift = (
            self.edge_regressor(edge_input).squeeze(-1).tanh()
            * self.config.max_edge_shift
        )
        normalized_shift = normalized_shift.reshape(
            batch_size,
            query_count,
            4,
        )
        if not return_auxiliary:
            return normalized_shift
        auxiliary = {
            "pred_consensus_attention_logits": attention_logits.reshape(
                batch_size,
                query_count,
                self.config.topk,
                4,
            ),
            "pred_consensus_neighbour_boxes": neighbour_boxes.reshape(
                batch_size,
                query_count,
                self.config.topk,
                4,
            ),
            "pred_consensus_neighbour_valid": selected_valid.reshape(
                batch_size,
                query_count,
                self.config.topk,
            ),
        }
        return normalized_shift, auxiliary


class QueryConsensusLWDETR(LWDETR):
    """Boundary alignment followed by collective query refinement."""

    boundary_evidence_head: BoundaryEvidenceHead
    query_consensus_head: EdgeWiseQueryConsensusHead
    _detach_consensus_neighbour_features = False
    _return_consensus_training_auxiliary = False

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

        handle = self.transformer.register_forward_hook(
            capture_decoder_states
        )
        try:
            outputs = super().forward(samples, targets)
        finally:
            handle.remove()
        hidden_states = captured.get("hs")
        if hidden_states is None:
            raise RuntimeError("RF-DETR transformer did not expose decoder states")

        initial_boxes = outputs["pred_boxes"]
        boundary_shift = self.boundary_evidence_head(
            samples.tensors,
            hidden_states[-1],
            initial_boxes,
        )
        boundary_boxes = apply_edge_shift(initial_boxes, boundary_shift)
        group_count = self.group_detr if self.training else 1
        consensus_result = self.query_consensus_head(
            hidden_states[-1],
            boundary_boxes,
            outputs["pred_logits"],
            group_count=group_count,
            detach_neighbour_features=(
                self._detach_consensus_neighbour_features
            ),
            return_auxiliary=(
                self.training
                and self._return_consensus_training_auxiliary
            ),
        )
        if isinstance(consensus_result, tuple):
            consensus_shift, consensus_auxiliary = consensus_result
            outputs.update(consensus_auxiliary)
        else:
            consensus_shift = consensus_result
        outputs["pred_boxes"] = apply_edge_shift(
            boundary_boxes,
            consensus_shift,
        )
        outputs["pred_edge_shift"] = boundary_shift
        outputs["pred_edge_ref_boxes"] = initial_boxes
        outputs["pred_consensus_shift"] = consensus_shift
        outputs["pred_consensus_ref_boxes"] = boundary_boxes
        return outputs


def install_query_consensus(
    model: LWDETR,
    config: QueryConsensusConfig,
) -> QueryConsensusLWDETR:
    """Install the identity-initialized boundary and consensus architecture."""

    config.validate()
    if not isinstance(model, LWDETR):
        raise TypeError(f"Expected LWDETR, received {type(model)!r}")
    if isinstance(model, QueryConsensusLWDETR):
        return model
    query_dim = int(model.transformer.d_model)
    model.boundary_evidence_head = BoundaryEvidenceHead(
        query_dim,
        config.boundary,
    )
    model.query_consensus_head = EdgeWiseQueryConsensusHead(
        query_dim,
        config,
    )
    model.__class__ = QueryConsensusLWDETR
    return model


def load_query_consensus_state_dict(
    model: LWDETR,
    checkpoint: dict[str, Any],
    config: QueryConsensusConfig,
) -> QueryConsensusLWDETR:
    """Install query consensus and restore its detector checkpoint exactly."""

    refined_model = install_query_consensus(model, config)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = refined_model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Query-consensus checkpoint mismatch: "
            f"{incompatible.missing_keys}, {incompatible.unexpected_keys}"
        )
    return refined_model


class QueryConsensusCriterion(nn.Module):
    """Boundary criterion plus direct final consensus-edge supervision."""

    supports_loss_normalizer_override: bool = True

    def __init__(
        self,
        base_criterion: BoundaryEvidenceCriterion,
        config: QueryConsensusConfig,
    ) -> None:
        super().__init__()
        self.base = base_criterion
        self.config = config
        self.weight_dict = copy.copy(base_criterion.weight_dict)
        self.weight_dict["loss_consensus_edge"] = float(
            config.edge_loss_weight
        )

    @property
    def detr_criterion(self) -> nn.Module:
        return self.base.base

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
        if "pred_consensus_shift" not in outputs:
            return losses

        detr = self.detr_criterion
        group_detr = detr.group_detr if self.training else 1
        match_outputs = {
            key: value
            for key, value in outputs.items()
            if key != "aux_outputs"
        }
        indices = detr.matcher(
            match_outputs,
            targets,
            group_detr=group_detr,
        )
        batch_indices, query_indices = detr._get_src_permutation_idx(
            indices
        )
        if query_indices.numel() == 0:
            losses["loss_consensus_edge"] = (
                outputs["pred_consensus_shift"].sum() * 0.0
            )
            return losses

        predicted_shift = outputs["pred_consensus_shift"][
            batch_indices,
            query_indices,
        ]
        reference = outputs["pred_consensus_ref_boxes"][
            batch_indices,
            query_indices,
        ].detach()
        target_boxes = torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, indices)
            ],
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
            normalizer = detr.num_boxes_for_targets(outputs, targets)
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
        losses["loss_consensus_edge"] = (
            F.smooth_l1_loss(
                predicted_shift,
                target_shift,
                beta=self.config.smooth_l1_beta,
                reduction="none",
            ).sum()
            / normalizer
        )
        return losses


_TRAINING_CONFIG: QueryConsensusConfig | None = None
_TRAINING_BBOX_ONLY_EVALUATION = False


def configure_training_patch(
    config: QueryConsensusConfig,
    *,
    bbox_only_evaluation: bool = False,
) -> None:
    """Patch RF-DETR's training factory for query-consensus learning."""

    config.validate()
    global _TRAINING_CONFIG
    global _TRAINING_BBOX_ONLY_EVALUATION
    _TRAINING_CONFIG = config
    _TRAINING_BBOX_ONLY_EVALUATION = bool(bbox_only_evaluation)

    import rfdetr.training as training_package
    from rfdetr.training.module_model import RFDETRModelModule

    class QueryConsensusRFDETRModelModule(RFDETRModelModule):
        def __init__(self, model_config: Any, train_config: Any) -> None:
            super().__init__(model_config, train_config)
            if _TRAINING_CONFIG is None:
                raise RuntimeError("Query consensus config was not installed")
            self.model = install_query_consensus(
                self.model,
                _TRAINING_CONFIG,
            )
            boundary_criterion = BoundaryEvidenceCriterion(
                self.criterion,
                _TRAINING_CONFIG.boundary,
            )
            self.criterion = QueryConsensusCriterion(
                boundary_criterion,
                _TRAINING_CONFIG,
            )
            if _TRAINING_BBOX_ONLY_EVALUATION:
                self.postprocess = BBoxOnlyPostProcess(self.postprocess)

    training_package.RFDETRModelModule = QueryConsensusRFDETRModelModule
    if _TRAINING_BBOX_ONLY_EVALUATION:
        original_build_trainer = training_package.build_trainer

        def build_bbox_evaluation_trainer(
            train_config: Any,
            model_config: Any,
            **trainer_kwargs: Any,
        ) -> Any:
            # Build only the bbox evaluator/checkpoint monitor while preserving
            # the segmentation flag for the data module, criterion and mask
            # auxiliary loss during fit.
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
