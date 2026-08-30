"""Query-conditioned high-resolution boundary decoder for DEIMv2.

The parent detector remains intact.  A zero-residual head samples fixed
outside/boundary/inside strips from the existing stride-4 spatial-prior stem
and the encoded stride-8 feature, then combines them with the final global
decoder query to refine four box edges.  The module is trained jointly during
the normal official-only target adaptation; it is not a checkpoint refiner.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from engine.core import register


class QueryConditionedBoundaryDecoder(nn.Module):
    """Fuse local high-resolution strips with a global DETR query."""

    def __init__(
        self,
        *,
        p2_in_channels: int = 64,
        p3_in_channels: int = 256,
        feature_channels: int = 32,
        query_dim: int = 256,
        hidden_dim: int = 128,
        tangent_samples: int = 7,
        normal_offsets: tuple[float, ...] = (-0.06, 0.0, 0.06),
        max_relative_residual: float = 0.08,
    ) -> None:
        super().__init__()
        if tangent_samples < 3 or len(normal_offsets) != 3:
            raise ValueError(
                "The decoder uses exactly three normal strips and at least "
                "three tangent samples"
            )
        self.p2_projection = nn.Sequential(
            nn.Conv2d(p2_in_channels, feature_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
        )
        self.p3_projection = nn.Sequential(
            nn.Conv2d(p3_in_channels, feature_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
        )
        self.register_buffer(
            "tangent_positions",
            torch.linspace(0.10, 0.90, tangent_samples),
            persistent=True,
        )
        self.register_buffer(
            "normal_offsets",
            torch.tensor(normal_offsets, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("edge_identity", torch.eye(4), persistent=True)
        local_dim = len(normal_offsets) * feature_channels * 2
        input_dim = local_dim + query_dim + 6 + 1 + 4
        self.norm = nn.LayerNorm(input_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.max_relative_residual = float(max_relative_residual)

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        center, size = boxes[..., :2], boxes[..., 2:]
        return torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1)

    def _edge_grid(self, boxes: torch.Tensor) -> torch.Tensor:
        """Return B,Q,4,N,K,2 grids with consistent outside-to-inside order."""
        xyxy = self._cxcywh_to_xyxy(boxes.detach())
        x1, y1, x2, y2 = (xyxy[..., index, None, None] for index in range(4))
        width = (x2 - x1).clamp_min(1e-4)
        height = (y2 - y1).clamp_min(1e-4)
        normal = self.normal_offsets.to(boxes)[None, None, :, None]
        tangent = self.tangent_positions.to(boxes)[None, None, None, :]

        # Negative normal is outside and positive normal is inside for all edges.
        left_x, left_y = x1 + normal * width, y1 + tangent * height
        right_x, right_y = x2 - normal * width, y1 + tangent * height
        top_x, top_y = x1 + tangent * width, y1 + normal * height
        bottom_x, bottom_y = x1 + tangent * width, y2 - normal * height
        edges = []
        for x_coord, y_coord in (
            (left_x, left_y),
            (top_x, top_y),
            (right_x, right_y),
            (bottom_x, bottom_y),
        ):
            x_coord, y_coord = torch.broadcast_tensors(x_coord, y_coord)
            edges.append(torch.stack((x_coord, y_coord), dim=-1))
        return torch.stack(edges, dim=2)

    def _sample_edges(self, feature: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        batch, queries, edges, normals, tangents, _ = grid.shape
        sampling_grid = grid.reshape(batch, queries * edges, normals * tangents, 2)
        sampled = F.grid_sample(
            feature,
            sampling_grid.mul(2.0).sub(1.0),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        channels = int(sampled.shape[1])
        sampled = sampled.reshape(batch, channels, queries, edges, normals, tangents)
        return sampled.mean(dim=-1).permute(0, 2, 3, 4, 1).flatten(3)

    def forward(
        self,
        *,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        query: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the small refiner numerically stable under the parent's BF16 AMP.
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            base = boxes.float()
            p2_feature = self.p2_projection(p2.float())
            p3_feature = self.p3_projection(p3.float())
            grid = self._edge_grid(base)
            local = torch.cat(
                (self._sample_edges(p2_feature, grid), self._sample_edges(p3_feature, grid)),
                dim=-1,
            )
            batch, queries = base.shape[:2]
            geometry = torch.cat(
                (base, base[..., 2:].clamp_min(1e-4).log()), dim=-1
            )[:, :, None, :].expand(-1, -1, 4, -1)
            global_query = query.float()[:, :, None, :].expand(-1, -1, 4, -1)
            score = logits.float().sigmoid().amax(dim=-1, keepdim=True).detach()
            score = score[:, :, None, :].expand(-1, -1, 4, -1)
            edge_identity = self.edge_identity.to(base)[None, None].expand(
                batch, queries, -1, -1
            )
            features = torch.cat(
                (local, global_query, geometry, score, edge_identity), dim=-1
            )
            residual = torch.tanh(self.residual_head(self.norm(features))).squeeze(-1)
            residual = residual * self.max_relative_residual

            # Analytic edge-to-cxcywh conversion preserves the parent bit for
            # bit at zero residual.  The bounded residual also guarantees
            # positive width/height without a clamp that could alter border
            # boxes before the new module has learned anything.
            left, top, right, bottom = residual.unbind(dim=-1)
            cx, cy, width, height = base.unbind(dim=-1)
            return torch.stack(
                (
                    cx + 0.5 * (left + right) * width,
                    cy + 0.5 * (top + bottom) * height,
                    width * (1.0 + right - left),
                    height * (1.0 + bottom - top),
                ),
                dim=-1,
            )


@register()
class QueryConditionedBoundaryDEIM(nn.Module):
    """DEIM wrapper whose final boxes consume global-query and local-edge evidence."""

    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        p2_in_channels: int = 64,
        p3_in_channels: int = 256,
        feature_channels: int = 32,
        query_dim: int = 256,
        boundary_hidden_dim: int = 128,
        tangent_samples: int = 7,
        normal_offsets: list[float] | tuple[float, ...] = (-0.06, 0.0, 0.06),
        max_relative_residual: float = 0.08,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        if not hasattr(backbone, "sta") or not hasattr(backbone.sta, "stem"):
            raise TypeError("This detector requires the DEIMv2 DINOv3 STA backbone")
        self.boundary_refiner = QueryConditionedBoundaryDecoder(
            p2_in_channels=p2_in_channels,
            p3_in_channels=p3_in_channels,
            feature_channels=feature_channels,
            query_dim=query_dim,
            hidden_dim=boundary_hidden_dim,
            tangent_samples=tangent_samples,
            normal_offsets=tuple(float(value) for value in normal_offsets),
            max_relative_residual=max_relative_residual,
        )

    def forward(self, x: torch.Tensor, targets: list[dict[str, Any]] | None = None):
        p2_holder: list[torch.Tensor] = []

        def capture_p2(_module, _inputs, output):
            p2_holder.append(output)

        p2_handle = self.backbone.sta.stem.register_forward_hook(capture_p2)
        try:
            backbone_features = self.backbone(x)
        finally:
            p2_handle.remove()
        if len(p2_holder) != 1:
            raise RuntimeError("Failed to capture exactly one stride-4 feature")
        encoded_features = self.encoder(backbone_features)

        hidden_holder: list[torch.Tensor] = []
        final_layer = self.decoder.decoder.layers[self.decoder.decoder.eval_idx]

        def capture_query(_module, _inputs, output):
            hidden_holder.append(output)

        query_handle = final_layer.register_forward_hook(capture_query)
        try:
            output = self.decoder(encoded_features, targets)
        finally:
            query_handle.remove()
        if len(hidden_holder) != 1:
            raise RuntimeError("Failed to capture exactly one final decoder query")
        base_boxes = output["pred_boxes"]
        query_count = int(base_boxes.shape[1])
        final_query = hidden_holder[0][:, -query_count:]
        refined_boxes = self.boundary_refiner(
            boxes=base_boxes,
            logits=output["pred_logits"],
            query=final_query,
            p2=p2_holder[0],
            p3=encoded_features[0],
        )
        output["base_pred_boxes"] = base_boxes
        output["pred_boxes"] = refined_boxes
        return output

    def deploy(self):
        self.eval()
        for module in self.modules():
            if module is not self and hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self
