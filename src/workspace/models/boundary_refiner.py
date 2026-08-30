"""Content-conditioned residual boundary refinement for a YOLO26x detector.

The detector remains unchanged. A frozen, shared copy of its shallow encoder
processes tight and wide proposal crops, while a small trainable head predicts
four non-uniform residual distributions for left/top/right/bottom boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_nonuniform_grid(
    bins: int = 33,
    residual_range: float = 0.25,
    beta: float = 2.5,
) -> torch.Tensor:
    """Return a symmetric grid with exponentially denser support near zero."""
    if bins < 3 or bins % 2 == 0:
        raise ValueError("bins must be an odd integer >= 3")
    if residual_range <= 0 or beta <= 0:
        raise ValueError("residual_range and beta must be positive")
    uniform = torch.linspace(-1.0, 1.0, bins)
    magnitude = torch.expm1(beta * uniform.abs()) / math.expm1(beta)
    grid = uniform.sign() * residual_range * magnitude
    grid[bins // 2] = 0.0
    return grid


class ConvNormAct(nn.Sequential):
    """Compact convolution block used only by the refinement head."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidual(nn.Module):
    """A low-parameter residual block that preserves spatial boundary cues."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FrozenYOLOShallowEncoder(nn.Module):
    """Reuse YOLO layers 0--4 and expose stride-4/stride-8 features."""

    def __init__(self, layers: Iterable[nn.Module]) -> None:
        super().__init__()
        selected = list(layers)
        if len(selected) != 5:
            raise ValueError(f"Expected exactly five YOLO layers, got {len(selected)}")
        self.layers = nn.ModuleList(selected)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenYOLOShallowEncoder":
        """Keep pretrained BatchNorm statistics frozen in every parent mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward_all(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1 = None
        p2 = None
        p3 = None
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index == 0:
                p1 = x
            elif index == 2:
                p2 = x
            elif index == 4:
                p3 = x
        if p1 is None or p2 is None or p3 is None:
            raise RuntimeError("Failed to collect YOLO shallow features")
        return p1, p2, p3

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, p2, p3 = self.forward_all(x)
        return p2, p3


class BoundaryDistributionRefiner(nn.Module):
    """Dual-context, four-edge distribution refinement head."""

    def __init__(
        self,
        encoder: FrozenYOLOShallowEncoder,
        bins: int = 33,
        residual_range: float = 0.25,
        grid_beta: float = 2.5,
        p2_channels: int = 384,
        p3_channels: int = 768,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bins = bins
        self.residual_range = residual_range
        self.grid_beta = grid_beta
        self.register_buffer(
            "residual_grid",
            make_nonuniform_grid(bins, residual_range, grid_beta),
            persistent=True,
        )

        self.reduce_p2 = ConvNormAct(p2_channels, 48, kernel_size=1)
        self.reduce_p3 = ConvNormAct(p3_channels, 48, kernel_size=1)
        self.context_fuser = nn.Sequential(
            ConvNormAct(96, 64),
            DepthwiseResidual(64),
            ConvNormAct(64, 96, stride=2),
        )
        self.dual_fuser = nn.Sequential(
            ConvNormAct(192, 128, stride=2),
            DepthwiseResidual(128),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128 * 5 * 5 + 32, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.classifier = nn.Linear(512, 4 * bins)

        # Symmetric uniform logits decode to exactly zero residual. This makes
        # the untrained composite detector identical to its parent.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def train(self, mode: bool = True) -> "BoundaryDistributionRefiner":
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode_context(self, crop: torch.Tensor) -> torch.Tensor:
        p2, p3 = self.encoder(crop)
        p2 = self.reduce_p2(p2)
        p3 = self.reduce_p3(p3)
        p3 = F.interpolate(
            p3,
            size=p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.context_fuser(torch.cat((p2, p3), dim=1))

    def forward(
        self,
        tight_crop: torch.Tensor,
        wide_crop: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if tight_crop.shape != wide_crop.shape:
            raise ValueError(
                f"Crop shapes differ: {tight_crop.shape} vs {wide_crop.shape}"
            )
        both = torch.cat((tight_crop, wide_crop), dim=0)
        encoded = self._encode_context(both)
        tight, wide = encoded.chunk(2, dim=0)
        spatial = self.dual_fuser(torch.cat((tight, wide), dim=1)).flatten(1)
        geometry_feature = self.geometry_encoder(geometry)
        logits = self.classifier(self.mlp(torch.cat((spatial, geometry_feature), dim=1)))
        return logits.view(-1, 4, self.bins)

    def distribution(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.float().softmax(dim=-1)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = self.distribution(logits)
        return (probabilities * self.residual_grid.float()).sum(dim=-1)

    def normalized_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = self.distribution(logits).clamp_min(1e-9)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        return entropy / math.log(self.bins)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize only the new module; base YOLO weights remain canonical."""
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("encoder.")
        }

    def load_trainable_state_dict(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        current = self.state_dict()
        expected = {key for key in current if not key.startswith("encoder.")}
        missing = expected.difference(state)
        unexpected = set(state).difference(expected)
        if missing or unexpected:
            raise RuntimeError(
                f"Refiner state mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for key, value in state.items():
                current[key].copy_(value)


class GatedBoundaryDistributionRefiner(BoundaryDistributionRefiner):
    """Global refiner with a learned mixture between correction and identity."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate_head = nn.Linear(512, 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        tight_crop: torch.Tensor,
        wide_crop: torch.Tensor,
        geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tight_crop.shape != wide_crop.shape:
            raise ValueError(
                f"Crop shapes differ: {tight_crop.shape} vs {wide_crop.shape}"
            )
        both = torch.cat((tight_crop, wide_crop), dim=0)
        encoded = self._encode_context(both)
        tight, wide = encoded.chunk(2, dim=0)
        spatial = self.dual_fuser(torch.cat((tight, wide), dim=1)).flatten(1)
        geometry_feature = self.geometry_encoder(geometry)
        hidden = self.mlp(torch.cat((spatial, geometry_feature), dim=1))
        logits = self.classifier(hidden).view(-1, 4, self.bins)
        return {
            "logits": logits,
            "gate_logits": self.gate_head(hidden),
        }

    def gate(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return output["gate_logits"].float().sigmoid()

    def distribution(
        self,
        output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        correction = output["logits"].float().softmax(dim=-1)
        gate = self.gate(output).unsqueeze(-1)
        identity = torch.zeros_like(correction)
        identity[..., self.bins // 2] = 1.0
        return gate * correction + (1.0 - gate) * identity

    def decode(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.distribution(output) * self.residual_grid.float()
        ).sum(dim=-1)

    def normalized_entropy(
        self,
        output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        probabilities = self.distribution(output).clamp_min(1e-9)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        return entropy / math.log(self.bins)


class SatelliteRelationBoundaryRefiner(BoundaryDistributionRefiner):
    """Global head exposing its spatial feature only for training distillation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_distillation_feature", None)

    def forward(
        self,
        tight_crop: torch.Tensor,
        wide_crop: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if tight_crop.shape != wide_crop.shape:
            raise ValueError(
                f"Crop shapes differ: {tight_crop.shape} vs {wide_crop.shape}"
            )
        both = torch.cat((tight_crop, wide_crop), dim=0)
        encoded = self._encode_context(both)
        tight, wide = encoded.chunk(2, dim=0)
        object.__setattr__(
            self,
            "_distillation_feature",
            torch.cat((tight, wide), dim=1),
        )
        spatial = self.dual_fuser(torch.cat((tight, wide), dim=1)).flatten(1)
        geometry_feature = self.geometry_encoder(geometry)
        logits = self.classifier(
            self.mlp(torch.cat((spatial, geometry_feature), dim=1))
        )
        return logits.view(-1, 4, self.bins)

    def take_distillation_feature(self) -> torch.Tensor:
        feature = object.__getattribute__(self, "_distillation_feature")
        if not torch.is_tensor(feature):
            raise RuntimeError("SAT student spatial feature was not captured")
        object.__setattr__(self, "_distillation_feature", None)
        return feature


class DirectionalBoundaryRefiner(nn.Module):
    """High-resolution directional strip search with an explicit identity expert."""

    def __init__(
        self,
        encoder: FrozenYOLOShallowEncoder,
        bins: int = 33,
        residual_range: float = 0.25,
        grid_beta: float = 2.5,
        tight_scale: float = 1.5,
        wide_scale: float = 2.0,
        edge_samples: int = 40,
        p1_channels: int = 96,
        p2_channels: int = 384,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bins = bins
        self.residual_range = residual_range
        self.grid_beta = grid_beta
        self.tight_scale = tight_scale
        self.wide_scale = wide_scale
        self.edge_samples = edge_samples
        self.register_buffer(
            "residual_grid",
            make_nonuniform_grid(bins, residual_range, grid_beta),
            persistent=True,
        )
        self.reduce_p1 = ConvNormAct(p1_channels, 24, kernel_size=1)
        self.reduce_p2 = ConvNormAct(p2_channels, 40, kernel_size=1)
        self.local_fuser = nn.Sequential(
            ConvNormAct(64, 64),
            DepthwiseResidual(64),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear((64 + 2) * 2 + 3, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
        )
        # Per candidate: mean/max strip statistics from tight and wide
        # contexts, a global semantic condition, and two position channels.
        candidate_channels = (64 + 2) * 4 + 64 + 2
        self.edge_scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(candidate_channels, 96, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(96, 64, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(64, 1, 1),
                )
                for _ in range(4)
            ]
        )
        self.identity_head = nn.Linear(64, 4)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
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

        # Both correction evidence and the identity prior start neutral.
        for scorer in self.edge_scorers:
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)
        nn.init.zeros_(self.identity_head.weight)
        nn.init.zeros_(self.identity_head.bias)

    def train(self, mode: bool = True) -> "DirectionalBoundaryRefiner":
        super().train(mode)
        self.encoder.eval()
        return self

    def _sobel_cues(self, crop: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        gray = (
            crop[:, 0:1] * 0.2989
            + crop[:, 1:2] * 0.5870
            + crop[:, 2:3] * 0.1140
        )
        padded = F.pad(gray.float(), (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, self.sobel_x).abs()
        grad_y = F.conv2d(padded, self.sobel_y).abs()
        cues = torch.cat((grad_x, grad_y), dim=1)
        return F.interpolate(
            cues,
            size=size,
            mode="bilinear",
            align_corners=True,
        )

    def _encode_local(self, crop: torch.Tensor) -> torch.Tensor:
        p1, p2, _ = self.encoder.forward_all(crop)
        p1 = self.reduce_p1(p1)
        p2 = self.reduce_p2(p2)
        p2 = F.interpolate(
            p2,
            size=p1.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        semantic = self.local_fuser(torch.cat((p1, p2), dim=1))
        return torch.cat(
            (semantic, self._sobel_cues(crop, semantic.shape[-2:])),
            dim=1,
        )

    def _edge_statistics(
        self,
        feature: torch.Tensor,
        edge_index: int,
        context_scale: float,
    ) -> torch.Tensor:
        """Return mean/max strip evidence as B x 2C x K."""
        batch_size = feature.shape[0]
        position = self.residual_grid.float() / context_scale
        span = torch.linspace(
            -0.55 / context_scale,
            0.55 / context_scale,
            self.edge_samples,
            device=feature.device,
            dtype=torch.float32,
        )
        if edge_index in (0, 2):
            base = -0.5 / context_scale if edge_index == 0 else 0.5 / context_scale
            candidate_x = 2.0 * (0.5 + base + position) - 1.0
            sample_y = 2.0 * (0.5 + span) - 1.0
            grid_x = candidate_x[None, None, :].expand(
                batch_size,
                self.edge_samples,
                -1,
            )
            grid_y = sample_y[None, :, None].expand(
                batch_size,
                -1,
                self.bins,
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)
            sampled = F.grid_sample(
                feature.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            mean = sampled.mean(dim=2)
            maximum = sampled.amax(dim=2)
        else:
            base = -0.5 / context_scale if edge_index == 1 else 0.5 / context_scale
            candidate_y = 2.0 * (0.5 + base + position) - 1.0
            sample_x = 2.0 * (0.5 + span) - 1.0
            grid_x = sample_x[None, None, :].expand(
                batch_size,
                self.bins,
                -1,
            )
            grid_y = candidate_y[None, :, None].expand(
                batch_size,
                -1,
                self.edge_samples,
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)
            sampled = F.grid_sample(
                feature.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            mean = sampled.mean(dim=3)
            maximum = sampled.amax(dim=3)
        return torch.cat((mean, maximum), dim=1)

    def forward(
        self,
        tight_crop: torch.Tensor,
        wide_crop: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if tight_crop.shape != wide_crop.shape:
            raise ValueError(
                f"Crop shapes differ: {tight_crop.shape} vs {wide_crop.shape}"
            )
        both = torch.cat((tight_crop, wide_crop), dim=0)
        encoded = self._encode_local(both)
        tight, wide = encoded.chunk(2, dim=0)
        global_context = self.context_encoder(
            torch.cat(
                (
                    tight.mean(dim=(-2, -1)),
                    wide.mean(dim=(-2, -1)),
                    geometry.float(),
                ),
                dim=1,
            )
        )
        grid_position = self.residual_grid.float() / self.residual_range
        position_features = torch.stack(
            (grid_position, grid_position.abs()),
            dim=0,
        ).unsqueeze(0).expand(tight.shape[0], -1, -1)
        edge_logits: list[torch.Tensor] = []
        for edge_index, scorer in enumerate(self.edge_scorers):
            tight_stats = self._edge_statistics(
                tight,
                edge_index,
                self.tight_scale,
            )
            wide_stats = self._edge_statistics(
                wide,
                edge_index,
                self.wide_scale,
            )
            candidates = torch.cat(
                (
                    tight_stats,
                    wide_stats,
                    global_context.unsqueeze(-1).expand(-1, -1, self.bins),
                    position_features,
                ),
                dim=1,
            )
            edge_logits.append(scorer(candidates).squeeze(1))
        logits = torch.stack(edge_logits, dim=1)
        identity = self.identity_head(global_context)
        identity_mask = F.one_hot(
            torch.tensor(
                self.bins // 2,
                device=logits.device,
            ),
            num_classes=self.bins,
        ).to(logits.dtype)
        return logits + identity.unsqueeze(-1) * identity_mask

    def distribution(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.float().softmax(dim=-1)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return (
            self.distribution(logits) * self.residual_grid.float()
        ).sum(dim=-1)

    def normalized_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = self.distribution(logits).clamp_min(1e-9)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        return entropy / math.log(self.bins)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("encoder.")
        }

    def load_trainable_state_dict(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        current = self.state_dict()
        expected = {key for key in current if not key.startswith("encoder.")}
        missing = expected.difference(state)
        unexpected = set(state).difference(expected)
        if missing or unexpected:
            raise RuntimeError(
                f"Refiner state mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for key, value in state.items():
                current[key].copy_(value)


class GlobalDirectionalBoundaryRefiner(BoundaryDistributionRefiner):
    """Global semantic refiner augmented with edge-aligned local evidence."""

    def __init__(
        self,
        *args,
        tight_scale: float = 1.5,
        wide_scale: float = 2.0,
        edge_samples: int = 40,
        p1_channels: int = 96,
        p2_channels: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(*args, p2_channels=p2_channels, **kwargs)
        self.tight_scale = tight_scale
        self.wide_scale = wide_scale
        self.edge_samples = edge_samples
        self.local_reduce_p1 = ConvNormAct(p1_channels, 24, kernel_size=1)
        self.local_reduce_p2 = ConvNormAct(p2_channels, 40, kernel_size=1)
        self.local_fuser = nn.Sequential(
            ConvNormAct(64, 64),
            DepthwiseResidual(64),
        )
        self.global_condition = nn.Sequential(
            nn.Linear(512, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
        )
        # For each context, mean/max strip pooling contributes 2*64 channels.
        candidate_channels = 64 * 4 + 64 + 2
        self.edge_scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(candidate_channels, 96, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(96, 64, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(64, 1, 1),
                )
                for _ in range(4)
            ]
        )
        for scorer in self.edge_scorers:
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)

    def _local_feature(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
    ) -> torch.Tensor:
        p1 = self.local_reduce_p1(p1)
        p2 = self.local_reduce_p2(p2)
        p2 = F.interpolate(
            p2,
            size=p1.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return self.local_fuser(torch.cat((p1, p2), dim=1))

    def _edge_statistics(
        self,
        feature: torch.Tensor,
        edge_index: int,
        context_scale: float,
    ) -> torch.Tensor:
        """Return mean/max tangent-strip evidence as B x 2C x K."""
        batch_size = feature.shape[0]
        position = self.residual_grid.float() / context_scale
        span = torch.linspace(
            -0.55 / context_scale,
            0.55 / context_scale,
            self.edge_samples,
            device=feature.device,
            dtype=torch.float32,
        )
        if edge_index in (0, 2):
            base = -0.5 / context_scale if edge_index == 0 else 0.5 / context_scale
            candidate_x = 2.0 * (0.5 + base + position) - 1.0
            sample_y = 2.0 * (0.5 + span) - 1.0
            grid_x = candidate_x[None, None, :].expand(
                batch_size, self.edge_samples, -1
            )
            grid_y = sample_y[None, :, None].expand(
                batch_size, -1, self.bins
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)
            sampled = F.grid_sample(
                feature.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            mean = sampled.mean(dim=2)
            maximum = sampled.amax(dim=2)
        else:
            base = -0.5 / context_scale if edge_index == 1 else 0.5 / context_scale
            candidate_y = 2.0 * (0.5 + base + position) - 1.0
            sample_x = 2.0 * (0.5 + span) - 1.0
            grid_x = sample_x[None, None, :].expand(
                batch_size, self.bins, -1
            )
            grid_y = candidate_y[None, :, None].expand(
                batch_size, -1, self.edge_samples
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)
            sampled = F.grid_sample(
                feature.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            mean = sampled.mean(dim=3)
            maximum = sampled.amax(dim=3)
        return torch.cat((mean, maximum), dim=1)

    def forward(
        self,
        tight_crop: torch.Tensor,
        wide_crop: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if tight_crop.shape != wide_crop.shape:
            raise ValueError(
                f"Crop shapes differ: {tight_crop.shape} vs {wide_crop.shape}"
            )
        both = torch.cat((tight_crop, wide_crop), dim=0)
        p1, p2, p3 = self.encoder.forward_all(both)

        global_p2 = self.reduce_p2(p2)
        global_p3 = self.reduce_p3(p3)
        global_p3 = F.interpolate(
            global_p3,
            size=global_p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        encoded = self.context_fuser(
            torch.cat((global_p2, global_p3), dim=1)
        )
        tight_global, wide_global = encoded.chunk(2, dim=0)
        spatial = self.dual_fuser(
            torch.cat((tight_global, wide_global), dim=1)
        ).flatten(1)
        geometry_feature = self.geometry_encoder(geometry)
        hidden = self.mlp(
            torch.cat((spatial, geometry_feature), dim=1)
        )
        logits = self.classifier(hidden).view(-1, 4, self.bins)

        local = self._local_feature(p1, p2)
        tight_local, wide_local = local.chunk(2, dim=0)
        condition = self.global_condition(hidden)
        grid_position = self.residual_grid.float() / self.residual_range
        position_features = torch.stack(
            (grid_position, grid_position.abs()), dim=0
        ).unsqueeze(0).expand(tight_crop.shape[0], -1, -1)
        local_logits: list[torch.Tensor] = []
        for edge_index, scorer in enumerate(self.edge_scorers):
            tight_stats = self._edge_statistics(
                tight_local, edge_index, self.tight_scale
            )
            wide_stats = self._edge_statistics(
                wide_local, edge_index, self.wide_scale
            )
            candidates = torch.cat(
                (
                    tight_stats,
                    wide_stats,
                    condition.unsqueeze(-1).expand(-1, -1, self.bins),
                    position_features,
                ),
                dim=1,
            )
            local_logits.append(scorer(candidates).squeeze(1))
        return logits + torch.stack(local_logits, dim=1)


def two_hot_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy against adjacent-bin linear interpolation targets."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"Shape mismatch: logits={logits.shape}, targets={targets.shape}")
    target = targets.float().clamp(float(grid[0]), float(grid[-1]))
    right = torch.searchsorted(grid.float(), target, right=False)
    right = right.clamp(1, len(grid) - 1)
    left = right - 1
    left_value = grid[left]
    right_value = grid[right]
    right_weight = (target - left_value) / (right_value - left_value).clamp_min(1e-9)
    left_weight = 1.0 - right_weight
    log_probabilities = logits.float().log_softmax(dim=-1)
    left_logp = log_probabilities.gather(-1, left.unsqueeze(-1)).squeeze(-1)
    right_logp = log_probabilities.gather(-1, right.unsqueeze(-1)).squeeze(-1)
    return (-(left_weight * left_logp + right_weight * right_logp)).mean()


def two_hot_probability_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Two-hot cross entropy for an already mixed probability distribution."""
    if probabilities.shape[:-1] != targets.shape:
        raise ValueError(
            f"Shape mismatch: probabilities={probabilities.shape}, "
            f"targets={targets.shape}"
        )
    target = targets.float().clamp(float(grid[0]), float(grid[-1]))
    right = torch.searchsorted(grid.float(), target, right=False)
    right = right.clamp(1, len(grid) - 1)
    left = right - 1
    left_value = grid[left]
    right_value = grid[right]
    right_weight = (target - left_value) / (
        right_value - left_value
    ).clamp_min(1e-9)
    left_weight = 1.0 - right_weight
    log_probabilities = probabilities.float().clamp_min(1e-9).log()
    left_logp = log_probabilities.gather(
        -1,
        left.unsqueeze(-1),
    ).squeeze(-1)
    right_logp = log_probabilities.gather(
        -1,
        right.unsqueeze(-1),
    ).squeeze(-1)
    return (-(left_weight * left_logp + right_weight * right_logp)).mean()


def residuals_to_local_boxes(residuals: torch.Tensor) -> torch.Tensor:
    """Convert dl/dt/dr/db to boxes in proposal-normalized coordinates."""
    return torch.stack(
        (
            residuals[..., 0],
            residuals[..., 1],
            1.0 + residuals[..., 2],
            1.0 + residuals[..., 3],
        ),
        dim=-1,
    )


def aligned_box_iou(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """IoU for aligned xyxy pairs."""
    top_left = torch.maximum(first[..., :2], second[..., :2])
    bottom_right = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    return intersection / union.clamp_min(eps)


def refinement_loss(
    model: nn.Module,
    output: torch.Tensor | dict[str, torch.Tensor],
    targets: torch.Tensor,
    entropy_weight: float = 0.10,
    iou_weight: float = 1.0,
    residual_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """UGS-inspired classification loss plus scale-normalized coordination."""
    if isinstance(output, dict):
        probabilities = model.distribution(output)
        classification = two_hot_probability_cross_entropy(
            probabilities,
            targets,
            model.residual_grid,
        )
        mean_gate = float(model.gate(output).detach().mean())
    else:
        classification = two_hot_cross_entropy(
            output,
            targets,
            model.residual_grid,
        )
        mean_gate = 1.0
    decoded = model.decode(output)
    entropy = model.normalized_entropy(output).mean()
    iou = aligned_box_iou(
        residuals_to_local_boxes(decoded),
        residuals_to_local_boxes(targets),
    )
    iou_loss = (1.0 - iou).mean()
    residual_loss = F.smooth_l1_loss(
        decoded,
        targets.float(),
        beta=0.01,
    )
    total = (
        classification
        + entropy_weight * entropy
        + iou_weight * iou_loss
        + residual_weight * residual_loss
    )
    stats = {
        "loss": float(total.detach()),
        "classification": float(classification.detach()),
        "entropy": float(entropy.detach()),
        "iou_loss": float(iou_loss.detach()),
        "residual_loss": float(residual_loss.detach()),
        "mean_train_iou": float(iou.detach().mean()),
        "mean_gate": mean_gate,
    }
    return total, stats


def apply_residuals_to_boxes(
    boxes_xyxy: torch.Tensor,
    residuals: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Apply normalized four-edge residuals and clamp to the source image."""
    boxes = boxes_xyxy.float()
    widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(2.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(2.0)
    refined = boxes.clone()
    refined[:, 0] += residuals[:, 0] * widths
    refined[:, 1] += residuals[:, 1] * heights
    refined[:, 2] += residuals[:, 2] * widths
    refined[:, 3] += residuals[:, 3] * heights
    refined[:, 0::2].clamp_(0.0, float(image_width))
    refined[:, 1::2].clamp_(0.0, float(image_height))
    refined[:, 2] = torch.maximum(refined[:, 2], refined[:, 0] + 1.0)
    refined[:, 3] = torch.maximum(refined[:, 3], refined[:, 1] + 1.0)
    refined[:, 2].clamp_(max=float(image_width))
    refined[:, 3].clamp_(max=float(image_height))
    return refined


def checkpoint_metadata(model: BoundaryDistributionRefiner) -> dict[str, Any]:
    return {
        "architecture": type(model).__name__,
        "bins": model.bins,
        "residual_range": model.residual_range,
        "grid_beta": model.grid_beta,
    }
