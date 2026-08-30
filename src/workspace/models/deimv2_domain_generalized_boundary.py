"""Boundary-protected semantic style generalization for DEIMv2.

Only the deepest backbone feature is stylized during training.  The
high-resolution stride-4 evidence consumed by the boundary decoder remains
untouched, so sensor/style diversity is injected into semantic recognition
without deliberately perturbing the precise edge path.  The style mixer has
no parameters and is an exact identity in evaluation/deployment mode.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from engine.core import register
from workspace.models.deimv2_query_conditioned_boundary import (
    QueryConditionedBoundaryDEIM,
)


class DeepSemanticMixStyle(nn.Module):
    """Mix per-channel feature statistics across images during training."""

    def __init__(
        self,
        *,
        probability: float = 0.5,
        alpha: float = 0.1,
        epsilon: float = 1e-6,
        feature_index: int = -1,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("style-mix probability must be in [0, 1]")
        if alpha <= 0.0 or epsilon <= 0.0:
            raise ValueError("style-mix alpha and epsilon must be positive")
        self.probability = float(probability)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.feature_index = int(feature_index)
        self.force_next = False
        self.last_applied = False
        self.last_mean_abs_delta = 0.0

    def _mix(self, feature: torch.Tensor) -> torch.Tensor:
        batch = int(feature.shape[0])
        if batch < 2:
            return feature
        with torch.autocast(device_type=feature.device.type, enabled=False):
            source = feature.float()
            mean = source.mean(dim=(2, 3), keepdim=True).detach()
            std = source.var(dim=(2, 3), keepdim=True, unbiased=False).add(
                self.epsilon
            ).sqrt().detach()
            # A cyclic shift is a derangement, so every example receives style
            # statistics from a different image even with the small batch size.
            shift = int(
                torch.randint(1, batch, (), device=feature.device).item()
            )
            donor_mean = mean.roll(shifts=shift, dims=0)
            donor_std = std.roll(shifts=shift, dims=0)
            concentration = torch.full(
                (batch,), self.alpha, dtype=torch.float32, device=feature.device
            )
            mixture = torch.distributions.Beta(
                concentration, concentration
            ).sample().view(batch, 1, 1, 1)
            mixed_mean = mixture * mean + (1.0 - mixture) * donor_mean
            mixed_std = mixture * std + (1.0 - mixture) * donor_std
            result = (source - mean) / std * mixed_std + mixed_mean
            self.last_mean_abs_delta = float(
                (result - source).abs().mean().detach().cpu()
            )
        return result.to(dtype=feature.dtype)

    def forward(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        values = tuple(features)
        self.last_applied = False
        self.last_mean_abs_delta = 0.0
        should_apply = self.training and (
            self.force_next
            or float(torch.rand((), device=values[0].device)) < self.probability
        )
        self.force_next = False
        if not should_apply:
            return values
        index = self.feature_index % len(values)
        mixed = list(values)
        mixed[index] = self._mix(mixed[index])
        self.last_applied = self.last_mean_abs_delta > 0.0
        return tuple(mixed)


@register()
class DomainGeneralizedBoundaryDEIM(QueryConditionedBoundaryDEIM):
    """Query-conditioned detector with train-only semantic style mixing."""

    def __init__(
        self,
        *args: Any,
        style_mix_probability: float = 0.5,
        style_mix_alpha: float = 0.1,
        style_mix_epsilon: float = 1e-6,
        style_feature_index: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_style_mixer = DeepSemanticMixStyle(
            probability=style_mix_probability,
            alpha=style_mix_alpha,
            epsilon=style_mix_epsilon,
            feature_index=style_feature_index,
        )

    def forward(
        self, x: torch.Tensor, targets: list[dict[str, Any]] | None = None
    ) -> dict[str, torch.Tensor]:
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
        backbone_features = self.semantic_style_mixer(backbone_features)
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
