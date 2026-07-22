"""State/subgoal value network used by the QGF-IQL critic.

The original HDP critic estimates Q(c, a), where c contains the object point
cloud, state history, and subgoal.  IQL additionally needs V(c), with the same
condition but no action input.  This module intentionally owns a separate point
cloud encoder so Q and V optimizers do not share parameters.
"""

import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn

from hiera_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder


logger = logging.getLogger(__name__)


class Value(nn.Module):
    """Estimate a scalar V(pcd, state, subgoal)."""

    def __init__(
        self,
        pcd_encoder: Optional[PointNetEncoder],
        state_dim: int,
        subgoal_dim: int,
        mlp_dims: Sequence[int] = (512, 256, 128),
    ):
        super().__init__()
        if not mlp_dims:
            raise ValueError("mlp_dims must contain at least one hidden dimension")

        self.pcd_encoder = pcd_encoder
        input_dim = int(state_dim) + int(subgoal_dim)
        if self.pcd_encoder is not None:
            input_dim += self.pcd_encoder.out_dim

        layers = []
        last_dim = input_dim
        for hidden_dim in mlp_dims:
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.ReLU()))
            last_dim = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(last_dim, 1)

        logger.info("parameters number of Value: %e", self.params_num())

    def forward(
        self,
        pcd: Optional[torch.Tensor],
        state: torch.Tensor,
        subgoal: torch.Tensor,
    ) -> torch.Tensor:
        if self.pcd_encoder is not None:
            if pcd is None:
                raise ValueError("pcd must be provided when Value has a pcd_encoder")
            x = torch.concat((self.pcd_encoder(pcd), state, subgoal), dim=1)
        else:
            x = torch.concat((state, subgoal), dim=1)
        return self.final_layer(self.layers(x))

    def params_num(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
