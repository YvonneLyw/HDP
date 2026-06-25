from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from hiera_diffusion_policy.model.diffusion.doser_selector_components import (
    CurrentImageEncoder,
    CurrentPcdEncoder,
    LatentStateDetector,
    LatentValueNet,
)


class GroundTruthDynamicsModel(nn.Module):
    """
    Predict normalized low-dimensional state and qpos at t + Tr directly.

    The transition is conditioned on all current selector observations. PCD and
    image features are inputs only. The predicted successor is
    concat(next_state, next_qpos).
    """

    def __init__(
        self,
        state_dim: int,
        subgoal_dim: int,
        qpos_dim: int,
        pcd_dim: int,
        action_eval_dim: int,
        hidden_dim: int = 256,
        pcd_feat_dim: int = 64,
        image_shape: Optional[Tuple[int, int, int, int]] = None,
        image_feat_dim: int = 64,
        predict_delta: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.subgoal_dim = int(subgoal_dim)
        self.qpos_dim = int(qpos_dim)
        self.pcd_dim = int(pcd_dim)
        self.action_eval_dim = int(action_eval_dim)
        self.successor_dim = self.state_dim + self.qpos_dim     ## s：(state,qpos)
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.pcd_feat_dim = int(pcd_feat_dim) if self.pcd_dim > 0 else 0
        self.image_feat_dim = int(image_feat_dim) if self.image_shape is not None else 0
        self.predict_delta = bool(predict_delta)

        self.pcd_encoder = (
            CurrentPcdEncoder(self.pcd_dim, self.pcd_feat_dim)
            if self.pcd_feat_dim > 0
            else None
        )
        self.image_encoder = (
            CurrentImageEncoder(self.image_shape, self.image_feat_dim)
            if self.image_feat_dim > 0
            else None
        )

        input_dim = (
            self.state_dim
            + self.subgoal_dim
            + self.qpos_dim
            + self.pcd_feat_dim
            + self.image_feat_dim
            + self.action_eval_dim
        )
        self.transition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.successor_dim),
        )

    def _current_features(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        action_eval: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = common["state"].reshape(common["state"].shape[0], -1)
        parts = [state]

        if self.subgoal_dim > 0:
            subgoal = common.get("subgoal", None)
            if subgoal is None:
                raise RuntimeError("GroundTruthDynamicsModel requires common['subgoal'].")
            parts.append(subgoal.reshape(state.shape[0], -1).to(state))

        if self.qpos_dim > 0:
            qpos_pair = common.get("doser_qpos_pair", None)
            if qpos_pair is None:
                qpos_pair = common.get("qpos_pair", None)
            if qpos_pair is None:
                raise RuntimeError(
                    "GroundTruthDynamicsModel requires common['doser_qpos_pair'] "
                    "or common['qpos_pair']."
                )
            qpos = qpos_pair[:, 0] if qpos_pair.dim() == 3 else qpos_pair
            qpos = qpos.reshape(state.shape[0], -1).to(state)
            parts.append(qpos)
        else:
            qpos = state.new_zeros((state.shape[0], 0)) ## s：(state,qpos)

        if self.pcd_encoder is not None:
            pcd = common.get("pcd", None)
            if pcd is None:
                raise RuntimeError("GroundTruthDynamicsModel requires common['pcd'].")
            parts.append(self.pcd_encoder(pcd.to(state)))

        if self.image_encoder is not None:
            image = common.get("doser_image_pair", None)
            if image is None:
                image = common.get("image_pair", None)
            if image is None:
                raise RuntimeError("GroundTruthDynamicsModel requires current images.")
            parts.append(self.image_encoder(image.to(state)))

        parts.append(action_eval.reshape(action_eval.shape[0], -1).to(state))
        return state, qpos, torch.cat(parts, dim=-1)

    def forward(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        action_eval: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        current_state, current_qpos, features = self._current_features(
            common,
            action_eval,
        )## s：(state,qpose)
        prediction = self.transition(features)
        state_prediction = prediction[:, :self.state_dim]
        qpos_prediction = prediction[:, self.state_dim:]
        if self.predict_delta:
            next_state = current_state + state_prediction
            next_qpos = current_qpos + qpos_prediction
        else:
            next_state = state_prediction
            next_qpos = qpos_prediction
        successor = torch.cat((next_state, next_qpos), dim=-1)  ## s'：(next_state,next_qpos)
        uncertainty = torch.zeros(
            successor.shape[0],
            device=successor.device,
            dtype=successor.dtype,
        )
        return {
            "next_state": next_state,
            "next_qpos": next_qpos,
            "successor": successor, ## s'：(next_state,next_qpos)
            "uncertainty": uncertainty,
        }


class GroundTruthStateDetector(LatentStateDetector):
    """Diffusion support detector over concat(normalized state, qpos)."""


class GroundTruthValueNet(LatentValueNet):
    """Standalone value estimator over concat(normalized state, qpos)."""
