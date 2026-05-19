from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _replace_bn_with_gn(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_groups = max(child.num_features // 16, 1)
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features),
            )
        else:
            _replace_bn_with_gn(child)
    return module


def build_resnet18_encoder(output_size: int, use_group_norm: bool) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    if use_group_norm:
        model = _replace_bn_with_gn(model)
    model.fc = nn.Linear(model.fc.in_features, output_size)
    return model


class MultiImageObsEncoder(nn.Module):
    """
    D3P-style multi-view image encoder:
      - encode each rgb view with (shared or independent) backbone
      - concatenate view features
    """
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (3, 84, 84),
        output_size: int = 64,
        share_rgb_model: bool = False,
        use_group_norm: bool = True,
        imagenet_norm: bool = True,
        # shape_meta: dict,
        # rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        # random_crop: bool = True,       ## False 不做随机裁剪数据增强
        # crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
    ):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.output_size = int(output_size)
        self.share_rgb_model = bool(share_rgb_model)
        self.imagenet_norm = bool(imagenet_norm)

        if self.share_rgb_model:
            shared = build_resnet18_encoder(self.output_size, use_group_norm)
            self.front_model = shared
            self.wrist_model = shared
        else:
            self.front_model = build_resnet18_encoder(self.output_size, use_group_norm)
            self.wrist_model = copy.deepcopy(self.front_model)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.output_size * 2

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1:] != self.image_shape:
            raise RuntimeError(
                f"image shape mismatch: expected (B,{self.image_shape[0]},{self.image_shape[1]},{self.image_shape[2]}), "
                f"got {tuple(img.shape)}"
            )
        if self.imagenet_norm:
            img = (img - self.imagenet_mean) / self.imagenet_std
        return img

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        front = self._preprocess(obs_dict["front_rgb"]) ## normalize
        wrist = self._preprocess(obs_dict["wrist_rgb"])

        front_feat = self.front_model(front)
        wrist_feat = self.wrist_model(wrist)
        return torch.cat((front_feat, wrist_feat), dim=-1)


class AttentionFusion(nn.Module):
    """
    Same fusion family as D3P's AttentionFusion.
    """
    def __init__(self, fea_act_dim: int, fea_vis_dim: int, out_dim: int):
        super().__init__()
        # dimension alignment
        self.fc_act = nn.Linear(fea_act_dim, out_dim)
        self.fc_vis = nn.Linear(fea_vis_dim, out_dim)
        self.attention_weight = nn.Linear(out_dim, 1)

    def forward(self, fea_act: torch.Tensor, fea_vis: torch.Tensor) -> torch.Tensor:
        bs = fea_act.shape[0]
        act_flat = fea_act.view(bs, -1)
        f1 = self.fc_act(act_flat)   # [batch_size, out_dim]
        f2 = self.fc_vis(fea_vis)    # [batch_size, out_dim]
        features = torch.stack((f1, f2), dim=1)        # (B, 2, out_dim)
        scores = self.attention_weight(features).squeeze(-1)   # (B, 2)
        weights = torch.softmax(scores, dim=1)         # (B, 2)
        fused = torch.sum(weights.unsqueeze(-1) * features, dim=1)  # (B, out_dim)
        return fused


class BranchConditionEncoder(nn.Module):
    """
    Branch condition builder:
      - vis_enc_pair: MultiImageObsEncoder on (t, t+h)
      - fea_fuse_pair: AttentionFusion(qpos, vis_enc)
      - extra_cond_pair for B1/B2 via branch-specific projector
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (84, 84),
        per_view_output_dim: int = 64,
        cond_hidden_dim: int = 64,
        qpos_dim: int = 9,
        subgoal_dim: int = 8,
        extra_cond_dim: int = 64,
        share_rgb_model: bool = False,
        use_group_norm: bool = True,
        imagenet_norm: bool = True,
    ):
        super().__init__()
        h, w = int(image_size[0]), int(image_size[1])
        self.subgoal_dim = int(subgoal_dim)
        self.qpos_dim = int(qpos_dim)
        self.extra_cond_dim = int(extra_cond_dim)
        self.cond_hidden_dim = int(cond_hidden_dim)

        self.vis_encoder = MultiImageObsEncoder(
            image_shape=(3, h, w),
            output_size=int(per_view_output_dim),
            share_rgb_model=share_rgb_model,
            use_group_norm=use_group_norm,
            imagenet_norm=imagenet_norm,
        )
        self.vis_dim = self.vis_encoder.out_dim

        self.b2_fuse_encoder = AttentionFusion(
            fea_act_dim=self.qpos_dim,
            fea_vis_dim=self.vis_dim,
            out_dim=self.cond_hidden_dim,
        )
        
        in_dim = self.cond_hidden_dim + self.subgoal_dim
        self.branch_cond_encoder = nn.Sequential(
            nn.Linear(in_dim, self.extra_cond_dim),
            nn.Mish(),
            nn.Linear(self.extra_cond_dim, self.extra_cond_dim),
        )

    def encode_vis_pair(self, image_pair: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image_pair is None:
            return None
        if image_pair.dim() != 6 or image_pair.shape[1] != 2 or image_pair.shape[2] != 2:
            raise RuntimeError(
                f"image_pair must be (B,2,2,3,H,W), got shape={tuple(image_pair.shape)}"
            )

        curr_vis_dict = {
            "front_rgb": image_pair[:, 0, 0],
            "wrist_rgb": image_pair[:, 0, 1],
        }
        next_vis_dict = {
            "front_rgb": image_pair[:, 1, 0],
            "wrist_rgb": image_pair[:, 1, 1],
        }
        curr_vis_enc = self.vis_encoder(curr_vis_dict)
        next_vis_enc = self.vis_encoder(next_vis_dict)
        return torch.stack((curr_vis_enc, next_vis_enc), dim=1)  # (B,2,vis_dim)

    def encode_fea_fuse_pair(
        self,
        qpos_pair: Optional[torch.Tensor],
        vis_enc_pair: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (qpos_pair is None) or (vis_enc_pair is None):
            return None
        if qpos_pair.dim() != 3 or qpos_pair.shape[1] != 2:
            raise RuntimeError(f"qpos_pair must be (B,2,D), got shape={tuple(qpos_pair.shape)}")
        if qpos_pair.shape[-1] != self.qpos_dim:
            raise RuntimeError(
                f"qpos_pair dim mismatch: got {qpos_pair.shape[-1]}, expected {self.qpos_dim}"
            )
        if vis_enc_pair.dim() != 3 or vis_enc_pair.shape[1] != 2:
            raise RuntimeError(f"vis_enc_pair must be (B,2,D), got shape={tuple(vis_enc_pair.shape)}")

        fuse_t = self.b2_fuse_encoder(qpos_pair[:, 0], vis_enc_pair[:, 0])
        fuse_th = self.b2_fuse_encoder(qpos_pair[:, 1], vis_enc_pair[:, 1])
        return torch.stack((fuse_t, fuse_th), dim=1)   # (B,2,cond_hidden_dim)

    def build_b1_extra_cond_pair(
        self,
        latent_act_pair: torch.Tensor,
        subgoal_pair: torch.Tensor,
    ) -> torch.Tensor:
        b1_input = torch.cat((latent_act_pair, subgoal_pair), dim=-1)
        return self.branch_cond_encoder(b1_input)

    def build_b2_extra_cond_pair(
        self,
        fea_fuse_pair: torch.Tensor,
        subgoal_pair: torch.Tensor,
    ) -> torch.Tensor:
        b2_input = torch.cat((fea_fuse_pair, subgoal_pair), dim=-1)
        return self.branch_cond_encoder(b2_input)
