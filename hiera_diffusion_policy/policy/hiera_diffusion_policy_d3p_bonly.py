from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hiera_diffusion_policy.model.diffusion.d3p_koopman import DeepKoopmanModule
from hiera_diffusion_policy.policy.hiera_diffusion_policy_d3p_fusion import (
    HieraDiffusionPolicyD3PFusion,
)

try:
    import imgaug.augmenters as iaa
except ImportError:
    iaa = None


class HieraDiffusionPolicyD3PBOnly(HieraDiffusionPolicyD3PFusion):
    """
    D3P-style B-only ablation policy.

    B1:
      - condition = latent acts from DKO on (t, t+h)
      - actor still receives zeroed state/pcd/subgoal to keep the fusion interface aligned

    B2:
      - condition = fea_fuse on (t, t+h)
      - DKO is still trained as an auxiliary objective
    """

    def __init__(
        self,
        koopman_latent_act_dim: Optional[int] = None,
        koopman_hidden_dim: int = 128,
        koopman_num_hidden_layers: int = 4,
        koopman_dropout: float = 0.0,
        koopman_activation: str = "ReLU",
        koopman_use_spectral_norm: bool = False,
        koopman_use_norm: bool = False,
        koopman_norm_style: str = "BatchNorm",
        koopman_kvp_weight: float = 0.3,
        koopman_fea_weight: float = 0.7,
        koopman_use_augmentation: bool = True,
        koopman_aug_crop_pad: float = 0.10,
        koopman_aug_rotate: float = 15.0,
        koopman_aug_noise_scale: float = 0.02,
        koopman_aug_brightness_min: float = 0.9,
        koopman_aug_brightness_max: float = 1.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        latent_act_dim = self.extra_cond_dim if koopman_latent_act_dim is None else int(koopman_latent_act_dim)
        if latent_act_dim != self.extra_cond_dim:
            raise ValueError(
                "D3P-style B-only expects koopman_latent_act_dim to match actor extra_cond_dim, "
                f"got {latent_act_dim} vs {self.extra_cond_dim}."
            )

        self.dko = DeepKoopmanModule(
            obs_dim=self.branch_condition_encoder.vis_dim,
            latent_act_dim=latent_act_dim,
            hidden_dim=int(koopman_hidden_dim),
            num_hidden_layers=int(koopman_num_hidden_layers),
            dropout=float(koopman_dropout),
            activation=str(koopman_activation),
            use_spectral_norm=bool(koopman_use_spectral_norm),
            use_norm=bool(koopman_use_norm),
            norm_style=str(koopman_norm_style),
        )

        self.koopman_kvp_weight = float(koopman_kvp_weight)
        self.koopman_fea_weight = float(koopman_fea_weight)
        self.koopman_use_augmentation = bool(koopman_use_augmentation)
        self.koopman_aug_crop_pad = float(koopman_aug_crop_pad)
        self.koopman_aug_rotate = float(koopman_aug_rotate)
        self.koopman_aug_noise_scale = float(koopman_aug_noise_scale)
        self.koopman_aug_brightness_min = float(koopman_aug_brightness_min)
        self.koopman_aug_brightness_max = float(koopman_aug_brightness_max)

        self._last_actor_aux_logs: Dict[str, float] = {}

    def _prepare_branch_inputs(
        self,
        raw_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        common = super()._prepare_branch_inputs(raw_batch)
        common["image_pair"] = raw_batch["image"] if "image" in raw_batch else None
        return common

    def _reduce_feature_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(target, pred, reduction="none")
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1)
        return loss.mean()

    def augment_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if images.dim() != 6 or images.shape[1] != 2 or images.shape[2] != 2:
            raise RuntimeError(f"image must be (B,2,2,3,H,W), got shape={tuple(images.shape)}")

        if iaa is None:
            return self._augment_images_torch(images)

        imgshape = images.shape
        device = images.device
        datatype = images.dtype
        img_augmenter = iaa.Sequential(
            [
                iaa.CropAndPad(percent=np.random.uniform(-self.koopman_aug_crop_pad, self.koopman_aug_crop_pad)),
                iaa.Fliplr(np.random.choice([0, 1], p=[0.5, 0.5])),
                iaa.Affine(rotate=np.random.uniform(-self.koopman_aug_rotate, self.koopman_aug_rotate)),
                iaa.AdditiveGaussianNoise(scale=self.koopman_aug_noise_scale),
                iaa.Multiply(np.random.uniform(self.koopman_aug_brightness_min, self.koopman_aug_brightness_max)),
            ]
        )

        input_images = images.permute(0, 1, 2, 4, 5, 3).reshape(-1, imgshape[4], imgshape[5], imgshape[3])
        input_images = input_images.detach().cpu().numpy()
        aug_images = img_augmenter(images=input_images)
        aug_images = np.clip(aug_images, 0.0, 1.0)

        aug_images = torch.tensor(aug_images, dtype=datatype, device=device)
        aug_images = aug_images.reshape(imgshape[0], imgshape[1], imgshape[2], imgshape[4], imgshape[5], imgshape[3])
        aug_images = aug_images.permute(0, 1, 2, 5, 3, 4)

        curr_vis_dict = {
            "front_rgb": aug_images[:, 0, 0],
            "wrist_rgb": aug_images[:, 0, 1],
        }
        next_vis_dict = {
            "front_rgb": aug_images[:, 1, 0],
            "wrist_rgb": aug_images[:, 1, 1],
        }
        return curr_vis_dict, next_vis_dict

    def _augment_images_torch(
        self,
        images: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        imgshape = images.shape
        h = imgshape[4]
        w = imgshape[5]
        aug_images = images.reshape(-1, imgshape[3], h, w)

        crop_pad_ratio = float(np.random.uniform(-self.koopman_aug_crop_pad, self.koopman_aug_crop_pad))
        if crop_pad_ratio >= 0:
            pad = int(round(crop_pad_ratio * min(h, w)))
            if pad > 0:
                aug_images = F.pad(aug_images, (pad, pad, pad, pad), mode="replicate")
                top = int(torch.randint(0, 2 * pad + 1, (1,), device=aug_images.device).item())
                left = int(torch.randint(0, 2 * pad + 1, (1,), device=aug_images.device).item())
                aug_images = aug_images[:, :, top:top + h, left:left + w]
        else:
            crop = int(round((-crop_pad_ratio) * min(h, w)))
            if crop > 0 and (2 * crop) < min(h, w):
                aug_images = aug_images[:, :, crop:h - crop, crop:w - crop]
                aug_images = F.interpolate(aug_images, size=(h, w), mode="bilinear", align_corners=False)

        if np.random.choice([0, 1], p=[0.5, 0.5]) == 1:
            aug_images = torch.flip(aug_images, dims=(-1,))

        angle = float(np.random.uniform(-self.koopman_aug_rotate, self.koopman_aug_rotate)) * np.pi / 180.0
        cos_theta = float(np.cos(angle))
        sin_theta = float(np.sin(angle))
        theta = torch.tensor(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0]],
            dtype=aug_images.dtype,
            device=aug_images.device,
        ).unsqueeze(0).repeat(aug_images.shape[0], 1, 1)
        grid = F.affine_grid(theta, aug_images.size(), align_corners=False)
        aug_images = F.grid_sample(
            aug_images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        aug_images = aug_images + self.koopman_aug_noise_scale * torch.randn_like(aug_images)
        brightness = float(np.random.uniform(self.koopman_aug_brightness_min, self.koopman_aug_brightness_max))
        aug_images = (aug_images * brightness).clamp(0.0, 1.0)

        aug_images = aug_images.reshape(imgshape[0], imgshape[1], imgshape[2], imgshape[3], h, w)
        curr_vis_dict = {
            "front_rgb": aug_images[:, 0, 0],
            "wrist_rgb": aug_images[:, 0, 1],
        }
        next_vis_dict = {
            "front_rgb": aug_images[:, 1, 0],
            "wrist_rgb": aug_images[:, 1, 1],
        }
        return curr_vis_dict, next_vis_dict

    def _compute_koopman_aux(
        self,
        common: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        vis_enc_pair = common["vis_enc_pair"]
        image_pair = common["image_pair"]
        if vis_enc_pair is None:
            raise RuntimeError("D3P-style B-only requires vis_enc_pair for Koopman losses.")
        if image_pair is None:
            raise RuntimeError("D3P-style B-only requires image_pair for Koopman losses.")

        curr_vis_enc = vis_enc_pair[:, 0]
        next_vis_enc = vis_enc_pair[:, 1]

        pred_next_vis_enc, curr_latent_acts = self.dko(curr_vis_enc, latent_act=None)
        koop_consis_kvp = self._reduce_feature_loss(next_vis_enc, pred_next_vis_enc)

        next_latent_acts = self.dko.get_latent_act(next_vis_enc)

        if self.koopman_use_augmentation:
            aug_curr_vis_dict, aug_next_vis_dict = self.augment_images(image_pair)
            aug_curr_vis_enc = self.branch_condition_encoder.vis_encoder(aug_curr_vis_dict)
            aug_next_vis_enc = self.branch_condition_encoder.vis_encoder(aug_next_vis_dict)
            pred_aug_next_vis_enc = self.dko(aug_curr_vis_enc, latent_act=curr_latent_acts)
            koop_consis_fea = self._reduce_feature_loss(aug_next_vis_enc, pred_aug_next_vis_enc)
        else:
            koop_consis_fea = torch.zeros((), dtype=curr_vis_enc.dtype, device=curr_vis_enc.device)

        return {
            "curr_latent_acts": curr_latent_acts,
            "next_latent_acts": next_latent_acts,
            "koop_consis_kvp": koop_consis_kvp,
            "koop_consis_fea": koop_consis_fea,
        }

    def _build_cond_by_branch(
        self,
        branch: str,
        common: Dict[str, Optional[torch.Tensor]],
        koop_aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if branch == "A":
            return super()._build_cond_by_branch(branch, common)

        if branch == "B1":
            if common["vis_enc_pair"] is None:
                raise RuntimeError("B1 requires vis_enc_pair, but image is missing.")
            if koop_aux is not None:
                curr_latent_acts = koop_aux["curr_latent_acts"]
                next_latent_acts = koop_aux["next_latent_acts"]
            else:
                curr_latent_acts = self.dko.get_latent_act(common["vis_enc_pair"][:, 0])
                next_latent_acts = self.dko.get_latent_act(common["vis_enc_pair"][:, 1])
            extra_cond_pair = torch.stack((curr_latent_acts, next_latent_acts), dim=1)
        elif branch == "B2":
            if common["fea_fuse_pair"] is None:
                raise RuntimeError("B2 requires fea_fuse_pair, but image/qpos is missing.")
            extra_cond_pair = common["fea_fuse_pair"]
        else:
            raise ValueError(f"Unknown branch: {branch}")

        cond = {
            "pcd": common["pcd_zero"],
            "state": common["state_zero"],
            "subgoal": common["subgoal_zero"],
            "extra_cond_pair": extra_cond_pair,
        }
        self._run_branch_checks(branch, cond)
        return cond

    def compute_loss_actor(self, batch: Dict[str, torch.Tensor]):
        common = self._prepare_branch_inputs(batch)
        branch = self._select_train_branch(common)

        koop_aux = None
        if branch in ("B1", "B2"):
            koop_aux = self._compute_koopman_aux(common)

        cond = self._build_cond_by_branch(branch, common, koop_aux=koop_aux)
        actor_loss, bc_loss, q_loss = self._compute_loss_actor_from_cond(branch=branch, cond=cond, common=common)

        if koop_aux is not None:
            actor_loss = actor_loss + self.koopman_kvp_weight * koop_aux["koop_consis_kvp"]
            actor_loss = actor_loss + self.koopman_fea_weight * koop_aux["koop_consis_fea"]
            self._last_actor_aux_logs = {
                "train_koop_consis_kvp": float(koop_aux["koop_consis_kvp"].detach().item()),
                "train_koop_consis_fea": float(koop_aux["koop_consis_fea"].detach().item()),
            }
        else:
            self._last_actor_aux_logs = {
                "train_koop_consis_kvp": 0.0,
                "train_koop_consis_fea": 0.0,
            }

        return actor_loss, bc_loss, q_loss

    def get_last_actor_aux_logs(self) -> Dict[str, float]:
        return dict(self._last_actor_aux_logs)
