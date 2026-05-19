from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hiera_diffusion_policy.model.diffusion.branch_condition_encoder import BranchConditionEncoder
from hiera_diffusion_policy.model.diffusion.d3p_koopman import DeepKoopmanModule
from hiera_diffusion_policy.policy.hiera_diffusion_policy import HieraDiffusionPolicy

try:
    import imgaug.augmenters as iaa
except ImportError:
    iaa = None


class HieraDiffusionPolicyD3PFusion(HieraDiffusionPolicy):
    """
    Debug-first fusion policy with explicit branch enumeration.

    Branch contracts:
      - A : state/pcd real, extra_cond=zeros
      - B1: state/pcd zero, dual-time extra_cond from (latent_act, subgoal)
      - B2: state/pcd zero, dual-time extra_cond from (fea_fuse, subgoal)

    Notes:
      - actor_small only receives extra_cond; no image/qpos/sub-branch args are passed into actor.
      - subgoal_pair prefers dataset-provided (t, t+h) subgoals and only falls back to
        duplicating current subgoal when explicit pair data is unavailable.
      - B branch BC loss uses two timestamps (t and t+h) with 0.5/0.5 weighting.
      - B1 uses DKO latent acts plus subgoal to build extra_cond_pair when DKO is enabled.
      - DKO auxiliary loss can be enabled for B branches.
      - Q loss is only applied on branch A; B branches always train with eta=0.
    """
    # A-only：policy.mode=SINGLE policy.single_branch=A
    # B1-only：policy.mode=SINGLE policy.single_branch=B1
    # B2-only：policy.mode=SINGLE policy.single_branch=B2
    # A/B1 switch+select：policy.mode=SWITCH policy.b_branch=B1 policy.switch_prob_b=0.5
    # A/B2 switch+select：policy.mode=SWITCH policy.b_branch=B2 policy.switch_prob_b=0.5
    
    def __init__(
        self,
        d3p_query_every: int = 4,
        mode: str = 'SINGLE',
        single_branch: str = 'A',
        b_branch: str = 'B1',
        switch_prob_b: float = 0.5,
        d3p_rollout_error_samples: int = 10,
        fusion_debug_checks: bool = True,
        use_koopman_aux: bool = False,
        extra_cond_dim: int = 64,
        image_feat_dim: int = 64,
        qpos_feat_dim: int = 9,
        image_size=(84, 84),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d3p_query_every = int(d3p_query_every)
        self.mode = str(mode).upper()
        self.single_branch = str(single_branch)
        self.b_branch = str(b_branch)
        self.switch_prob_b = float(switch_prob_b)

        self.d3p_rollout_error_samples = int(d3p_rollout_error_samples)
        self.fusion_debug_checks = bool(fusion_debug_checks)
        self._debug_checked_once = False
        self.use_koopman_aux = bool(use_koopman_aux)

        self.extra_cond_dim = int(extra_cond_dim)
        self.image_feat_dim = int(image_feat_dim)  # per-view output dim
        self.qpos_feat_dim = int(qpos_feat_dim)
        self.image_size = tuple(image_size)

        if self.mode not in ('SINGLE', 'SWITCH'):
            raise ValueError(f"mode must be one of ['SINGLE','SWITCH'], got {self.mode}")
        if self.single_branch not in ('A', 'B1', 'B2'):
            raise ValueError(f"single_branch must be one of ['A','B1','B2'], got {self.single_branch}")
        if self.b_branch not in ('B1', 'B2'):
            raise ValueError(f"b_branch must be one of ['B1','B2'], got {self.b_branch}")
        if not (0.0 <= self.switch_prob_b <= 1.0):
            raise ValueError(f"switch_prob_b must be in [0,1], got {self.switch_prob_b}")
        if self.d3p_query_every != 4:
            raise ValueError(f"d3p_query_every is fixed to 4 in current fusion stage, got {self.d3p_query_every}")
        if self.d3p_rollout_error_samples < 1:
            raise ValueError(f"d3p_rollout_error_samples must be >= 1, got {self.d3p_rollout_error_samples}")

        actor_extra_cond_dim = int(getattr(self.actor, 'extra_cond_dim', 0))
        if actor_extra_cond_dim != self.extra_cond_dim:
            raise RuntimeError(
                f"actor.extra_cond_dim ({actor_extra_cond_dim}) != policy.extra_cond_dim ({self.extra_cond_dim})"
            )

        self.branch_condition_encoder = BranchConditionEncoder(
            image_size=self.image_size,
            per_view_output_dim=self.image_feat_dim,
            cond_hidden_dim=self.extra_cond_dim,
            qpos_dim=self.qpos_feat_dim,
            subgoal_dim=self.subgoal_dim,
            extra_cond_dim=self.extra_cond_dim,
            share_rgb_model=False,
            use_group_norm=True,
            imagenet_norm=True,
        )

        self.koopman_kvp_weight = 0.3
        self.koopman_fea_weight = 0.7
        self.koopman_use_augmentation = True
        self.koopman_aug_crop_pad = 0.10
        self.koopman_aug_rotate = 15.0
        self.koopman_aug_noise_scale = 0.02
        self.koopman_aug_brightness_min = 0.9
        self.koopman_aug_brightness_max = 1.1
        self._last_actor_aux_logs: Dict[str, float] = {
            'train_koop_consis_kvp': 0.0,
            'train_koop_consis_fea': 0.0,
        }

        if self.use_koopman_aux:
            self.dko = DeepKoopmanModule(
                obs_dim=self.branch_condition_encoder.vis_dim,
                latent_act_dim=self.extra_cond_dim,
                hidden_dim=128,
                num_hidden_layers=4,
                dropout=0.0,
                activation='ReLU',
                use_spectral_norm=False,
                use_norm=False,
                norm_style='BatchNorm',
            )
        else:
            self.dko = None

    ########## 模型（actor,branch_condition_encoder，dko）参数加入optimizer############ D3P fusion 把actor+ branch_condition_encoder （+dko）一起训
    def get_actor_training_parameters(self):
        # B-branch actor losses depend on branch_condition_encoder outputs, and
        # optionally DKO auxiliary losses, so they must be optimized together
        # during actor stage.
        params = list(self.actor.parameters()) + list(self.branch_condition_encoder.parameters())
        if self.dko is not None:
            params += list(self.dko.parameters())
        return params

    # =========================
    # Pair feature builders
    # =========================
    def _fit_action_and_pad_to_horizon(
        self,
        d3p_actions: torch.Tensor,
        act_is_pad: Optional[torch.Tensor],
    ):
        """
        Fit D3P action chunk length L to actor horizon.
        d3p_actions: (B, L, Dim_a)
        act_is_pad: (B, L) or None
        """
        B, L, Dim_a = d3p_actions.shape
        if act_is_pad is None:
            act_is_pad = torch.zeros((B, L), dtype=torch.bool, device=d3p_actions.device)
        else:
            act_is_pad = act_is_pad.bool()

        if L == self.horizon:
            return d3p_actions, act_is_pad
        if L > self.horizon:
            return d3p_actions[:, :self.horizon], act_is_pad[:, :self.horizon]
        if L < self.horizon:
            pad_len = self.horizon - L
            action_pad = torch.zeros((B, pad_len, Dim_a), dtype=d3p_actions.dtype, device=d3p_actions.device)
            mask_pad = torch.ones((B, pad_len), dtype=torch.bool, device=d3p_actions.device)
            return (
                torch.concat((d3p_actions, action_pad), dim=1),
                torch.concat((act_is_pad, mask_pad), dim=1),
            )
        raise RuntimeError(f"Unexpected action length L={L}, horizon={self.horizon}")

    def _select_train_branch(self, common: Dict[str, Optional[torch.Tensor]]) -> str:
        if self.mode == 'SINGLE':
            return self._resolve_branch(self.single_branch, common)
        use_b = np.random.uniform() < self.switch_prob_b
        return self._resolve_branch(self.b_branch if use_b else 'A', common)

    def _resolve_branch(self, branch: str, common: Dict[str, Optional[torch.Tensor]]) -> str:
        if branch == 'B1' and common['vis_enc_pair'] is None:
            return 'A'
        if branch == 'B2' and common['fea_fuse_pair'] is None:
            return 'A'
        if branch not in ('A', 'B1', 'B2'):
            return 'A'
        return branch

    # =========================
    # DKO
    # =========================
    def _reduce_feature_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(target, pred, reduction='none')
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
            'front_rgb': aug_images[:, 0, 0],
            'wrist_rgb': aug_images[:, 0, 1],
        }
        next_vis_dict = {
            'front_rgb': aug_images[:, 1, 0],
            'wrist_rgb': aug_images[:, 1, 1],
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
                aug_images = F.pad(aug_images, (pad, pad, pad, pad), mode='replicate')
                top = int(torch.randint(0, 2 * pad + 1, (1,), device=aug_images.device).item())
                left = int(torch.randint(0, 2 * pad + 1, (1,), device=aug_images.device).item())
                aug_images = aug_images[:, :, top:top + h, left:left + w]
        else:
            crop = int(round((-crop_pad_ratio) * min(h, w)))
            if crop > 0 and (2 * crop) < min(h, w):
                aug_images = aug_images[:, :, crop:h - crop, crop:w - crop]
                aug_images = F.interpolate(aug_images, size=(h, w), mode='bilinear', align_corners=False)

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
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )

        aug_images = aug_images + self.koopman_aug_noise_scale * torch.randn_like(aug_images)
        brightness = float(np.random.uniform(self.koopman_aug_brightness_min, self.koopman_aug_brightness_max))
        aug_images = (aug_images * brightness).clamp(0.0, 1.0)

        aug_images = aug_images.reshape(imgshape[0], imgshape[1], imgshape[2], imgshape[3], h, w)
        curr_vis_dict = {
            'front_rgb': aug_images[:, 0, 0],
            'wrist_rgb': aug_images[:, 0, 1],
        }
        next_vis_dict = {
            'front_rgb': aug_images[:, 1, 0],
            'wrist_rgb': aug_images[:, 1, 1],
        }
        return curr_vis_dict, next_vis_dict

    def _compute_koopman_aux(
        self,
        common: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if self.dko is None:
            raise RuntimeError('DKO auxiliary loss requested but self.dko is not initialized.')

        vis_enc_pair = common['vis_enc_pair']
        image_pair = common['image_pair']
        if vis_enc_pair is None:
            raise RuntimeError('Fusion Koopman auxiliary loss requires vis_enc_pair.')
        if image_pair is None:
            raise RuntimeError('Fusion Koopman auxiliary loss requires image_pair.')

        curr_vis_enc = vis_enc_pair[:, 0]
        next_vis_enc = vis_enc_pair[:, 1]

        latent_act_pair = common.get('b1_latent_act_pair', None)
        if latent_act_pair is not None:
            curr_latent_acts = latent_act_pair[:, 0]
            next_latent_acts = latent_act_pair[:, 1]
            self.dko.enable_kv_grad(True)
            pred_next_vis_enc = self.dko.K(curr_vis_enc.detach() + self.dko.V(curr_latent_acts))
        else:
            pred_next_vis_enc, curr_latent_acts = self.dko(curr_vis_enc, latent_act=None)
            next_latent_acts = self.dko.get_latent_act(next_vis_enc)
        koop_consis_kvp = self._reduce_feature_loss(next_vis_enc, pred_next_vis_enc)

        if self.koopman_use_augmentation:
            aug_curr_vis_dict, aug_next_vis_dict = self.augment_images(image_pair)
            aug_curr_vis_enc = self.branch_condition_encoder.vis_encoder(aug_curr_vis_dict)
            aug_next_vis_enc = self.branch_condition_encoder.vis_encoder(aug_next_vis_dict)
            pred_aug_next_vis_enc = self.dko(aug_curr_vis_enc, latent_act=curr_latent_acts)
            koop_consis_fea = self._reduce_feature_loss(aug_next_vis_enc, pred_aug_next_vis_enc)
        else:
            koop_consis_fea = torch.zeros((), dtype=curr_vis_enc.dtype, device=curr_vis_enc.device)

        return {
            'curr_latent_acts': curr_latent_acts,
            'next_latent_acts': next_latent_acts,
            'koop_consis_kvp': koop_consis_kvp,
            'koop_consis_fea': koop_consis_fea,
        }

    # =========================
    # Unified condition prep
    # =========================
    def _prepare_branch_inputs(
        self,
        raw_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Merge of old normalize + `_extract_d3p_payload` + `_prepare_common_inputs`.
        Important: image/qpos/is_pad are read from raw batch (normalizer drops them).
        """
        nbatch = self.normalizer.normalize(raw_batch, self.subgoal_dim_nocont)  ## HDP数据统一标准化
        B = nbatch['state'].shape[0]

        # HDP-native conditions#######################
        if self.use_pcd:
            pcd = nbatch['pcd'].transpose(1, 2).reshape(            ##(B, H=2, N=1024, D=3) -> (B, N, H*D=2*3)
                (B, -1, self.pcd_dim*self.observation_history_num)
            )
            if 'pcd_id' in nbatch:
                pcd = torch.concat((pcd, nbatch['pcd_id']), dim=-1)
            pcd_zero = torch.zeros_like(pcd)
        else:
            pcd = None
            pcd_zero = None
        
        state = nbatch['state'].reshape((B, -1))    # (B, n*S)    ##(B, H=2, 27) -> (B, 展平：2*27)
        state_zero = torch.zeros_like(state)

        if 'subgoal' in nbatch:
            subgoal = nbatch['subgoal']
        else:
            subgoal = torch.zeros((B, self.subgoal_dim), dtype=state.dtype, device=state.device)    ##这个原版是None
        subgoal_zero = torch.zeros_like(subgoal)

        image_pair = raw_batch['image'] if 'image' in raw_batch else None
        qpos_pair = raw_batch['qpos'] if 'qpos' in raw_batch else None  ## (B, 2时间, 9=7jiont+2爪宽)
        # Pair-style D3P subgoal pair. Prefer explicit (t, t+h) pair from dataset.
        if 'd3p_subgoal_pair' in raw_batch:
            subgoal_pair = self.normalizer.normalize({'subgoal': raw_batch['d3p_subgoal_pair']}, self.subgoal_dim_nocont)['subgoal']
        else:
            subgoal_pair = torch.stack((subgoal, subgoal), dim=1)  # rollout / legacy fallback
        
        d3p_action_pair = (
            self.normalizer.normalize({'action': raw_batch['d3p_action_pair']})['action']
            if 'd3p_action_pair' in raw_batch else None
        )
        act_is_pad_pair = raw_batch['act_is_pad_pair'] if 'act_is_pad_pair' in raw_batch else None

        if d3p_action_pair is not None:
            if d3p_action_pair.dim() != 4:
                raise RuntimeError(f"d3p_action_pair must be (B,2,L,Dim_a), got shape={tuple(d3p_action_pair.shape)}")
            if d3p_action_pair.shape[1] != 2:
                raise RuntimeError(f"d3p_action_pair second dim must be 2, got {d3p_action_pair.shape[1]}")
            if d3p_action_pair.shape[-1] != self.action_dim:
                raise RuntimeError(
                    f"d3p_action_pair action dim mismatch: got {d3p_action_pair.shape[-1]}, expected {self.action_dim}"
                )
        if act_is_pad_pair is not None:
            if act_is_pad_pair.dim() != 3:
                raise RuntimeError(f"act_is_pad_pair must be (B,2,L), got shape={tuple(act_is_pad_pair.shape)}")
            if act_is_pad_pair.shape[1] != 2:
                raise RuntimeError(f"act_is_pad_pair second dim must be 2, got {act_is_pad_pair.shape[1]}")
            if d3p_action_pair is not None:
                if act_is_pad_pair.shape[0] != d3p_action_pair.shape[0] or act_is_pad_pair.shape[2] != d3p_action_pair.shape[2]:
                    raise RuntimeError(
                        "act_is_pad_pair shape mismatch with d3p_action_pair: "
                        f"{tuple(act_is_pad_pair.shape)} vs {tuple(d3p_action_pair.shape)}"
                    )
        if subgoal_pair.dim() != 3:
            raise RuntimeError(f"subgoal_pair must be (B,2,subgoal_dim), got shape={tuple(subgoal_pair.shape)}")
        if subgoal_pair.shape[1] != 2:
            raise RuntimeError(f"subgoal_pair second dim must be 2, got {subgoal_pair.shape[1]}")
        if subgoal_pair.shape[-1] != self.subgoal_dim:
            raise RuntimeError(
                f"subgoal_pair subgoal dim mismatch: got {subgoal_pair.shape[-1]}, expected {self.subgoal_dim}"
            )

        vis_enc_pair = self.branch_condition_encoder.encode_vis_pair(image_pair)

        if vis_enc_pair is not None and vis_enc_pair.shape[-1] != self.branch_condition_encoder.vis_dim:
            raise RuntimeError(
                f"vis_enc_pair dim mismatch: got {vis_enc_pair.shape[-1]}, expected {self.branch_condition_encoder.vis_dim}"
            )
        if qpos_pair is not None and qpos_pair.shape[-1] != self.qpos_feat_dim:
            raise RuntimeError(
                f"qpos_pair dim mismatch: got {qpos_pair.shape[-1]}, expected {self.qpos_feat_dim}"
            )

        # D3P-style condition2 in B2: fea_fuse = fuse(qpos, vis_enc)
        fea_fuse_pair = self.branch_condition_encoder.encode_fea_fuse_pair(qpos_pair, vis_enc_pair)
        
        extra_cond_zero = torch.zeros((B, self.extra_cond_dim), dtype=state.dtype, device=state.device)

        return {
            'nbatch': nbatch,

            'pcd': pcd,
            'pcd_zero': pcd_zero,
            'state': state,
            'state_zero': state_zero,
            'subgoal': subgoal,
            'subgoal_zero': subgoal_zero,
            'subgoal_pair': subgoal_pair,
            
            'vis_enc_pair': vis_enc_pair,
            'fea_fuse_pair': fea_fuse_pair,
            'b1_latent_act_pair': None,
            'image_pair': image_pair,

            'd3p_action_pair': d3p_action_pair,
            'act_is_pad_pair': act_is_pad_pair,

            'extra_cond_zero': extra_cond_zero,
        }

    def _is_zero_tensor(self, x: Optional[torch.Tensor]) -> bool:
        if x is None:
            return True
        return bool((x.abs().max() == 0).item())

    def _run_branch_checks(self, branch: str, cond: Dict[str, Optional[torch.Tensor]]):
        if (not self.fusion_debug_checks) or self._debug_checked_once:
            return

        if branch == 'A':
            if cond['extra_cond'] is None:
                raise RuntimeError(f'Branch {branch} missing extra_cond.')
            if cond['extra_cond'].shape[-1] != self.extra_cond_dim:
                raise RuntimeError(
                    f"Branch {branch} extra_cond dim mismatch: {cond['extra_cond'].shape[-1]} != {self.extra_cond_dim}"
                )
        else:
            if cond['extra_cond_pair'] is None:
                raise RuntimeError(f'Branch {branch} missing extra_cond_pair.')
            if cond['extra_cond_pair'].dim() != 3 or cond['extra_cond_pair'].shape[1] != 2:
                raise RuntimeError(f'Branch {branch} extra_cond_pair must be (B,2,D), got {cond["extra_cond_pair"].shape}')
            if cond['extra_cond_pair'].shape[-1] != self.extra_cond_dim:
                raise RuntimeError(
                    f"Branch {branch} extra_cond_pair dim mismatch: {cond['extra_cond_pair'].shape[-1]} != {self.extra_cond_dim}"
                )

        if branch == 'A':
            if self._is_zero_tensor(cond['state']):
                raise RuntimeError('Branch A expects non-zero state.')
            if self.use_pcd and self._is_zero_tensor(cond['pcd']):
                raise RuntimeError('Branch A expects non-zero pcd.')
            if not self._is_zero_tensor(cond['extra_cond']):
                raise RuntimeError('Branch A expects zero extra_cond.')
        else:
            if not self._is_zero_tensor(cond['state']):
                raise RuntimeError(f'Branch {branch} expects zero state.')
            if self.use_pcd and not self._is_zero_tensor(cond['pcd']):
                raise RuntimeError(f'Branch {branch} expects zero pcd.')

        self._debug_checked_once = True

    def _ensure_b1_latent_act_pair(
        self,
        common: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        latent_act_pair = common.get('b1_latent_act_pair', None)
        if latent_act_pair is not None:
            return latent_act_pair

        if self.dko is None:
            raise RuntimeError('B1 requires use_koopman_aux=true because it uses DKO latent acts.')
        if common['vis_enc_pair'] is None:
            raise RuntimeError('B1 requires vis_enc_pair, but image is missing.')

        latent_act_pair = torch.stack(
            (
                self.dko.get_latent_act(common['vis_enc_pair'][:, 0]),
                self.dko.get_latent_act(common['vis_enc_pair'][:, 1]),
            ),
            dim=1,
        )
        common['b1_latent_act_pair'] = latent_act_pair
        return latent_act_pair

    def _build_cond_by_branch(
        self,
        branch: str,
        common: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if branch == 'A':
            cond = {
                'pcd': common['pcd'],
                'state': common['state'],
                'subgoal': common['subgoal'],
                'extra_cond': common['extra_cond_zero'],
            }
        elif branch == 'B1':
            if common['vis_enc_pair'] is None:
                raise RuntimeError('B1 requires vis_enc_pair, but image is missing.')
            latent_act_pair = self._ensure_b1_latent_act_pair(common)
            extra_cond_pair = self.branch_condition_encoder.build_b1_extra_cond_pair(
                latent_act_pair,
                common['subgoal_pair'],
            )
            cond = {
                'pcd': common['pcd_zero'],
                'state': common['state_zero'],
                'subgoal': common['subgoal_zero'],
                'extra_cond_pair': extra_cond_pair,
            }
        elif branch == 'B2':
            if common['fea_fuse_pair'] is None:
                raise RuntimeError('B2 requires fea_fuse_pair, but image/qpos is missing.')
            extra_cond_pair = self.branch_condition_encoder.build_b2_extra_cond_pair(
                common['fea_fuse_pair'],
                common['subgoal_pair'],
            )
            cond = {
                'pcd': common['pcd_zero'],
                'state': common['state_zero'],
                'subgoal': common['subgoal_zero'],
                'extra_cond_pair': extra_cond_pair,
            }
        else:
            raise ValueError(f'Unknown branch: {branch}')

        self._run_branch_checks(branch, cond)
        return cond
    
    ##B分支extra_cond双时刻改单时刻
    def _rollout_cond_from_branch_cond(
        self,
        cond: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if 'extra_cond_pair' in cond:
            return {
                'pcd': cond['pcd'],
                'state': cond['state'],
                'subgoal': cond['subgoal'],
                'extra_cond': cond['extra_cond_pair'][:, 0],
            }
        return cond

    # ================================================================================
    # Actor core
    # ================================================================================
    def _compute_masked_bc_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_is_pad is None:
            return F.mse_loss(pred, target)

        loss = F.mse_loss(pred, target, reduction='none')
        ##把 padding 部分的 loss 去掉
        valid = (~action_is_pad).unsqueeze(-1).to(loss.dtype)# exclude losses from the padding actions    
        loss = loss * valid

        denom = valid.sum() * loss.shape[-1]
        if float(denom.item()) <= 0:
            return loss.mean()
        return loss.sum() / denom   ##只对非 padding 的动作维度求平均

    # 用纯噪A_k前向###########+完整逆扩散---->生成最终action
    def conditional_sample_action(
        self,
        cond: Dict[str, Optional[torch.Tensor]],
        action_init: Optional[torch.Tensor] = None,
        model=None,
    ) -> torch.Tensor:
        """
        Branch-aware diffusion sampling with unified entry:
          - model is None  -> actor_target (rollout/inference behavior)
          - model is actor -> online actor (training Q-loss branch)
        """
        B = cond['state'].shape[0]
        if action_init is None:
            action = torch.randn(         ##纯噪A_k
                size=(B, self.horizon, self.action_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            action = action_init

        timesteps = self.noise_scheduler_actor.timesteps
        for t in timesteps:
            if model is None:   ## rollout前向############################3
                pred_noise = self.actor_target(
                    cond['pcd'],
                    cond['state'],
                    cond['subgoal'],
                    action,
                    t,
                    extra_cond=cond['extra_cond'],
                )
            else:               ## trainin前向###########################3
                pred_noise = model(
                    cond['pcd'],
                    cond['state'],
                    cond['subgoal'],
                    action,
                    t,
                    extra_cond=cond['extra_cond'],
                )
            ## 反推一步:x_k到x_k-1
            action = self.noise_scheduler_actor.step(
                pred_noise, t, action, generator=None
            ).prev_sample
        return action

    def _compute_bc_step(
        self,
        cond: Dict[str, Optional[torch.Tensor]],
        action_target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        One diffusion training step for a given condition and target action chunk.
        """
        B = action_target.shape[0]
         # diffusion     ## （batchsize个样本）一次加噪（timesteps：batchsize 个随机 k）
        timesteps = torch.randint(
            0, self.noise_scheduler_actor.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        noise = torch.randn(action_target.shape, device=self.device)
        noisy_action = self.noise_scheduler_actor.add_noise(action_target, noise, timesteps)

        pred = self.actor(              #######BC前向###############################
            cond['pcd'],
            cond['state'],
            cond['subgoal'],
            noisy_action,
            timesteps,
            extra_cond=cond['extra_cond'],
        )
        bc_loss = self._compute_masked_bc_loss(pred, noise, action_is_pad)
        return {
            'bc_loss': bc_loss,
            'pred': pred,
            'timesteps': timesteps,
            'noisy_action': noisy_action,
        }

    def _compute_loss_actor_from_cond(
        self,
        branch: str,
        cond: Dict[str, Optional[torch.Tensor]],
        common: Dict[str, Optional[torch.Tensor]],
    ):
        B = common['nbatch']['state'].shape[0]

        # ******** bc loss ********
        if branch == 'A':
            cond_q = cond
            step_out = self._compute_bc_step(
                cond=cond,
                action_target=common['nbatch']['action'],
                action_is_pad=None,
            )
            bc_loss = step_out['bc_loss']
        else:
            if common['d3p_action_pair'] is None:
                raise RuntimeError(f'Branch {branch} requires d3p_action_pair for dual-time BC loss.')
            action_t = common['d3p_action_pair'][:, 0]
            action_th = common['d3p_action_pair'][:, 1]
            pad_t = common['act_is_pad_pair'][:, 0] if common['act_is_pad_pair'] is not None else None
            pad_th = common['act_is_pad_pair'][:, 1] if common['act_is_pad_pair'] is not None else None
            action_t, pad_t = self._fit_action_and_pad_to_horizon(action_t, pad_t)
            action_th, pad_th = self._fit_action_and_pad_to_horizon(action_th, pad_th)
            cond_t = {
                'pcd': cond['pcd'],
                'state': cond['state'],
                'subgoal': cond['subgoal'],
                'extra_cond': cond['extra_cond_pair'][:, 0],
            }
            cond_th = {
                'pcd': cond['pcd'],
                'state': cond['state'],
                'subgoal': cond['subgoal'],
                'extra_cond': cond['extra_cond_pair'][:, 1],
            }

            step_t = self._compute_bc_step(
                cond=cond_t,
                action_target=action_t,
                action_is_pad=pad_t,
            )
            step_th = self._compute_bc_step(
                cond=cond_th,
                action_target=action_th,
                action_is_pad=pad_th,
            )
            bc_loss = 0.5 * step_t['bc_loss'] + 0.5 * step_th['bc_loss']
            # Q-loss stays on current-t branch condition.
            cond_q = cond_t
            step_out = step_t

        # ******** q loss ********
        effective_eta = self.eta if branch == 'A' else 0.0
        if effective_eta != 0:
            if self.single_step_reverse_diffusion:  # 单次逆扩散，由Xt直接生成X0    ## 一次性反推:x_k到x_0
                pred_x0_list = []
                for i in range(B):
                    step_output = self.noise_scheduler_actor.step(
                        step_out['pred'][i:i+1],
                        int(step_out['timesteps'][i].item()),
                        step_out['noisy_action'][i:i+1],
                        generator=None,
                    )
                    pred_x0_list.append(step_output.pred_original_sample)
                new_action_seq = torch.cat(pred_x0_list, dim=0)
            else:                                   # 重新用纯噪A_k前向###########+完整逆扩散
                new_action_seq = self.conditional_sample_action(
                    cond=cond_q,
                    model=self.actor
                )

            new_action = new_action_seq[:, self.observation_history_num-1:
                                        self.observation_history_num-1+self.Tr]
            new_action = new_action.reshape((B, -1))
            q1_new_action, q2_new_action = self.critic(
                common['pcd'], common['state'], common['subgoal'], new_action
            )

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            actor_loss = bc_loss + effective_eta * q_loss
        else:
            if (branch != 'A') and (self.eta != 0):
                q_loss = torch.zeros((), device=self.device)
            else:
                q_loss = torch.tensor(-1, device=self.device)
            actor_loss = bc_loss

        return actor_loss, bc_loss, q_loss

    ## 把归一化动作还原成真实动作'action_pred'和'action'
    def _format_action_from_normalized(
        self,
        action_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        action = self.normalizer.unnormalize(naction=action_norm)
        start = self.observation_history_num - 1
        end = start + self.n_action_steps
        action_run = action[:, start:end]
        return {
            'action_pred': action,  ## 完整预测AC
            'action': action_run,   ## 真实要执行的 action 段
        }

    # ===========================================================================
    # Public actor API
    # ===========================================================================
    def compute_loss_actor(self, batch: Dict[str, torch.Tensor]):
        common = self._prepare_branch_inputs(batch)
        branch = self._select_train_branch(common)
        cond = self._build_cond_by_branch(branch, common)
        actor_loss, bc_loss, q_loss = self._compute_loss_actor_from_cond(branch=branch, cond=cond, common=common)

        # DKO
        koop_aux = None
        if self.use_koopman_aux and branch in ('B1', 'B2'):
            koop_aux = self._compute_koopman_aux(common)
            if branch == 'B1':
                common['b1_latent_act_pair'] = torch.stack(
                    (koop_aux['curr_latent_acts'], koop_aux['next_latent_acts']),
                    dim=1,
                )
        if koop_aux is not None:
            actor_loss = actor_loss + self.koopman_kvp_weight * koop_aux['koop_consis_kvp']
            actor_loss = actor_loss + self.koopman_fea_weight * koop_aux['koop_consis_fea']
            self._last_actor_aux_logs = {
                'train_koop_consis_kvp': float(koop_aux['koop_consis_kvp'].detach().item()),
                'train_koop_consis_fea': float(koop_aux['koop_consis_fea'].detach().item()),
            }
        else:
            self._last_actor_aux_logs = {
                'train_koop_consis_kvp': 0.0,
                'train_koop_consis_fea': 0.0,
            }

        return actor_loss, bc_loss, q_loss

    def get_last_actor_aux_logs(self) -> Dict[str, float]:
        return dict(self._last_actor_aux_logs)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        common = self._prepare_branch_inputs(obs_dict)
        resolved_b = self._resolve_branch(self.b_branch, common)
        use_dual = (self.mode == 'SWITCH') and (resolved_b == self.b_branch)
        if not use_dual:
            branch = self._resolve_branch('A' if self.mode == 'SWITCH' else self.single_branch, common)
            cond = self._build_cond_by_branch(branch, common)
            cond_run = self._rollout_cond_from_branch_cond(cond)    ## B分支extra_cond双时刻改单时刻
            with torch.no_grad():
                action_norm = self.conditional_sample_action(cond=cond_run, model=None)
            err = self.compute_test_time_ddpm_error(cond=cond_run, action_norm=action_norm)
            out = self._format_action_from_normalized(action_norm)  ## 把归一化动作还原成真实动作'action_pred'和'action'
            if branch == 'A':
                out['branch_err_A'] = err.unsqueeze(-1)
                out['branch_err_B'] = torch.full_like(err.unsqueeze(-1), 1e6)   ## 很大的假误差
                out['selected_branch'] = torch.zeros_like(err.unsqueeze(-1))    ## 0
            else:
                out['branch_err_A'] = torch.full_like(err.unsqueeze(-1), 1e6)
                out['branch_err_B'] = err.unsqueeze(-1)
                out['selected_branch'] = torch.ones_like(err.unsqueeze(-1))     ## 1
            return out
            # out: = {
            #     "action_pred": action,          # full predicted action sequence
            #     "action": action_run,           # action segment to execute
            #     "branch_err_A": branch_err_A,   # branch A reconstruction / DDPM error
            #     "branch_err_B": branch_err_B,   # branch B reconstruction / DDPM error
            #     "selected_branch": selected_branch,
            # }
        ########################A分支############################
        cond_A = self._build_cond_by_branch('A', common)
        cond_A_run = self._rollout_cond_from_branch_cond(cond_A)

        with torch.no_grad():
            action_A_norm = self.conditional_sample_action(cond=cond_A_run, model=None)
        err_A = self.compute_test_time_ddpm_error(cond=cond_A_run, action_norm=action_A_norm)

        ########################B分支############################
        cond_B = self._build_cond_by_branch(self.b_branch, common)
        cond_B_run = self._rollout_cond_from_branch_cond(cond_B)
        with torch.no_grad():
            action_B_norm = self.conditional_sample_action(cond=cond_B_run, model=None)
        err_B = self.compute_test_time_ddpm_error(cond=cond_B_run, action_norm=action_B_norm)

        select_B = (err_B < err_A).view(-1, 1, 1)   ## 选 B：True，选 A：False 
        action_sel_norm = torch.where(select_B, action_B_norm, action_A_norm)

        out = self._format_action_from_normalized(action_sel_norm)  ## 'action_pred'完整预测AC，'action'真实要执行的 action 段
        out['branch_err_A'] = err_A.unsqueeze(-1)
        out['branch_err_B'] = err_B.unsqueeze(-1)
        out['selected_branch'] = select_B[:, 0, 0].to(dtype=action_sel_norm.dtype).unsqueeze(-1)
        return out

    # =========================
    # Inference aggregation hooks
    # =========================
    @torch.no_grad()
    def compute_test_time_ddpm_error(
        self,
        cond: Dict[str, Optional[torch.Tensor]],
        action_norm: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        D3P-style test-time DDPM error:
          E_t,eps [ || eps_theta(x_t, t, cond) - eps ||^2 ]
        Returns per-batch error with shape (B,).
        """
        if num_samples is None:
            num_samples = self.d3p_rollout_error_samples    ## 10份 action_norm
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        B = action_norm.shape[0]
        action_rep = action_norm.repeat_interleave(num_samples, dim=0)  ## 复制10份   (B*num_samples, H, D)
        cond_rep: Dict[str, Optional[torch.Tensor]] = {}
        for k, v in cond.items():                                       ## cond 每个key 也重复 num_samples 次
            if torch.is_tensor(v):
                cond_rep[k] = v.repeat_interleave(num_samples, dim=0)
            else:
                cond_rep[k] = v

        noise = torch.randn(action_rep.shape, device=action_rep.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler_actor.config.num_train_timesteps,
            (B * num_samples,),
            device=action_rep.device
        ).long()
        noisy_action = self.noise_scheduler_actor.add_noise(action_rep, noise, timesteps)   ## 一步加噪
        pred_noise = self.actor_target(   ## 前向 ######
            cond_rep['pcd'],
            cond_rep['state'],
            cond_rep['subgoal'],
            noisy_action,
            timesteps,
            extra_cond=cond_rep['extra_cond'],
        )   ## (B*K, H, D)
        loss = F.mse_loss(pred_noise, noise, reduction='none')
        loss = loss.mean(dim=(1, 2)).reshape(B, num_samples).mean(dim=1)
        return loss
