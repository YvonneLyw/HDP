from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from hiera_diffusion_policy.model.diffusion.branch_condition_encoder import BranchConditionEncoder
from hiera_diffusion_policy.model.diffusion.d3p_koopman import DeepKoopmanModule
from hiera_diffusion_policy.policy.doser_branch_selector import DoserBranchSelector
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
      - Q loss is always applied on branch A; B branches can optionally share the same Q loss.
      - SWITCH branch arbitration can use diffusion error, critic score, or hybrid selectors.
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
        b_branch_use_q_loss: bool = False,
        branch_selector: str = 'err',   ##['err','q','hybrid_gate','hybrid_linear']
        use_action_smoothing: bool = False,
        use_test_time_aggregation: bool = False,
        test_time_agg_beta: float = 0.97,
        test_time_agg_tau: float = 0.1,
        extra_cond_dim: int = 64,
        image_feat_dim: int = 64,
        qpos_feat_dim: int = 9,
        image_size=(84, 84),
        doser_selector: Optional[Dict] = None,
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
        self.b_branch_use_q_loss = bool(b_branch_use_q_loss)
        self.branch_selector = str(branch_selector).lower()
        self.use_action_smoothing = bool(use_action_smoothing)
        self.use_test_time_aggregation = bool(use_test_time_aggregation)
        self.test_time_agg_beta = float(test_time_agg_beta)
        self.test_time_agg_tau = float(test_time_agg_tau)

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
        if self.branch_selector not in ('err', 'q', 'hybrid_gate', 'hybrid_linear', 'doser'):
            raise ValueError(
                "branch_selector must be one of ['err','q','hybrid_gate','hybrid_linear','doser'], "
                f"got {self.branch_selector}"
            )
        if self.test_time_agg_beta <= 0.0:
            raise ValueError(f"test_time_agg_beta must be > 0, got {self.test_time_agg_beta}")
        if self.test_time_agg_tau <= 0.0:
            raise ValueError(f"test_time_agg_tau must be > 0, got {self.test_time_agg_tau}")
        if self.use_test_time_aggregation and self.use_action_smoothing:
            raise ValueError(
                "use_test_time_aggregation and use_action_smoothing should not be enabled together "
                "in the first fusion version."
            )

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
        doser_selector_cfg = dict(doser_selector or {})
        self.doser_selector = DoserBranchSelector(**doser_selector_cfg)
        self._smoothing_prev_exec_norm: Optional[torch.Tensor] = None   ## 上一次最终执行的动作序列，用于判断这是不是当前 episode 的第一次动作聚合（没有历史执行动作，所以逻辑会特殊处理）
        self._agg_action_buffer: Optional[torch.Tensor] = None
        self._agg_weight_buffer: Optional[torch.Tensor] = None
        self._agg_branch_buffer: Optional[torch.Tensor] = None

    ########## 模型（actor,branch_condition_encoder，dko）参数加入optimizer############ D3P fusion 把actor+ branch_condition_encoder （+dko）一起训
    def get_actor_training_parameters(self):
        # B-branch actor losses depend on branch_condition_encoder outputs, and
        # optionally DKO auxiliary losses, so they must be optimized together
        # during actor stage.
        params = list(self.actor.parameters()) + list(self.branch_condition_encoder.parameters())
        if self.dko is not None:
            params += list(self.dko.parameters())
        return params

    def reset(self):
        super().reset()
        self._smoothing_prev_exec_norm = None
        self._agg_action_buffer = None
        self._agg_weight_buffer = None
        self._agg_branch_buffer = None

    def needs_rollout_image_qpos(self) -> bool:
        if self.mode == 'SINGLE':
            return self.single_branch in ('B1', 'B2')
        return True

    
    # =========================
    # A/B分支AC的起始点及截取
    # =========================
    def _get_action_start(self, branch: str) -> int:
        if branch in ('B1', 'B2'):
            return 0
        return self.observation_history_num - 1

    def _extract_action_segment(
        self,
        action_seq: torch.Tensor,
        start_idx,
        length: int,
    ) -> torch.Tensor:
        """
        Extract a fixed-length action segment from a batched sequence.
        start_idx can be:
          - int: same start for the whole batch
          - Tensor(B,): branch-specific / sample-specific starts
        """
        B, T, D = action_seq.shape
        length = int(length)    ## 16/4
        if isinstance(start_idx, int):
            start = int(start_idx)
            end = min(start + length, T)
            segment = action_seq[:, start:end]
            if segment.shape[1] == length:
                return segment
            pad = segment[:, -1:].expand(-1, length - segment.shape[1], -1) ## 不足16/4的用最后一个动作pad补齐
            return torch.cat((segment, pad), dim=1)

        start_idx = start_idx.reshape(-1).to(device=action_seq.device, dtype=torch.long)
        offsets = torch.arange(length, device=action_seq.device, dtype=torch.long).view(1, -1)
        gather_idx = (start_idx[:, None] + offsets).clamp(min=0, max=T-1)
        gather_idx = gather_idx.unsqueeze(-1).expand(B, length, D)
        return torch.gather(action_seq, dim=1, index=gather_idx)

    def _write_action_segment(
        self,
        action_seq: torch.Tensor,
        start_idx,
        segment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Write a fixed-length segment back into a batched action sequence.
        """
        out = action_seq.clone()
        B, T, _ = action_seq.shape
        if isinstance(start_idx, int):
            start_idx = torch.full(
                (B,),
                int(start_idx),
                device=action_seq.device,
                dtype=torch.long,
            )
        else:
            start_idx = start_idx.reshape(-1).to(device=action_seq.device, dtype=torch.long)

        for i in range(B):
            start = int(start_idx[i].item())
            end = min(start + segment.shape[1], T)
            out[i, start:end] = segment[i, :end-start]
        return out

    # =========================
    # aggregate test time
    # =========================
    def _get_rollout_temporal_weights(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        steps = torch.arange(length, device=device, dtype=dtype)
        beta_base = torch.full((length,), self.test_time_agg_beta, device=device, dtype=dtype)
        weights = torch.pow(beta_base, steps)
        weights = weights / weights.sum().clamp_min(1e-8)
        return weights

    def _compute_rollout_branch_confidence(
        self,
        score_A: torch.Tensor,
        score_B: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.stack((score_A, score_B), dim=-1)
        ##return F.softmax(scores / self.test_time_agg_tau, dim=-1)

        # scaled_scores = (scores / self.test_time_agg_tau).clamp(min=-50.0, max=50.0)
        # return torch.exp(scaled_scores)
        scaled_scores = (scores / self.test_time_agg_tau).clamp(min=-20.0, max=20.0)
        pairwise_confidence = F.softmax(scaled_scores, dim=-1)  ## 负责这轮里 A/B 谁更占优
        query_reliability = torch.exp(torch.max(scaled_scores, dim=-1, keepdim=True).values)    ## 负责这轮整体值不值得信
        return pairwise_confidence * query_reliability

    def _ensure_rollout_agg_buffers(
        self,
        batch_size: int,
        chunk_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shift = max(1, int(self.n_action_steps))    ## n_action_steps：4
        rows = int(np.ceil(chunk_len / shift)) * 2  ## 走完当前预测AC（长=16），需要ceil(chunk_len / shift)个rollout    ## *2是两个branch
        rows = max(rows, 2)
        needs_init = (
            self._agg_action_buffer is None
            or self._agg_weight_buffer is None
            or self._agg_branch_buffer is None
            or self._agg_action_buffer.shape != (batch_size, rows, chunk_len, action_dim)
            or self._agg_action_buffer.device != device
            or self._agg_action_buffer.dtype != dtype
        )
        if needs_init:
            self._agg_action_buffer = torch.zeros(
                (batch_size, rows, chunk_len, action_dim),
                device=device,
                dtype=dtype,
            )
            self._agg_weight_buffer = torch.zeros(
                (batch_size, rows, chunk_len),
                device=device,
                dtype=dtype,
            )
            self._agg_branch_buffer = torch.full(
                (batch_size, rows),
                fill_value=-1,
                device=device,
                dtype=torch.long,
            )

    def _aggregate_rollout_candidates(
        self,
        candidate_actions_norm: torch.Tensor,
        score_A: torch.Tensor,
        score_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Minimal D3P-style test-time aggregator:
          - keep A/B rollout-aligned trajectories from the latest overlapping queries
          - convert current query scores to pairwise_confidence * query_reliability
          - multiply by temporal decay weights
          - choose the max-weight trajectory per future step
        """
        if candidate_actions_norm.ndim != 4 or candidate_actions_norm.shape[1] != 2:
            raise RuntimeError(
                f"candidate_actions_norm must have shape (B,2,H,D), got {tuple(candidate_actions_norm.shape)}"
            )

        B, _, L, D = candidate_actions_norm.shape   ##(B, 2, L=chunk_len=4, D)
        self._ensure_rollout_agg_buffers(
            batch_size=B,
            chunk_len=L,
            action_dim=D,
            device=candidate_actions_norm.device,
            dtype=candidate_actions_norm.dtype,
        )

        assert self._agg_action_buffer is not None
        assert self._agg_weight_buffer is not None
        assert self._agg_branch_buffer is not None

        shift = min(max(1, int(self.n_action_steps)), L)
        rows_to_insert = candidate_actions_norm.shape[1]    ## 2

        self._agg_action_buffer = torch.roll(self._agg_action_buffer, shifts=-rows_to_insert, dims=1)
        self._agg_weight_buffer = torch.roll(self._agg_weight_buffer, shifts=-rows_to_insert, dims=1)
        self._agg_branch_buffer = torch.roll(self._agg_branch_buffer, shifts=-rows_to_insert, dims=1)

        self._agg_action_buffer[:, :, :-shift] = self._agg_action_buffer[:, :, shift:].clone()
        self._agg_action_buffer[:, :, -shift:] = 0
        self._agg_weight_buffer[:, :, :-shift] = self._agg_weight_buffer[:, :, shift:].clone()
        self._agg_weight_buffer[:, :, -shift:] = 0

        temporal_weights = self._get_rollout_temporal_weights(
            length=L,
            device=candidate_actions_norm.device,
            dtype=candidate_actions_norm.dtype,
        )
        conf = self._compute_rollout_branch_confidence(score_A, score_B)  # (B,2)
        candidate_weights = conf.unsqueeze(-1) * temporal_weights.view(1, 1, L)

        self._agg_action_buffer[:, -rows_to_insert:] = candidate_actions_norm
        self._agg_weight_buffer[:, -rows_to_insert:] = candidate_weights
        self._agg_branch_buffer[:, -rows_to_insert:] = torch.tensor(
            [0, 1],
            device=candidate_actions_norm.device,
            dtype=torch.long,
        ).view(1, rows_to_insert).expand(B, rows_to_insert)

        best_row_idx = torch.argmax(self._agg_weight_buffer, dim=1)  ## (B, L=4)
        action_rows = self._agg_action_buffer.permute(0, 2, 1, 3)    ## (B, L=4, Row数=query数*2分支, D)
        aggregated = torch.gather(
            action_rows,
            dim=2,
            index=best_row_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, D),
        ).squeeze(2)    ## (B, L=4, D)

        branch_rows = self._agg_branch_buffer.unsqueeze(1).expand(B, L, -1)
        chosen_branch = torch.gather(
            branch_rows,
            dim=2,
            index=best_row_idx.unsqueeze(-1),
        ).squeeze(-1)   ## (B, L=4)   ## 0=A, 1=B
        return aggregated, chosen_branch

    # =========================
    # DKO
    # =========================
    def _reduce_feature_loss(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(target, pred, reduction='none')
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1)
        return loss.mean()

    ## 图像增强后的 两时刻的（front_rgb，wrist_rgb）
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
            aug_curr_vis_dict, aug_next_vis_dict = self.augment_images(image_pair)  ## 图像增强
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

        # Branch-B / DOSER side channels are intentionally kept outside the HDP
        # Normalizer: images are float [0,1] and normalized inside the image
        # encoder; qpos is standardized by dataset/runner; action/subgoal below
        # are explicitly mapped through the HDP action/state normalizer.
        image_pair = raw_batch['image'] if 'image' in raw_batch else None
        doser_image_pair = raw_batch['doser_image_pair'] if 'doser_image_pair' in raw_batch else None
        qpos_pair = raw_batch['qpos'] if 'qpos' in raw_batch else None  ## (B, 2时间, 9=7jiont+2爪宽)
        
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
            'doser_image_pair': doser_image_pair,
            'qpos_pair': qpos_pair,

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
    
    # =========================
    # train branch selection
    # =========================
    
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
        effective_eta = self.eta if (branch == 'A' or self.b_branch_use_q_loss) else 0.0
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

            new_action = self._extract_action_segment(
                new_action_seq,
                self._get_action_start(branch),
                self.Tr,
            )
            new_action = new_action.reshape((B, -1))
            q1_new_action, q2_new_action = self.critic(
                common['pcd'], common['state'], common['subgoal'], new_action
            )

            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()

            actor_loss = bc_loss + effective_eta * q_loss
        else:   ## B不使用Q，无q_loss
            if (branch != 'A') and (self.eta != 0):
                q_loss = torch.zeros((), device=self.device)
            else:
                q_loss = torch.tensor(-1, device=self.device)
            actor_loss = bc_loss

        return actor_loss, bc_loss, q_loss

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
            actor_loss = actor_loss + self.koopman_kvp_weight * koop_aux['koop_consis_kvp'] \
                                    + self.koopman_fea_weight * koop_aux['koop_consis_fea']
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
            err = self.compute_ddpm_error(cond=cond_run, action_norm=action_norm)
            ## action_smoothing和还原成真实动作
            exec_start_idx = self._get_action_start(branch) ## A:1, B:0
            if self.use_action_smoothing:
                action_norm = self._smooth_selected_action_sequence(
                    action_norm,
                    start_idx=exec_start_idx,
                )   ## A：把 [1:1+n_action_steps] 平滑后放回 [1:...] / B: 把 [0:0+n_action_steps] 平滑后放回 [0:...]
            action = self.normalizer.unnormalize(naction=action_norm)
            out = {
                'action_pred': action,  ## 完整预测AC
                'action': self._extract_action_segment(action, exec_start_idx, self.n_action_steps),
            }   ## 'action_pred'完整预测AC，'action'真实要执行的 action 段 (B,AC长=4,dimA）

            if branch == 'A':
                out['branch_score_A'] = -err.unsqueeze(-1)
                out['branch_score_B'] = torch.full_like(err.unsqueeze(-1), -1e6)   ## 很差的假分数
                out['selected_branch'] = torch.zeros_like(err.unsqueeze(-1))    ## 0
            else:
                out['branch_score_A'] = torch.full_like(err.unsqueeze(-1), -1e6)
                out['branch_score_B'] = -err.unsqueeze(-1)
                out['selected_branch'] = torch.ones_like(err.unsqueeze(-1))     ## 1
            return out
            # out: = {
            #     "action_pred": action,          # full predicted action sequence
            #     "action": action_run,           # action segment to execute
            #     "branch_score_A": branch_score_A,   # branch A selector score
            #     "branch_score_B": branch_score_B,   # branch B selector score
            #     "selected_branch": selected_branch,
            # }
        ########################A分支############################
        cond_A = self._build_cond_by_branch('A', common)
        cond_A_run = self._rollout_cond_from_branch_cond(cond_A)

        with torch.no_grad():
            action_A_norm = self.conditional_sample_action(cond=cond_A_run, model=None)

        ########################B分支############################
        cond_B = self._build_cond_by_branch(self.b_branch, common)
        cond_B_run = self._rollout_cond_from_branch_cond(cond_B)
        with torch.no_grad():
            action_B_norm = self.conditional_sample_action(cond=cond_B_run, model=None)
        #################### A/B 各自 start index 对齐 #####################
        start_A = self._get_action_start('A')                 ## 1
        start_B = self._get_action_start(self.b_branch)       ## 0
        aligned_A_norm = self._extract_action_segment(action_A_norm, start_A, self.horizon,)
        aligned_B_norm = self._extract_action_segment(action_B_norm, start_B, self.horizon,)

        if self.branch_selector == 'doser':
            selector_out = self.doser_selector.select(
                common=common,
                aligned_A_norm=aligned_A_norm,
                aligned_B_norm=aligned_B_norm,
                critic_target=self.critic_target,
                Tr=self.Tr,
            )
        else:
            err_A = self.compute_ddpm_error(cond=cond_A_run, action_norm=action_A_norm)   ## (B,1)
            err_B = self.compute_ddpm_error(cond=cond_B_run, action_norm=action_B_norm)   ## (B,1)
            selector_out = self._select_branch_in_switch(
                common=common,
                err_A=err_A,
                err_B=err_B,
                aligned_A_norm=aligned_A_norm,
                aligned_B_norm=aligned_B_norm,
            )
            # return {
            #     'select_B': select_B.view(-1, 1, 1),
            #     'select_score_A': select_score_A,
            #     'select_score_B': select_score_B,
            #     'select_source': select_source,
            # }

        ## aggregate test time
        if self.use_test_time_aggregation:
            aggregated_norm, chosen_branch = self._aggregate_rollout_candidates(
                candidate_actions_norm=torch.stack((aligned_A_norm, aligned_B_norm), dim=1),    ##(B, 2, H=chunk_len=4, D)
                score_A=selector_out['select_score_A'],
                score_B=selector_out['select_score_B'],
            )
            out = {
                'action': self.normalizer.unnormalize(naction=aggregated_norm[:, :self.n_action_steps]),
                'selected_branch_exec_ratio': chosen_branch[:, :self.n_action_steps].float().mean(  ## (B,1) e.g.[0.5,0.25,0.75,0,1,...]######
                    dim=1, keepdim=True
                ),
            }
            out['branch_score_A'] = selector_out['select_score_A'].unsqueeze(-1)   ## (B,1)
            out['branch_score_B'] = selector_out['select_score_B'].unsqueeze(-1)   ## (B,1)
            if 'select_source' in selector_out:
                out['branch_select_source'] = selector_out['select_source'].unsqueeze(-1)
                # 'hybrid_gate'模式： 0 = 比较 diffusion errors    1 = 比较 critic Q scores
                # 'doser'模式：0 = A/B action 都 ID，按 Q 选择      1 = 一个 ID 一个 OOD    2 = A/B action 都 OOD，使用 state/value/fallback
            return out

        ## 直接按score选
        select_B = selector_out['select_B']
        action_sel_norm = torch.where(select_B, action_B_norm, action_A_norm)   ## 原始hard-select完整AC
        action_sel_aligned_norm = torch.where(select_B, aligned_B_norm, aligned_A_norm)

        ## action_smoothing
        if self.use_action_smoothing:
            action_sel_aligned_norm = self._smooth_selected_action_sequence(
                action_sel_aligned_norm,
            )   ## rollout-aligned后，当前执行段总是从 index 0 开始

        ## 还原成真实动作（unnormalize+截取n_action_steps）
        out = {
            'action_pred': self.normalizer.unnormalize(naction=action_sel_norm),   ## 保持原始hard-select完整AC，避免改动train诊断语义  #(B,AC长=16,dimA）
            'action': self.normalizer.unnormalize(
                naction=action_sel_aligned_norm[:, :self.n_action_steps]
            ),
            'selected_branch': select_B[:, 0, 0].to(dtype=action_sel_norm.dtype).unsqueeze(-1),    ## (B,1) e.g.[1,0,1,1,0,0...]    #########
        }
        out['branch_score_A'] = selector_out['select_score_A'].unsqueeze(-1)   ## (B,1)
        out['branch_score_B'] = selector_out['select_score_B'].unsqueeze(-1)   ## (B,1)
        if 'select_source' in selector_out:
            out['branch_select_source'] = selector_out['select_source'].unsqueeze(-1)
                # 'hybrid_gate'模式： 0 = 比较 diffusion errors    1 = 比较 critic Q scores
                # 'doser'模式：0 = A/B action 都 ID，按 Q 选择      1 = 一个 ID 一个 OOD    2 = A/B action 都 OOD，使用 state/value/fallback
        return out

    # =========================
    # 
    # =========================
    @torch.no_grad()
    def compute_ddpm_error(
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
    
    @torch.no_grad()
    def _compute_branch_q_score(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        aligned_action_norm: torch.Tensor,
    ) -> torch.Tensor:
        B = aligned_action_norm.shape[0]
        q_action = aligned_action_norm[:, :self.Tr].reshape((B, -1))
        q1, q2 = self.critic_target(
            common['pcd'], common['state'], common['subgoal'], q_action
        )
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        return torch.minimum(q1, q2)

    @torch.no_grad()
    def _select_branch_in_switch(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        err_A: torch.Tensor,
        err_B: torch.Tensor,
        aligned_A_norm: torch.Tensor,
        aligned_B_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.branch_selector == 'doser':
            raise RuntimeError("branch_selector='doser' should call DoserBranchSelector.select().")
        if self.branch_selector == 'err':
            select_B = (err_B < err_A)
            select_score_A = -err_A
            select_score_B = -err_B
            select_source = torch.zeros_like(err_A, dtype=torch.long)   # 0 = 比较 diffusion errors
        else:
            q_A = self._compute_branch_q_score(common, aligned_action_norm=aligned_A_norm)
            q_B = self._compute_branch_q_score(common, aligned_action_norm=aligned_B_norm)

            if self.branch_selector == 'q':
                select_B = q_B > q_A
                select_score_A = q_A
                select_score_B = q_B
                # 1 = 比较 critic Q scores
                select_source = torch.ones_like(err_A, dtype=torch.long)
            elif self.branch_selector == 'hybrid_gate':
                err_scale = (0.5 * (err_A.abs() + err_B.abs())).clamp_min(1e-6)
                err_rel_gap = (err_A - err_B).abs() / err_scale     ## err_A和err_B的差距

                select_B_err = (err_B < err_A)
                select_B_q = (q_B > q_A)
                err_gate_ratio = 0.25
                use_err = err_rel_gap > err_gate_ratio
                
                select_B = torch.where(use_err, select_B_err, select_B_q)
                select_score_A = torch.where(use_err, -err_A, q_A)
                select_score_B = torch.where(use_err, -err_B, q_B)
                select_source = torch.where(
                    use_err,
                    torch.zeros_like(err_A, dtype=torch.long),  # 0 = 比较 diffusion errors
                    torch.ones_like(err_A, dtype=torch.long),   # 1 = 比较 critic Q scores
                )
            else:
                q_weight = 1.0
                err_weight = 1.0
                q_scale = torch.maximum(
                    torch.maximum(q_A.abs(), q_B.abs()),
                    torch.full_like(q_A, 1e-6),
                )
                err_scale = torch.maximum(
                    torch.maximum(err_A.abs(), err_B.abs()),
                    torch.full_like(err_A, 1e-6),
                )
                score_A = q_weight * (q_A / q_scale) - err_weight * (err_A / err_scale)
                score_B = q_weight * (q_B / q_scale) - err_weight * (err_B / err_scale)
                select_B = score_B > score_A
                select_score_A = score_A
                select_score_B = score_B
                # 2 = 比较 hybrid linear score (Q and err combined)
                select_source = torch.full_like(err_A, 2, dtype=torch.long)

        return {
            'select_B': select_B.view(-1, 1, 1),
            'select_score_A': select_score_A,
            'select_score_B': select_score_B,
            'select_source': select_source,
        }


    @torch.no_grad()
    def _smooth_selected_action_sequence(
        self,
        action_norm: torch.Tensor,
        start_idx=0,
    ) -> torch.Tensor:
        """
        D3P-style smoothing-only post-processing:
          - smooth only the action chunk that will actually be executed
          - first rollout chunk: prepend the local pre-execution anchor from the predicted sequence
          - subsequent chunks: prepend the last action from the previous smoothed chunk
          - apply Savitzky-Golay filtering along the execution axis

        This is applied only after branch selection, so selector scores and
        DDPM/Q arbitration still operate on the raw sampled trajectories.
        """
        if action_norm.ndim != 3:
            return action_norm
        if savgol_filter is None:
            raise ImportError(
                "use_action_smoothing=true requires scipy.signal.savgol_filter, but scipy is not available."
            )

        B, T, D = action_norm.shape
        if isinstance(start_idx, int):
            start_idx = torch.full(
                (B,),
                int(start_idx),
                device=action_norm.device,
                dtype=torch.long,
            )   ## A:1, B:0
        else:
            start_idx = start_idx.reshape(-1).to(device=action_norm.device, dtype=torch.long)

        ## 取出这次 rollout 真的要执行的n_action_steps    ## A：action_norm[:, 1:5] / B：action_norm[:, 0:4]
        exec_seq = self._extract_action_segment(action_norm, start_idx, self.n_action_steps)    ## (B, 4, D)    
        if exec_seq.shape[1] == 0:
            return action_norm

        if self._smoothing_prev_exec_norm is None:  ## 当前 episode 是第一次 rollout chunk
            anchor_idx = (start_idx - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, D) ## A：添加action_norm[:, 0] / B：添加action_norm[:, 0]
            current_anchor = torch.gather(action_norm, dim=1, index=anchor_idx) ##(B, 1, D) ##按每个 batch 的 anchor_idx 从时间维度取出一个 action：action_norm[:, 0:1, :]
            aug = torch.cat((current_anchor, exec_seq), dim=1)  ## (B, 5, D)
            ##即（action_norm[:, 0, :]拼action_norm[:, 1:5或0:4， ：]）
        else:   ## 后续 rollout chunk
            prev_last = self._smoothing_prev_exec_norm[:, -1:, :].to(
                device=action_norm.device, dtype=action_norm.dtype
            )
            aug = torch.cat((prev_last, exec_seq), dim=1)   ## 拼上一段的最后一步
            ##即（_smoothing_prev_exec_norm[:, -1, :]拼action_norm[:, 1:5或0:4， ：]）

        ##savgol_filter平滑窗口：AC越长，平滑窗口可以稍大；AC短时，用最小窗口 3。
        window = min(max(3, self.n_action_steps // 2 + 1), aug.shape[1])
        if window % 2 == 0:
            window -= 1
        if window < 3:
            self._smoothing_prev_exec_norm = exec_seq.detach()
            return action_norm
        poly_order = min(2, window - 1)

        aug_np = aug.detach().to('cpu').numpy()
        smoothed_np = aug_np.copy()
        if D > 1:
            smoothed_np[:, :, :-1] = savgol_filter(
                aug_np[:, :, :-1],      ## 只平滑除了最后一维以外的所有 action 维度（gripper不参与平滑）
                window_length=window,
                polyorder=poly_order,
                axis=1,                 ## 沿时间轴平滑
            )
        else:
            smoothed_np = savgol_filter(
                aug_np,
                window_length=window,
                polyorder=poly_order,
                axis=1,
            )

        smoothed_exec = torch.from_numpy(smoothed_np[:, 1:]).to(
            device=action_norm.device, dtype=action_norm.dtype
        )   ## 去掉之前拼的current_anchor/prev_last
        smoothed_action = self._write_action_segment(action_norm, start_idx, smoothed_exec) ##(B, 4, D)
        self._smoothing_prev_exec_norm = smoothed_exec.detach()
        return smoothed_action

   
