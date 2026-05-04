from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hiera_diffusion_policy.model.diffusion.branch_condition_encoder import BranchConditionEncoder
from hiera_diffusion_policy.policy.hiera_diffusion_policy import HieraDiffusionPolicy


class HieraDiffusionPolicyD3PFusion(HieraDiffusionPolicy):
    """
    Debug-first fusion policy with explicit branch enumeration.

    Branch contracts:
      - A : state/pcd real, extra_cond=zeros
      - B1: state/pcd zero, dual-time extra_cond from (vis_enc, subgoal)
      - B2: state/pcd zero, dual-time extra_cond from (fea_fuse, subgoal)

    Notes:
      - actor_small only receives extra_cond; no image/qpos/sub-branch args are passed into actor.
      - subgoal_pair keeps two timesteps but currently duplicates current subgoal for both slots.
      - B branch BC loss uses two timestamps (t and t+h) with 0.5/0.5 weighting.
    """

    def __init__(
        self,
        d3p_query_every: int = 4,
        d3p_train_branch: str = 'A',
        d3p_b_branch: str = 'B1',
        d3p_switch_signal: float = 1.0,
        d3p_enable_switching: bool = False,
        fusion_debug_checks: bool = True,
        extra_cond_dim: int = 64,
        image_feat_dim: int = 64,
        qpos_feat_dim: int = 9,
        image_size=(84, 84),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d3p_query_every = int(d3p_query_every)
        self.d3p_train_branch = str(d3p_train_branch)
        self.d3p_b_branch = str(d3p_b_branch)
        self.d3p_switch_signal = float(d3p_switch_signal)
        self.d3p_enable_switching = bool(d3p_enable_switching)
        self.fusion_debug_checks = bool(fusion_debug_checks)
        self._debug_checked_once = False

        self.extra_cond_dim = int(extra_cond_dim)
        self.image_feat_dim = int(image_feat_dim)  # per-view output dim
        self.qpos_feat_dim = int(qpos_feat_dim)
        self.image_size = tuple(image_size)

        if self.d3p_train_branch not in ('A', 'B1', 'B2'):
            raise ValueError(f"d3p_train_branch must be one of ['A','B1','B2'], got {self.d3p_train_branch}")
        if self.d3p_b_branch not in ('B1', 'B2'):
            raise ValueError(f"d3p_b_branch must be one of ['B1','B2'], got {self.d3p_b_branch}")
        if self.d3p_query_every != 4:
            raise ValueError(f"d3p_query_every is fixed to 4 in current fusion stage, got {self.d3p_query_every}")

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

    def _select_train_branch(self) -> str:
        # Stage-0: fixed branch for deterministic debugging.
        return self.d3p_train_branch

    def _select_rollout_branch(self, common: Dict[str, Optional[torch.Tensor]]) -> str:
        # Stage-0: same as train branch, with safe fallback to A when required payload is missing.
        branch = self._select_train_branch()
        if branch == 'A':
            return 'A'
        if common['vis_enc_pair'] is None:
            return 'A'
        if (branch == 'B2') and (common['fea_fuse_pair'] is None):
            return 'A'
        return branch

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

        # Pair-style D3P placeholders.##################
        subgoal_pair = torch.stack((subgoal, subgoal), dim=1)  # (B, 2, subgoal_dim=8=6爪pos+2接触)

        image_pair = raw_batch['image'] if 'image' in raw_batch else None
        qpos_pair = raw_batch['qpos'] if 'qpos' in raw_batch else None  ## (B, 2时间, 9=7jiont+2爪宽)
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
            extra_cond_pair = self.branch_condition_encoder.build_b1_extra_cond_pair(
                common['vis_enc_pair'],
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

    # =========================
    # Actor core
    # =========================
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
        if self.eta != 0:
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

            actor_loss = bc_loss + self.eta*q_loss
        else:
            q_loss = torch.tensor(-1, device=self.device)
            actor_loss = bc_loss

        return actor_loss, bc_loss, q_loss

    def _predict_action_from_cond(
        self,
        cond: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        cond_run = cond
        if 'extra_cond_pair' in cond:
            cond_run = {
                'pcd': cond['pcd'],
                'state': cond['state'],
                'subgoal': cond['subgoal'],
                'extra_cond': cond['extra_cond_pair'][:, 0],
            }

        with torch.no_grad():
            action = self.conditional_sample_action(cond=cond_run, model=None)
        action = self.normalizer.unnormalize(naction=action)

        # get action
        start = self.observation_history_num - 1
        end = start + self.n_action_steps   # 1 + 8
        action_run = action[:, start:end]   # (B, 1:9, A)

        return {
            'action': action_run,
            'action_pred': action,
        }

    # =========================
    # Public actor API
    # =========================
    def compute_loss_actor(self, batch: Dict[str, torch.Tensor]):
        common = self._prepare_branch_inputs(batch)
        branch = self._select_train_branch()
        cond = self._build_cond_by_branch(branch, common)
        return self._compute_loss_actor_from_cond(branch=branch, cond=cond, common=common)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        common = self._prepare_branch_inputs(obs_dict)
        branch = self._select_rollout_branch(common)
        cond = self._build_cond_by_branch(branch, common)
        return self._predict_action_from_cond(cond=cond)

    # =========================
    # Inference aggregation hooks
    # =========================
    def compute_test_time_ddpm_error(self, *args, **kwargs):
        # Stage-1 placeholder: add DDPM error based branch selection here.
        raise NotImplementedError('Stage-0 skeleton: test-time DDPM error is not implemented yet.')
