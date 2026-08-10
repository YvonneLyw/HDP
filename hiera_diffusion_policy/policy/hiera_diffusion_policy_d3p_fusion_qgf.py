"""Isolated B2 test-time Q guidance for D3P Fusion.

The base ``HieraDiffusionPolicyD3PFusion`` is intentionally untouched.  This
subclass preserves its actor training losses (A: BC+Q; B2: BC when configured)
and changes only B2's inference-time DDPM sampling.
"""

from typing import Dict, Optional

import torch

from hiera_diffusion_policy.policy.hiera_diffusion_policy_d3p_fusion import (
    HieraDiffusionPolicyD3PFusion,
)


class HieraDiffusionPolicyD3PFusionQGF(HieraDiffusionPolicyD3PFusion):
    """Fusion A/B2 policy with optional clean/noisy Q guidance on B2 only."""

    _GUIDANCE_MODES = {"none", "noisy", "clean"}
    _GUIDANCE_SCALES = {"posterior_variance", "constant"}

    def __init__(
        self,
        b2_q_guidance_mode: str = "none",
        b2_q_guidance_weight: float = 1.0,
        b2_q_guidance_scale: str = "posterior_variance",
        b2_q_guidance_grad_clip_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.b2_q_guidance_mode = str(b2_q_guidance_mode).lower()
        self.b2_q_guidance_weight = float(b2_q_guidance_weight)
        self.b2_q_guidance_scale = str(b2_q_guidance_scale).lower()
        self.b2_q_guidance_grad_clip_norm = (
            None
            if b2_q_guidance_grad_clip_norm is None
            else float(b2_q_guidance_grad_clip_norm)
        )

        if self.b2_q_guidance_mode not in self._GUIDANCE_MODES:
            raise ValueError(
                "b2_q_guidance_mode must be one of "
                f"{sorted(self._GUIDANCE_MODES)}, got {b2_q_guidance_mode!r}"
            )
        if self.b2_q_guidance_scale not in self._GUIDANCE_SCALES:
            raise ValueError(
                "b2_q_guidance_scale must be one of "
                f"{sorted(self._GUIDANCE_SCALES)}, got {b2_q_guidance_scale!r}"
            )
        if self.b2_q_guidance_weight < 0.0:
            raise ValueError("b2_q_guidance_weight must be non-negative")
        if (
            self.b2_q_guidance_grad_clip_norm is not None
            and self.b2_q_guidance_grad_clip_norm <= 0.0
        ):
            raise ValueError("b2_q_guidance_grad_clip_norm must be positive or null")

        # Keep the YAML weight at 1.0, while `none` remains an exact old-Fusion
        # control.  Its predict_action delegates directly to the base class.
        if self.b2_q_guidance_mode == "none":
            self.b2_q_guidance_weight = 0.0
        else:
            if self.mode != "SWITCH" or self.b_branch != "B2":
                raise ValueError(
                    "B2 QGF requires policy.mode=SWITCH and policy.b_branch=B2."
                )
            # Autograd below must differentiate only with respect to action.
            self.critic_target.requires_grad_(False)
            self.critic_target.eval()

    def _critic_action_slice_b2(self, action: torch.Tensor) -> torch.Tensor:
        """Return B2's executable Tr-step chunk: indices [0, Tr)."""
        if self.Tr > action.shape[1]:
            raise RuntimeError(
                "B2 critic action chunk exceeds sampled horizon: "
                f"Tr={self.Tr}, horizon={action.shape[1]}"
            )
        return action[:, :self.Tr].reshape(action.shape[0], -1)

    def _b2_q_gradient(
        self,
        critic_common: Dict[str, Optional[torch.Tensor]],
        action_candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute d min(Q1,Q2) / d(full B2 action trajectory).

        B2's actor receives zero pcd/state/subgoal plus ``extra_cond``.  The
        critic must instead receive the real normalized current pcd/state/
        subgoal stored in ``critic_common``.
        """
        with torch.enable_grad():
            action_input = action_candidate.detach().requires_grad_(True)
            q1, q2 = self.critic_target(
                critic_common["pcd"],
                critic_common["state"],
                critic_common["subgoal"],
                self._critic_action_slice_b2(action_input),
            )
            q = torch.minimum(q1, q2)
            gradient = torch.autograd.grad(
                outputs=q.sum(),
                inputs=action_input,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
        return gradient.detach()

    def _clip_b2_q_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.b2_q_guidance_grad_clip_norm is None:
            return gradient
        flat = gradient.reshape(gradient.shape[0], -1)
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
        scale = torch.clamp(
            self.b2_q_guidance_grad_clip_norm / norm.clamp_min(1e-12),
            max=1.0,
        )
        return gradient * scale.reshape(
            gradient.shape[0], *([1] * (gradient.ndim - 1))
        )

    def _b2_q_guidance_step_scale(
        self,
        timestep,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if self.b2_q_guidance_scale == "constant":
            return torch.ones((), device=action.device, dtype=action.dtype)
        timestep_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        variance = self.noise_scheduler_actor._get_variance(timestep_int)
        if not torch.is_tensor(variance):
            variance = torch.as_tensor(variance)
        return variance.to(device=action.device, dtype=action.dtype)

    def conditional_sample_action(
        self,
        cond: Dict[str, Optional[torch.Tensor]],
        action_init: Optional[torch.Tensor] = None,
        model=None,
        b2_q_critic_common: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Base Fusion sampler, with an optional QGF DDPM transition for B2.

        Training calls have ``model=self.actor`` and omit
        ``b2_q_critic_common``; hence they remain exactly ordinary Fusion
        actor training.
        """
        if (
            model is not None
            or b2_q_critic_common is None
            or self.b2_q_guidance_mode == "none"
            or self.b2_q_guidance_weight <= 0.0
        ):
            return super().conditional_sample_action(
                cond=cond,
                action_init=action_init,
                model=model,
            )

        B = cond["state"].shape[0]
        if action_init is None:
            action = torch.randn(
                size=(B, self.horizon, self.action_dim),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            action = action_init

        for t in self.noise_scheduler_actor.timesteps:
            pred_noise = self.actor_target(
                cond["pcd"],
                cond["state"],
                cond["subgoal"],
                action,
                t,
                extra_cond=cond["extra_cond"],
            )

            step_output = self.noise_scheduler_actor.step(
                pred_noise, t, action, generator=None
            )
            next_action = step_output.prev_sample
            action_for_q = (
                step_output.pred_original_sample
                if self.b2_q_guidance_mode == "clean"
                else action
            )
            gradient = self._clip_b2_q_gradient(
                self._b2_q_gradient(b2_q_critic_common, action_for_q)
            )
            next_action = next_action + (
                self.b2_q_guidance_weight
                * self._b2_q_guidance_step_scale(t, next_action)
                * gradient
            )
            action = next_action
        return action

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The no-guidance control delegates to the original Fusion policy.
        if self.b2_q_guidance_mode == "none" or self.b2_q_guidance_weight <= 0.0:
            return super().predict_action(obs_dict)

        common = self._prepare_branch_inputs(obs_dict)
        # If B2 side inputs are unavailable, the base policy falls back to A.
        # Do the same instead of attempting a QGF B2 action with invalid input.
        if self._resolve_branch(self.b_branch, common) != "B2":
            return super().predict_action(obs_dict)

        cond_A = self._build_cond_by_branch("A", common)
        cond_A_run = self._rollout_cond_from_branch_cond(cond_A)
        with torch.no_grad():
            action_A_norm = self.conditional_sample_action(
                cond=cond_A_run,
                model=None,
            )

        cond_B = self._build_cond_by_branch("B2", common)
        cond_B_run = self._rollout_cond_from_branch_cond(cond_B)
        with torch.no_grad():
            action_B_norm = self.conditional_sample_action(
                cond=cond_B_run,
                model=None,
                b2_q_critic_common=common,
            )

        start_A = self._get_action_start("A")
        start_B = self._get_action_start("B2")
        aligned_A_norm = self._extract_action_segment(action_A_norm, start_A, self.horizon)
        aligned_B_norm = self._extract_action_segment(action_B_norm, start_B, self.horizon)

        if self.branch_selector in ("doser_latent", "doser_gt", "doser_err"):
            if self.branch_selector == "doser_err":
                selector_out = self.doser_gt_selector.select_by_action_percentile(
                    common=common,
                    aligned_A_norm=aligned_A_norm,
                    aligned_B_norm=aligned_B_norm,
                    Tr=self.Tr,
                    require_branch_detectors=True,
                )
            else:
                active_selector = (
                    self.doser_selector
                    if self.branch_selector == "doser_latent"
                    else self.doser_gt_selector
                )
                selector_out = active_selector.select(
                    common=common,
                    aligned_A_norm=aligned_A_norm,
                    aligned_B_norm=aligned_B_norm,
                    critic_target=self.critic_target,
                    Tr=self.Tr,
                )
        else:
            err_A = self.compute_ddpm_error(cond=cond_A_run, action_norm=action_A_norm)
            err_B = self.compute_ddpm_error(cond=cond_B_run, action_norm=action_B_norm)
            selector_out = self._select_branch_in_switch(
                common=common,
                err_A=err_A,
                err_B=err_B,
                aligned_A_norm=aligned_A_norm,
                aligned_B_norm=aligned_B_norm,
            )

        if self.use_test_time_aggregation:
            aggregated_norm, chosen_branch = self._aggregate_rollout_candidates(
                candidate_actions_norm=torch.stack(
                    (aligned_A_norm, aligned_B_norm), dim=1
                ),
                score_A=selector_out["select_score_A"],
                score_B=selector_out["select_score_B"],
            )
            out = {
                "action": self.normalizer.unnormalize(
                    naction=aggregated_norm[:, :self.n_action_steps]
                ),
                "selected_branch_exec_ratio": chosen_branch[:, :self.n_action_steps]
                .float()
                .mean(dim=1, keepdim=True),
            }
        else:
            select_B = selector_out["select_B"]
            action_sel_norm = torch.where(select_B, action_B_norm, action_A_norm)
            action_sel_aligned_norm = torch.where(
                select_B, aligned_B_norm, aligned_A_norm
            )
            if self.use_action_smoothing:
                action_sel_aligned_norm = self._smooth_selected_action_sequence(
                    action_sel_aligned_norm,
                )
            out = {
                "action_pred": self.normalizer.unnormalize(naction=action_sel_norm),
                "action": self.normalizer.unnormalize(
                    naction=action_sel_aligned_norm[:, :self.n_action_steps]
                ),
                "selected_branch": select_B[:, 0, 0]
                .to(dtype=action_sel_norm.dtype)
                .unsqueeze(-1),
            }

        out["branch_score_A"] = selector_out["select_score_A"].unsqueeze(-1)
        out["branch_score_B"] = selector_out["select_score_B"].unsqueeze(-1)
        if "select_source" in selector_out:
            out["branch_select_source"] = selector_out["select_source"].unsqueeze(-1)
        if self.branch_selector in ("doser_latent", "doser_gt", "doser_err"):
            for key in (
                "action_percentile_A",
                "action_percentile_B",
                "action_id_A",
                "action_id_B",
                "state_percentile_A",
                "state_percentile_B",
                "q_A",
                "q_B",
                "v_A",
                "v_B",
            ):
                if key in selector_out:
                    value = selector_out[key]
                    out[f"doser_{key}"] = (
                        value.unsqueeze(-1) if value.ndim == 1 else value
                    )
        return out
