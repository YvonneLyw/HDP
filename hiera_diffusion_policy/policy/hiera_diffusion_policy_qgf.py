"""Test-time Q-gradient guidance for the original hierarchical diffusion policy.

The class deliberately keeps HDP's training losses unchanged.  It only augments
action sampling with three inference modes:

* ``none``: delegate exactly to the original sampler (BC control).
* ``noisy``: query Q on the current DDPM latent action ``x_t`` (QFQL-style
  OOD-gradient ablation).
* ``clean``: query Q on the scheduler's predicted clean action ``x_hat_0``
  and use the QGF identity-Jacobian approximation.

Both guidance modes use the same DDPM posterior-mean shift; their only
algorithmic difference is the action that the critic receives.
"""

from typing import Optional

import torch

from hiera_diffusion_policy.model.diffusion.actor import Actor
from hiera_diffusion_policy.policy.hiera_diffusion_policy import HieraDiffusionPolicy


class HieraDiffusionPolicyQGF(HieraDiffusionPolicy):
    """HDP with optional test-time Q-gradient guidance during DDPM sampling."""

    _GUIDANCE_MODES = {"none", "noisy", "clean"}
    _GUIDANCE_SCALES = {"posterior_variance", "constant"}

    def __init__(
        self,
        q_guidance_mode: str = "none",
        q_guidance_weight: float = 1.0,
        q_guidance_scale: str = "posterior_variance",
        q_guidance_grad_clip_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.q_guidance_mode = str(q_guidance_mode).lower()
        self.q_guidance_weight = float(q_guidance_weight)
        self.q_guidance_scale = str(q_guidance_scale).lower()
        self.q_guidance_grad_clip_norm = (
            None
            if q_guidance_grad_clip_norm is None
            else float(q_guidance_grad_clip_norm)
        )

        if self.q_guidance_mode not in self._GUIDANCE_MODES:
            raise ValueError(
                f"q_guidance_mode must be one of {sorted(self._GUIDANCE_MODES)}, "
                f"got {q_guidance_mode!r}"
            )
        if self.q_guidance_scale not in self._GUIDANCE_SCALES:
            raise ValueError(
                f"q_guidance_scale must be one of {sorted(self._GUIDANCE_SCALES)}, "
                f"got {q_guidance_scale!r}"
            )
        if self.q_guidance_weight < 0.0:
            raise ValueError("q_guidance_weight must be non-negative")
        if (
            self.q_guidance_grad_clip_norm is not None
            and self.q_guidance_grad_clip_norm <= 0.0
        ):
            raise ValueError("q_guidance_grad_clip_norm must be positive or null")

        # `none` is the strict Actor_BC control: never retain a non-zero
        # coefficient merely because the shared YAML default is 1.0.
        if self.q_guidance_mode == "none":
            self.q_guidance_weight = 0.0

        # Q is an inference-time scorer.  Its parameters must remain frozen while
        # autograd computes dQ / d(action input).
        self.critic_target.requires_grad_(False)
        self.critic_target.eval()

    @property
    def q_guidance_enabled(self) -> bool:
        return self.q_guidance_mode != "none" and self.q_guidance_weight > 0.0

    # AC截取Tr步(a_t:t+Tr-1)
    def _critic_action_slice(self, action: torch.Tensor) -> torch.Tensor:
        """Return the normalized Tr-step action chunk expected by HDP's critic."""
        start = self.observation_history_num - 1
        end = start + self.Tr
        if end > action.shape[1]:
            raise RuntimeError(
                "Critic action chunk exceeds sampled horizon: "
                f"start={start}, Tr={self.Tr}, horizon={action.shape[1]}"
            )
        return action[:, start:end].reshape(action.shape[0], -1)

    # Q guidance
    def _q_gradient(
        self,
        pcd: Optional[torch.Tensor],
        state: torch.Tensor,
        subgoal: Optional[torch.Tensor],
        action_candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute d min(Q1,Q2) / d(full normalized action trajectory).

        ``autograd.grad`` returns the required input gradient directly and does
        not accumulate gradients into frozen critic parameters.  Gradients outside
        the critic's Tr-step chunk are zero by construction.
        """
        with torch.enable_grad():
            action_input = action_candidate.detach().requires_grad_(True)   # 让A可导
            critic_action = self._critic_action_slice(action_input)
            q1, q2 = self.critic_target(pcd, state, subgoal, critic_action)###################
            q = torch.minimum(q1, q2)
            # 求梯度
            gradient = torch.autograd.grad(
                outputs=q.sum(),
                inputs=action_input,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
        return gradient.detach()

    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Optionally clip the full-trajectory L2 norm independently per sample."""
        if self.q_guidance_grad_clip_norm is None:
            return gradient

        flat = gradient.reshape(gradient.shape[0], -1)
        norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
        scale = torch.clamp(
            self.q_guidance_grad_clip_norm / norm.clamp_min(1e-12),
            max=1.0,
        )
        return gradient * scale.reshape(
            gradient.shape[0], *([1] * (gradient.ndim - 1))
        )

    def _guidance_step_scale(self, timestep, action: torch.Tensor) -> torch.Tensor:
        """Return the common DDPM scale used by both noisy and clean guidance.

        A classifier-guided DDPM transition shifts its reverse mean by posterior
        variance times the guidance score.  ``constant`` is provided only for a
        controlled ablation; the default is the posterior variance.
        """
        if self.q_guidance_scale == "constant":
            return torch.ones((), device=action.device, dtype=action.dtype)

        scheduler = self.noise_scheduler_actor
        timestep_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        variance = scheduler._get_variance(timestep_int)
        if not torch.is_tensor(variance):
            variance = torch.as_tensor(variance)
        return variance.to(device=action.device, dtype=action.dtype)

    def conditional_sample_action(
        self,
        pcd,
        state,
        subgoal=None,
        action_init=None,
        model: Actor = None,
    ):
        """Sample an action sequence and optionally guide every reverse DDPM step.

        Guidance is inference-only.  When ``model`` is supplied by a training
        path, or mode is ``none``, the original HDP sampler is used unchanged.
        """
        if model is not None or not self.q_guidance_enabled:
            return super().conditional_sample_action(
                pcd=pcd,
                state=state,
                subgoal=subgoal,
                action_init=action_init,
                model=model,
            )

        batch_size = state.shape[0]
        action = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )

        for timestep in self.noise_scheduler_actor.timesteps:
            # The actor and DDPM base transition stay outside the autograd graph.
            with torch.no_grad():
                action_noise = self.actor_target(pcd, state, subgoal, action, timestep)
                step_output = self.noise_scheduler_actor.step(
                    action_noise, timestep, action, generator=None
                )
                base_prev_action = step_output.prev_sample          # x_t-1
                if self.q_guidance_mode == "clean":
                    action_for_q = step_output.pred_original_sample # x0_prev
                else:  # noisy: query Q directly on x_t (the OOD-gradient ablation)
                    action_for_q = action                           # x_t

            # clean mode uses dQ/d(x_hat_0) as an approximation to dQ/d(x_t),
            # i.e. d(x_hat_0)/d(x_t) is replaced by the identity.
            q_gradient = self._q_gradient(pcd, state, subgoal, action_for_q)
            q_gradient = self._clip_gradient(q_gradient)
            scale = self._guidance_step_scale(timestep, action)
            action = base_prev_action + self.q_guidance_weight * scale * q_gradient

        return action
