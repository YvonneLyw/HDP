from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from hiera_diffusion_policy.model.diffusion.doser_selector_components import LatentValueNet


class PercentileCalibrator(nn.Module):
    """
    Converts raw reconstruction errors to empirical percentile ranks.

    reference_errors should be computed on the offline training set after the
    frozen detector has been trained. The returned percentile is in [0, 1].
    """
    def __init__(self, reference_errors: Optional[torch.Tensor] = None):
        super().__init__()
        if reference_errors is None:
            self.register_buffer("reference_errors", torch.empty(0))
        else:
            ## 保存在离线训练集上计算出的 detector error [e1, e2, ..., eN]
            ref = torch.as_tensor(reference_errors, dtype=torch.float32).reshape(-1)        ## 转换为一维并排序
            self.register_buffer("reference_errors", torch.sort(ref).values)

    def has_reference(self) -> bool:
        return self.reference_errors.numel() > 0

    @torch.no_grad()
    def percentile(self, errors: torch.Tensor) -> torch.Tensor:
        if not self.has_reference():
            raise RuntimeError(
                "PercentileCalibrator has no reference errors. "
                "Compute detector errors on offline data and load them first."
            )
        ref = self.reference_errors.to(device=errors.device, dtype=errors.dtype)
        flat = errors.reshape(-1)   ## 当前当前 error
        ranks = torch.searchsorted(ref, flat, right=True).to(dtype=errors.dtype) ## 当前 error 在数组中的排序
        percentile = ranks / float(ref.numel())     #### 当前 error 在数组中的percentile
        return percentile.reshape_as(errors)


class DoserBranchSelector(nn.Module):
    """
    Wrapper for DOSER-style A/B branch selection.

    This class does not train detectors. It only consumes frozen components:
      - shared full-state action support detector
      - latent dynamics
      - latent state support detector
      - standalone latent value net

    Required detector output convention:
      action_detector.score(common, action) -> {"error": ..., "percentile": ...}
      detector.score(latent) -> {"error": ..., "percentile": ...}
    If a custom detector returns only raw errors, attach a PercentileCalibrator
    through set_components(); tuple/raw tensor fallback is intentionally not
    supported.
    """
    def __init__(
        self,
        action_ood_percentile: float = 0.95,
        state_ood_percentile: float = 0.95,
        q_margin: float = 0.0,
        v_margin: float = 0.0,
        fallback_when_both_ood: str = "A",
        require_dynamics_uncertainty: bool = False,
        dynamics_uncertainty_threshold: float = float("inf"),
        components_path: Optional[str] = None,
    ):
        super().__init__()
        self.action_ood_percentile = float(action_ood_percentile)
        self.state_ood_percentile = float(state_ood_percentile)
        self.q_margin = float(q_margin)
        self.v_margin = float(v_margin)
        self.fallback_when_both_ood = str(fallback_when_both_ood).upper()
        self.require_dynamics_uncertainty = bool(require_dynamics_uncertainty)
        self.dynamics_uncertainty_threshold = float(dynamics_uncertainty_threshold)
        self.components_path = components_path

        if self.fallback_when_both_ood not in ("A", "B"):
            raise ValueError("fallback_when_both_ood must be 'A' or 'B'.")

        self.action_detector = None
        self.state_detector = None
        self.dynamics_model = None
        self.value_net = None

        self.action_pct_calibrator = PercentileCalibrator()
        self.state_pct_calibrator = PercentileCalibrator()

        if self.components_path not in (None, ""):
            self.load_components_from_checkpoint(self.components_path, map_location="cpu")

    ## 注入预训练好的 action detector / dynamics / state detector / V net
    def set_components(
        self,
        action_detector=None,
        state_detector=None,
        dynamics_model=None,
        value_net=None,
        action_pct_calibrator: Optional[PercentileCalibrator] = None,
        state_pct_calibrator: Optional[PercentileCalibrator] = None,
    ) -> None:
        self.action_detector = action_detector
        self.state_detector = state_detector
        self.dynamics_model = dynamics_model
        self.value_net = value_net
        if action_pct_calibrator is not None:
            self.action_pct_calibrator = action_pct_calibrator
        if state_pct_calibrator is not None:
            self.state_pct_calibrator = state_pct_calibrator

        for module in (
            self.action_detector,
            self.state_detector,
            self.dynamics_model,
            self.value_net,
        ):
            if isinstance(module, nn.Module):
                module.eval()
                module.requires_grad_(False)

    def load_components_from_checkpoint(self, path: str, map_location="cpu") -> Dict:
        """
        Load the frozen selector components produced by
        pretrain_doser_selector_components.py.

        This is called before rollout/training inference, not during rollout.
        """
        import pathlib
        try:
            import dill
        except ImportError:
            import pickle as dill
        from hiera_diffusion_policy.model.diffusion.doser_selector_components import (
            FullStateActionDetector,
            LatentDynamicsModel,
            LatentStateDetector,
        )

        ckpt_path = pathlib.Path(path)
        payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location=map_location)
        metadata = payload["metadata"]
        state_dicts = payload["state_dicts"]
        image_shape = metadata.get("image_shape", None)
        if image_shape is None:
            raise RuntimeError("DOSER selector checkpoint metadata must contain image_shape.")

        action_detector = FullStateActionDetector(
            state_dim=metadata["state_dim"],
            subgoal_dim=metadata["subgoal_dim"],
            qpos_dim=metadata.get("qpos_dim", 0),
            action_eval_dim=metadata["action_eval_dim"],
            image_shape=image_shape,
            image_feat_dim=metadata.get("image_feat_dim", 64),
            hidden_dim=metadata.get("detector_hidden_dim", 256),
            time_embed_dim=metadata.get("time_embed_dim", 32),
            score_samples=metadata.get("score_samples", 4),
        )
        dynamics_model = LatentDynamicsModel(
            state_dim=metadata["state_dim"],
            subgoal_dim=metadata["subgoal_dim"],
            action_eval_dim=metadata["action_eval_dim"],
            latent_dim=metadata["latent_dim"],
            hidden_dim=metadata.get("dynamics_hidden_dim", 256),
            image_shape=image_shape,
            image_feat_dim=metadata.get("dynamics_image_feat_dim", 64),
        )
        state_detector = LatentStateDetector(
            latent_dim=metadata["latent_dim"],
            hidden_dim=metadata.get("detector_hidden_dim", 256),
            time_embed_dim=metadata.get("time_embed_dim", 32),
            score_samples=metadata.get("score_samples", 4),
        )
        value_net = LatentValueNet(
            latent_dim=metadata["latent_dim"],
            subgoal_dim=metadata.get("value_subgoal_dim", 0),
            hidden_dim=metadata.get("value_hidden_dim", 256),
        )

        def load_detector_state(module, state_dict):
            reference_errors = state_dict.get("reference_errors", None)
            clean_state = {k: v for k, v in state_dict.items() if k != "reference_errors"}
            module.load_state_dict(clean_state, strict=False)
            if reference_errors is not None and hasattr(module, "set_reference_errors"):
                module.set_reference_errors(reference_errors)

        load_detector_state(action_detector, state_dicts["action_detector"])
        dynamics_model.load_state_dict(state_dicts["dynamics_model"])
        load_detector_state(state_detector, state_dicts["state_detector"])
        value_net.load_state_dict(state_dicts["value_net"])

        self.set_components(
            action_detector=action_detector,
            state_detector=state_detector,
            dynamics_model=dynamics_model,
            value_net=value_net,
        )
        return payload

    ## 注入组件检查
    def _require(self, module, name: str):
        if module is None:
            raise RuntimeError(
                f"DOSER branch selector requires frozen {name}. "
                "Train/load it first and call set_components()."
            )
        return module

    @torch.no_grad()
    def _score_detector(
        self,
        detector,
        calibrator: PercentileCalibrator,
        threshold: float,
        *args,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(detector, "score"):
            raise RuntimeError(
                "DOSER detectors must implement score(...)->dict with "
                "'error' and calibrated 'percentile'."
            )
        raw = detector.score(*args)
        if not isinstance(raw, dict):
            raise RuntimeError("DOSER detector score(...) must return a dict.")
        error = raw.get("error", raw.get("reconstruction_error", None))
        percentile = raw.get("percentile", raw.get("percentile_rank", None))
        is_id = raw.get("is_id", None)
        if error is None:
            raise RuntimeError("Detector score dict must contain 'error' or 'reconstruction_error'.")

        error = error.reshape(-1)
        if percentile is None:
            percentile = calibrator.percentile(error)
        else:
            percentile = percentile.reshape(-1).to(device=error.device, dtype=error.dtype)

        if is_id is None:
            is_id = percentile <= threshold         ## id/ood判断
        else:
            is_id = is_id.reshape(-1).to(device=error.device, dtype=torch.bool)
        return error, percentile, is_id

    @torch.no_grad()
    def _compute_q_score(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        action_eval_norm: torch.Tensor,
        critic_target,
    ) -> torch.Tensor:
        B = action_eval_norm.shape[0]
        q_action = action_eval_norm.reshape((B, -1))
        q1, q2 = critic_target(common["pcd"], common["state"], common["subgoal"], q_action)
        return torch.minimum(q1.squeeze(-1), q2.squeeze(-1))

    @torch.no_grad()
    def _predict_latent_transition(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        action_eval_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamics = self._require(self.dynamics_model, "latent dynamics model")
        if hasattr(dynamics, "predict"):
            raw = dynamics.predict(common, action_eval_norm)
        else:
            raw = dynamics(common, action_eval_norm)

        if isinstance(raw, dict):
            latent = raw.get("latent", raw.get("next_latent", None))
            uncertainty = raw.get("uncertainty", None)
            if latent is None:
                raise RuntimeError("Dynamics output dict must contain 'latent' or 'next_latent'.")
        elif isinstance(raw, (tuple, list)):
            latent = raw[0]
            uncertainty = raw[1] if len(raw) > 1 else None
        else:
            latent = raw
            uncertainty = None

        if uncertainty is None:
            uncertainty = torch.zeros(latent.shape[0], device=latent.device, dtype=latent.dtype)
        else:
            uncertainty = uncertainty.reshape(-1).to(device=latent.device, dtype=latent.dtype)
        return latent, uncertainty

    @torch.no_grad()
    def _compute_value(
        self,
        latent: torch.Tensor,
        common: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        value_net = self._require(self.value_net, "latent value net")
        if hasattr(value_net, "score"):
            value = value_net.score(latent, common)
        else:
            try:
                value = value_net(latent, common["subgoal"])
            except TypeError:
                value = value_net(latent)
        return value.reshape(-1)

    @torch.no_grad()
    def select(
        self,
        common: Dict[str, Optional[torch.Tensor]],
        aligned_A_norm: torch.Tensor,
        aligned_B_norm: torch.Tensor,
        critic_target,
        Tr: int,
    ) -> Dict[str, torch.Tensor]:
        action_detector = self._require(self.action_detector, "full-state action detector")
        state_detector = self._require(self.state_detector, "latent state support detector")

        aA_eval = aligned_A_norm[:, :int(Tr)]
        aB_eval = aligned_B_norm[:, :int(Tr)]

        ## 第一步：判断动作是否 ID
        error_A, p_A, id_A = self._score_detector(
            action_detector,
            self.action_pct_calibrator,
            self.action_ood_percentile,
            common,
            aA_eval,
        )
        error_B, p_B, id_B = self._score_detector(
            action_detector,
            self.action_pct_calibrator,
            self.action_ood_percentile,
            common,
            aB_eval,
        )

        q_A = self._compute_q_score(common, aA_eval, critic_target)
        q_B = self._compute_q_score(common, aB_eval, critic_target)

        zA_next, dyn_unc_A = self._predict_latent_transition(common, aA_eval)
        zB_next, dyn_unc_B = self._predict_latent_transition(common, aB_eval)

        state_error_A, state_p_A, state_id_A = self._score_detector(
            state_detector,
            self.state_pct_calibrator,
            self.state_ood_percentile,
            zA_next,
        )
        state_error_B, state_p_B, state_id_B = self._score_detector(
            state_detector,
            self.state_pct_calibrator,
            self.state_ood_percentile,
            zB_next,
        )

        dyn_ok_A = dyn_unc_A <= self.dynamics_uncertainty_threshold
        dyn_ok_B = dyn_unc_B <= self.dynamics_uncertainty_threshold
        if not self.require_dynamics_uncertainty:
            dyn_ok_A = torch.ones_like(state_id_A, dtype=torch.bool)
            dyn_ok_B = torch.ones_like(state_id_B, dtype=torch.bool)

        v_A = self._compute_value(zA_next, common)
        v_B = self._compute_value(zB_next, common)

        both_action_id = id_A & id_B
        a_id_b_ood = id_A & (~id_B)
        a_ood_b_id = (~id_A) & id_B
        both_action_ood = (~id_A) & (~id_B)

        select_B_both_action_id = q_B > (q_A + self.q_margin)
        select_B_a_id_b_ood = state_id_B & dyn_ok_B & (v_B > (v_A + self.v_margin))
        select_B_a_ood_b_id = ~(state_id_A & dyn_ok_A & (v_A > (v_B + self.v_margin)))

        one_state_id = state_id_A ^ state_id_B
        select_B_one_state_id = state_id_B & dyn_ok_B
        select_B_both_state_id = v_B > (v_A + self.v_margin)
        fallback_B = torch.ones_like(id_A, dtype=torch.bool) if self.fallback_when_both_ood == "B" \
            else torch.zeros_like(id_A, dtype=torch.bool)
        select_B_both_action_ood = torch.where(
            one_state_id,
            select_B_one_state_id,
            torch.where(
                state_id_A & state_id_B,
                select_B_both_state_id,
                fallback_B,
            ),
        )

        select_B = torch.where(
            both_action_id,
            select_B_both_action_id,
            torch.where(
                a_id_b_ood,
                select_B_a_id_b_ood,
                torch.where(
                    a_ood_b_id,
                    select_B_a_ood_b_id,
                    select_B_both_action_ood,
                ),
            ),
        )

        score_A = torch.where(id_A, q_A, v_A)
        score_B = torch.where(id_B, q_B, v_B)

        select_source = torch.full_like(p_A, 3, dtype=torch.long)
        select_source = torch.where(both_action_id, torch.zeros_like(select_source), select_source)
        select_source = torch.where(a_id_b_ood | a_ood_b_id, torch.ones_like(select_source), select_source)
        select_source = torch.where(both_action_ood, torch.full_like(select_source, 2), select_source)

        return {
            "select_B": select_B.view(-1, 1, 1),
            "select_score_A": score_A,
            "select_score_B": score_B,
            "select_source": select_source,
            "action_error_A": error_A,
            "action_error_B": error_B,
            "action_percentile_A": p_A,
            "action_percentile_B": p_B,
            "action_id_A": id_A,
            "action_id_B": id_B,
            "state_error_A": state_error_A,
            "state_error_B": state_error_B,
            "state_percentile_A": state_p_A,
            "state_percentile_B": state_p_B,
            "state_id_A": state_id_A,
            "state_id_B": state_id_B,
            "dynamics_uncertainty_A": dyn_unc_A,
            "dynamics_uncertainty_B": dyn_unc_B,
            "q_A": q_A,
            "q_B": q_B,
            "v_A": v_A,
            "v_B": v_B,
        }
