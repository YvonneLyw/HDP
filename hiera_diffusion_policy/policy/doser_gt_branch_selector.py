from typing import Dict

import torch

from hiera_diffusion_policy.policy.doser_branch_selector import DoserBranchSelector


class GroundTruthDoserBranchSelector(DoserBranchSelector):
    """DOSER branch selector whose dynamics and support space are real state."""

    def load_components_from_checkpoint(self, path: str, map_location="cpu") -> Dict:
        import pathlib
        try:
            import dill
        except ImportError:
            import pickle as dill

        from hiera_diffusion_policy.model.diffusion.doser_selector_components import (
            FullStateActionDetector,
        )
        from hiera_diffusion_policy.model.diffusion.doser_gt_selector_components import (
            GroundTruthDynamicsModel,
            GroundTruthStateDetector,
            GroundTruthValueNet,
        )

        ckpt_path = pathlib.Path(path)
        payload = torch.load(
            ckpt_path.open("rb"),
            pickle_module=dill,
            map_location=map_location,
        )
        metadata = payload["metadata"]
        if metadata.get("selector_type", None) != "ground_truth_state":
            raise RuntimeError(
                "GroundTruthDoserBranchSelector requires a ground_truth_state checkpoint. "
                "Run pretrain_doser_gt_selector_components.py."
            )
        if int(metadata.get("format_version", 1)) < 2:
            raise RuntimeError(
                "This DOSER-GT checkpoint predicts only low-dimensional state. "
                "Re-run pretrain_doser_gt_selector_components.py to create the "
                "state+qpos successor checkpoint."
            )
        state_dicts = payload["state_dicts"]
        image_shape = metadata.get("image_shape", None)
        successor_dim = int(metadata["state_dim"]) + int(metadata["qpos_dim"])
        action_detector_mode = str(metadata.get("action_detector_mode", "shared")).lower()
        use_branch_action_detectors = action_detector_mode == "branch"

        if use_branch_action_detectors:
            action_detector = None
            action_detector_A = FullStateActionDetector(
                state_dim=metadata["state_dim"],
                subgoal_dim=metadata["subgoal_dim"],
                qpos_dim=0,
                action_eval_dim=metadata["action_eval_dim"],
                pcd_dim=metadata["pcd_dim"],
                image_shape=None,
                image_feat_dim=0,
                pcd_feat_dim=metadata["action_pcd_feat_dim"],
                hidden_dim=metadata["detector_hidden_dim"],
                time_embed_dim=metadata["time_embed_dim"],
                score_samples=metadata["score_samples"],
            )
            action_detector_B2 = FullStateActionDetector(
                state_dim=0,
                subgoal_dim=metadata["subgoal_dim"],
                qpos_dim=metadata["qpos_dim"],
                action_eval_dim=metadata["action_eval_dim"],
                pcd_dim=0,
                image_shape=image_shape,
                image_feat_dim=metadata["action_image_feat_dim"],
                pcd_feat_dim=0,
                hidden_dim=metadata["detector_hidden_dim"],
                time_embed_dim=metadata["time_embed_dim"],
                score_samples=metadata["score_samples"],
            )
        else:
            action_detector = FullStateActionDetector(
                state_dim=metadata["state_dim"],
                subgoal_dim=metadata["subgoal_dim"],
                qpos_dim=metadata["qpos_dim"],
                action_eval_dim=metadata["action_eval_dim"],
                pcd_dim=metadata["pcd_dim"],
                image_shape=image_shape,
                image_feat_dim=metadata["action_image_feat_dim"],
                pcd_feat_dim=metadata["action_pcd_feat_dim"],
                hidden_dim=metadata["detector_hidden_dim"],
                time_embed_dim=metadata["time_embed_dim"],
                score_samples=metadata["score_samples"],
            )
            action_detector_A = None
            action_detector_B2 = None
        dynamics_model = GroundTruthDynamicsModel(
            state_dim=metadata["state_dim"],
            subgoal_dim=metadata["subgoal_dim"],
            qpos_dim=metadata["qpos_dim"],
            pcd_dim=metadata["pcd_dim"],
            action_eval_dim=metadata["action_eval_dim"],
            hidden_dim=metadata["dynamics_hidden_dim"],
            pcd_feat_dim=metadata["dynamics_pcd_feat_dim"],
            image_shape=image_shape,
            image_feat_dim=metadata["dynamics_image_feat_dim"],
            predict_delta=metadata["predict_delta"],
            split_heads=metadata.get("dynamics_split_heads", False),
            observation_history_num=metadata.get("observation_history_num", 1),
        )
        state_detector = GroundTruthStateDetector(
            latent_dim=successor_dim,
            hidden_dim=metadata["detector_hidden_dim"],
            time_embed_dim=metadata["time_embed_dim"],
            score_samples=metadata["score_samples"],
        )
        value_net = GroundTruthValueNet(
            latent_dim=successor_dim,
            subgoal_dim=metadata["value_subgoal_dim"],
            hidden_dim=metadata["value_hidden_dim"],
        )

        def load_detector_state(module, state_dict):
            reference_errors = state_dict.get("reference_errors", None)
            clean_state = {
                key: value
                for key, value in state_dict.items()
                if key != "reference_errors"
            }
            module.load_state_dict(clean_state, strict=False)
            if reference_errors is not None:
                module.set_reference_errors(reference_errors)

        if use_branch_action_detectors:
            load_detector_state(action_detector_A, state_dicts["action_detector_A"])
            load_detector_state(action_detector_B2, state_dicts["action_detector_B2"])
        else:
            load_detector_state(action_detector, state_dicts["action_detector"])
        dynamics_model.load_state_dict(state_dicts["dynamics_model"])
        load_detector_state(state_detector, state_dicts["state_detector"])
        value_net.load_state_dict(state_dicts["value_net"])

        self.action_detector = action_detector
        self.action_detector_A = action_detector_A
        self.action_detector_B2 = action_detector_B2
        self.use_branch_action_detectors = use_branch_action_detectors
        self.dynamics_model = dynamics_model
        self.state_detector = state_detector
        self.value_net = value_net
        modules = (
            self.action_detector,
            self.action_detector_A,
            self.action_detector_B2,
            self.dynamics_model,
            self.state_detector,
            self.value_net,
        )
        for module in modules:
            if module is None:
                continue
            module.eval()
            module.requires_grad_(False)
        return payload

    @torch.no_grad()
    def _score_action_candidates(self, common, aA_eval, aB_eval):
        if not getattr(self, "use_branch_action_detectors", False):
            return super()._score_action_candidates(
                common,
                aA_eval,
                aB_eval,
            )
        action_detector_A = self._require(
            self.action_detector_A,
            "branch-A action detector",
        )
        action_detector_B2 = self._require(
            self.action_detector_B2,
            "branch-B2 action detector",
        )
        score_A = self._score_detector(
            action_detector_A,
            self.action_ood_percentile,
            common,
            aA_eval,
        )
        score_B = self._score_detector(
            action_detector_B2,
            self.action_ood_percentile,
            common,
            aB_eval,
        )
        return score_A, score_B

    @torch.no_grad()
    def _predict_latent_transition(self, common, action_eval_norm):
        dynamics = self._require(
            self.dynamics_model,
            "ground-truth state dynamics model",
        )
        raw = dynamics(common, action_eval_norm)
        successor = raw["successor"]
        uncertainty = raw.get("uncertainty", None)
        if uncertainty is None:
            uncertainty = torch.zeros(
                successor.shape[0],
                device=successor.device,
                dtype=successor.dtype,
            )
        return successor, uncertainty.reshape(-1)
