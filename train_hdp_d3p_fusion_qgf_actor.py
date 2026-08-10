"""Train the isolated Fusion A/B2 actor used by the QGF evaluation.

The resulting checkpoint is shared by the no-QGF and B2-QGF evaluations:
A is trained with the original actor-Q loss and B2 with BC only.  This entry
point never runs environment rollout and never enables QGF during training.
"""

import pathlib
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra
from omegaconf import OmegaConf

from hiera_diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        "hiera_diffusion_policy", "config"
    )),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    if float(cfg.eta) <= 0.0 or float(cfg.policy.eta) <= 0.0:
        raise ValueError("Fusion A must use eta>0 for its original actor-Q loss.")
    if str(cfg.policy.mode).upper() != "SWITCH" or str(cfg.policy.b_branch) != "B2":
        raise ValueError("This entry point requires policy.mode=SWITCH and policy.b_branch=B2.")
    if bool(cfg.policy.b_branch_use_q_loss):
        raise ValueError("B2 must remain BC-only: set policy.b_branch_use_q_loss=false.")
    if cfg.guider_path is None or cfg.critic_path is None:
        raise ValueError("Both guider_path and critic_path are required for Fusion actor training.")

    # QGF is an evaluation-only intervention.  The actor checkpoint therefore
    # cannot depend on whether later evaluation chooses mode=none or clean.
    cfg.policy.b2_q_guidance_mode = "none"
    cfg.train_model = "actor"
    cfg.test_run = False
    OmegaConf.update(cfg, "training.enable_actor_rollout", False, force_add=True)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
