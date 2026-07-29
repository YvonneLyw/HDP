"""Train only the original HDP actor with its training-time Q loss.

This is the matched baseline counterpart of ``train_hdp_actor_bc.py``.  It
reuses frozen, already-trained original HDP guider and TD critic checkpoints,
then optimizes a freshly initialized actor with
``L_actor = L_BC + eta * L_Q``.  Environment rollout is forcibly disabled;
use ``eval_hdp.py`` after training for the single, controlled evaluation.
"""

import pathlib
import sys

# Match the existing long-running training entry points.
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
        raise ValueError(
            "train_hdp_actor_q.py requires eta>0 so the original actor Q loss is active."
        )
    if cfg.guider_path is None:
        raise ValueError("guider_path is required for original actor-Q training.")
    if cfg.critic_path is None:
        raise ValueError("critic_path is required for original actor-Q training.")

    # Unlike train_hdp.py, do not retrain guider/critic first.  The workspace
    # loads the two supplied checkpoints, then only enters its actor loop.
    cfg.train_model = "actor"
    cfg.test_run = False
    OmegaConf.update(cfg, "training.enable_actor_rollout", False, force_add=True)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
