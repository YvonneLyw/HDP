"""Train only a behavior-cloning actor for the QGF-on-HDP experiment.

This deliberately bypasses train_hdp.py's guider -> critic -> actor orchestration.
It reuses the standard TrainWorkspace and requires eta=0, so the actor is trained
only with the existing diffusion BC objective.
"""

import pathlib
import sys

# Use line-buffered output for long training jobs, matching the existing entries.
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf

from hiera_diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'hiera_diffusion_policy', 'config')),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    if float(cfg.eta) != 0.0 or float(cfg.policy.eta) != 0.0:
        raise ValueError(
            "train_hdp_actor_bc.py requires eta=0 so Actor_BC has no training-time Q loss."
        )
    if cfg.guider_path is None:
        raise ValueError(
            "guider_path is required: pass the existing guider checkpoint with guider_path=/abs/path.ckpt"
        )

    cfg.train_model = 'actor'
    cfg.test_run = False

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
