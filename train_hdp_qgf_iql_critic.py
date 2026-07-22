"""Train only the IQL Q/V critic used by the QGF-on-HDP experiment.

This is deliberately separate from ``train_hdp.py``: the latter orchestrates
guider -> TD critic -> actor, whereas step 6a must train a fresh IQL critic
without modifying the fixed guider or Actor_BC.
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

    cfg.train_model = "critic_iql"
    cfg.critic_training_mode = "iql"
    cfg.test_run = False

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
