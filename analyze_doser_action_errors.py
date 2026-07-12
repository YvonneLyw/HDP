"""
Analyze DOSER action reconstruction-error distributions on a dataset.

Examples:
python analyze_doser_action_errors.py \
  --config-name hdp_d3p_can_mh \
  --components-path outputs/doser_selector_components/can_mh/doser_selector_components_gt_split_demo50.ckpt \
  --output outputs/analysis/can_mh_action_errors.npz \
  --plot

For a fair PH vs MH data-domain comparison, keep --components-path fixed and
run the script once with the PH config and once with the MH config.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    os.chdir(str(ROOT_DIR))

import argparse
import json
import pathlib
import pickle
from typing import Dict, List

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.policy.doser_branch_selector import DoserBranchSelector
from hiera_diffusion_policy.policy.doser_gt_branch_selector import (
    GroundTruthDoserBranchSelector,
)
from pretrain_doser_selector_components import _prepare_selector_batch

try:
    import dill as checkpoint_pickle
except ImportError:
    checkpoint_pickle = pickle


OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--config-dir",
        default="hiera_diffusion_policy/config",
    )
    parser.add_argument("--components-path", required=True)
    parser.add_argument(
        "--selector-type",
        choices=("auto", "latent", "gt"),
        default="auto",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--score-samples", type=int, default=None)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def load_cfg(config_dir: str, config_name: str):
    config_path = pathlib.Path(config_dir)
    if not config_path.is_absolute():
        config_path = pathlib.Path.cwd() / config_path
    name = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
    cfg = OmegaConf.load(config_path / name)
    return cfg


def instantiate_policy_and_loader(cfg, args, device: torch.device):
    model_guider = hydra.utils.instantiate(cfg.model_guider)
    model_actor = hydra.utils.instantiate(cfg.model_actor)
    model_critic = hydra.utils.instantiate(cfg.model_critic)

    policy_cfg = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=False))
    with open_dict(policy_cfg):
        if "branch_selector" in policy_cfg:
            policy_cfg.branch_selector = "err"
        if "doser_selector" in policy_cfg:
            policy_cfg.doser_selector.components_path = None
        if "doser_gt_selector" in policy_cfg:
            policy_cfg.doser_gt_selector.components_path = None

    model = hydra.utils.instantiate(
        policy_cfg,
        guider=model_guider,
        actor=model_actor,
        critic=model_critic,
    )

    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=False))
    if args.max_train_episodes is not None:
        with open_dict(dataset_cfg):
            dataset_cfg.max_train_episodes = int(args.max_train_episodes)
    dataset = hydra.utils.instantiate(dataset_cfg)
    if args.validation:
        dataset = dataset.get_validation_dataset()

    loader_cfg = cfg.get("dataloader_noshuff", cfg.dataloader)
    loader_kwargs = OmegaConf.to_container(loader_cfg, resolve=True)
    loader_kwargs["shuffle"] = False
    loader = DataLoader(dataset, **loader_kwargs)

    model.set_normalizer(dataset.get_normalizer())
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, loader


def infer_selector_type(components_path: pathlib.Path) -> str:
    payload = torch.load(
        components_path.open("rb"),
        pickle_module=checkpoint_pickle,
        map_location="cpu",
    )
    metadata = payload.get("metadata", {})
    if metadata.get("selector_type", None) == "ground_truth_state":
        return "gt"
    return "latent"


def load_selector(args, device: torch.device):
    components_path = pathlib.Path(args.components_path)
    selector_type = (
        infer_selector_type(components_path)
        if args.selector_type == "auto"
        else args.selector_type
    )
    if selector_type == "gt":
        selector = GroundTruthDoserBranchSelector(components_path=str(components_path))
    else:
        selector = DoserBranchSelector(components_path=str(components_path))
    selector.to(device)
    selector.eval()
    selector.requires_grad_(False)
    return selector_type, selector


def extract_b_action_eval(model, common):
    d3p_action_pair = common.get("d3p_action_pair", None)
    if d3p_action_pair is None:
        raise RuntimeError("Split B2 action detector requires common['d3p_action_pair'].")
    return model._extract_action_segment(
        d3p_action_pair[:, 0],
        0,
        model.Tr,
    )


@torch.no_grad()
def collect_errors(model, loader, selector, args, device: torch.device) -> Dict[str, np.ndarray]:
    errors: Dict[str, List[torch.Tensor]] = {}

    def append_error(name: str, value: torch.Tensor):
        errors.setdefault(name, []).append(value.detach().cpu())

    pbar = tqdm.tqdm(loader, desc="Scoring action reconstruction errors", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= int(args.max_batches):
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        common, action_eval, _, _, _ = _prepare_selector_batch(model, batch)
        score_samples = args.score_samples

        if isinstance(selector, GroundTruthDoserBranchSelector) and getattr(
            selector,
            "use_branch_action_detectors",
            False,
        ):
            action_b_eval = extract_b_action_eval(model, common)
            append_error(
                "action_A",
                selector.action_detector_A.reconstruction_error(
                    common,
                    action_eval,
                    score_samples,
                ),
            )
            append_error(
                "action_B2",
                selector.action_detector_B2.reconstruction_error(
                    common,
                    action_b_eval,
                    score_samples,
                ),
            )
        else:
            append_error(
                "action",
                selector.action_detector.reconstruction_error(
                    common,
                    action_eval,
                    score_samples,
                ),
            )

    if not errors:
        raise RuntimeError("No errors were collected. Check the dataset and max-batches.")
    return {
        key: torch.cat(values, dim=0).numpy()
        for key, values in errors.items()
    }


def summarize(values: np.ndarray):
    percentiles = [1, 5, 25, 50, 75, 90, 95, 98, 99]
    out = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for percentile in percentiles:
        out[f"p{percentile}"] = float(np.percentile(values, percentile))
    return out


def save_plot(error_arrays: Dict[str, np.ndarray], output_path: pathlib.Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc

    plt.figure(figsize=(8, 5))
    for key, values in error_arrays.items():
        plt.hist(values, bins=80, alpha=0.45, density=True, label=key)
    plt.xlabel("Action reconstruction error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_cfg(args.config_dir, args.config_name)
    OmegaConf.resolve(cfg)

    model, loader = instantiate_policy_and_loader(cfg, args, device)
    selector_type, selector = load_selector(args, device)
    error_arrays = collect_errors(model, loader, selector, args, device)

    config_stem = pathlib.Path(args.config_name).stem
    ckpt_stem = pathlib.Path(args.components_path).stem
    output_path = pathlib.Path(
        args.output
        or f"outputs/analysis/{config_stem}_{ckpt_stem}_action_errors.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config_name": args.config_name,
        "components_path": args.components_path,
        "selector_type": selector_type,
        "validation": bool(args.validation),
        "max_train_episodes": args.max_train_episodes,
        "max_batches": args.max_batches,
        "score_samples": args.score_samples,
        "errors": {
            key: summarize(values)
            for key, values in error_arrays.items()
        },
    }

    np.savez_compressed(output_path, **error_arrays)
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path = None
    if args.plot:
        plot_path = save_plot(error_arrays, output_path)

    print(json.dumps(summary, indent=2))
    print(f"Saved arrays to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    if plot_path is not None:
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
