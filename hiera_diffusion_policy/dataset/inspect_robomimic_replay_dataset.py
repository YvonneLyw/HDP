if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import pathlib
import sys
import types
import importlib.util

import hydra
from omegaconf import OmegaConf


def _shape_str(value):
    shape = tuple(value.shape) if hasattr(value, "shape") else "N/A"
    dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
    return f"shape={shape}, dtype={dtype}"


def _resolve_dataset_path(path_str):
    path = pathlib.Path(path_str)
    if path.is_file():
        return str(path)

    fallback = pathlib.Path("data/image_abs_pcd.hdf5")
    if fallback.is_file():
        print(f"[warn] dataset_path not found: {path_str}")
        print(f"[warn] use local fallback: {fallback}")
        return str(fallback)

    return path_str


def _patch_optional_deps():
    # Inspector-only fallback: dataset dependency chain imports heavy geometry deps.
    # The inspection path used here does not rely on their APIs.
    for module_name in ["open3d", "trimesh"]:
        if importlib.util.find_spec(module_name) is None:
            sys.modules[module_name] = types.ModuleType(module_name)
            print(f"[warn] {module_name} not found; patched a dummy module for inspector only.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="hiera_diffusion_policy/config/hdp_can_ph_image.yaml",
        help="Path to yaml config",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to inspect",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.task.dataset.dataset_path = _resolve_dataset_path(cfg.task.dataset.dataset_path)

    _patch_optional_deps()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    sample = dataset[args.index]

    print(f"config: {args.config}")
    print(f"dataset_path: {cfg.task.dataset.dataset_path}")
    print(f"dataset_len: {len(dataset)}")
    print("")

    print("=== HDP fields ===")
    hdp_keys = [
        "id",
        "scene_pcd",
        "object_pcd",
        "pcd",
        "state",
        "action",
        "next_pcd",
        "next_state",
        "next_action",
        "subgoal",
        "next_subgoal",
        "reward",
    ]
    for key in hdp_keys:
        if key in sample:
            print(f"{key:12s}: {_shape_str(sample[key])}")

    print("")
    print("=== D3P-compatible fields ===")
    for key in ["image", "qpos", "d3p_action_pair", "act_is_pad_pair"]:
        if key in sample:
            print(f"{key:12s}: {_shape_str(sample[key])}")

    if "image" in sample:
        image = sample["image"]
        print(f"image min/max: {image.min().item():.6f} / {image.max().item():.6f}")
        print(f"front frame shape (C,H,W): {tuple(image[0, 0].shape)}")
        print(f"wrist frame shape (C,H,W): {tuple(image[0, 1].shape)}")

    if "qpos" in sample:
        qpos = sample["qpos"]
        print(f"qpos dim breakdown: joint_pos(7) + gripper_qpos(2) = {qpos.shape[-1]}")
        print(f"qpos min/max: {qpos.min().item():.6f} / {qpos.max().item():.6f}")

    if "act_is_pad_pair" in sample:
        print(f"act_is_pad_pair[:,0] values: {sample['act_is_pad_pair'][:, 0].tolist()}")


if __name__ == "__main__":
    main()
