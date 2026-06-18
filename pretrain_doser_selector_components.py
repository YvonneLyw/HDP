"""
Pretrain frozen DOSER-style branch selector components.

Important:
  The action detector is shared by A/B and reads full-state/raw inputs from the
  offline batch. It does not depend on branch_condition_encoder / DKO, so it
  does not require a pretrained actor checkpoint.

Example:
python pretrain_doser_selector_components.py --config-name=hdp_d3p_can_ph \
  critic_path=/path/to/critic_run \
  +doser_pretrain.output_dir=outputs/doser_selector_components \
  +doser_pretrain.max_train_episodes=50 \
  +doser_pretrain.train_epochs=50
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pathlib
import pickle
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf, open_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.model.diffusion.doser_selector_components import (
    FullStateActionDetector,
    LatentDynamicsModel,
    LatentValueNet,
    LatentStateDetector,
)

try:
    import dill as checkpoint_pickle
except ImportError:
    checkpoint_pickle = pickle


DEFAULT_PRETRAIN_CFG = {
    "output_dir": "outputs/doser_selector_components",  # 输出文件 
    "checkpoint_name": "doser_selector_components.ckpt",
    "critic_path": None,    # 必须加载 Critic ，因为 ValueNet 的训练 target 来自 Critic
    "require_critic_checkpoint": True,
    "train_epochs": 50,
    "max_train_episodes": None,
    "max_train_steps": None,
    "max_batches_per_epoch": None,
    "calibration_batches": None,

    "latent_dim": 128,
    "detector_hidden_dim": 256,
    "dynamics_hidden_dim": 256,
    "value_hidden_dim": 256,
    "action_detector_image_feat_dim": 64,
    "dynamics_image_feat_dim": 64,

    "time_embed_dim": 32,   # Detector score 配置
    "score_samples": 8,     # 计算 reconstruction error 时，可以对随机 timestep 或随机噪声采样 8 次，再进行聚合
    # 训练配置    
    "lr": 1.0e-4,   # AdamW optimizer
    "weight_decay": 1.0e-6,
    # 总损失权重
    "action_loss_weight": 1.0, 
    "dynamics_loss_weight": 1.0,
    "dynamics_recon_loss_weight": 0.1,  # Dynamics 模型中“状态自编码重建损失” （encoder 必须让 latent 保留足够的信息）
    "state_loss_weight": 1.0,
    "value_loss_weight": 1.0,

    "value_expectile": 0.7,
    "value_subgoal_dim": 0,
}


def _resolve_checkpoint_path(path_like, default_name: str) -> Optional[pathlib.Path]:
    if path_like in (None, ""):
        return None
    path = pathlib.Path(str(path_like))
    if path.suffix != ".ckpt":
        path = path.joinpath("checkpoints", default_name)
    return path


def _resolve_pretrain_demo_count(cfg: OmegaConf, pre_cfg: OmegaConf) -> Optional[int]:
    if pre_cfg.max_train_episodes is not None:
        return int(pre_cfg.max_train_episodes)
    dataset_max_train_episodes = cfg.task.dataset.get("max_train_episodes", None)
    if dataset_max_train_episodes is None:
        return None
    return int(dataset_max_train_episodes)


def _checkpoint_name_with_demo_count(checkpoint_name: str, demo_count: Optional[int]) -> str:
    path = pathlib.Path(str(checkpoint_name))
    tag = f"demo{int(demo_count)}" if demo_count is not None else "demo_all"
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    return str(path.with_name(f"{stem}_{tag}{suffix}"))


def _choose_state_dict(payload: Dict, prefer_ema: bool = False) -> Dict[str, torch.Tensor]:
    state_dicts = payload.get("state_dicts", {})
    keys = ("ema_model", "model") if prefer_ema else ("model", "ema_model")
    for key in keys:
        if key in state_dicts:
            return state_dicts[key]
    raise RuntimeError("Checkpoint does not contain state_dicts['model'] or state_dicts['ema_model'].")


def _load_model_prefixes(
    model: torch.nn.Module,
    ckpt_path: pathlib.Path,
    prefixes: Tuple[str, ...],
    required_prefixes: Tuple[str, ...] = tuple(),
    prefer_ema: bool = False,
) -> Dict[str, bool]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # 加载 checkpoint
    payload = torch.load(ckpt_path.open("rb"), pickle_module=checkpoint_pickle, map_location="cpu")
    # 选择 model 还是 ema_model
    source = _choose_state_dict(payload, prefer_ema=prefer_ema)
    target = model.state_dict()
    matched = {}
    matched_prefix = {prefix: False for prefix in prefixes}

    for key, value in source.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if key not in target or target[key].shape != value.shape:
            continue
        matched[key] = value
        for prefix in prefixes:
            if key.startswith(prefix):
                matched_prefix[prefix] = True

    if not matched:
        raise RuntimeError(f"No matching parameters loaded from {ckpt_path} for prefixes={prefixes}.")

    missing_required = [prefix for prefix in required_prefixes if not matched_prefix.get(prefix, False)]
    if missing_required:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is missing required prefixes: {missing_required}."
        )

    target.update(matched)
    model.load_state_dict(target, strict=False)
    return matched_prefix


# 实例化完整 Policy（Guider、Actor 和 Critic）为了复用接口方法（没用Guider、Actor模型本身） 和 DataLoader
def _instantiate_policy_and_data(cfg: OmegaConf, pre_cfg: OmegaConf, device: torch.device):
    model_guider = hydra.utils.instantiate(cfg.model_guider)
    model_actor = hydra.utils.instantiate(cfg.model_actor)
    model_critic = hydra.utils.instantiate(cfg.model_critic)
    policy_cfg = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=False))
    if "doser_selector" in policy_cfg:
        with open_dict(policy_cfg):
            del policy_cfg.doser_selector
    if "branch_selector" in policy_cfg and str(policy_cfg.branch_selector).lower() == "doser":
        with open_dict(policy_cfg):
            policy_cfg.branch_selector = "err"
    model = hydra.utils.instantiate(
        policy_cfg,
        guider=model_guider,
        actor=model_actor,
        critic=model_critic,    # 此时 Critic checkpoint 还没有加载，但冻结状态下后面只是把 checkpoint 里的数值复制到现有参数中
    )
    # 创建 Dataset。DOSER selector components 可以单独控制预训练使用多少条 demo。
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=False))
    if pre_cfg.max_train_episodes is not None:
        with open_dict(dataset_cfg):
            dataset_cfg.max_train_episodes = int(pre_cfg.max_train_episodes)
    dataset = hydra.utils.instantiate(dataset_cfg)
    # 设置 Normalizer                               ##############D3P信息怎么办?
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    # 冻结所有 Policy 参数
    model.to(device)
    model.eval()    # model.training == False，主要影响：Dropout，BatchNorm，其他区分 train/eval 行为的层
    model.requires_grad_(False)
    # 创建训练 DataLoader：取一个 batch 推断网络输入维度，训练模型参数
    train_loader = DataLoader(dataset, **OmegaConf.to_container(cfg.dataloader, resolve=True))
    # 训练完成后统计 reference errors
    if "dataloader_noshuff" in cfg:
        calib_loader = DataLoader(dataset, **OmegaConf.to_container(cfg.dataloader_noshuff, resolve=True))
    else:
        calib_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        calib_cfg["shuffle"] = False
        calib_cfg["drop_last"] = False
        calib_loader = DataLoader(dataset, **calib_cfg)
    return model, train_loader, calib_loader

def _prepare_selector_batch(model, batch: Dict[str, torch.Tensor]):
    with torch.no_grad():
        common = model._prepare_branch_inputs(batch)
        action_eval = model._extract_action_segment(
            common["nbatch"]["action"],
            model._get_action_start("A"),
            model.Tr,
        )   # (B , Tr长=8 , dimA）
        nbatch = common["nbatch"]
        next_state = nbatch["next_state"].reshape(nbatch["next_state"].shape[0], -1)
        next_subgoal = nbatch.get("next_subgoal", None)
        next_image = common.get("doser_image_pair", None)
        if next_image is not None:
            next_image = next_image[:, 1]
    return common, action_eval, next_state, next_subgoal, next_image


def _infer_dims(model, train_loader, device: torch.device) -> Dict[str, int]:
    batch = next(iter(train_loader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    common, action_eval, _, _, _ = _prepare_selector_batch(model, batch)
    image = common.get("image_pair", common.get("doser_image_pair", None))
    return {
        "state_dim": int(common["state"].shape[-1]),
        "subgoal_dim": int(common["subgoal"].shape[-1]) if common.get("subgoal", None) is not None else 0,
        "qpos_dim": int(common["qpos_pair"].shape[-1]) if common.get("qpos_pair", None) is not None else 0,
        "image_shape": tuple(int(x) for x in image.shape[2:]) if image is not None else None,
        "action_eval_dim": int(action_eval.reshape(action_eval.shape[0], -1).shape[-1]),    # [B, Tr * action_dim]
    }


########################核心################################
def _build_components(dims: Dict[str, int], pre_cfg: OmegaConf, device: torch.device):
    if dims["image_shape"] is None:
        raise RuntimeError("DOSER selector pretraining requires image observations.")
    action_detector = FullStateActionDetector(
        state_dim=dims["state_dim"],
        subgoal_dim=dims["subgoal_dim"],
        qpos_dim=dims["qpos_dim"],
        action_eval_dim=dims["action_eval_dim"],
        image_shape=dims["image_shape"],
        image_feat_dim=pre_cfg.action_detector_image_feat_dim,
        hidden_dim=pre_cfg.detector_hidden_dim,
        time_embed_dim=pre_cfg.time_embed_dim,
        score_samples=pre_cfg.score_samples,
    ).to(device)
    dynamics_model = LatentDynamicsModel(
        state_dim=dims["state_dim"],
        subgoal_dim=dims["subgoal_dim"],
        action_eval_dim=dims["action_eval_dim"],
        image_shape=dims["image_shape"],
        image_feat_dim=pre_cfg.dynamics_image_feat_dim,
        latent_dim=pre_cfg.latent_dim,
        hidden_dim=pre_cfg.dynamics_hidden_dim,
        
    ).to(device)
    state_detector = LatentStateDetector(
        latent_dim=pre_cfg.latent_dim,
        hidden_dim=pre_cfg.detector_hidden_dim,
        time_embed_dim=pre_cfg.time_embed_dim,
        score_samples=pre_cfg.score_samples,
    ).to(device)
    value_net = LatentValueNet(
        latent_dim=pre_cfg.latent_dim,
        subgoal_dim=pre_cfg.value_subgoal_dim,
        hidden_dim=pre_cfg.value_hidden_dim,
    ).to(device)
    return action_detector, dynamics_model, state_detector, value_net


def _train_components(
    model,
    train_loader,
    components,
    pre_cfg: OmegaConf,
    device: torch.device,
) -> None:
    action_detector, dynamics_model, state_detector, value_net = components
    params = []
    for module in components:
        params.extend(module.parameters())
    optimizer = AdamW(params, lr=pre_cfg.lr, weight_decay=pre_cfg.weight_decay)

    global_step = 0
    max_train_steps = pre_cfg.max_train_steps
    for epoch in range(int(pre_cfg.train_epochs)):
        losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"DOSER selector pretrain epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_train_steps is not None and global_step >= int(max_train_steps):
                return
            if pre_cfg.max_batches_per_epoch is not None and batch_idx >= int(pre_cfg.max_batches_per_epoch):
                break

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            common, action_eval, next_state, next_subgoal, next_image = _prepare_selector_batch(model, batch)
            if dynamics_model.image_encoder is not None and next_image is None:
                raise RuntimeError("LatentDynamicsModel uses images, but Tr-aligned next image is missing.")

            z_current = dynamics_model.encode(common=common)
            z_next = dynamics_model.encode(state=next_state, subgoal=next_subgoal, image=next_image)
            pred_next = dynamics_model(common, action_eval)["next_latent"]

            action_loss = action_detector.denoising_loss(common, action_eval)
            dynamics_loss = F.mse_loss(pred_next, z_next.detach())
            dynamics_recon_loss = 0.5 * (
                dynamics_model.reconstruction_loss(
                    common["state"],
                    common.get("subgoal", None),
                    common.get("doser_image_pair", common.get("image_pair", None)),
                )
                + dynamics_model.reconstruction_loss(next_state, next_subgoal, next_image)
            )
            state_loss = state_detector.denoising_loss(
                torch.cat((z_current.detach(), z_next.detach()), dim=0)
            )

            with torch.no_grad():
                q1, q2 = model.critic_target(
                    common["pcd"],
                    common["state"],
                    common["subgoal"],
                    action_eval.reshape(action_eval.shape[0], -1),
                )
                q_target = torch.minimum(q1.squeeze(-1), q2.squeeze(-1))
            value_subgoal = common.get("subgoal", None) if int(pre_cfg.value_subgoal_dim) > 0 else None
            value_pred = value_net(z_current.detach(), value_subgoal)
            value_loss = value_net.expectile_loss(q_target - value_pred, float(pre_cfg.value_expectile))

            loss = (
                float(pre_cfg.action_loss_weight) * action_loss
                + float(pre_cfg.dynamics_loss_weight) * dynamics_loss
                + float(pre_cfg.dynamics_recon_loss_weight) * dynamics_recon_loss
                + float(pre_cfg.state_loss_weight) * state_loss
                + float(pre_cfg.value_loss_weight) * value_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().item()))
            pbar.set_postfix(
                loss=f"{losses[-1]:.4f}",
                action=f"{float(action_loss.detach().item()):.4f}",
                dyn=f"{float(dynamics_loss.detach().item()):.4f}",
                value=f"{float(value_loss.detach().item()):.4f}",
            )
            global_step += 1


@torch.no_grad()        # 不反传，不修改模型参数
def _calibrate_components(model, calib_loader, components, pre_cfg: OmegaConf, device: torch.device):
    action_detector, dynamics_model, state_detector, _ = components
    for module in components:
        module.eval()

    action_errors = []
    state_errors = []
    for batch_idx, batch in enumerate(tqdm.tqdm(calib_loader, desc="Calibrating DOSER detectors", leave=False)):
        if pre_cfg.calibration_batches is not None and batch_idx >= int(pre_cfg.calibration_batches):
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        common, action_eval, next_state, next_subgoal, next_image = _prepare_selector_batch(model, batch)
        if dynamics_model.image_encoder is not None and next_image is None:
            raise RuntimeError("LatentDynamicsModel uses images, but Tr-aligned next image is missing.")

        action_errors.append(action_detector.reconstruction_error(common, action_eval, pre_cfg.score_samples).detach().cpu())

        z_current = dynamics_model.encode(common=common)
        z_next = dynamics_model.encode(state=next_state, subgoal=next_subgoal, image=next_image)
        state_latents = torch.cat((z_current, z_next), dim=0)
        state_errors.append(state_detector.reconstruction_error(state_latents, pre_cfg.score_samples).detach().cpu())

    action_errors = torch.cat(action_errors, dim=0)
    state_errors = torch.cat(state_errors, dim=0)
    # action/state_errors保存到 detector 中
    action_detector.set_reference_errors(action_errors) 
    state_detector.set_reference_errors(state_errors)
    return action_errors, state_errors


def _save_components(
    cfg: OmegaConf,
    pre_cfg: OmegaConf,
    dims: Dict[str, int],
    components,
    reference_errors,
) -> pathlib.Path:
    action_detector, dynamics_model, state_detector, value_net = components
    action_errors, state_errors = reference_errors
    output_dir = pathlib.Path(str(pre_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_count = _resolve_pretrain_demo_count(cfg, pre_cfg)
    checkpoint_name = _checkpoint_name_with_demo_count(pre_cfg.checkpoint_name, demo_count)
    output_path = output_dir.joinpath(checkpoint_name)

    metadata = {
        **dims,
        "latent_dim": int(pre_cfg.latent_dim),
        "detector_hidden_dim": int(pre_cfg.detector_hidden_dim),
        "dynamics_hidden_dim": int(pre_cfg.dynamics_hidden_dim),
        "value_hidden_dim": int(pre_cfg.value_hidden_dim),
        "image_feat_dim": int(pre_cfg.action_detector_image_feat_dim),
        "dynamics_image_feat_dim": int(pre_cfg.dynamics_image_feat_dim),
        "time_embed_dim": int(pre_cfg.time_embed_dim),
        "score_samples": int(pre_cfg.score_samples),
        "value_subgoal_dim": int(pre_cfg.value_subgoal_dim),
        "max_train_episodes": demo_count,
    }
    metadata["image_shape"] = (
        tuple(action_detector.image_shape)
        if action_detector.image_shape is not None
        else None
    )

    def cpu_state_dict(module):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in module.state_dict().items()
        }

    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "doser_pretrain_cfg": OmegaConf.to_container(pre_cfg, resolve=True),
        "metadata": metadata,
        "state_dicts": {
            "action_detector": cpu_state_dict(action_detector),
            "dynamics_model": cpu_state_dict(dynamics_model),
            "state_detector": cpu_state_dict(state_detector),
            "value_net": cpu_state_dict(value_net),
        },
        "reference_errors": {
            "action": action_errors,
            "state": state_errors,
        },
    }
    torch.save(payload, output_path.open("wb"), pickle_module=pickle)
    return output_path

##########################入口#############################
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("hiera_diffusion_policy", "config")),
)   # 加载主配置 cfg 传入 main()

def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    # 创建配置 DEFAULT_PRETRAIN_CFG 传入 cfg.doser_pretrain.xxxxxx = xxxx
    pre_cfg = OmegaConf.merge(
        OmegaConf.create(DEFAULT_PRETRAIN_CFG),
        cfg.get("doser_pretrain", {}),
    )   # 最终： pre_cfg.xxxxxx = xxxx
    device = torch.device(cfg.training.device)

    # 实例化完整 Policy（Guider、Actor 和 Critic）为了复用接口方法 和 DataLoader
    model, train_loader, calib_loader = _instantiate_policy_and_data(cfg, pre_cfg, device)

    # 加载已经训练好的 Critic
    critic_path = _resolve_checkpoint_path(
        pre_cfg.critic_path if pre_cfg.critic_path is not None else cfg.critic_path,
        "critic_latest.ckpt",
    )
    if critic_path is None and bool(pre_cfg.require_critic_checkpoint):
        raise RuntimeError("critic_path is required for IQL-style ValueNet targets.")
    if critic_path is not None:
        matched = _load_model_prefixes(
            model,
            critic_path,
            prefixes=("critic.", "critic_target."), # 只提取 Critic 参数
            required_prefixes=("critic.",),
        )
        if not matched.get("critic_target.", False):
            model.critic_target.load_state_dict(model.critic.state_dict())
    # 再次冻结完整 Policy（只用critic）
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    dims = _infer_dims(model, train_loader, device)

    components = _build_components(dims, pre_cfg, device)

    _train_components(model, train_loader, components, pre_cfg, device)
    reference_errors = _calibrate_components(model, calib_loader, components, pre_cfg, device)
    output_path = _save_components(cfg, pre_cfg, dims, components, reference_errors)

    print(f"Saved DOSER selector components to: {output_path}")
    print(f"Metadata: {OmegaConf.to_yaml(OmegaConf.create({**dims, 'latent_dim': pre_cfg.latent_dim}))}")


if __name__ == "__main__":
    main()
