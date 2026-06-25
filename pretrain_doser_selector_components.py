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
  +doser_pretrain.stage1_epochs=50 \
  +doser_pretrain.stage2_epochs=50
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
    "checkpoint_name": "doser_selector_components_pcd_staged.ckpt",
    "critic_path": None,    # 必须加载 Critic ，因为 ValueNet 的训练 target 来自 Critic
    "require_critic_checkpoint": True,
    # If stage-specific values are unset, train_epochs is used for each stage.
    "train_epochs": 500,
    "stage1_epochs": None,
    "stage2_epochs": None,
    "max_train_episodes": None,
    "max_train_steps": None,
    "max_batches_per_epoch": None,
    "calibration_batches": None,
    "validation_batches": None,

    "latent_dim": 128,
    "detector_hidden_dim": 256,
    "dynamics_hidden_dim": 256,
    "value_hidden_dim": 256,
    "action_detector_image_feat_dim": 64,
    "action_detector_pcd_feat_dim": 64,
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
    "dynamics_state_loss_weight": 1.0,
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
    target_prefixed_keys = [
        key for key in target
        if any(key.startswith(prefix) for prefix in prefixes)
    ]

    for key, value in source.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if key not in target or target[key].shape != value.shape:
            continue
        matched[key] = value

    if not matched:
        raise RuntimeError(f"No matching parameters loaded from {ckpt_path} for prefixes={prefixes}.")

    complete_prefix = {}
    for prefix in prefixes:
        target_keys = [key for key in target_prefixed_keys if key.startswith(prefix)]
        loaded_keys = [key for key in target_keys if key in matched]
        complete_prefix[prefix] = (
            len(target_keys) > 0 and len(loaded_keys) == len(target_keys)
        )

    missing_required = [
        prefix for prefix in required_prefixes
        if not complete_prefix.get(prefix, False)
    ]
    if missing_required:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} did not completely load required prefixes: "
            f"{missing_required}."
        )

    target.update(matched)
    model.load_state_dict(target, strict=False)
    return complete_prefix


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
    # 创建val DataLoader：
    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        raise RuntimeError(
            "DOSER dynamics validation set is empty. Increase task.dataset.val_ratio "
            "so at least one complete demo is reserved for validation."
        )
    val_loader_cfg = cfg.val_dataloader if "val_dataloader" in cfg else cfg.dataloader_noshuff
    val_loader = DataLoader(
        val_dataset,
        **OmegaConf.to_container(val_loader_cfg, resolve=True),
    )
    # calib_loader（不打乱）：训练完成后统计 reference errors
    if "dataloader_noshuff" in cfg:
        calib_loader = DataLoader(dataset, **OmegaConf.to_container(cfg.dataloader_noshuff, resolve=True))
    else:
        calib_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        calib_cfg["shuffle"] = False
        calib_cfg["drop_last"] = False
        calib_loader = DataLoader(dataset, **calib_cfg)
    return model, train_loader, calib_loader, val_loader

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
        if next_image is None:
            next_image = batch.get("doser_image_pair", None)
        if next_image is not None:
            next_image = next_image[:, 1]
    return common, action_eval, next_state, next_subgoal, next_image


def _infer_dims(model, train_loader, device: torch.device) -> Dict[str, int]:
    batch = next(iter(train_loader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    common, action_eval, _, _, _ = _prepare_selector_batch(model, batch)
    image = common.get("image_pair", None)
    if image is None:
        image = common.get("doser_image_pair", None)
    return {
        "state_dim": int(common["state"].shape[-1]),
        "subgoal_dim": int(common["subgoal"].shape[-1]) if common.get("subgoal", None) is not None else 0,
        "qpos_dim": int(common["qpos_pair"].shape[-1]) if common.get("qpos_pair", None) is not None else 0,
        "pcd_dim": int(common["pcd"].shape[-1]) if common.get("pcd", None) is not None else 0,
        "image_shape": tuple(int(x) for x in image.shape[2:]) if image is not None else None,
        "action_eval_dim": int(action_eval.reshape(action_eval.shape[0], -1).shape[-1]),    # [B, Tr * action_dim]
    }


########################################### 核心 ##########################################
def _build_components(dims: Dict[str, int], pre_cfg: OmegaConf, device: torch.device):
    if dims["image_shape"] is None:
        raise RuntimeError("DOSER selector pretraining requires image observations.")
    action_detector = FullStateActionDetector(
        state_dim=dims["state_dim"],
        subgoal_dim=dims["subgoal_dim"],
        qpos_dim=dims["qpos_dim"],
        action_eval_dim=dims["action_eval_dim"],
        pcd_dim=dims["pcd_dim"],
        image_shape=dims["image_shape"],
        image_feat_dim=pre_cfg.action_detector_image_feat_dim,
        pcd_feat_dim=pre_cfg.action_detector_pcd_feat_dim,
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


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    module.train(trainable)
    module.requires_grad_(trainable)


def _stage_epochs(pre_cfg: OmegaConf, key: str) -> int:
    value = pre_cfg.get(key, None)
    return int(pre_cfg.train_epochs if value is None else value)


def _stage_limit_reached(pre_cfg: OmegaConf, step: int, batch_idx: int) -> bool:
    if pre_cfg.max_train_steps is not None and step >= int(pre_cfg.max_train_steps):
        return True
    return (
        pre_cfg.max_batches_per_epoch is not None
        and batch_idx >= int(pre_cfg.max_batches_per_epoch)
    )

################## stage1：Action Detector + Dynamics #####################
def _train_action_and_dynamics(
    model,
    train_loader,
    components,
    pre_cfg: OmegaConf,
    device: torch.device,
) -> None:
    action_detector, dynamics_model, state_detector, value_net = components
    _set_module_trainable(action_detector, True)
    _set_module_trainable(dynamics_model, True)
    _set_module_trainable(state_detector, False)
    _set_module_trainable(value_net, False)
    # 仅 action_detector 和 dynamics_model
    optimizer = AdamW(
        list(action_detector.parameters()) + list(dynamics_model.parameters()),
        lr=pre_cfg.lr,
        weight_decay=pre_cfg.weight_decay,
    )
    step = 0
    for epoch in range(_stage_epochs(pre_cfg, "stage1_epochs")):
        pbar = tqdm.tqdm(
            train_loader,
            desc=f"DOSER stage 1 action+dynamics epoch {epoch}",
            leave=False,
        )
        for batch_idx, batch in enumerate(pbar):
            if _stage_limit_reached(pre_cfg, step, batch_idx):
                if pre_cfg.max_train_steps is not None and step >= int(pre_cfg.max_train_steps):
                    return
                break

            # 准备训练数据
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            common, action_eval, next_state, next_subgoal, next_image = _prepare_selector_batch(model, batch)
            
            # action_loss ###########
            action_loss = action_detector.denoising_loss(common, action_eval)

            # dynamics_loss & dynamics_recon_loss & dynamics_state_loss ###########
            z_next = dynamics_model.encode(
                state=next_state,
                subgoal=next_subgoal,
                image=next_image,
            )   # 下一状态 latent
            dynamics_out = dynamics_model(common, action_eval)
            dynamics_loss = F.mse_loss(dynamics_out["next_latent"], z_next.detach())    ###### z'
            dynamics_state_loss = F.mse_loss(dynamics_out["next_state"], next_state)    #新增 next_state_head（dynamics_state_loss）
            current_image = common.get("doser_image_pair", None)
            if current_image is None:
                current_image = common.get("image_pair", None)
            dynamics_recon_loss = 0.5 * (
                dynamics_model.reconstruction_loss(
                    common["state"],
                    common.get("subgoal", None),
                    current_image,
                )
                + dynamics_model.reconstruction_loss(next_state, next_subgoal, next_image)
            )   # encoder

            loss = (
                float(pre_cfg.action_loss_weight) * action_loss
                + float(pre_cfg.dynamics_loss_weight) * dynamics_loss
                + float(pre_cfg.dynamics_recon_loss_weight) * dynamics_recon_loss
                + float(pre_cfg.dynamics_state_loss_weight) * dynamics_state_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                action=f"{float(action_loss.detach().item()):.4f}",
                dyn=f"{float(dynamics_loss.detach().item()):.4f}",
                state_pred=f"{float(dynamics_state_loss.detach().item()):.4f}",
                recon=f"{float(dynamics_recon_loss.detach().item()):.4f}",
            )
            step += 1

################## stage2：State Detector + ValueNet #####################
def _train_state_detector_and_value(
    model,
    train_loader,
    components,
    pre_cfg: OmegaConf,
    device: torch.device,
) -> None:
    action_detector, dynamics_model, state_detector, value_net = components
    _set_module_trainable(action_detector, False)
    _set_module_trainable(dynamics_model, False)
    _set_module_trainable(state_detector, True)
    _set_module_trainable(value_net, True)

    optimizer = AdamW(
        list(state_detector.parameters()) + list(value_net.parameters()),
        lr=pre_cfg.lr,
        weight_decay=pre_cfg.weight_decay,
    )
    step = 0
    for epoch in range(_stage_epochs(pre_cfg, "stage2_epochs")):
        pbar = tqdm.tqdm(
            train_loader,
            desc=f"DOSER stage 2 state+value epoch {epoch}",
            leave=False,
        )
        for batch_idx, batch in enumerate(pbar):
            if _stage_limit_reached(pre_cfg, step, batch_idx):
                if pre_cfg.max_train_steps is not None and step >= int(pre_cfg.max_train_steps):
                    return
                break

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            common, action_eval, next_state, next_subgoal, next_image = _prepare_selector_batch(model, batch)
            with torch.no_grad():
                # 构造当前和下一状态 latent
                z_current = dynamics_model.encode(common=common)
                z_next = dynamics_model.encode(
                    state=next_state,
                    subgoal=next_subgoal,
                    image=next_image,
                )

                q1, q2 = model.critic_target(
                    common["pcd"],
                    common["state"],
                    common["subgoal"],
                    action_eval.reshape(action_eval.shape[0], -1),
                )
                q_target = torch.minimum(q1.squeeze(-1), q2.squeeze(-1))
            # state_loss ##########     信息：(z,z')
            state_loss = state_detector.denoising_loss(
                torch.cat((z_current, z_next), dim=0)
            )
            # value_loss #########      信息：（ z,subgoal)
            value_subgoal = common.get("subgoal", None) if int(pre_cfg.value_subgoal_dim) > 0 else None
            value_pred = value_net(z_current, value_subgoal)
            value_loss = value_net.expectile_loss(
                q_target - value_pred,
                float(pre_cfg.value_expectile),
            )

            loss = (
                float(pre_cfg.state_loss_weight) * state_loss
                + float(pre_cfg.value_loss_weight) * value_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                state=f"{float(state_loss.detach().item()):.4f}",
                value=f"{float(value_loss.detach().item()):.4f}",
            )
            step += 1


def _train_components(
    model,
    train_loader,
    components,
    pre_cfg: OmegaConf,
    device: torch.device,
) -> None:
    _train_action_and_dynamics(model, train_loader, components, pre_cfg, device)
    _train_state_detector_and_value(model, train_loader, components, pre_cfg, device)

################## calibration #####################
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

        # 整个校准数据上的 action reference error 分布
        action_errors.append(action_detector.reconstruction_error(common, action_eval, pre_cfg.score_samples).detach().cpu())

        # 整个校准数据上的 state_latents reference error 分布
        z_current = dynamics_model.encode(common=common)
        z_next = dynamics_model.encode(state=next_state, subgoal=next_subgoal, image=next_image)
        state_latents = torch.cat((z_current, z_next), dim=0)
        state_errors.append(state_detector.reconstruction_error(state_latents, pre_cfg.score_samples).detach().cpu())

    action_errors = torch.cat(action_errors, dim=0)
    state_errors = torch.cat(state_errors, dim=0)
    # action/state_errorserror 拉平、排序、存到 detector 自己的 buffer 里。
    action_detector.set_reference_errors(action_errors) 
    state_detector.set_reference_errors(state_errors)
    return action_errors, state_errors

################## validation #####################
@torch.no_grad()
def _validate_components(
    model,
    val_loader,
    components,
    pre_cfg: OmegaConf,
    device: torch.device,
    state_ood_percentile: float,
) -> Dict[str, float]:
    action_detector, dynamics_model, state_detector, value_net = components
    for module in components:
        module.eval()

    metric_sums = {
        "val_dyn_latent_mse": 0.0,
        "val_dyn_copy_baseline_mse": 0.0,
        "val_dyn_next_state_mse": 0.0,
        "val_state_id_agreement": 0.0,
        "val_pred_state_id_rate": 0.0,
        "val_true_state_id_rate": 0.0,
        "val_value_pred_true_next_mae": 0.0,
        "val_value_current_q_mae": 0.0,
    }
    sample_count = 0
    pbar = tqdm.tqdm(val_loader, desc="Validating DOSER dynamics", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if pre_cfg.validation_batches is not None and batch_idx >= int(pre_cfg.validation_batches):
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        common, action_eval, next_state, next_subgoal, next_image = _prepare_selector_batch(model, batch)

        z_current = dynamics_model.encode(common=common)
        z_next = dynamics_model.encode(
            state=next_state,
            subgoal=next_subgoal,
            image=next_image,
        )
        dynamics_out = dynamics_model(common, action_eval)
        pred_z_next = dynamics_out["next_latent"]
        pred_next_state = dynamics_out["next_state"]

        pred_state_score = state_detector.score(pred_z_next)
        true_state_score = state_detector.score(z_next)
        pred_state_id = pred_state_score["percentile"] <= float(state_ood_percentile)
        true_state_id = true_state_score["percentile"] <= float(state_ood_percentile)

        value_subgoal = common.get("subgoal", None) if int(pre_cfg.value_subgoal_dim) > 0 else None
        pred_next_value = value_net(pred_z_next, value_subgoal)
        true_next_value = value_net(z_next, value_subgoal)
        current_value = value_net(z_current, value_subgoal)
        q1, q2 = model.critic_target(
            common["pcd"],
            common["state"],
            common["subgoal"],
            action_eval.reshape(action_eval.shape[0], -1),
        )
        q_target = torch.minimum(q1.squeeze(-1), q2.squeeze(-1))

        batch_size = int(action_eval.shape[0])
        batch_metrics = {
            "val_dyn_latent_mse": F.mse_loss(pred_z_next, z_next),
            "val_dyn_copy_baseline_mse": F.mse_loss(z_current, z_next),     # 真z' <-> 真z（范围）
            "val_dyn_next_state_mse": F.mse_loss(pred_next_state, next_state),
            "val_state_id_agreement": (pred_state_id == true_state_id).float().mean(),
            "val_pred_state_id_rate": pred_state_id.float().mean(),
            "val_true_state_id_rate": true_state_id.float().mean(),
            "val_value_pred_true_next_mae": (pred_next_value - true_next_value).abs().mean(),   #V(pred_z') 与 V(true_z') 的差距
            "val_value_current_q_mae": (current_value - q_target).abs().mean(),
        }
        for key, value in batch_metrics.items():
            metric_sums[key] += float(value.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("DOSER dynamics validation produced no samples.")

    metrics = {
        key: value / float(sample_count)
        for key, value in metric_sums.items()
    }
    copy_mse = metrics["val_dyn_copy_baseline_mse"]
    metrics["val_dyn_vs_copy_ratio"] = (
        metrics["val_dyn_latent_mse"] / copy_mse
        if copy_mse > 0.0
        else float("inf")
    )
    metrics["val_samples"] = float(sample_count)
    print("DOSER validation metrics:")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics)))
    return metrics


def _save_components(
    cfg: OmegaConf,
    pre_cfg: OmegaConf,
    dims: Dict[str, int],
    components,
    reference_errors,
    validation_metrics: Dict[str, float],
) -> pathlib.Path:
    action_detector, dynamics_model, state_detector, value_net = components
    action_errors, state_errors = reference_errors  # 未排序
    # 创建输出目录
    output_dir = pathlib.Path(str(pre_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_count = _resolve_pretrain_demo_count(cfg, pre_cfg)
    checkpoint_name = _checkpoint_name_with_demo_count(pre_cfg.checkpoint_name, demo_count)
    output_path = output_dir.joinpath(checkpoint_name)

    # 网络结构
    metadata = {
        **dims,
        "format_version": 2,
        "latent_dim": int(pre_cfg.latent_dim),
        "detector_hidden_dim": int(pre_cfg.detector_hidden_dim),
        "dynamics_hidden_dim": int(pre_cfg.dynamics_hidden_dim),
        "value_hidden_dim": int(pre_cfg.value_hidden_dim),
        "image_feat_dim": int(pre_cfg.action_detector_image_feat_dim),
        "pcd_feat_dim": int(pre_cfg.action_detector_pcd_feat_dim),
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

    # 所有网络参数移到 CPU，这样 checkpoint 不会绑定当前 GPU
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
        # 未排序
        "reference_errors": {
            "action": action_errors,
            "state": state_errors,
        },
        "validation_metrics": validation_metrics,
    }
    torch.save(payload, output_path.open("wb"), pickle_module=pickle)
    return output_path

##########################入口##############################################################################
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

    if not 0.0 < float(pre_cfg.value_expectile) < 1.0:
        raise ValueError(
            f"doser_pretrain.value_expectile must be in (0, 1), got {pre_cfg.value_expectile}."
        )
    for key in ("stage1_epochs", "stage2_epochs"):
        if _stage_epochs(pre_cfg, key) < 0:
            raise ValueError(f"doser_pretrain.{key} must be non-negative.")
    device = torch.device(cfg.training.device)

    # 实例化完整 Policy（Guider、Actor 和 Critic）为了复用接口方法 和 DataLoader
    model, train_loader, calib_loader, val_loader = _instantiate_policy_and_data(
        cfg,
        pre_cfg,
        device,
    )

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

    # 分 2 stage 训练
    _train_components(model, train_loader, components, pre_cfg, device)
    # calibration
    reference_errors = _calibrate_components(model, calib_loader, components, pre_cfg, device)
    # validation
    selector_cfg = cfg.policy.get("doser_selector", {})
    validation_metrics = _validate_components(
        model,
        val_loader,
        components,
        pre_cfg,
        device,
        state_ood_percentile=float(selector_cfg.get("state_ood_percentile", 0.95)),
    )
    output_path = _save_components(
        cfg,
        pre_cfg,
        dims,
        components,
        reference_errors,
        validation_metrics, # 新增
    )

    print(f"Saved DOSER selector components to: {output_path}")
    print(f"Metadata: {OmegaConf.to_yaml(OmegaConf.create({**dims, 'latent_dim': pre_cfg.latent_dim}))}")


if __name__ == "__main__":
    main()
