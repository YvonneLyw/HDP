"""
Pretrain a DOSER selector that predicts and evaluates normalized simulator state.

This is a parallel alternative to pretrain_doser_selector_components.py. It
does not replace or modify the latent-state checkpoint format.

Example:
python pretrain_doser_gt_selector_components.py --config-name=hdp_d3p_can_ph \
  critic_path=/path/to/critic_latest.ckpt \
  +doser_gt_pretrain.max_train_episodes=50 \
  +doser_gt_pretrain.train_epochs=500
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
from typing import Dict

import hydra
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf, open_dict
from torch.optim import AdamW

from hiera_diffusion_policy.common.pytorch_util import dict_apply
from hiera_diffusion_policy.model.diffusion.doser_selector_components import (
    FullStateActionDetector,
)
from hiera_diffusion_policy.model.diffusion.doser_gt_selector_components import (
    GroundTruthDynamicsModel,
    GroundTruthStateDetector,
    GroundTruthValueNet,
)
from pretrain_doser_selector_components import (
    _checkpoint_name_with_demo_count,
    _infer_dims,
    _instantiate_policy_and_data as _instantiate_doser_policy_and_data,
    _load_model_prefixes,
    _prepare_selector_batch,
    _resolve_checkpoint_path,
    _resolve_pretrain_demo_count,
)


DEFAULT_PRETRAIN_CFG = {
    "output_dir": "outputs/doser_selector_components",
    "checkpoint_name": "doser_selector_components_gt.ckpt",
    "critic_path": None,
    "require_critic_checkpoint": True,
    "max_train_episodes": None,
    "train_epochs": 500,
    "max_train_steps": None,
    "max_batches_per_epoch": None,
    "calibration_batches": None,
    "validation_batches": None,
    "lr": 1.0e-4,
    "weight_decay": 1.0e-6,
    "detector_hidden_dim": 256,
    "dynamics_hidden_dim": 256,
    "value_hidden_dim": 256,
    "time_embed_dim": 32,
    "score_samples": 8,
    "action_image_feat_dim": 64,
    "action_pcd_feat_dim": 64,
    "dynamics_image_feat_dim": 64,
    "dynamics_pcd_feat_dim": 64,
    "predict_delta": True,
    "split_action_detectors": False,
    "action_loss_weight": 1.0,
    "dynamics_loss_weight": 1.0,
    "dynamics_state_loss_weight": 0.7,
    "dynamics_qpos_loss_weight": 0.3,
    "state_loss_weight": 1.0,
    "value_loss_weight": 1.0,
    "value_expectile": 0.7,
    "value_use_subgoal": True,
}


def _instantiate_policy_and_data(cfg, pre_cfg, device):
    pretrain_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(pretrain_cfg.policy):
        pretrain_cfg.policy.branch_selector = "err"
    return _instantiate_doser_policy_and_data(pretrain_cfg, pre_cfg, device)


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    module.train(trainable)
    module.requires_grad_(trainable)


def _limit_reached(cfg, step: int, batch_idx: int) -> bool:
    if cfg.max_train_steps is not None and step >= int(cfg.max_train_steps):
        return True
    return (
        cfg.max_batches_per_epoch is not None
        and batch_idx >= int(cfg.max_batches_per_epoch)
    )

########################################### 核心 ##########################################
def _build_components(dims: Dict, cfg, device: torch.device):
    if dims["image_shape"] is None:
        raise RuntimeError("Ground-truth DOSER pretraining requires image observations.")
    if bool(cfg.split_action_detectors):
        action_detector = torch.nn.ModuleDict({
            "A": FullStateActionDetector(
                state_dim=dims["state_dim"],
                subgoal_dim=dims["subgoal_dim"],
                qpos_dim=0,
                action_eval_dim=dims["action_eval_dim"],
                pcd_dim=dims["pcd_dim"],
                image_shape=None,
                image_feat_dim=0,
                pcd_feat_dim=cfg.action_pcd_feat_dim,
                hidden_dim=cfg.detector_hidden_dim,
                time_embed_dim=cfg.time_embed_dim,
                score_samples=cfg.score_samples,
            ),
            "B2": FullStateActionDetector(
                state_dim=0,
                subgoal_dim=dims["subgoal_dim"],
                qpos_dim=dims["qpos_dim"],
                action_eval_dim=dims["action_eval_dim"],
                pcd_dim=0,
                image_shape=dims["image_shape"],
                image_feat_dim=cfg.action_image_feat_dim,
                pcd_feat_dim=0,
                hidden_dim=cfg.detector_hidden_dim,
                time_embed_dim=cfg.time_embed_dim,
                score_samples=cfg.score_samples,
            ),
        }).to(device)
    else:
        action_detector = FullStateActionDetector(
            state_dim=dims["state_dim"],
            subgoal_dim=dims["subgoal_dim"],
            qpos_dim=dims["qpos_dim"],
            action_eval_dim=dims["action_eval_dim"],
            pcd_dim=dims["pcd_dim"],
            image_shape=dims["image_shape"],
            image_feat_dim=cfg.action_image_feat_dim,
            pcd_feat_dim=cfg.action_pcd_feat_dim,
            hidden_dim=cfg.detector_hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            score_samples=cfg.score_samples,
        ).to(device)
    dynamics_model = GroundTruthDynamicsModel(
        state_dim=dims["state_dim"],
        subgoal_dim=dims["subgoal_dim"],
        qpos_dim=dims["qpos_dim"],
        pcd_dim=dims["pcd_dim"],
        action_eval_dim=dims["action_eval_dim"],
        hidden_dim=cfg.dynamics_hidden_dim,
        pcd_feat_dim=cfg.dynamics_pcd_feat_dim,
        image_shape=dims["image_shape"],
        image_feat_dim=cfg.dynamics_image_feat_dim,
        predict_delta=cfg.predict_delta,
    ).to(device)
    state_detector = GroundTruthStateDetector(
        latent_dim=dims["state_dim"] + dims["qpos_dim"],
        hidden_dim=cfg.detector_hidden_dim,
        time_embed_dim=cfg.time_embed_dim,
        score_samples=cfg.score_samples,
    ).to(device)
    value_subgoal_dim = dims["subgoal_dim"] if bool(cfg.value_use_subgoal) else 0
    value_net = GroundTruthValueNet(
        latent_dim=dims["state_dim"] + dims["qpos_dim"],
        subgoal_dim=value_subgoal_dim,
        hidden_dim=cfg.value_hidden_dim,
    ).to(device)
    return action_detector, dynamics_model, state_detector, value_net


def _prepare_gt_batch(model, batch):
    common, action_eval, next_state, next_subgoal, next_image = (
        _prepare_selector_batch(model, batch)
    )
    qpos_pair = common.get("doser_qpos_pair", None)
    if qpos_pair is None:
        raise RuntimeError(
            "DOSER-GT pretraining requires Tr-aligned common['doser_qpos_pair']."
        )
    current_qpos = qpos_pair[:, 0]
    next_qpos = qpos_pair[:, 1]
    current_successor = torch.cat((common["state"], current_qpos), dim=-1)  # s
    next_successor = torch.cat((next_state, next_qpos), dim=-1)             # s'
    return (
        common,
        action_eval,
        next_state,
        current_qpos,
        next_qpos,
        current_successor,
        next_successor,
        next_subgoal,
        next_image,
    )


def _extract_b_action_eval(model, common):
    d3p_action_pair = common.get("d3p_action_pair", None)
    if d3p_action_pair is None:
        raise RuntimeError("Split B2 action detector requires common['d3p_action_pair'].")
    return model._extract_action_segment(
        d3p_action_pair[:, 0],
        0,
        model.Tr,
    )


def _action_denoising_loss(model, common, action_eval, action_detector, cfg):
    if not bool(cfg.split_action_detectors):
        return action_detector.denoising_loss(common, action_eval)
    action_B_eval = _extract_b_action_eval(model, common)   #   (B, Tr=8, 10)
    loss_A = action_detector["A"].denoising_loss(common, action_eval)
    loss_B = action_detector["B2"].denoising_loss(common, action_B_eval)
    return 0.5 * (loss_A + loss_B)


def _train_components(model, loader, components, cfg, device):
    action_detector, dynamics_model, state_detector, value_net = components
    _set_trainable(action_detector, True)
    _set_trainable(dynamics_model, True)
    _set_trainable(state_detector, True)
    _set_trainable(value_net, True)
    optimizer = AdamW(
        list(action_detector.parameters())
        + list(dynamics_model.parameters())
        + list(state_detector.parameters())
        + list(value_net.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    step = 0
    for epoch in range(int(cfg.train_epochs)):
        pbar = tqdm.tqdm(
            loader,
            desc=f"DOSER-GT joint pretrain epoch {epoch}",
            leave=False,
        )
        for batch_idx, batch in enumerate(pbar):
            if _limit_reached(cfg, step, batch_idx):
                if cfg.max_train_steps is not None and step >= int(cfg.max_train_steps):
                    return
                break
            # 准备训练数据
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            (
                common,
                action_eval,
                next_state,
                _,
                next_qpos,
                current_successor,
                next_successor,
                _,
                _,
            ) = _prepare_gt_batch(model, batch)
            # action_loss ###########
            action_loss = _action_denoising_loss(
                model,
                common,
                action_eval,
                action_detector,
                cfg,
            )
            # dynamics_loss ###########
            dynamics_out = dynamics_model(common, action_eval)
            dynamics_state_loss = F.mse_loss(
                dynamics_out["next_state"],
                next_state,
            )
            dynamics_qpos_loss = F.mse_loss(
                dynamics_out["next_qpos"],
                next_qpos,
            )
            dynamics_loss = (
                float(cfg.dynamics_state_loss_weight) * dynamics_state_loss
                + float(cfg.dynamics_qpos_loss_weight) * dynamics_qpos_loss
            )
            with torch.no_grad():
                q1, q2 = model.critic_target(
                    common["pcd"],
                    common["state"],
                    common["subgoal"],
                    action_eval.reshape(action_eval.shape[0], -1),
                )
                q_target = torch.minimum(q1.squeeze(-1), q2.squeeze(-1))
            # state_loss #########      信息：([state,qpos], [next_state,next_qpos])
            state_loss = state_detector.denoising_loss(
                torch.cat((current_successor, next_successor), dim=0)
            )
            # value_loss #########      信息：([state,qpos], subgoal)
            value_subgoal = common["subgoal"] if value_net.subgoal_dim > 0 else None
            value_pred = value_net(current_successor, value_subgoal)
            value_loss = value_net.expectile_loss(
                q_target - value_pred,
                float(cfg.value_expectile),
            )

            loss = (
                float(cfg.action_loss_weight) * action_loss
                + float(cfg.dynamics_loss_weight) * dynamics_loss
                + float(cfg.state_loss_weight) * state_loss
                + float(cfg.value_loss_weight) * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                action=f"{float(action_loss.detach().item()):.4f}",
                dyn=f"{float(dynamics_loss.detach().item()):.4f}",
                dyn_state=f"{float(dynamics_state_loss.detach().item()):.4f}",
                dyn_qpos=f"{float(dynamics_qpos_loss.detach().item()):.4f}",
                state=f"{float(state_loss.detach().item()):.4f}",
                value=f"{float(value_loss.detach().item()):.4f}",
            )
            step += 1

################## calibration #####################
@torch.no_grad()
def _calibrate(model, loader, components, cfg, device):
    action_detector, _, state_detector, _ = components
    for module in components:
        module.eval()
    split_action_detectors = bool(cfg.split_action_detectors)
    action_errors = {"A": [], "B2": []} if split_action_detectors else []
    state_errors = []
    pbar = tqdm.tqdm(loader, desc="Calibrating DOSER-GT detectors", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if cfg.calibration_batches is not None and batch_idx >= int(cfg.calibration_batches):
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        (
            common,
            action_eval,
            _,
            _,
            _,
            current_successor,  # s
            next_successor,     # s'
            _,
            _,
        ) = _prepare_gt_batch(model, batch)
        # 整个校准数据上的 action reference error 分布
        if split_action_detectors:
            action_B_eval = _extract_b_action_eval(model, common)
            action_errors["A"].append(
                action_detector["A"].reconstruction_error(
                    common,
                    action_eval,
                    cfg.score_samples,
                ).cpu()
            )
            action_errors["B2"].append(
                action_detector["B2"].reconstruction_error(
                    common,
                    action_B_eval,
                    cfg.score_samples,
                ).cpu()
            )
        else:
            action_errors.append(
                action_detector.reconstruction_error(
                    common,
                    action_eval,
                    cfg.score_samples,
                ).cpu()
            )
        # Reference errors over true [state, qpos] successors.
        state_errors.append(
            state_detector.reconstruction_error(
                torch.cat((current_successor, next_successor), dim=0),
                cfg.score_samples,
            ).cpu()
        )
    if split_action_detectors:
        action_errors = {
            "A": torch.cat(action_errors["A"]),
            "B2": torch.cat(action_errors["B2"]),
        }
    else:
        action_errors = torch.cat(action_errors)
    state_errors = torch.cat(state_errors)
    # action/state_errorserror 拉平、排序、存到 detector 自己的 buffer 里。
    if split_action_detectors:
        action_detector["A"].set_reference_errors(action_errors["A"])
        action_detector["B2"].set_reference_errors(action_errors["B2"])
    else:
        action_detector.set_reference_errors(action_errors)
    state_detector.set_reference_errors(state_errors)
    return action_errors, state_errors

################## validation #####################
@torch.no_grad()
def _validate(model, loader, components, cfg, device, state_percentile):
    _, dynamics_model, state_detector, value_net = components
    for module in components:
        module.eval()
    sums = {
        "val_dyn_successor_mse": 0.0,
        "val_dyn_state_mse": 0.0,
        "val_dyn_qpos_mse": 0.0,
        "val_dyn_successor_copy_mse": 0.0,
        "val_dyn_state_copy_mse": 0.0,
        "val_dyn_qpos_copy_mse": 0.0,
        "val_state_id_agreement": 0.0,
        "val_true_state_id_self_agreement": 0.0,
        "val_pred_state_id_rate": 0.0,
        "val_true_state_id_rate": 0.0,
        "val_value_pred_true_next_mae": 0.0,
        "val_value_current_q_mae": 0.0,
    }
    count = 0
    q_values = []
    pbar = tqdm.tqdm(loader, desc="Validating DOSER-GT dynamics", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if cfg.validation_batches is not None and batch_idx >= int(cfg.validation_batches):
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        (
            common,
            action_eval,
            next_state,
            current_qpos,
            next_qpos,
            current_successor,
            next_successor,
            _,
            _,
        ) = _prepare_gt_batch(model, batch)
        dynamics_out = dynamics_model(common, action_eval)
        pred_next_state = dynamics_out["next_state"]
        pred_next_qpos = dynamics_out["next_qpos"]
        pred_successor = dynamics_out["successor"]

        pred_state_score = state_detector.score(pred_successor)
        true_state_score = state_detector.score(next_successor)
        true_state_score_repeat = state_detector.score(next_successor)
        pred_state_id = pred_state_score["percentile"] <= float(state_percentile)
        true_state_id = true_state_score["percentile"] <= float(state_percentile)
        true_state_id_repeat = (
            true_state_score_repeat["percentile"] <= float(state_percentile)
        )

        value_subgoal = common["subgoal"] if value_net.subgoal_dim > 0 else None
        pred_next_value = value_net(pred_successor, value_subgoal)
        true_next_value = value_net(next_successor, value_subgoal)
        current_value = value_net(current_successor, value_subgoal)
        q1, q2 = model.critic_target(
            common["pcd"],
            common["state"],
            common["subgoal"],
            action_eval.reshape(action_eval.shape[0], -1),
        )
        q_target = torch.minimum(q1.squeeze(-1), q2.squeeze(-1))
        q_values.append(q_target.detach().cpu())

        metrics = {
            "val_dyn_successor_mse": F.mse_loss(
                pred_successor,
                next_successor,
            ),
            "val_dyn_state_mse": F.mse_loss(pred_next_state, next_state),
            "val_dyn_qpos_mse": F.mse_loss(pred_next_qpos, next_qpos),
            "val_dyn_successor_copy_mse": F.mse_loss(
                current_successor,
                next_successor,
            ),
            "val_dyn_state_copy_mse": F.mse_loss(
                common["state"],
                next_state,
            ),
            "val_dyn_qpos_copy_mse": F.mse_loss(
                current_qpos,
                next_qpos,
            ),
            "val_state_id_agreement": (pred_state_id == true_state_id).float().mean(),
            "val_true_state_id_self_agreement": (
                true_state_id == true_state_id_repeat
            ).float().mean(),
            "val_pred_state_id_rate": pred_state_id.float().mean(),
            "val_true_state_id_rate": true_state_id.float().mean(),
            "val_value_pred_true_next_mae": (pred_next_value - true_next_value).abs().mean(),   #V(pred_s') 与 V(true_s') 的差距
            "val_value_current_q_mae": (current_value - q_target).abs().mean(),
        }

        batch_size = int(action_eval.shape[0])
        for key, value in metrics.items():
            sums[key] += float(value.item()) * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("DOSER-GT validation produced no samples.")
    metrics = {key: value / float(count) for key, value in sums.items()}
    q_values = torch.cat(q_values).float()
    metrics.update({
        "val_q_mean": float(q_values.mean().item()),
        "val_q_min": float(q_values.min().item()),
        "val_q_max": float(q_values.max().item()),
    })
    copy_mse = metrics["val_dyn_successor_copy_mse"]
    metrics["val_dyn_vs_copy_ratio"] = (
        metrics["val_dyn_successor_mse"] / copy_mse
        if copy_mse > 0.0
        else float("inf")
    )
    metrics["val_samples"] = float(count)
    print("DOSER-GT validation metrics:")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics)))
    return metrics


def _save(cfg, pre_cfg, dims, components, reference_errors, metrics):
    action_detector, dynamics_model, state_detector, value_net = components
    demo_count = _resolve_pretrain_demo_count(cfg, pre_cfg)
    split_action_detectors = bool(pre_cfg.split_action_detectors)
    # 创建输出目录
    output_dir = pathlib.Path(str(pre_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(pre_cfg.checkpoint_name)
    if (
        split_action_detectors
        and checkpoint_name == str(DEFAULT_PRETRAIN_CFG["checkpoint_name"])
    ):
        checkpoint_name = "doser_selector_components_gt_split.ckpt"
    output_path = output_dir / _checkpoint_name_with_demo_count(
        checkpoint_name,
        demo_count,
    )

    # 所有网络参数移到 CPU，这样 checkpoint 不会绑定当前 GPU
    def cpu_state_dict(module):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in module.state_dict().items()
        }
    
    # 网络结构
    metadata = {
        **dims,
        "format_version": 2,
        "selector_type": "ground_truth_state",
        "successor_dim": int(dims["state_dim"] + dims["qpos_dim"]),
        "detector_hidden_dim": int(pre_cfg.detector_hidden_dim),
        "dynamics_hidden_dim": int(pre_cfg.dynamics_hidden_dim),
        "value_hidden_dim": int(pre_cfg.value_hidden_dim),
        "time_embed_dim": int(pre_cfg.time_embed_dim),
        "score_samples": int(pre_cfg.score_samples),
        "action_image_feat_dim": int(pre_cfg.action_image_feat_dim),
        "action_pcd_feat_dim": int(pre_cfg.action_pcd_feat_dim),
        "dynamics_image_feat_dim": int(pre_cfg.dynamics_image_feat_dim),
        "dynamics_pcd_feat_dim": int(pre_cfg.dynamics_pcd_feat_dim),
        "predict_delta": bool(pre_cfg.predict_delta),
        "value_subgoal_dim": int(value_net.subgoal_dim),
        "max_train_episodes": demo_count,
        "action_detector_mode": "branch" if split_action_detectors else "shared",
    }

    state_dicts = {
        "dynamics_model": cpu_state_dict(dynamics_model),
        "state_detector": cpu_state_dict(state_detector),
        "value_net": cpu_state_dict(value_net),
    }
    if split_action_detectors:
        state_dicts["action_detector_A"] = cpu_state_dict(action_detector["A"])
        state_dicts["action_detector_B2"] = cpu_state_dict(action_detector["B2"])
    else:
        state_dicts["action_detector"] = cpu_state_dict(action_detector)

    payload = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "doser_gt_pretrain_cfg": OmegaConf.to_container(pre_cfg, resolve=True),
        "metadata": metadata,
        "state_dicts": state_dicts,
        "reference_errors": {
            "action": reference_errors[0],  # 未排序
            "state": reference_errors[1],   # 未排序
        },
        "validation_metrics": metrics,
    }
    torch.save(payload, output_path.open("wb"), pickle_module=pickle)
    return output_path

##########################入口##############################################################################
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent / "hiera_diffusion_policy" / "config"),
) # 加载主配置 cfg 传入 main()

def main(cfg):
    OmegaConf.resolve(cfg)# 创建配置 DEFAULT_PRETRAIN_CFG 传入 cfg.doser_pretrain.xxxxxx = xxxx
    pre_cfg = OmegaConf.merge(
        OmegaConf.create(DEFAULT_PRETRAIN_CFG),
        cfg.get("doser_gt_pretrain", {}),
    )

    if not 0.0 < float(pre_cfg.value_expectile) < 1.0:
        raise ValueError("doser_gt_pretrain.value_expectile must be in (0, 1).")
    if int(pre_cfg.train_epochs) <= 0:
        raise ValueError("doser_gt_pretrain.train_epochs must be positive.")
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
        raise RuntimeError("critic_path is required for DOSER-GT ValueNet targets.")
    if critic_path is not None:
        matched = _load_model_prefixes(
            model,
            critic_path,
            prefixes=("critic.", "critic_target."),# 只提取 Critic 参数
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
    # Fixed state/qpos coordinates allow all four independent modules to train
    # in one pass over the dataset.
    _train_components(model, train_loader, components, pre_cfg, device)
    # calibration
    reference_errors = _calibrate(
        model,
        calib_loader,
        components,
        pre_cfg,
        device,
    )
    # validation
    selector_cfg = cfg.policy.get("doser_gt_selector", {})
    metrics = _validate(
        model,
        val_loader,
        components,
        pre_cfg,
        device,
        state_percentile=float(selector_cfg.get("state_ood_percentile", 0.95)),
    )

    output_path = _save(
        cfg,
        pre_cfg,
        dims,
        components,
        reference_errors,
        metrics,
    )
    print(f"Saved DOSER-GT selector components to: {output_path}")


if __name__ == "__main__":
    main()
