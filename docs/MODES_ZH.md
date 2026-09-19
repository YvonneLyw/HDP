# 模式与开关速查

这是读代码时按需查阅的技术参考。第一次了解项目，先看[项目导读](GETTING_STARTED_ZH.md)即可。

这份说明按当前代码整理，主要针对 `hdp_d3p_can_mh` 的 A/B2 仿真主线。
先区分三个问题：**训练哪个条件分支、如何选候选动作、是否修改最终执行动作**。这些开关不是同一层。
源码索引：[Fusion policy](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py)、[MH 配置](../hiera_diffusion_policy/config/hdp_d3p_can_mh.yaml)、[训练 workspace](../hiera_diffusion_policy/workspace/train_workspace.py)。

## 1. A、B1、B2 和 SINGLE、SWITCH

A/B 共用同一个 actor，区别是输入条件；不能理解为两套独立训练的完整策略。

| 分支 | actor 获得的条件 | 用途 |
| --- | --- | --- |
| A | state、可选 pcd（`use_pcd=true`）、subgoal；extra_cond 为零 | HDP 条件路径 |
| B1 | 图像经 DKO 得到 latent action，再结合 subgoal 编成 extra_cond；其他条件置零 | DKO latent 条件消融 |
| B2 | 图像与 qpos 融合，再结合 subgoal 编成 extra_cond；其他条件置零 | 当前 A/B2 主线 |

见 [条件构造](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1099)。B2 仍依赖共享 subgoal，不能简单称为完全独立的视觉策略。

| 开关 | 训练阶段 | rollout 阶段 |
| --- | --- | --- |
| `policy.mode=SINGLE` | 使用 `single_branch` | 使用 `single_branch` |
| `policy.mode=SWITCH` | 每次按 `switch_prob_b` 选择 A 或指定 B 分支训练 | 同时生成 A/B 候选，由 `branch_selector` 选择 |

`switch_prob_b=0.5` 不代表推理有一半时间选 B。见 [训练分支采样](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1162)与[双分支推理](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1486)。
B 的训练条件包含 `t`、`t+4` 两时刻，两份 BC loss 各占 0.5；rollout 使用当前条件，不读取未来观测。
A 候选执行起点默认索引 1，B 为索引 0；比较动作时要先按执行时刻对齐，见 [action 对齐](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L553)。

## 2. 当前 YAML 默认不是最小基线

运行 Hydra 配置时，应以解析后的配置、入口脚本覆盖和实际加载的 checkpoint 为准。下表的 Python 默认指直接构造 Fusion 类时的默认值。

| 开关 | 当前 `hdp_d3p_can_mh.yaml` | Fusion Python 默认 |
| --- | --- | --- |
| `mode` / `single_branch` / `b_branch` | `SWITCH` / `A` / `B2` | `SINGLE` / `A` / `B1` |
| `switch_prob_b` | `0.5` | `0.5` |
| `branch_selector` | `doser_gt` | `err` |
| `b_branch_use_q_loss` | `false` | `false` |
| `use_koopman_aux` | `true` | `false` |
| `use_action_smoothing` | `false` | `false` |
| `use_test_time_aggregation` | `false` | `false` |
| `test_time_agg_beta` / `test_time_agg_tau` | `0.97` / `0.1` | `0.97` / `0.1` |
| `d3p_rollout_error_samples` | `10` | `10` |
| `doser_critic_refresh.enabled` | `true` | `false` |
| `fusion_debug_checks` | `true` | `true` |

DOSER 子配置另有自己的构造默认；当前 YAML 将 action/state percentile 阈值均设为 `0.98`，Q/V margin 均设为 `0.05`，fallback 为 A。
主线时间参数为 `horizon=16`、`n_action_steps=4`、`Tr=8`、`d3p_query_every=4`；最后一个参数当前强制为 4。

## 3. 选择器只回答“选哪个候选”

| `branch_selector` | 决策依据 | 额外依赖 |
| --- | --- | --- |
| `err` | 候选在 actor 下的扩散重构误差，越小越优 | 不需 DOSER 组件 |
| `q` | critic 的 Q，越大越优 | 匹配的 critic |
| `hybrid_gate` | error 相对差超过 0.25 用 error，否则 Q | critic；0.25 当前写在代码里 |
| `hybrid_linear` | 归一化 Q 减 error | critic；两项权重当前均为 1 |
| `doser_latent` | action 支持度、Q、预测 latent 的支持度和 V | 独立 latent 组件 checkpoint |
| `doser_gt` | 同一决策树，后继评估空间换成 state+qpos | GT 组件 checkpoint；支持共享或分支 detector |
| `doser_err` | 仅比较 A/B action detector 的校准 percentile，越小越优 | GT split-action-detector checkpoint |

基础选择器见 [实现](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1690)；DOSER 见 [决策树](../hiera_diffusion_policy/policy/doser_branch_selector.py#L328)。
DOSER 两个 action 都 ID 时比较 Q；出现 OOD 时还会看预测后继的支持度和 V，因此并非“发现一个 OOD 就一定选择另一个”。
`doser_err` 的阈值用于 ID/OOD 诊断；实际选谁只由 percentile 比较决定，见 [percentile 选择](../hiera_diffusion_policy/policy/doser_branch_selector.py#L277)。
`err` 小或 percentile 低不等于任务一定成功；必须同时检查仿真任务得分。

## 4. 把训练、选择和执行分开看

| 层次 | 开关 | 实际作用 |
| --- | --- | --- |
| 训练阶段 | `train_model`、入口脚本 | 决定训练 guider、critic 或 actor；注意下节的入口覆盖 |
| actor 训练 | `eta` | Q-loss 权重；`eta=0` 才关闭 actor 的 Q-loss |
| B 分支训练 | `b_branch_use_q_loss` | 是否也给 B 加 Q-loss；为 false 时 A 仍可使用 Q-loss |
| 表征与辅助训练 | `use_koopman_aux` | 控制 DKO；B1 的条件依赖 DKO，B2 可将其作为辅助训练项 |
| 推理候选选择 | `branch_selector` | 比较 A/B 候选；与训练 Q-loss 是独立设置 |
| 执行动作 | `use_action_smoothing` | 选择之后再平滑动作 |
| 历史候选选择 | `use_test_time_aggregation` | 根据分数聚合当前和历史预测，逐步选择执行候选 |
| 训练期间 Q/V 更新 | `doser_critic_refresh.*` | 使用离线 batch 刷新 critic，并可同步 V；不是 rollout 在线更新 |

当前 `eta=0.001` 且 `b_branch_use_q_loss=false` 表示 **B 不用 Q-loss，A 使用 Q-loss**，不是整个实验都没有 Q。
refresh 当前从 actor epoch 100 开始，每 100 epoch 运行；见 [workspace 更新位置](../hiera_diffusion_policy/workspace/train_workspace.py#L645)。
aggregation 会使用 `select_score_A/B` 重选历史候选，绕过 DOSER 的 `select_B` 硬决策；这会改变实验含义，见 [aggregation 分支](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1542)。
smoothing 与 aggregation 当前不能同时开启；初学阶段两者都关闭。

## 5. 常见的隐藏行为

1. **SINGLE 仍可能加载 DOSER 组件。** [构造器](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L180)按 selector 决定加载，不按 mode 决定；仅设 SINGLE/A 不能摆脱组件依赖。
2. **MH 的 latent 路径需要修正。** 当前 `doser_selector.components_path` 与 GT 路径同指 GT split checkpoint；切到 latent 前必须换成 latent 预训练产物。两种格式不可互换，见 [latent loader](../hiera_diffusion_policy/policy/doser_branch_selector.py#L56)和[GT loader](../hiera_diffusion_policy/policy/doser_gt_branch_selector.py#L11)。
3. **B 输入缺失会回退 A。** 缺图像或融合特征时，SINGLE B 与 SWITCH 都可能实际执行 A，见 [回退逻辑](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1168)。应检查运行日志，不能只看配置名称。
4. **B1 必须有 DKO。** Fusion 中 B1 配 `use_koopman_aux=false` 会报错；见 [B1 条件检查](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L1084)。
5. **`training.resume=true` 不代表自动续训。** 当前 [workspace 的自动 resume 块](../hiera_diffusion_policy/workspace/train_workspace.py#L106)被注释；后面的 guider/critic/actor 路径加载仍在，但不应等同于完整恢复训练。
6. **`train_hdp.py train_model=actor` 不能限制只训练 actor。** [多阶段入口](../train_hdp.py#L32)会重写 `train_model`，按 `use_subgoal`、`eta` 决定前置阶段，并覆盖 actor 阶段的 guider/critic 路径。
7. **refresh 不检查 SINGLE/SWITCH。** 只检查 enabled、selector 和 epoch；固定分支对照也应显式关闭，见 [调度条件](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_fusion.py#L223)。
8. **独立 B-only 类不是 Fusion SINGLE B 的别名。** [B-only 条件](../hiera_diffusion_policy/policy/hiera_diffusion_policy_d3p_bonly.py#L237)不在 extra_cond 中加入 subgoal，属于另一种消融。

## 6. 查阅配置时怎样对应到代码

配置中的模式决定走哪条实现路径；checkpoint 还决定模型实际学到的参数。DKO、模型维度和条件编码结构需要与相应 checkpoint 匹配。
固定某个分支是在同一个融合模型中选择一种条件路径，与单独训练一套 A-only/B-only 模型的含义不同。
本表用于解释已有代码；具体运行组合需要结合对应实验的配置理解。

## 7. DOSER latent 已有什么，还需验证什么

已有独立预训练的 action detector、latent dynamics、state detector 和 V；组件加载后冻结，见 [预训练入口](../pretrain_doser_selector_components.py)与[组件实现](../hiera_diffusion_policy/model/diffusion/doser_selector_components.py)。
latent 来自独立的 state/subgoal/image encoder，不是 B 分支的 DKO latent；已实现分阶段训练和 next-state 辅助预测。
GT 版本学习预测 `t+Tr` 的标准化 state+qpos；它没有在推理时直接获取真实未来状态，见 [GT dynamics](../hiera_diffusion_policy/model/diffusion/doser_gt_selector_components.py#L14)。
待验证：预测 latent 与真实后继编码的误差和分布差、离线与 rollout 的 percentile 校准差、Q/V 排序是否对应实际效果。
这些是研究问题，当前不能直接断言“latent 已经崩塌”或某个修补已经有效。
latent 与 GT dynamics 返回的 `uncertainty` 当前都固定为 0；仅打开 uncertainty gate 无法得到有意义的不确定性估计，见 [latent 输出](../hiera_diffusion_policy/model/diffusion/doser_selector_components.py#L433)与[GT 输出](../hiera_diffusion_policy/model/diffusion/doser_gt_selector_components.py#L228)。

## 8. QGF 留到基础选择验证之后

[Fusion QGF 配置](../hiera_diffusion_policy/config/hdp_d3p_can_mh_qgf.yaml)使用 `b2_q_guidance_mode=none/noisy/clean`，只在 rollout 修改 B2 候选的扩散采样；不是新的分支选择器。
`none` 会强制 guidance weight 为 0 并委托基础 Fusion；启用 guidance 要求 SWITCH+B2。训练 B 的 Q-loss、QGF 采样、DOSER 选择应分别做对照。
另有[原 HDP QGF 配置](../hiera_diffusion_policy/config/hdp_can_mh_qgf.yaml)，用 `q_guidance_mode`，还含 BFN/IQL 设置；不要与 Fusion 的 `b2_q_guidance_mode` 混用。
