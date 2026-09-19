# HDP-D3P 融合定稿方案（v1）

## 0. 目标与边界
- 目标：保持 HDP 的 hierarchical contact-guided 框架不变，在 low-level actor 中引入 D3P 风格双分支鲁棒机制。
- 当前阶段优先级：先跑通数据链路与 B-lite，再上 DKO 正式分支。
- 不改项（直到你明确下令）：guider/critic 主体结构、env runner、rollout 主流程。

## 1. 统一数据契约（当前 dataset 输出）

### 1.1 HDP 旧字段（保留）
- `id`: `(1,)`
- `scene_pcd`: `(1024, 3)`
- `object_pcd`: `(1024, 3)`
- `pcd`: `(obs_history_num, 1024, 3)`
- `state`: `(obs_history_num, 27)`
- `action`: `(horizon, 10)`
- `next_pcd`: `(obs_history_num, 1024, 3)`
- `next_state`: `(obs_history_num, 27)`
- `next_action`: `(horizon, 10)`
- `subgoal`: `(8,)`
- `next_subgoal`: `(8,)`
- `reward`: `(1,)`

### 1.2 D3P 新字段（新增，不覆盖旧字段）
- `image`: `(2, 2, 3, 84, 84)`
- `qpos`: `(2, 9)`
- `obs_is_pad`: `(2,)`
- `d3p_action_pair`: `(2, L, 10)`，`L=d3p_action_chunk_len`（默认 `horizon`）
- `act_is_pad_pair`: `(2, L)`

### 1.3 字段语义（必须固定）
- `image[0,0]`: `t` 的 front
- `image[0,1]`: `t` 的 wrist
- `image[1,0]`: `t+h` 的 front
- `image[1,1]`: `t+h` 的 wrist
- `h = d3p_query_every`
- `qpos[k] = concat(robot0_joint_pos(7), robot0_gripper_qpos(2))`
- `obs_is_pad[1]=True` 当 `t+h` 越界（值复制最后有效帧）
- `act_is_pad_pair` 是动作 chunk 的 mask，不是观测 mask

### 1.4 预处理规范
- 图像：`HWC uint8 -> CHW float32 -> /255` 到 `[0,1]`
- `qpos`：默认标准化 `(qpos-mean)/std`（mean/std 从 replay buffer 统计）
- 注意：当前 `pcd/next_pcd` 仍是 `float64`（后续可改 `float32` 节省显存）

## 2. 模型输入变量命名（统一到融合实现）
- `O_geo`: HDP 几何条件（最短路径定义：沿用现有 `pcd + state` 编码）
- `O_img`: `image` 中每时刻双相机图像
- `O_prop`: 机器人本体（本阶段使用 `qpos`）
- `C_t`: contact/guider 条件（沿用 HDP 的 subgoal/contact 条件）
- `A`: 动作 chunk（维度 `10`，EEF pose + gripper）

## 3. 网络架构定稿

### 3.1 高层（不改）
- `Guider`: 原始 HDP，不接 RGB。
- 输出 `C_t`（subgoal/contact condition）。

### 3.2 Critic（不改）
- 原始 HDP critic，用于 contact-conditioned Q-learning。
- 在 aggregation v2 中额外作为分支选择打分。

### 3.3 低层 Actor（改为双分支+共享扩散骨干）
- 共享：一个 diffusion actor backbone（建议继续用现有 actor 的 UNet/1D diffusion 主干）。
- 分支条件编码器输出同维 `cond_vec`，喂给同一个 backbone。

#### Branch A（HDP 原版）
- 输入：`O_geo + O_prop(HDP原state路径) + C_t`
- 实现：先模块化现有 low-level actor 作为 `BranchAEncoder`。

#### Branch B-lite（先验证 RGB）
- `VisionEncoder(O_img_t) -> f_v`
- `B1-lite`: `Fuse(f_v, C_t) -> cond_B1`
- `B2-lite`: `Fuse(f_v, O_prop_t, C_t) -> cond_B2`
- 说明：B-lite 阶段不引入 DKO 动力学损失。

#### Branch B 正式版（DKO）
- `VisionEncoder(O_img_t, O_img_{t+h}) -> (f_v_t, f_v_{t+h})`
- `DKO(f_v_t, f_v_{t+h}) -> f_u + dko_loss`
- `B1`: `Fuse(f_u, C_t) -> cond_B1`
- `B2`: `Fuse(f_u, O_prop_t, C_t) -> cond_B2`

## 4. 训练流程（分阶段）

### 阶段 C：HDP baseline 复现
- 只开 Branch A 路径，记录 success/contact error/rollout。

### 阶段 D：Branch A 模块化
- 把原 actor 条件编码抽成 `BranchAEncoder`，行为不变。

### 阶段 E-F：A/B-lite + switching training
- 每个 batch 采样一个 `branch_id in {A, B1-lite, B2-lite}`。
- 只计算当前分支条件下 diffusion loss，更新：
  - 共享 diffusion backbone 参数
  - 当前分支 encoder 参数
- 先做消融再做混合：
  - `A only`
  - `B1-lite only`
  - `B2-lite only`
  - `A+B1+B2 switching`

### 阶段 I-J：引入 DKO 并替换 B-lite
- 增加 `dko_loss`，总损失：
  - `L = L_diff + lambda_dko * L_dko`
- 用 B1/B2 正式版替换 B-lite。

## 5. 推理与分支选择

### 5.1 aggregation v1（先上）
- 每个候选分支 `b`：
  - 生成动作 `A_b`
  - 计算 test-time DDPM error `E_b`
- 选 `argmin_b E_b`

### 5.2 aggregation v2（正式）
- 每个候选分支 `b`：
  - `A_b` from branch `b`
  - `E_b` = test-time diffusion error
  - `Q_b` = critic value（同一状态条件下评估 `A_b`）
- 分支内归一化（同一步三分支上做）：
  - `Qn_b = normalize(Q_b)`
  - `En_b = normalize(E_b)`
- 打分：
  - `Score_b = Qn_b - alpha * En_b`
- 选 `argmax_b Score_b`
- `alpha` 初值建议：`0.5`，再网格搜索 `{0.25, 0.5, 1.0}`。

## 5.3 Loss 定稿（A/B 可比口径）
- 结论 1：`A` 分支不引入 `DKO`，保持其作为 HDP 几何/接触分支的语义纯净。
- 结论 2：`B` 分支引入 `DKO`，图像增强项先不加（后续可做消融补充）。
- 结论 3：用于分支选择时，不直接比较训练总 loss；统一用 `Q + DDPM error` 打分（见 5.2）。

### 当前可执行版本（先跑通）
- `A`：单时刻 BC + Q（沿用当前 HDP actor 训练口径）。
- `B`：双时刻 BC + Q（先不加 DKO 也可先跑通）。

### 正式版本（进入 DKO 阶段）
- `A` 分支训练目标（建议升级到双时刻，保证与 B 的时间口径一致）：
  - `L_A = 0.5 * L_bc^t + 0.5 * L_bc^{t+h} + eta * L_q`
- `B` 分支训练目标：
  - `L_B = 0.5 * L_bc^t + 0.5 * L_bc^{t+h} + lambda_dko * L_dko + eta * L_q`
- 超参建议：
  - `lambda_dko` 初值 `0.1~0.3`（不要一开始过大）
  - `eta` 与 HDP baseline 保持一致，避免引入额外变量

### 可比性说明
- `A/B` 训练项不必完全同构（`B` 多一个 `DKO` 正则是允许的）。
- 分支选择可比性通过统一推理打分保证：
  - `Score_b = normalize(Q_b) - alpha * normalize(E_b)`
  - 其中 `E_b` 是 test-time DDPM error。

## 6. 关于 t+h 与 next_xx 的最终选择
- `next_xx`（HDP）与 `(t,t+h)`（D3P）语义不同。
- 结论：
  - 训练 HDP 主路径仍按原 `next_xx` 语义。
  - D3P 分支严格使用显式 `(t, t+h)`，由 `d3p_query_every` 控制。
- 建议：保留两个独立间隔参数，不强绑同一个量。

## 7. 关于 pad 量的最终选择
- `obs_is_pad`: 只对应观测对 `(t,t+h)` 是否越界。
- `act_is_pad_pair`: 只对应动作 chunk 的逐步 padding mask。
- 不再输出 `is_pad` 别名，避免歧义。

## 8. 实验矩阵（最终）
- Baseline: `A only`
- RGB验证: `B1-lite only`, `B2-lite only`, `A+B-lite switching + agg v1`
- 正式版: `A+B(DKO) switching + agg v1/v2`
- 指标：
  - success rate
  - recovery rate
  - contact error
  - branch selection ratio

## 9. 实现原则（固定）
- 最短路径优先，先跑通再加复杂项。
- 不做补丁式分叉：统一字段契约、统一分支接口。
- 先证据后扩展：每阶段必须有可复现实验输出再进入下一阶段。
