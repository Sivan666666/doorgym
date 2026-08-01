# A2W WC4 ACT / DP / DP3：数据规模、数据多样性及 EE/Joint 控制评测汇总

日期：2026-07-20，更新：2026-07-21  
任务：汇总使用 WC4 depth-only 专家轨迹训练的 ACT、LeRobot Diffusion Policy 和官方 Full DP3，比较不同数据规模、跨门数据多样性，以及 EE10/Joint9 动作表示在 seen WC4 和 unseen Fire Door 上的成功率。

## 0. 导师汇报摘要

### 0.1 核心结果

为保证数据规模比较公平，本节统一采用训练 seed `1000` 和同一组 WC4 评测 seeds `615455575～615455578`，每个模型评测 `64 trials`。成功条件均为 `abs(door_hinge_angle) >= 80°`。ACT 与 LeRobot DP 的评测命令使用 horizon 25；官方 Full DP3 使用表中单独标注的、受其 `n_action_steps=16` 限制的 horizon。

| 模型 | WC4-only 50 | WC4 50 + 其他4门50 | WC4 50 + 单一其他门50 | WC4-only 200 | 50→200 提升 |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 31/64 = 48.44% | 34/64 = 53.13% | 26/64 = 40.63% | 40/64 = 62.50% | **+14.06 pp** |
| Plücker ACT FOV55 | 32/64 = 50.00% | 35/64 = 54.69% | 45/64 = 70.31% | 45/64 = 70.31% | **+20.31 pp** |
| Plücker + Interaction | 34/64 = 53.13% | 43/64 = 67.19% | 44/64 = 68.75% | 53/64 = 82.81% | **+29.68 pp** |

![WC4 50条到200条训练数据的成功率提升](figures/a2w_wc4_50_to_200_success.png)

*图0-1：在相同训练 seed 和评测 seeds 下，三种 EE10 双深度 ACT 从50条 WC4 数据增加到200条后的成功率变化。*

### 0.2 200条数据各模型结果

下表统一报告 `base_seed=615455575` 的 64 次 WC4 评测。不同观测和动作表示之间不是严格单变量对照，因此应在各自分组内比较。

| 观测与动作表示 | 模型 | 训练 seed | WC4成功率 |
|---|---|---:|---:|
| 双深度 + EE10 | Baseline ACT | 1000 | 40/64 = 62.50% |
| 双深度 + EE10 | Baseline + Interaction | 1000 | 41/64 = 64.06% |
| 双深度 + EE10 | Plücker ACT FOV55 | 1000 | 45/64 = 70.31% |
| 双深度 + EE10 | Plücker + Interaction | 1000 | **53/64 = 82.81%** |
| 双深度 + Joint9 | Baseline ACT | 1000 | 41/64 = 64.06% |
| 双深度 + Joint9 | Plücker + Interaction | 1000 | **64/64 = 100.00%** |
| Front点云 + EE10 | DP3 Global Encoder + ACT | 1000 | 10/64 = 15.63% |
| Front点云 + EE10 | OBSBench Local ACT | 1000 | **62/64 = 96.88%** |
| Front点云 + EE10 | OBSBench Local + Interaction | 1000 | 61/64 = 95.31% |
| Front点云 + EE10 | OBSBench Local ACT | 2000 | 30/64 = 46.88% |
| 双深度 + EE10 | LeRobot Diffusion Policy | 1000 | 33/64 = 51.56% |
| Front点云 + EE10 | 官方 Full DP3，最佳 horizon=16 | 42 | **49/64 = 76.56%** |
| Front点云 + Joint9 | OBSBench Local ACT | 1000 | 44/64 = 68.75% |
| Front点云 + Joint9 | OBSBench Local + Interaction | 1000 | 28/64 = 43.75% |
| 双深度 + Joint9 | LeRobot Diffusion Policy | 1000 | **42/64 = 65.63%** |
| Front点云 + Joint9 | 官方 Full DP3，horizon=4 | 42 | **42/64 = 65.63%** |

### 0.3 简要分析

1. **增加同门数据量最稳定有效。** 从50条增加到200条后，Baseline、Plücker、Plücker + Interaction 分别提升 `14.06`、`20.31` 和 `29.68 pp`；Interaction 模型对数据量增加最敏感。
2. **跨门数据的收益依赖模型结构。** 其他4门各取约12/13条时，Plücker + Interaction 提升到 `67.19%`；加入单一其他门50条时，Plücker 达到 `70.31%`，但 Baseline 反而降至 `40.63%`。因此多样性本身不保证提升，模型需要能够利用几何差异。
3. **200条 EE10 双深度模型中，Plücker + Interaction 最好。** 它达到 `82.81%`，相对同数据 Baseline 提高 `20.31 pp`。
4. **局部点云 token 有很高上限，但训练非常不稳定。** OBSBench Local ACT 在训练 seed 1000 达到 `96.88%`，换成 seed 2000 后只有 `46.88%`；两次评测 seeds 完全一致，说明主要问题是训练方差。当前不能把 `96.88%` 视为稳定结论。
5. **DP3 Global Encoder + ACT 的点云压缩过强。** 该 ACT 变体的单个全局点云 token 只有 `15.63%`，明显不如保留局部三维 token 的 OBSBench Local；这条结论不适用于第12章的官方 Full DP3 diffusion policy。点云 Interaction 目前也没有稳定正收益。
6. **官方 Full DP3 不等于 DP3 Global Encoder + ACT。** 前者是点云条件扩散策略；后者仍然以 ACT Transformer 为策略后端，只借用了 DP3 风格的 PointNet 全局编码器。
7. **200条 EE10 数据上，官方 Full DP3 优于 LeRobot DP。** 两者分别为 `49/64 = 76.56%` 和 `33/64 = 51.56%`，但视觉输入、历史长度和执行 horizon 不同，不是严格单变量对照。
8. **200条 Joint9 数据上，DP 与 DP3 本轮持平。** 两者都是 `42/64 = 65.63%`，仅 batch 分布不同；相对 Joint9 Baseline ACT 的 `41/64` 也只多成功一次。

## 1. 实验范围

本次汇总包含 6 个 50K ACT checkpoint：4 个 EE10 模型和 2 个 Joint9 模型。所有模型的训练 seed 均为 `1000`。

| 属性 | EE10 模型 | Joint9 模型 |
|---|---|---|
| 训练数据 | 200 条 WC4 depth-only 轨迹 | 200 条 WC4 depth-only 轨迹 |
| State / action 表示 | 10D：底盘命令、base-frame EE pose、夹爪 | 9D：底盘命令、6 个机械臂关节、夹爪 |
| 训练步数 | 50K | 50K |
| 训练 seed | 1000 | 1000 |
| ACT chunk size | 100 | 100 |
| 训练时执行步数配置 | 50 | 50 |

## 2. Checkpoint

| 动作表示 | 模型 | Checkpoint |
|---|---|---|
| EE10 | Baseline ACT | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_baseline_seed1000_50k_chunk100_exec50_bs16_0717/checkpoints/050000` |
| EE10 | Baseline + Interaction | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_baseline_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0717/checkpoints/050000` |
| EE10 | Plücker ACT FOV55 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_plucker_fov55_seed1000_50k_chunk100_exec50_bs16_0717/checkpoints/050000` |
| EE10 | Plücker + Interaction decoder-chunk | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0717/checkpoints/050000` |
| Joint9 | Baseline ACT | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_jointstate9_gymjacobian_200_baseline_seed1000_50k_chunk100_exec50_bs16_0719/checkpoints/050000` |
| Joint9 | Plücker + Interaction decoder-chunk | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_jointstate9_gymjacobian_200_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0719/checkpoints/050000` |

## 3. 评测参数

| 参数 | WC4 seen | Fire Door unseen |
|---|---:|---:|
| `num_envs` | 16 | 16 |
| 单组 trials | 64 | 64 |
| 汇总 trials | 128（两组 base seed） | 64（一组 base seed） |
| base seeds | 615455575、824731906 | 615455575 |
| 实际 batch seeds | 每个 base seed 连续取 4 个 seed | 615455575～615455578 |
| rollout steps | 1000 | 1200 |
| `dp_action_horizon` | 25 | 25 |
| `dp_fps` | 25 | 25 |
| 成功指标 | `abs(door_hinge_angle) >= 80°` | `abs(door_hinge_angle) >= 80°` |
| 深度噪声 | 关闭 | 关闭 |
| Gaussian blur | 关闭 | 关闭 |
| 相机随机化 | 位置 0.02 m、旋转 5° | 位置 0.02 m、旋转 5° |
| Robot pitch | 0，随机化关闭 | 0，随机化关闭 |
| Depth clip | 0.2～1.5 m | 0.2～1.5 m |

说明：WC4 结果汇总了两组评测 seed，共 128 次；Fire Door 当前只有一组评测 seed，共 64 次。因此跨门成功率差值用于描述趋势，不应直接视为严格配对统计量。

## 4. 总体结果

| 动作表示 | 模型 | WC4 seen | Fire Door unseen |
|---|---|---:|---:|
| EE10 | Baseline ACT | 81/128 = 63.28% | 19/64 = 29.69% |
| EE10 | Baseline + Interaction | 84/128 = 65.63% | 17/64 = 26.56% |
| EE10 | Plücker ACT FOV55 | 94/128 = 73.44% | **25/64 = 39.06%** |
| EE10 | Plücker + Interaction decoder-chunk | **106/128 = 82.81%** | 18/64 = 28.13% |
| Joint9 | Baseline ACT | 79/128 = 61.72% | **58/64 = 90.63%** |
| Joint9 | Plücker + Interaction decoder-chunk | **128/128 = 100.00%** | **57/64 = 89.06%** |

## 5. EE10 结果

### 5.1 WC4 与 Fire Door

| 模型 | WC4 成功率 | Fire Door 成功率 | Fire Door 相对 WC4 |
|---|---:|---:|---:|
| Baseline ACT | 81/128 = 63.28% | 19/64 = 29.69% | -33.59 pp |
| Baseline + Interaction | 84/128 = 65.63% | 17/64 = 26.56% | -39.07 pp |
| Plücker ACT FOV55 | 94/128 = 73.44% | **25/64 = 39.06%** | -34.38 pp |
| Plücker + Interaction decoder-chunk | **106/128 = 82.81%** | 18/64 = 28.13% | -54.68 pp |

### 5.2 Fire Door 分批结果

| 模型 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | 合计 |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 4/16 | 5/16 | 7/16 | 3/16 | 19/64 = 29.69% |
| Baseline + Interaction | 6/16 | 3/16 | 5/16 | 3/16 | 17/64 = 26.56% |
| Plücker ACT FOV55 | 7/16 | 9/16 | 6/16 | 3/16 | **25/64 = 39.06%** |
| Plücker + Interaction decoder-chunk | 6/16 | 5/16 | 4/16 | 3/16 | 18/64 = 28.13% |

### 5.3 EE10 模块增益

| 对比 | WC4 变化 | Fire Door 变化 |
|---|---:|---:|
| Baseline → Baseline + Interaction | +2.35 pp | -3.13 pp |
| Baseline → Plücker | +10.16 pp | +9.37 pp |
| Plücker → Plücker + Interaction | +9.37 pp | -10.94 pp |

## 6. Joint9 结果

### 6.1 WC4 两组 seed

| 模型 | base seed 615455575 | base seed 824731906 | 两轮合计 |
|---|---:|---:|---:|
| Baseline ACT | 41/64 = 64.06% | 38/64 = 59.38% | 79/128 = 61.72% |
| Plücker + Interaction decoder-chunk | 64/64 = 100.00% | 64/64 = 100.00% | **128/128 = 100.00%** |

### 6.2 Fire Door 分批结果

| 模型 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | 合计 |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 16/16 | 14/16 | 14/16 | 14/16 | **58/64 = 90.63%** |
| Plücker + Interaction decoder-chunk | 13/16 | 15/16 | 15/16 | 14/16 | **57/64 = 89.06%** |

### 6.3 WC4 与 Fire Door

| 模型 | WC4 成功率 | Fire Door 成功率 | Fire Door 相对 WC4 |
|---|---:|---:|---:|
| Baseline ACT | 79/128 = 61.72% | 58/64 = 90.63% | +28.91 pp |
| Plücker + Interaction decoder-chunk | 128/128 = 100.00% | 57/64 = 89.06% | -10.94 pp |

## 7. 结论

1. 在 EE10 模型中，Plücker 对 WC4 和 Fire Door 都有稳定正收益：相对 Baseline 分别提升 `10.16 pp` 和 `9.37 pp`。
2. EE10 的 Interaction decoder-chunk 具有明显的 seen-door specialization。它在 WC4 上提高成功率，但在 Fire Door 上使 Baseline 和 Plücker 分别下降 `3.13 pp` 与 `10.94 pp`。
3. Joint9 的 Plücker + Interaction 在两组 WC4 seed 上均达到 `64/64`，是当前 WC4 评测中最稳定的模型。
4. 两个 Joint9 模型在 Fire Door 上都接近 90%，明显高于四个 EE10 模型。该差异同时包含动作表示、模型结构和闭环误差传播方式的变化，不能只归因于 Plücker 或 Interaction。
5. Joint9 Baseline 在 Fire Door 上高于 WC4，说明当前 Fire Door 的几何、机器人初始位姿及关节轨迹可能恰好更适配该策略；还需要增加 Fire Door base seed 和其他 unseen door 才能判断这种优势是否稳定。
6. 当前结果不支持“Interaction 一定提高 unseen 泛化”。对 EE10，它在 Fire Door 上是负收益；对 Joint9，Fire Door 上两个模型只相差 1 次成功，无法证明 Interaction 带来提升。

## 8. Fire Door 日志位置

| 模型 | 日志目录 |
|---|---|
| Joint9 Baseline | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_jointstate9_baseline50k_seed615455575` |
| Joint9 Plücker + Interaction | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_jointstate9_plucker_interaction50k_seed615455575` |
| EE10 Baseline，前 32 次 | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eecommand_wc4_200_baseline50k_seed615455575` |
| EE10 Baseline，后 32 次补跑 | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eecommand_wc4_200_baseline50k_seed615455577_part2_fixed` |
| EE10 Baseline + Interaction，前 32 次 | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eecommand_wc4_200_baseline_interaction50k_seed615455575` |
| EE10 Baseline + Interaction，后 32 次补跑 | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eecommand_wc4_200_baseline_interaction50k_seed615455577_part2_fixed` |
| EE10 Plücker | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eecommand_wc4_200_plucker50k_seed615455575` |
| EE10 Plücker + Interaction | `/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eecommand_wc4_200_plucker_interaction50k_seed615455575` |

## 9. 50/100 条数据规模与跨门多样性实验

### 9.1 实验范围

本节汇总早期使用 WC4 50 条数据及两种 100 条混合数据训练的 9 个 EE10 ACT checkpoint。所有模型均训练 50K step，训练 seed 为 `1000`。

| 数据设置 | Episodes | 数据组成 | 模型数量 |
|---|---:|---|---:|
| WC4-only 50 | 50 | WC4 50 条 | 3 |
| WC4 50 + Other-4Door 50 | 100 | WC4 50 条；其余四扇 PartNet 门各约 12/13 条 | 3 |
| WC4 50 + Single-Door 50 | 100 | WC4 50 条；门 `99650089960001` 50 条 | 3 |

统一评测参数如下：

| 参数 | 数值 |
|---|---:|
| 评测门 | WC4 |
| `base_seed` | 615455575 |
| 实际 batch seeds | 615455575～615455578 |
| `num_envs` | 16 |
| `total_trials` | 64 |
| rollout steps | 1000 |
| `dp_action_horizon` | 25 |
| `dp_fps` | 25 |
| 成功指标 | `abs(door_hinge_angle) >= 80°` |

说明：这 9 个 checkpoint 当时只完成了统一的 WC4 评测，没有统一的 unseen Fire Door 结果。

### 9.2 总体结果

| 训练数据 | 模型 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | WC4 合计 |
|---|---|---:|---:|---:|---:|---:|
| WC4-only 50 | Baseline ACT | 8/16 | 6/16 | 7/16 | 10/16 | 31/64 = 48.44% |
| WC4-only 50 | Plücker ACT FOV55 | 8/16 | 6/16 | 9/16 | 9/16 | 32/64 = 50.00% |
| WC4-only 50 | Plücker + Interaction | 10/16 | 7/16 | 9/16 | 8/16 | 34/64 = 53.13% |
| WC4 50 + Other-4Door 50 | Baseline ACT | 7/16 | 7/16 | 8/16 | 12/16 | 34/64 = 53.13% |
| WC4 50 + Other-4Door 50 | Plücker ACT FOV55 | 7/16 | 9/16 | 9/16 | 10/16 | 35/64 = 54.69% |
| WC4 50 + Other-4Door 50 | Plücker + Interaction | 10/16 | 11/16 | 11/16 | 11/16 | **43/64 = 67.19%** |
| WC4 50 + Single-Door 50 | Baseline ACT | 3/16 | 6/16 | 8/16 | 9/16 | 26/64 = 40.63% |
| WC4 50 + Single-Door 50 | Plücker ACT FOV55 | 11/16 | 10/16 | 11/16 | 13/16 | **45/64 = 70.31%** |
| WC4 50 + Single-Door 50 | Plücker + Interaction | 9/16 | 11/16 | 11/16 | 13/16 | 44/64 = 68.75% |

### 9.3 Checkpoint

| 训练数据 | 模型 | Checkpoint |
|---|---|---|
| WC4-only 50 | Baseline ACT | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4only50_baseline_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4-only 50 | Plücker ACT FOV55 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4only50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4-only 50 | Plücker + Interaction | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4only50_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4 50 + Other-4Door 50 | Baseline ACT | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_balanced100_baseline_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4 50 + Other-4Door 50 | Plücker ACT FOV55 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_balanced100_plucker_fov55_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4 50 + Other-4Door 50 | Plücker + Interaction | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_balanced100_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4 50 + Single-Door 50 | Baseline ACT | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_50_door99650089960001_50_baseline_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4 50 + Single-Door 50 | Plücker ACT FOV55 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_50_door99650089960001_50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |
| WC4 50 + Single-Door 50 | Plücker + Interaction | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_50_door99650089960001_50_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0716/checkpoints/050000` |

对应 LeRobot 数据集：

| 数据设置 | LeRobot repo_id |
|---|---|
| WC4-only 50 | `local/door_a2w_wc4_robotbasefull_contactcheck_50` |
| WC4 50 + Other-4Door 50 | `local/door_a2w_wc4_50_other4_balanced50_interaction_100` |
| WC4 50 + Single-Door 50 | `local/door_a2w_wc4_50_door99650089960001_50_interaction_100` |

### 9.4 相对 WC4-only 50 的数据多样性收益

| 模型 | WC4-only 50 | + Other-4Door 50 | 变化 | + Single-Door 50 | 变化 |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 48.44% | 53.13% | +4.69 pp | 40.63% | -7.81 pp |
| Plücker ACT FOV55 | 50.00% | 54.69% | +4.69 pp | 70.31% | **+20.31 pp** |
| Plücker + Interaction | 53.13% | 67.19% | **+14.06 pp** | 68.75% | +15.63 pp |

### 9.5 Interaction 模块在不同数据设置下的变化

| 训练数据 | Plücker | Plücker + Interaction | Interaction 相对变化 |
|---|---:|---:|---:|
| WC4-only 50 | 50.00% | 53.13% | +3.13 pp |
| WC4 50 + Other-4Door 50 | 54.69% | 67.19% | **+12.50 pp** |
| WC4 50 + Single-Door 50 | 70.31% | 68.75% | -1.56 pp |

### 9.6 结果分析

1. 9 个模型中，WC4 成功率最高的是 `WC4 50 + Single-Door 50` 的 Plücker ACT，达到 `45/64 = 70.31%`。
2. 四门混合数据对 Plücker + Interaction 的帮助最明显：相对 WC4-only 50 提升 `14.06 pp`，并且 Interaction 相对同数据 Plücker 提升 `12.50 pp`。
3. 单一其他门数据对模型的影响高度依赖架构：Baseline 下降 `7.81 pp`，但 Plücker 提升 `20.31 pp`。这说明仅加入跨门数据不会自动提升策略；模型还需要能够利用跨门几何变化。
4. Plücker 在两种混合数据中均优于对应 Baseline，尤其在 Single-Door 100 条数据上提升 `29.68 pp`，表明相机射线编码对跨门几何差异非常重要。
5. Interaction 并非始终有益。它在四门混合数据上收益最大，在 WC4-only 50 上只有小幅收益，在单一其他门数据上反而比纯 Plücker 低 `1.56 pp`。
6. 从当前单一训练 seed 和单一 WC4 评测 seed 还不能判断这些差异是否稳定。若用于正式结论，应至少补充 3 个训练 seed，并对这些 checkpoint 统一测试 Fire Door 和其他 unseen doors。

## 10. WC4 200 条数据：EE10 Front 点云 ACT

### 10.1 实验设置

本节使用 WC4 200 条专家轨迹，将 Front depth 反投影为 `robot_base` 坐标系下的 1024 点 XYZ 点云。模型输入为单路 Front 点云和 EE10 state，Wrist camera 关闭；动作仍为原来的 10D EE motion action。所有模型训练 50K step，ACT chunk size 为 100。

评测沿用本文的标准 WC4 配置：

```text
num_envs=16
total_trials=64
base_seed=615455575
实际 batch seeds=615455575～615455578
steps=1000
dp_action_horizon=25
dp_fps=25
success=abs(door_hinge_angle) >= 80°
depth_noise=off
camera_randomization=0.02 m / 5°
Front camera=on
Wrist camera=off
```

### 10.2 WC4 评测结果

| 模型 | 训练 seed | Batch 0 | Batch 1 | Batch 2 | Batch 3 | WC4 合计 |
|---|---:|---:|---:|---:|---:|---:|
| DP3 Global Encoder + ACT | 1000 | 4/16 | 3/16 | 2/16 | 1/16 | **10/64 = 15.63%** |
| OBSBench Local ACT | 1000 | 15/16 | 16/16 | 15/16 | 16/16 | **62/64 = 96.88%** |
| OBSBench Local + Interaction decoder-chunk | 1000 | 15/16 | 15/16 | 15/16 | 16/16 | **61/64 = 95.31%** |
| OBSBench Local ACT | 2000 | 9/16 | 8/16 | 5/16 | 8/16 | **30/64 = 46.88%** |

对应 checkpoint：

| 模型 | Checkpoint |
|---|---|
| DP3 Global Encoder + ACT，seed 1000 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_pcd_front_dp3global_exactdp3_50k_seed1000_0719_1844/checkpoints/050000` |
| OBSBench Local ACT，seed 1000 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_pcd_front_obsbenchlocal_postsample_exact_50k_seed1000_0719_1844/checkpoints/050000` |
| OBSBench Local + Interaction，seed 1000 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_pcd_front_obsbenchlocal_interaction_decoderchunk_w005_50k_seed1000_0719_2059/checkpoints/050000` |
| OBSBench Local ACT，seed 2000 | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_200_pcd_front_obsbenchlocal_postsample_exact_50k_seed2000_0720/checkpoints/050000` |

### 10.3 结果分析

1. `DP3 Global Encoder + ACT` 只有一个全局点云 token，成功率仅为 `15.63%`。这表明将整幅门场景压缩成单个全局向量会严重损失把手位置、门板结构和局部接触几何，不适合作为当前门操作任务的最终点云表示。
2. `OBSBench Local ACT` 使用局部点云 token 后，训练 seed 1000 达到 `62/64 = 96.88%`，说明保留局部三维结构的点云 ACT 在 WC4 上具有很高的性能上限。
3. 同为训练 seed 1000，加入 Interaction decoder-chunk 后从 `62/64` 变成 `61/64`，下降 1 次成功。该差异很小，当前结果不能证明 Interaction 对点云 ACT 有稳定收益。
4. OBSBench Local ACT 仅改变训练 seed，从 1000 换为 2000 后，成功率由 `96.88%` 降至 `46.88%`，相差整整 `50.00 pp`。两次评测使用相同的 WC4 seeds，因此差异主要来自训练随机性，而不是评测随机性。
5. 两个训练 seed 的简单平均成功率为 `71.88%`，但样本数过少且方差极大。当前不能把 `96.88%` 作为稳定性能结论；至少还需要补充 seed 3000 及更多训练 seed，并报告均值、标准差和最差 seed。

评测日志：

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_pcdact_dp3global_exactdp3_50k_seed615455575_0719_rerun
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_pcdact_obsbenchlocal_postsample_exact_50k_seed615455575_0719_rerun
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_pcdact_obsbenchlocal_interaction_decoderchunk_w005_50k_seed615455575_0719_retry2
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_pcdact_obsbenchlocal_postsample_exact_50k_trainseed2000_evalseed615455575_0720
```

## 11. Joint9 Front 点云 ACT 实验

### 11.1 实验设置与结果

本节将机器人 state/action 从 EE10 改为 Joint9：底盘命令、6 个机械臂关节和夹爪。视觉输入仍为 Front-only 的 1024 点 XYZ 点云，点云编码器采用 OBSBench Local post-sampling 结构。训练数据为 200 条 WC4 Joint9 + Gym Jacobian 专家轨迹，训练 seed 为 `1000`，训练步数为 50K。

评测参数与第 10 章完全一致。

| 模型 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | WC4 合计 |
|---|---:|---:|---:|---:|---:|
| OBSBench Local ACT | 8/16 | 13/16 | 12/16 | 11/16 | **44/64 = 68.75%** |
| OBSBench Local + Interaction decoder-chunk | 5/16 | 9/16 | 6/16 | 8/16 | **28/64 = 43.75%** |

对应 checkpoint：

| 模型 | Checkpoint |
|---|---|
| OBSBench Local ACT | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_jointstate9_gymjacobian_200_pcd_front_obsbenchlocal_50k_seed1000_0720_0015/checkpoints/050000` |
| OBSBench Local + Interaction | `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_jointstate9_gymjacobian_200_pcd_front_obsbenchlocal_interaction_decoderchunk_w005_50k_seed1000_0720_0015/checkpoints/050000` |

### 11.2 结果分析

1. Joint9 OBSBench Local ACT 达到 `44/64 = 68.75%`，说明局部点云 token 与关节动作表示可以正常联合训练，但该结果明显低于 EE10 seed 1000 的 `96.88%`。
2. 加入 Interaction decoder-chunk 后，成功率从 `68.75%` 降至 `43.75%`，减少 16 次成功，即下降 `25.00 pp`。在这组 Joint9 点云实验中，Interaction 是明确的负收益。
3. 该下降可能来自 Interaction auxiliary loss 与 motion action loss 的梯度竞争，也可能来自未来 interaction chunk 标签和 Joint9 动作动态不匹配。当前结果不支持在 Joint9 点云 ACT 中默认启用 Interaction head。
4. Joint9 点云实验目前只有一个训练 seed；结合第 10 章观察到的巨大训练方差，仍需补充相同配置下的其他训练 seed，才能区分结构问题和单次训练随机性。

评测日志：

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_jointstate9_pcd_obsbenchlocal_50k_seed615455575_0720
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_jointstate9_pcd_obsbenchlocal_interaction_50k_seed615455575_0720
```

## 12. LeRobot DP 与官方 Full DP3 对比

### 12.1 模型定义与对比范围

本节只统计真正以 diffusion policy 为策略后端的两个模型：

1. `LeRobot Diffusion Policy`：以双深度图和 state 为输入的 LeRobot DP。
2. `官方 Full DP3`：将 Front depth 转为 `robot_base` 坐标系下的 `1024×3` XYZ 点云，经 PointNet 编码后条件化 diffusion policy。

第10章的 `DP3 Global Encoder + ACT` 不属于官方 DP3。它只复用了 DP3 风格的 PointNet 全局编码器，动作生成后端仍然是 ACT，因此不计入本节 DP3 主结果。

当前完成的训练/评测组合如下：

| 数据规模与动作表示 | LeRobot DP | 官方 Full DP3 |
|---|---|---|
| WC4-only 50，EE10 | 未训练/未评测 | 未训练/未评测 |
| WC4-only 50，Joint9 | 无对应数据集 | 无对应数据集 |
| WC4-only 200，EE10 | 已完成 50K 和 64 次评测 | 已完成 50K 和 horizon sweep |
| WC4-only 200，Joint9 | 已完成 50K 和 64 次评测 | 已完成 50K 和 64 次评测 |

因此，50条数据当前只有第9章的 ACT 系列结果；没有证据可以填写“50条 DP/DP3 成功率”。

### 12.2 统一评测条件

除执行 horizon 和模型观测结构外，以下评测条件保持一致：

```text
door=wc4
num_envs=16
total_trials=64
base_seed=615455575
batch seeds=615455575, 615455576, 615455577, 615455578
steps=1000
dp_fps=25
success_metric=abs
success threshold=80 deg
depth noise=off
Gaussian blur=off
camera randomization=position 0.02 m, rotation 5 deg
robot pitch=0.0, pitch randomization=off
```

LeRobot DP 的训练配置为 `prediction horizon=40`、`n_obs_steps=16`、`n_action_steps=20`。评测命令沿用之前 DP 基准的 `dp_action_horizon=25`，但模型单个 chunk 最多实际提供20个动作。

官方 Full DP3 的训练配置为 `prediction horizon=32`、`n_obs_steps=2`、`n_action_steps=16`。因此 DP3 只在 `1～16` 范围内测试执行 horizon。

### 12.3 200条 EE10 数据结果

训练数据为 `door_a2w_wc4_robotbasefull_contactcheck_200`。LeRobot DP 使用 Front/Wrist 双深度；官方 Full DP3 只使用 Front depth 点云。两者 state/action 均为 EE10。

| 模型 | 执行 horizon | Batch 0 | Batch 1 | Batch 2 | Batch 3 | WC4 合计 |
|---|---:|---:|---:|---:|---:|---:|
| LeRobot Diffusion Policy 50K | 25（单 chunk 最多20步） | 8/16 | 7/16 | 8/16 | 10/16 | **33/64 = 51.56%** |
| 官方 Full DP3 50K | 4 | 9/16 | 7/16 | 4/16 | 11/16 | 31/64 = 48.44% |
| 官方 Full DP3 50K | 8 | 14/16 | 10/16 | 9/16 | 15/16 | 48/64 = 75.00% |
| 官方 Full DP3 50K | 12 | 12/16 | 9/16 | 9/16 | 12/16 | 42/64 = 65.63% |
| 官方 Full DP3 50K | **16** | **15/16** | **12/16** | **9/16** | **13/16** | **49/64 = 76.56%** |

在这批200条 EE10 数据上，官方 Full DP3 的最佳结果为 `49/64 = 76.56%`，比 LeRobot DP 的 `33/64 = 51.56%` 高 `25.00 pp`。但该差异不能直接解释为纯模型架构收益，因为两者的视觉输入、观测历史和执行 horizon 不同。

双视角官方 DP3 目前只有一个可核验的单 batch 重测：`5/16 = 31.25%`（horizon=16，seed=615455576）。它没有完整可靠的64次汇总，因此不进入主表，也不能据此判断双视角一定弱于 Front-only。

对应 checkpoint：

```text
# EE10 LeRobot DP
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/lerodp_a2w_wc4_200_h40_exec20_obs16_seed1000_50k_bs8_0718/checkpoints/050000

# EE10 Front-only 官方 Full DP3 Door checkpoint
/home/sivan/whole_body/visual_whole_body/high-level/dp/logs/door-auto-wrapped/a2w_wc4_front_dp3_full_50k_0718_0209/050000/model_latest.pt
```

评测日志：

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_lerodp_wc4_200_50k_h25_seed615455575
/home/sivan/whole_body/visual_whole_body/high-level/logs/door-policy-success/wc4_dp3_front200_50k_h4_seed615455575
/home/sivan/whole_body/visual_whole_body/high-level/logs/door-policy-success/wc4_dp3_front200_50k_h8_seed615455575
/home/sivan/whole_body/visual_whole_body/high-level/logs/door-policy-success/wc4_dp3_front200_50k_h12_seed615455575
/home/sivan/whole_body/visual_whole_body/high-level/logs/door-policy-success/wc4_dp3_front200_50k_h16_seed615455575
```

### 12.4 200条 Joint9 数据结果

训练数据为 `door_a2w_wc4_jointstate9_gymjacobian_200`。state/action 都是9D：底盘命令2D、机械臂六关节和夹爪。LeRobot DP 使用双深度；官方 Full DP3 使用 Front-only 点云。

| 模型 | 执行 horizon | Batch 0 | Batch 1 | Batch 2 | Batch 3 | WC4 合计 |
|---|---:|---:|---:|---:|---:|---:|
| Joint9 Baseline ACT 50K（参考） | 25 | 13/16 | 8/16 | 11/16 | 9/16 | 41/64 = 64.06% |
| Joint9 LeRobot Diffusion Policy 50K | 25（单 chunk 最多20步） | 11/16 | 9/16 | 9/16 | 13/16 | **42/64 = 65.63%** |
| Joint9 Front-only 官方 Full DP3 50K | 4 | 12/16 | 12/16 | 10/16 | 8/16 | **42/64 = 65.63%** |

Joint9 LeRobot DP 和官方 Full DP3 都是 `42/64 = 65.63%`，仅分 batch 成功数不同。它们相对 Joint9 Baseline ACT 的 `41/64` 只多成功一次；在当前单组64次评测下，没有证据表明 DP 或 DP3 对 Joint9 有实质性提升。

对应 checkpoint：

```text
# Joint9 LeRobot DP
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/lerodp_wc4_jointstate9_gymjacobian_200_h40_exec20_obs16_bs8_50k_0720/checkpoints/050000

# Joint9 Front-only 官方 Full DP3 Door checkpoint
/home/ps/workspace/txc/doorgym/high-level/dp/logs/door-auto-wrapped/a2w_wc4_joint9_front_dp3_50k_0720/050000/model_latest.pt
```

评测日志：

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_joint9_lerodp_50k_h25_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_joint9_dp3_50k_h4_seed615455575
```

### 12.5 DP / DP3 总结

| 数据 | 动作表示 | LeRobot DP | 官方 Full DP3 | 本组较高结果 |
|---|---|---:|---:|---:|
| WC4-only 50 | EE10 | 未训练/未评测 | 未训练/未评测 | 无 |
| WC4-only 200 | EE10 | 33/64 = 51.56% | **49/64 = 76.56%**（h=16） | DP3 |
| WC4-only 200 | Joint9 | **42/64 = 65.63%** | **42/64 = 65.63%**（h=4） | 持平 |

当前可以确认：官方 Full DP3 在200条 EE10 数据上明显高于对应 LeRobot DP；但换成 Joint9 后两者完全持平。由于两种策略的输入模态、观测历史和最优执行 horizon 不一致，后续若要形成严格架构结论，应统一 Front-only 观测、action horizon 和训练 seed，并补充至少3个独立训练 seed。
