# A2W ACT Camera Gating + Keyframe W3/R3 50k 评测（2026-07-01）

## 1. 新模型

Checkpoint：

`high-level/dp/logs/lerobot-train/leroact_a2w_gating_keyframe_w3_r3_sample30_chunk100_exec50_bs16_0630_2234/checkpoints/050000`

训练配置：

- ACT ResNet18，chunk size 100。
- 关键帧窗口半径 3，action loss weight 3。
- 关键帧窗口采样 30%，普通帧采样 70%。
- Camera input gating 开启，hidden dim 128，temperature 1.0。
- 50k checkpoint，batch size 16。

## 2. 评测设置

- Horizon：8–15。
- 每个 horizon 64 次，共 512 次 rollout。
- 4 批 × 16 env，batch seed 62000–62003；每个 env 使用独立派生 seed。
- WC4，1000 steps，门轴绝对角度达到 80° 为成功。
- 环境随机化与相机位姿随机化开启：位置 ±0.02 m，角度 ±5°。
- 深度范围 0.2–1.5 m。
- 不加 depth noise，不加 Gaussian blur。

原始日志：

`high-level/logs/door-policy-success/gating_w3r3_sample30_50k_horizon_sweep_20260701`

## 3. Horizon 扫描

| Horizon | 成功数 | 成功率 | Wilson 95% CI |
|---:|---:|---:|---:|
| 8 | 30/64 | 46.88% | 35.2%–58.9% |
| 9 | 31/64 | 48.44% | 36.6%–60.4% |
| 10 | 31/64 | 48.44% | 36.6%–60.4% |
| 11 | 34/64 | 53.12% | 41.1%–64.8% |
| 12 | **39/64** | **60.94%** | 48.7%–71.9% |
| 13 | 35/64 | 54.69% | 42.6%–66.3% |
| 14 | **39/64** | **60.94%** | 48.7%–71.9% |
| 15 | 36/64 | 56.25% | 44.1%–67.7% |

H12 与 H14 成功数完全相同，但并不是相同的 39 个环境：

- 共同成功：32。
- H12 独占成功：7。
- H14 独占成功：7。
- 共同失败：18。

动作质量辅助指标：

| Horizon | 首次完全闭合夹爪 | 闭合后重开再闭合 | 成功 trial 首次开门平均 step |
|---:|---:|---:|---:|
| 12 | 51/64 | **20/64** | **686.8** |
| 14 | 53/64 | 24/64 | 699.9 |

因此选择 **H12**：成功率与 H14 相同，但夹爪重开更少，平均成功时间也略早。

## 4. 三个 50k 模型对比

| 模型 | 关键帧设置 | Gating | 最佳 horizon | 成功率 |
|---|---|---:|---:|---:|
| 无关键帧 baseline | 无 | 关闭 | H15 | **42/64（65.62%）** |
| 关键帧、无 gating | W8/R3，sample 20% | 关闭 | H9/H11/H14 | 29/64（45.31%） |
| 当前模型 | W3/R3，sample 30% | 开启 | H12/H14 | **39/64（60.94%）** |

同 horizon 配对比较：

- 当前 H12 对无关键帧 baseline H12：39 对 34，提升 5/64；exact McNemar `p=0.302`。
- 当前 H14 对无关键帧 baseline H14：39 对 36，提升 3/64；exact McNemar `p=0.607`。
- 当前 H12 对关键帧无 gating H12：39 对 26，提升 13/64；exact McNemar `p=0.0072`。
- 当前 H14 对关键帧无 gating H14：39 对 29，提升 10/64；exact McNemar `p=0.0525`。
- 当前最佳 H12 对 baseline 最佳 H15：39 对 42，低 3/64；配对 `p=0.629`。

## 5. 结论

当前 W3/R3 + sample 30% + camera gating 模型明显恢复了 W8 关键帧模型损失的成功率：

- 相比旧关键帧无 gating 模型，最佳成功率从 45.31% 提升到 60.94%。
- 相比无关键帧 baseline，当前模型仍低 4.69 个百分点，但 64 次测试下差异不明显。
- 推荐推理 horizon 为 12。

不能把全部提升单独归因于 camera gating，因为新模型同时把 keyframe weight 从 8 降到 3、采样比例从 20% 改到 30%。若要严格测量 gating 的贡献，需要额外训练一份完全相同的 W3/R3 + sample 30%、但关闭 gating 的模型。

## 6. 当前模型为何仍低于无关键帧 baseline

对当前 H12 和 baseline H15 的相同 64 个随机环境逐 trial 对齐：

- 共同成功 32。
- 当前模型独占成功 7。
- baseline 独占成功 10。
- 共同失败 15。

### 6.1 失败集中在夹爪闭合和把手解锁

| 指标 | 当前 W3 + gating H12 | 无关键帧 baseline H15 |
|---|---:|---:|
| 夹爪 command 完全闭合 | 51/64 | **57/64** |
| 把手达到 40° 解锁角 | 39/64 | **42/64** |
| 门达到 80° | 39/64 | **42/64** |
| Gripper command 总变差均值 | 8.57 | **6.11** |
| Gripper command 跳变 P95 均值 | 0.103 | **0.049** |

两个模型一旦把手达到 40°，本次测试中都能把门打开。因此差距不在最终推门，而在此前的抓握与转把手。

在 10 个“baseline 成功、当前模型失败”的环境中：

- 当前模型只有 7/10 完全闭合夹爪，baseline 为 10/10。
- 当前模型有 7/10 出现闭合后重开再闭合，baseline 为 3/10。
- 当前模型最大把手角中位数只有 7.0°，baseline 接近 45°。
- 当前模型 gripper command 总变差中位数为 11.23，baseline 为 4.01。

这说明当前模型的主要输出问题是 gripper action 更跳、更不稳定，导致夹爪没有持续夹住把手。

### 6.2 Base 和 EE 不是主要差异

全体 64 个环境的输出：

- 最大 `vx` 均值：当前 0.338，baseline 0.339。
- EE target 轨迹总变差：当前 3.868，baseline 3.859。
- EE 平均跟踪误差：当前 0.071 m，baseline 0.068 m。

这些差异很小。当前模型失败后 EE 路径较短，主要是策略停滞在抓握阶段，而不是 IK 本身明显跟不上。

成功 trial 中，当前模型夹爪完全闭合更早：

- 当前模型中位 step 348。
- baseline 中位 step 362。

但从完全闭合到把手解锁：

- 当前模型中位 143 steps。
- baseline 中位 123 steps。

当前模型虽然更早发出闭合动作，后续转把手却更慢、更容易发生 gripper command 反复。

### 6.3 关键帧监督仍然偏强

当前训练中，30% 的 batch anchor 来自关键帧窗口，权重为 3。归一化加权 loss 中关键帧窗口的期望贡献约为：

`0.3 × 3 / (0.3 × 3 + 0.7 × 1) = 56.25%`

数据中关键帧窗口实际只占 18.82%，而 baseline 对所有帧均匀训练。因此当前模型仍把超过一半的 action loss 压在关键帧窗口。

此外，`loss.action_weight` 加权的是以当前帧为 anchor 的整个 100-step action chunk，不只是夹爪变化那一帧。相邻关键帧 anchor 中，夹爪闭合出现在 chunk 的不同时间位置；普通 L1 回归容易学出时间被平均化、chunk 边界不连续的 gripper action。

### 6.4 Gating 没有塌缩，但缺少时间一致性约束

训练到 50k 时 W&B 记录的 batch 平均 gate：

- `g_front ≈ 0.939`
- `g_wrist ≈ 1.061`

两个 gate 之和保持为 2，只是轻微偏向 wrist，没有出现某个相机被完全关闭。因此当前差距不像是 gating 塌缩。

但 gating 仅由 action loss 端到端学习，没有跨帧平滑或阶段监督。它可能使相邻观测的视觉特征缩放发生变化，而推理又没有 temporal ensemble，这会进一步放大 action chunk 边界处的 gripper 跳变。

严格判断 gating 本身是帮助还是伤害，仍需要训练完全同配置的 W3/R3 + sample 30% + gating off 对照模型。
