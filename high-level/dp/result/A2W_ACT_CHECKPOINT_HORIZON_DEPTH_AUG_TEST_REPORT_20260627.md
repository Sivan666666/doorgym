# A2W ACT 策略：Checkpoint、Action Horizon 与深度增强测试报告

**日期：** 2026-06-27  
**任务：** A2W-Z1 打开 `wc4` 下压把手门  
**策略：** LeRobot ACT，双深度相机输入，10D state / 10D action  

## 1. 报告范围

本报告汇总本轮四部分实验：

1. 三个不同训练结果的 checkpoint 对比；
2. `dp_action_horizon` 对闭环开门成功率的影响；
3. 不同 Gaussian blur 参数的影响；
4. 深度随机噪声与 Gaussian blur 的组合影响。

本文只采用已经保存的实验日志，不把人工观察到的未记录结果混入统计。

## 2. 统一评测定义

除非单独说明，测试设置如下：

| 项目 | 设置 |
|---|---|
| 机器人与门 | A2W-Z1，`wc4` |
| 仿真长度 | 1000 simulation steps |
| 策略频率 | 25 Hz |
| 深度裁剪 | 0.2–1.5 m |
| base pitch | 固定为 0° |
| 环境随机化 | 开启 |
| 成功条件 | 1000 steps 内门轴最大绝对角度达到 80° |

成功判定统一使用：

```text
max_t(abs(door_hinge_angle[t])) >= 80°
```

使用绝对值是因为历史 `eval` 的 signed metric 曾与门的实际开启方向不一致，出现“画面中门已打开，但 summary 仍为 0”的问题。本报告中的 horizon 结果直接从 `horizon_sweep_direct/*.jsonl` 重新计算。

## 3. 三个 checkpoint

| 简称 | 训练数据 | 训练步数 | Checkpoint |
|---|---|---:|---|
| Old-200-100k | 旧版 new-FOV randomized，200 episodes | 100000 | `...randomized_200.../100000/model_latest.pt` |
| New-100-50k | `stop_distance=0.15`、`grasp_x_offset=-0.015`，100 episodes | 50000 | `...stop015_graspx015_100.../050000/model_latest.pt` |
| New-100-100k | 与上一行相同 | 100000 | `...stop015_graspx015_100.../100000/model_latest.pt` |

完整目录：

```text
high-level/dp/logs/door-auto-wrapped/
  leroact_a2w_state10_wc4_newfov_randomized_200_depthonly_chunk100_exec50_bs16_0625_0212/
  leroact_a2w_state10_wc4_newfov_randomized_stop015_graspx015_100_depthonly_chunk100_exec50_bs16_0626_2341/
```

## 4. Checkpoint 与 horizon 测试

### 4.1 Seed 12345，4-env 初筛

| Checkpoint | Horizon | 成功数 | 成功率 |
|---|---:|---:|---:|
| Old-200-100k | 10 | 1/4 | 25% |
| New-100-100k | 10 | 0/4 | 0% |
| New-100-100k | 15 | 1/4 | 25% |
| New-100-100k | 20 | 1/4 | 25% |
| New-100-100k | 25 | 0/4 | 0% |
| New-100-100k | 30 | 0/4 | 0% |
| New-100-100k | 40 | 1/4 | 25% |
| New-100-100k | 50 | 0/4 | 0% |
| New-100-100k | 75 | 1/4 | 25% |
| New-100-100k | 100 | 0/4 | 0% |
| New-100-50k | 10 | 4/4 | 100% |
| New-100-50k | 15 | 4/4 | 100% |
| New-100-50k | 20 | 3/4 | 75% |
| New-100-50k | 25 | 2/4 | 50% |
| New-100-50k | 30 | 3/4 | 75% |
| New-100-50k | 40 | 1/4 | 25% |
| New-100-50k | 50 | 2/4 | 50% |
| New-100-50k | 75 | 1/4 | 25% |

初筛显示：

- checkpoint 差异比 horizon 微调更显著；
- New-100-50k 明显优于 New-100-100k；
- horizon 变长后整体表现下降，说明策略需要较频繁地根据新观测重新规划；
- 4-env 只适合筛选候选参数，不能作为最终成功率。

### 4.2 Seed 23456，New-100-50k 的 16-env 细筛

| Horizon | 成功数 | 成功率 | 失败 env |
|---:|---:|---:|---|
| 10 | 12/16 | 75.0% | 4, 10, 11, 14 |
| 12 | **14/16** | **87.5%** | 1, 11 |
| 14 | **14/16** | **87.5%** | 2, 11 |
| 15 | 13/16 | 81.25% | 0, 2, 11 |

因此后续深度增强实验采用：

```text
checkpoint = New-100-50k
dp_action_horizon = 12
```

### 4.3 为什么 horizon 很敏感

ACT 每次预测一个 100-step action chunk，但 `dp_action_horizon` 决定实际连续执行多少步后才读取新观测并重新推理。

- horizon 太大时，策略长时间执行旧观测生成的动作。抓把手、闭合夹爪和转把手稍有偏差，后续动作仍会继续放大误差；
- horizon 较小时，策略能更快使用新深度图与当前 EE 状态纠偏；
- horizon 也不能无限缩短。过于频繁地切换新 chunk，可能破坏闭合夹爪、转把手等需要短时连续性的动作；
- 当前策略的平衡点在 12–14 步附近，不代表其他 checkpoint 也具有相同最优值。

## 5. 为什么 50k 优于 100k

这不表示 100k 的训练 loss 一定更高，而是说明 open-loop 平均 loss 与 closed-loop 开门成功率并不等价。

ACT 的 action loss 会在 100-step chunk、10 个 action 维度上进行聚合。关键事件存在明显的不平衡：

- `close gripper` 和 `rotate handle` 只占整段轨迹中的少量帧；
- gripper 只占 10D action 中的 1 维；
- walk、hold 和平滑 EE 移动占据大部分训练帧和 loss；
- 继续训练可能让常见、平滑阶段的平均误差更低，却让短暂关键事件的时序变得更脆弱；
- closed-loop 执行中，一次夹爪没有及时闭合或把手旋转不足，会改变之后的观测分布并形成连锁失败。

本轮现象与上述机制一致：

- New-100-100k 经常在到达 grasp 后夹爪不闭合，或出现“闭合—张开—再次闭合”；
- New-100-50k 的 gripper 与 rotate 时序更稳定；
- horizon=12 能进一步减少旧 chunk 对关键阶段的错误延续。

因此模型选择不能只看最终训练 loss，应同时保存中间 checkpoint，并用固定 seed 的 closed-loop 成功率选模。后续加入关键帧重采样和关键帧加权 loss，正是针对这种短时关键事件被平均 loss 稀释的问题。

## 6. Gaussian blur 参数实验

### 6.1 设置

- checkpoint：New-100-50k；
- horizon：12；
- seed：34567；
- 16 env；
- 非模糊随机噪声项全部置零；
- Gaussian blur 作用在 0.2–1.5 m 裁剪并归一化后的 policy depth 上。

### 6.2 结果

| Gaussian blur | 成功数 | 成功率 | 失败 env |
|---|---:|---:|---|
| 不开启 | 12/16 | 75.0% | 1, 2, 7, 10 |
| 3×3, σ=0.2 | 12/16 | 75.0% | 1, 2, 7, 10 |
| 3×3, σ=0.5 | 10/16 | 62.5% | 1, 2, 3, 7, 10, 14 |
| 5×5, σ=1.0（训练同款） | 10/16 | 62.5% | 1, 2, 3, 7, 10, 14 |
| 9×9, σ=2.0 | **13/16** | **81.25%** | 1, 7, 14 |
| 15×15, σ=4.0 | **13/16** | **81.25%** | 2, 7, 8 |

在 seed 34567 上，较大的模糊优于训练同款 5×5/σ=1.0。但这一排序没有在另一个 seed 上完全复现，不能据此直接断言“大模糊始终更好”。

## 7. 深度噪声与 Gaussian blur 组合实验

### 7.1 设置与实现语义

- checkpoint：New-100-50k；
- horizon：12；
- seed：45678；
- 16 env；
- depth noise 打开时，固定选择 50% env，即 8/16 env 的每一帧都加噪；
- “无噪声 + blur”通过启用增强管线、选择全部 env、将随机噪声强度置零，仅保留 blur 来实现。

当前实现中 Gaussian blur 位于 depth-noise 管线末尾，并受 `env_selected` 控制。因此：

- `noise on + blur on`：8 个被选中的 env 同时接受噪声和 blur，另外 8 个两者都不接受；
- `noise off + blur on`：人为选择全部 16 个 env，但把随机噪声项置零，因此全部 env 只接受 blur；
- 这不是完全独立的标准 2×2 因子实验。若要严格研究交互作用，应把 blur 的 env 选择从 noise 的 env 选择中解耦。

### 7.2 结果

| Depth noise | Gaussian blur | 成功数 | 成功率 |
|---|---|---:|---:|
| 关 | 关 | 10/16 | 62.5% |
| 开 | 关 | 7/16 | 43.75% |
| 关 | 5×5, σ=1.0 | **13/16** | **81.25%** |
| 开 | 5×5, σ=1.0 | 10/16 | 62.5% |
| 关 | 9×9, σ=2.0 | 12/16 | 75.0% |
| 开 | 9×9, σ=2.0 | 11/16 | 68.75% |

关闭 blur 时，表中的 kernel 和 sigma 不参与计算。因此：

- `noise on + blur off` 的 5×5 与 9×9 运行完全相同，均为 7/16；
- `noise off + blur off` 的两次运行完全相同，均为 10/16。

### 7.3 组合实验结论

1. **当前随机深度噪声会降低成功率。**  
   无 blur 时从 10/16 降至 7/16。

2. **Gaussian blur 能明显缓解噪声影响。**  
   加噪条件下，5×5 从 7/16 提升到 10/16，9×9 提升到 11/16。

3. **无随机噪声时，blur 也可能提升成功率。**  
   5×5 为 13/16，9×9 为 12/16，均高于无 blur 的 10/16。

4. **当前单 seed 下，5×5/σ=1.0 是最好的 blur-only 设置。**  
   但它在 seed 34567 的上一组实验中只有 10/16，说明结果对随机环境组合敏感。

## 8. 为什么同一个 5×5/σ=1.0 会得到 10/16 和 13/16

两次 Gaussian 参数相同，但随机 seed 不同：

| 实验 | Seed | 结果 |
|---|---:|---:|
| Gaussian blur sweep | 34567 | 10/16 |
| Noise/blur grid | 45678 | 13/16 |

seed 会同时改变：

- robot x/y/z/yaw；
- door x/y 与墙体偏移；
- 门轴与把手的 friction、damping；
- 是否施加门回弹阻力及其大小；
- 不同 env 到达关键阶段时的具体状态。

16 个 env 中相差 3 次成功，尚不足以证明策略性能发生了稳定变化。两组结果的置信区间高度重叠，Fisher exact test 的双侧 `p≈0.43`。更重要的是，无 blur 的基线也从 seed 34567 的 12/16 变为 seed 45678 的 10/16，说明两批随机环境难度本身不同。

此外，抓取和转把手属于接触敏感、闭环敏感过程。GPU PhysX、相机观测和临界接触的小差异也可能改变最终结果，但当前最主要、可确认的差别仍然是随机 seed。

## 9. 总体结论

1. **当前最可靠的组合是 New-100-50k + horizon=12。**  
   在 seed 23456 的 16-env 测试中达到 14/16。

2. **New-100-100k 明显弱于 50k。**  
   更长训练降低平均监督误差，不保证短暂关键事件的闭环时序更好。

3. **horizon 是闭环控制参数，不是纯粹的推理性能参数。**  
   它改变重新观测与重新规划的频率，当前策略在 12–14 附近最好。

4. **深度随机噪声当前总体有负面影响。**  
   模型尚不能完全适应模拟噪声，尤其是没有 blur 时。

5. **Gaussian blur 通常能缓解噪声，并可能提高无噪声条件下的稳定性。**  
   但 5×5 与 9×9 的相对优劣会随 seed 改变。

6. **目前不应把单个 16-env 结果当作最终成功率。**  
   现有结果适合筛选候选设置，不足以给出稳定泛化结论。

## 10. 推荐的下一轮严谨评测

建议固定一组共享 seeds，例如：

```text
34567, 45678, 56789, 67890, 78901
```

每个配置执行 `5 seeds × 16 env = 80 trials`，并保证不同配置使用完全相同的 per-env randomization。至少评测：

| 组别 | Checkpoint | Horizon | Noise | Blur |
|---|---|---:|---|---|
| A | New-100-50k | 12 | 关 | 关 |
| B | New-100-50k | 12 | 关 | 5×5/1.0 |
| C | New-100-50k | 12 | 关 | 9×9/2.0 |
| D | New-100-50k | 12 | 开 | 关 |
| E | New-100-50k | 12 | 开 | 5×5/1.0 |
| F | New-100-50k | 12 | 开 | 9×9/2.0 |

同时记录阶段指标：

- 是否到达 stop distance；
- 是否到达 pregrasp / grasp；
- 夹爪是否完成闭合；
- 把手最大旋转角；
- 门轴最大旋转角；
- 首次达到 80° 的 step。

最终报告应提供总成功率、各 seed 成功率和 Wilson 95% 置信区间。代码层面建议先将 Gaussian blur 与 depth-noise env selection 解耦，保证 noise 和 blur 可以真正独立开关。

## 11. 原始日志

```text
high-level/logs/horizon_sweep_direct/
high-level/logs/gaussian_blur_sweep/summary.json
high-level/logs/depth_noise_blur_grid/summary.json
```

