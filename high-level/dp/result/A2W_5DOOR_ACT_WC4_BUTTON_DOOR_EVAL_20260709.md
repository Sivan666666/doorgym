# A2W 5-door ACT 策略在 WC4 与 Button Door 上的评测汇总

日期：2026-07-09  
任务：比较 4 种 50K checkpoint 在训练门 `wc4` 和 unseen door `button_door` 上的成功率。  

## 1. 对比的 4 种策略

| 名称 | 主要改动 | checkpoint |
|---|---|---|
| Baseline ACT | 标准 ACT，depth-only 双相机输入，无 keyframe loss，无 Plücker | 本地 `high-level/dp/logs/lerobot-train/leroact_a2w_5door_baseline_act_50k_chunk100_exec50_bs16_0709_0206/checkpoints/050000` |
| Keyframe ACT | 关键帧重采样 + keyframe action loss，`W=3, R=3, sample=30%` | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_keyframe_w3_r3_sample30_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |
| Plücker ACT FOV55 | Plücker ray camera geometry conditioning，FOV=55° | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |
| Keyframe + Plücker ACT | Keyframe ACT + Plücker FOV55 | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |

## 2. 统一评测参数

| 参数 | 设置 |
|---|---|
| robot | `a2wz1` |
| mode | `ikpush` |
| door cfg | `high-level/data/cfg/b1z1_opendoor.yaml` |
| eval doors | `wc4`, `button_door` |
| trials | `64` |
| num envs | `16` |
| steps | `1000` |
| seed | `615455575` |
| success threshold | `pass_open_angle_deg = 80` |
| success metric | `abs` |
| ACT action horizon | `dp_action_horizon = 25` |
| ACT fps | `dp_fps = 25` |
| observation | depth-only |
| depth clip | `[0.2, 1.5] m` |
| depth noise | off |
| Gaussian blur | off |
| camera randomization | on |
| camera pos rand | `0.02 m` |
| camera rot rand | `5 deg` |
| robot pitch | fixed `0.0` |
| robot pitch rand | `[0.0, 0.0]` |
| EE action frame | `robot_base_full` |

说明：

- `wc4` 是训练门之一，用来检查 seen-door 性能。
- `button_door` 没有参与这批 5-door 训练，用来检查 unseen-door 泛化。
- 本地 baseline 的 `wc4` 命令里虽然传了 `--success_metric auto`，日志实际解析为 `metric=abs`，因此和其他模型一致。
- ps1 上第一次评测 `button_door` 失败是因为缺少 `high-level/data/asset/door_set/button_door/bounding_box.json`。补齐 `button_door` asset 后已重新评测，下面结果来自重新评测后的有效日志。

## 3. 成功率结果

### 3.1 汇总表

| 策略 | WC4 成功率 | Button Door 成功率 | WC4 → Button Door 泛化下降 |
|---|---:|---:|---:|
| Baseline ACT | 58/64 = 90.62% | 3/64 = 4.69% | -85.94 pp |
| Keyframe ACT | 43/64 = 67.19% | 6/64 = 9.38% | -57.81 pp |
| Plücker ACT FOV55 | 59/64 = 92.19% | 13/64 = 20.31% | -71.88 pp |
| Keyframe + Plücker ACT | 49/64 = 76.56% | 19/64 = 29.69% | -46.88 pp |

### 3.2 WC4 分 batch 结果

| 策略 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Total |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 16/16 | 12/16 | 14/16 | 16/16 | 58/64 |
| Keyframe ACT | 12/16 | 11/16 | 9/16 | 11/16 | 43/64 |
| Plücker ACT FOV55 | 16/16 | 14/16 | 13/16 | 16/16 | 59/64 |
| Keyframe + Plücker ACT | 14/16 | 10/16 | 11/16 | 14/16 | 49/64 |

### 3.3 Button Door 分 batch 结果

| 策略 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Total |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 1/16 | 1/16 | 1/16 | 0/16 | 3/64 |
| Keyframe ACT | 1/16 | 2/16 | 2/16 | 1/16 | 6/64 |
| Plücker ACT FOV55 | 3/16 | 4/16 | 4/16 | 2/16 | 13/64 |
| Keyframe + Plücker ACT | 3/16 | 5/16 | 6/16 | 5/16 | 19/64 |

## 4. 初步分析

### 4.1 WC4 上：Baseline 和 Plücker 最强，Keyframe 反而伤性能

在 `wc4` 上，Baseline ACT 已经很强，达到 `58/64 = 90.62%`。Plücker ACT 是 `59/64 = 92.19%`，只比 baseline 多 1 次成功，这个差异很小，不能说有显著提升，但至少说明 Plücker conditioning 没有破坏 seen-door 性能。

Keyframe ACT 降到 `43/64 = 67.19%`，Keyframe + Plücker 也只有 `49/64 = 76.56%`。这和之前观察到的现象一致：关键帧 loss / sampling 容易让 gripper 或阶段切换动作变得更激进，可能强化了短时关键动作，但也可能破坏 ACT 原本平滑的 closed-loop 行为，尤其在 seen door 上不如 baseline 稳。

### 4.2 Button Door 上：所有模型都掉得很厉害，但 Plücker 和 Keyframe+Plücker 有明显优势

`button_door` 是 unseen door，四个模型都出现很大泛化下降。Baseline 从 `90.62%` 掉到 `4.69%`，几乎失效。这说明标准 depth-only ACT 很可能学到了较强的门实例/轨迹分布依赖，对新门的几何、把手位置、局部接触模式不够鲁棒。

Plücker ACT 在 button door 上达到 `13/64 = 20.31%`，比 baseline 高 `+15.62 pp`。这说明显式相机几何输入确实帮助模型理解双相机 depth 与机器人 base frame 之间的空间关系，对 unseen geometry 有一定帮助。

Keyframe + Plücker 达到 `19/64 = 29.69%`，是 button door 上最好的一组。一个合理解释是：Plücker 提供空间几何对齐，Keyframe 机制又强化了 grasp / rotate / push 这些短关键阶段；在 seen door 上这可能破坏平滑性，但在 unseen door 上反而提供了一些“关键动作不能丢”的归纳偏置。

### 4.3 Keyframe 单独作用有限

Keyframe ACT 从 baseline 的 `3/64` 提升到 `6/64`，有一点改善，但仍然很低。这说明只强调关键帧，不解决相机几何和空间对应关系，泛化帮助有限。换句话说，模型知道“关键阶段重要”还不够，它还需要知道“这个像素/这个把手在机器人坐标系里到底在哪里”。

### 4.4 Horizon 固定为 25 的影响

这次四个模型都使用 `dp_action_horizon=25`，因此结果可比。之前我们发现 ACT 的 closed-loop 效果对 horizon 很敏感，尤其 gripper 可能出现 close-open-close 的时序问题。这里固定 horizon=25 的好处是消除了 horizon sweep 带来的变量；但缺点是没有保证每个模型都在各自最优 horizon 下评测。

如果后续要更严谨，可以对每个模型在 button door 上 sweep：

```text
horizon = 10, 12, 14, 15, 25, 50
```

然后报告每个模型的 best-horizon success rate。但如果论文/文档里想强调公平对比，固定 horizon=25 是更干净的设置。

## 5. 结论

1. `wc4` 上最强的是 Plücker ACT FOV55：`59/64 = 92.19%`，但和 baseline `58/64 = 90.62%` 差距很小。
2. `button_door` unseen 泛化最强的是 Keyframe + Plücker：`19/64 = 29.69%`。
3. Baseline ACT 在 seen door 很强，但 unseen door 泛化非常差：`3/64 = 4.69%`。
4. Plücker conditioning 对 unseen door 有明确帮助：`3/64 → 13/64`。
5. Keyframe 单独不够强，但和 Plücker 结合后在 unseen door 上最好：`19/64`。
6. 目前所有 unseen-door 成功率仍偏低，说明新门泛化瓶颈还没有根本解决；下一步更值得排查的是门几何差异、把手/按钮交互模式、接触判断、轨迹阶段和训练数据覆盖，而不是只继续调 loss。

## 6. 日志位置

本地 baseline：

```text
high-level/logs/door-policy-success/wc4_5door_baseline_act_50k_seed615455575
high-level/logs/door-policy-success/button_door_5door_baseline_act_50k_seed615455575
```

ps1 三个模型：

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_5door_keyframe_w3_r3_sample30_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_5door_plucker_fov55_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_seed615455575

/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/button_door_5door_keyframe_w3_r3_sample30_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/button_door_5door_plucker_fov55_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/button_door_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_seed615455575
```
