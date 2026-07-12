# A2W 5-door ACT 策略在 WC4 / Button Door / Fire Door 上的评测汇总

日期：2026-07-09，更新：2026-07-12  
任务：比较 4 种 50K ACT checkpoint 在训练门 `wc4`、unseen door `button_door`、unseen `fire_door` 上的成功率，并记录 `fire_door` 50 条数据 finetune 后的变化；另补充 π0.5 在同一评测设置下的结果。  

## 1. 对比的 4 种策略

| 名称 | 主要改动 | checkpoint |
|---|---|---|
| Baseline ACT | 标准 ACT，depth-only 双相机输入，无 keyframe loss，无 Plücker | 本地 `high-level/dp/logs/lerobot-train/leroact_a2w_5door_baseline_act_50k_chunk100_exec50_bs16_0709_0206/checkpoints/050000` |
| Keyframe ACT | 关键帧重采样 + keyframe action loss，`W=3, R=3, sample=30%` | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_keyframe_w3_r3_sample30_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |
| Plücker ACT FOV55 | Plücker ray camera geometry conditioning，FOV=55° | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |
| Keyframe + Plücker ACT | Keyframe ACT + Plücker FOV55 | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |

补充评测：

| 名称 | 主要改动 | checkpoint |
|---|---|---|
| π0.5 | π0.5 policy，在 250 条 5-door 数据上从 `005000` warm-start 继续训练 25k step；目录 step 为 `025000`，语义上约等于 30k | ps1 `/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/pi05_a2w_5door_robotbasefull_contactcheck_250_from005000_to030000_bs16_gpu3_0710_2132/checkpoints/025000` |

## 2. 统一评测参数

| 参数 | 设置 |
|---|---|
| robot | `a2wz1` |
| mode | `ikpush` |
| door cfg | `high-level/data/cfg/b1z1_opendoor.yaml` |
| eval doors | `wc4`, `button_door`, `fire_door` |
| trials | `64` |
| num envs | `16` |
| steps | `wc4/button_door: 1000`, `fire_door: 1200` |
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
- `fire_door` 没有参与 5-door 训练，用来检查另一类 unseen-door 泛化；因为 `fire_door` skill 包含 traverse / return_home，评测 `steps=1200`，其余参数保持一致。
- 本地 baseline 的 `wc4` 命令里虽然传了 `--success_metric auto`，日志实际解析为 `metric=abs`，因此和其他模型一致。
- ps1 上第一次评测 `button_door` 失败是因为缺少 `high-level/data/asset/door_set/button_door/bounding_box.json`。补齐 `button_door` asset 后已重新评测，下面结果来自重新评测后的有效日志。

## 3. 成功率结果

### 3.1 汇总表

| 策略 | WC4 seen | Button Door unseen | Fire Door unseen | WC4 → Button | WC4 → Fire |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 58/64 = 90.62% | 3/64 = 4.69% | 56/64 = 87.50% | -85.94 pp | -3.12 pp |
| Keyframe ACT | 43/64 = 67.19% | 6/64 = 9.38% | 未测 | -57.81 pp | - |
| Plücker ACT FOV55 | 59/64 = 92.19% | 13/64 = 20.31% | 61/64 = 95.31% | -71.88 pp | +3.12 pp |
| Keyframe + Plücker ACT | 49/64 = 76.56% | 19/64 = 29.69% | 61/64 = 95.31% | -46.88 pp | +18.75 pp |
| π0.5 | 31/64 = 48.44% | 31/64 = 48.44% | 58/64 = 90.62% | +0.00 pp | +42.19 pp |

### 3.1.1 250 条 5-door 数据训练完 checkpoint 的 WC4 结果

| 策略 | checkpoint step | Horizon | WC4 成功率 |
|---|---:|---:|---:|
| Baseline ACT | 50k | 25 | 58/64 = 90.62% |
| Keyframe ACT | 50k | 25 | 43/64 = 67.19% |
| Plücker ACT FOV55 | 50k | 25 | 59/64 = 92.19% |
| Keyframe + Plücker ACT | 50k | 25 | 49/64 = 76.56% |
| π0.5 | 005000 + 25k continued | 25 | 31/64 = 48.44% |

### 3.1.2 Unseen Door 结果

| 策略 | Button Door 成功率 | Fire Door 成功率 |
|---|---:|---:|
| Baseline ACT | 3/64 = 4.69% | 56/64 = 87.50% |
| Keyframe ACT | 6/64 = 9.38% | 未测 |
| Plücker ACT FOV55 | 13/64 = 20.31% | 61/64 = 95.31% |
| Keyframe + Plücker ACT | 19/64 = 29.69% | 61/64 = 95.31% |
| π0.5 | 31/64 = 48.44% | 58/64 = 90.62% |

### 3.2 WC4 分 batch 结果

| 策略 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Total |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 16/16 | 12/16 | 14/16 | 16/16 | 58/64 |
| Keyframe ACT | 12/16 | 11/16 | 9/16 | 11/16 | 43/64 |
| Plücker ACT FOV55 | 16/16 | 14/16 | 13/16 | 16/16 | 59/64 |
| Keyframe + Plücker ACT | 14/16 | 10/16 | 11/16 | 14/16 | 49/64 |
| π0.5 | 10/16 | 8/16 | 5/16 | 8/16 | 31/64 |

### 3.3 Button Door 分 batch 结果

| 策略 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Total |
|---|---:|---:|---:|---:|---:|
| Baseline ACT | 1/16 | 1/16 | 1/16 | 0/16 | 3/64 |
| Keyframe ACT | 1/16 | 2/16 | 2/16 | 1/16 | 6/64 |
| Plücker ACT FOV55 | 3/16 | 4/16 | 4/16 | 2/16 | 13/64 |
| Keyframe + Plücker ACT | 3/16 | 5/16 | 6/16 | 5/16 | 19/64 |
| π0.5 | 5/16 | 9/16 | 7/16 | 10/16 | 31/64 |

### 3.4 Fire Door 直接评测结果

| 策略 | Horizon | Steps | 成功率 | 备注 |
|---|---:|---:|---:|---|
| Baseline ACT | 25 | 1200 | 56/64 = 87.50% | 本地评测 |
| Keyframe ACT | 25 | 1200 | 未测 | 目前没有有效 fire_door 日志 |
| Plücker ACT FOV55 | 25 | 1200 | 61/64 = 95.31% | ps1 评测 |
| Keyframe + Plücker ACT | 25 | 1200 | 61/64 = 95.31% | ps1 评测 |
| π0.5 | 25 | 1200 | 58/64 = 90.62% | ps1 评测；从 005000 warm-start 后继续训练 25k step 的 checkpoint |

确认：上表 `fire_door` 直接评测使用的是 **5-door 原始 50k checkpoint**，没有经过 fire door finetune。

| 策略 | Fire Door 直接评测 checkpoint |
|---|---|
| Baseline ACT | `leroact_a2w_5door_baseline_act_50k_chunk100_exec50_bs16_0709_0206/checkpoints/050000` |
| Plücker ACT FOV55 | `leroact_a2w_5door_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |
| Keyframe + Plücker ACT | `leroact_a2w_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000` |
| π0.5 | `pi05_a2w_5door_robotbasefull_contactcheck_250_from005000_to030000_bs16_gpu3_0710_2132/checkpoints/025000` |

### 3.4.1 π0.5 结果补充

| Door | Success | Success Rate | Batch 0 | Batch 1 | Batch 2 | Batch 3 | 失败角度特征 |
|---|---:|---:|---:|---:|---:|---:|---|
| `wc4` | 31/64 | 48.44% | 10/16 | 8/16 | 5/16 | 8/16 | 失败 max open 平均约 `1.81°`，基本是门没真正打开 |
| `fire_door` | 58/64 | 90.62% | 16/16 | 14/16 | 13/16 | 15/16 | 失败 max open 平均约 `2.31°`，失败样本也基本没打开 |
| `button_door` | 31/64 | 48.44% | 5/16 | 9/16 | 7/16 | 10/16 | 失败 max open 最大到 `61.29°`，部分样本是推开不够而非完全没动 |

π0.5 评测参数与上文一致：

```text
base_seed=615455575
num_envs=16
total_trials=64
dp_action_horizon=25
dp_fps=25
success_metric=abs
pass_open_angle_deg=80
depth_only
depth_noise=off
gaussian_blur=off
camera_randomization=on, pos=0.02m, rot=5deg
ee_pose_frame=robot_base_full
steps: wc4/button_door=1000, fire_door=1200
```

### 3.5 Fire Door 50 条数据 Finetune 后结果

| 初始策略 | Finetune 数据 | Finetune step | Horizon | Steps | Fire Door 成功率 | 对直接评测变化 |
|---|---|---:|---:|---:|---:|---:|
| Baseline ACT | `a2w_fire_door_robotbasefull_skill_contactcheck_50_step1200` | 5k | 25 | 1200 | 56/64 = 87.50% | +0.00 pp |
| Baseline ACT | 同上 | 10k | 25 | 1200 | 52/64 = 81.25% | -6.25 pp |
| Baseline ACT | 同上 | 20k | 25 | 1200 | 52/64 = 81.25% | -6.25 pp |
| Plücker ACT FOV55 | 同上 | 20k | 25 | 1200 | 58/64 = 90.62% | -4.69 pp |
| Keyframe + Plücker ACT | 同上 | 20k | 25 | 1200 | 59/64 = 92.19% | -3.12 pp |

### 3.6 Baseline ACT Fire Door Finetune Step 对比

| Baseline 版本 | Checkpoint | Fire Door 成功率 |
|---|---|---:|
| 5-door baseline，未 finetune | `leroact_a2w_5door_baseline_act_50k.../checkpoints/050000` | 56/64 = 87.50% |
| Fire door finetune 5k | `finetune_firedoor50_from_5door_baseline.../checkpoints/005000` | 56/64 = 87.50% |
| Fire door finetune 10k | `finetune_firedoor50_from_5door_baseline.../checkpoints/010000` | 52/64 = 81.25% |
| Fire door finetune 20k | `finetune_firedoor50_from_5door_baseline.../checkpoints/020000` | 52/64 = 81.25% |

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

### 4.4 Fire Door 上没有出现 Button Door 那种泛化崩溃

`fire_door` 上 baseline 直接达到 `56/64 = 87.50%`，Plücker 和 Keyframe+Plücker 都达到 `61/64 = 95.31%`。这说明 fire door 虽然是 unseen door，但它的几何和当前 scripted skill / 训练分布更接近模型已经学到的模式；不像 `button_door` 那样把交互模式推到明显 OOD。

Plücker 在 fire door 上有小幅正收益：

```text
Baseline: 56/64
Plucker:  61/64
KF+Plucker: 61/64
```

这和 button door 的趋势一致：显式相机几何更容易在 unseen geometry 上帮到策略。但 fire door 的 baseline 已经很强，所以绝对提升空间比 button door 小。

`button_door` 和 `fire_door` 都是 unseen door，但难度不是同一级别：

1. `button_door` 的交互模式和 lever-handle 数据分布差异更大，策略不仅要定位新几何，还要产生不同的接触/按压动作。
2. `fire_door` 虽然没参与 5-door 训练，但 skill 仍然更像 lever-handle push/traverse，和训练轨迹的阶段结构更接近。
3. `fire_door` 评测给了 `1200` steps，而 `button_door` 是 `1000` steps；fire door 的 traverse / return_home skill 更长，客观上给了策略更多后续调整时间。
4. `button_door` 的失败通常更像“接触模式不对”，不是单纯门开角不够；这类 OOD 不是靠多走几步就能修回来。
5. 因此 `fire_door` 高成功率不能简单解释成“见过 fire door”，日志确认没有见过；更合理的解释是 fire door 这个 unseen 更接近训练分布，而 button door 是更强 OOD。

### 4.5 π0.5 的表现：Fire Door 强，但 WC4 / Button Door 不稳

π0.5 在 `fire_door` 上达到 `58/64 = 90.62%`，说明它对 fire door 这种 unseen door 有不错表现；但在 `wc4` 和 `button_door` 上都只有 `31/64 = 48.44%`。这个分布很不均衡：

- `fire_door`：接近 ACT baseline / Plücker 的高成功率区间，说明 π0.5 并非整体不会开门。
- `wc4`：作为 seen door 只有 48.44%，明显低于 ACT baseline 的 90.62%，说明 π0.5 当前 checkpoint 对原训练门的动作相位/抓取稳定性还没学好。
- `button_door`：48.44%，显著高于 ACT baseline 的 4.69%，也高于 Keyframe+Plücker ACT 的 29.69%，说明 π0.5 在这个 OOD 门上反而更强。

一个合理解释是：π0.5 的策略分布更“宽”，对 button door 这种偏离训练分布的门更容易探索到有效推门动作；但它对 wc4 的稳定闭环时序不如专门训练的 ACT。换句话说，它的泛化形状和 ACT 不一样：ACT 在 seen wc4 很稳，但 button door 崩；π0.5 在 button door 有明显提升，但牺牲了 wc4 稳定性。

### 4.6 Fire Door finetune 反而退化的可能原因

Baseline ACT 用 50 条 fire door 数据 finetune 后没有提升：5k 与未 finetune 持平，10k/20k 下降到 `52/64`。这更像是小数据 finetune 的 closed-loop 过拟合，而不是模型容量不够。

可能原因：

1. 50 条 fire door 轨迹太少，finetune 很容易把原本 5-door 训练得到的泛化能力往单一轨迹模式拉窄。
2. Fire door 直接评测已经有 `87.50%`，基线很高，finetune 的收益空间小，但破坏已有行为的风险大。
3. ACT 的 chunk policy 对阶段时序敏感，finetune 会改变 gripper / rotate / push 的相对时序；即使 open-loop loss 更低，closed-loop 不一定更稳。
4. 当前 finetune 没有 replay 原 250 条数据，容易出现轻微 forgetting；更稳的做法是混合 `250 old + 50 fire`，或者降低 LR / 冻结视觉 backbone / 减少训练步数。

### 4.7 Horizon 固定为 25 的影响

这次四个模型都使用 `dp_action_horizon=25`，因此结果可比。之前我们发现 ACT 的 closed-loop 效果对 horizon 很敏感，尤其 gripper 可能出现 close-open-close 的时序问题。这里固定 horizon=25 的好处是消除了 horizon sweep 带来的变量；但缺点是没有保证每个模型都在各自最优 horizon 下评测。

如果后续要更严谨，可以对每个模型在 button door 上 sweep：

```text
horizon = 10, 12, 14, 15, 25, 50
```

然后报告每个模型的 best-horizon success rate。但如果论文/文档里想强调公平对比，固定 horizon=25 是更干净的设置。

## 5. 结论

1. `wc4` 上最强的是 Plücker ACT FOV55：`59/64 = 92.19%`，但和 baseline `58/64 = 90.62%` 差距很小。
2. `button_door` unseen 泛化最强的是 Keyframe + Plücker：`19/64 = 29.69%`。
3. `fire_door` unseen 上 Plücker 和 Keyframe + Plücker 最强，都是 `61/64 = 95.31%`。
4. Baseline ACT 在 seen door 很强，但对 button door 泛化非常差：`3/64 = 4.69%`。
5. Baseline ACT 对 fire door 直接评测已经很强：`56/64 = 87.50%`。
6. Fire door 50 条数据 finetune 没有提升 baseline：5k 持平，10k/20k 退化。
7. Plücker conditioning 对 unseen door 有明确帮助：button door `3/64 → 13/64`，fire door `56/64 → 61/64`。
8. Keyframe 单独不够强，但和 Plücker 结合后在 button door 上最好：`19/64`。
9. π0.5 在 fire door 上 `58/64 = 90.62%`，在 button door 上 `31/64 = 48.44%`，但在 seen `wc4` 上也只有 `31/64 = 48.44%`，说明它对 OOD door 有潜力，但当前 checkpoint 还不如 ACT baseline 稳定。
10. 目前最值得继续做的是更稳的 finetune protocol：混合旧数据 replay、降低 LR、冻结部分视觉 encoder、或只 finetune action head。

## 6. 日志位置

本地 baseline：

```text
high-level/logs/door-policy-success/wc4_5door_baseline_act_50k_seed615455575
high-level/logs/door-policy-success/button_door_5door_baseline_act_50k_seed615455575
high-level/logs/door-policy-success/firedoor_eval_5door_baseline_act_50k_seed615455575
high-level/logs/door-policy-success/fire_door_finetune5k_baseline_seed615455575
high-level/logs/door-policy-success/fire_door_finetune10k_baseline_seed615455575
high-level/logs/door-policy-success/fire_door_finetune20k_baseline_seed615455575
```

ps1 三个模型：

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_5door_keyframe_w3_r3_sample30_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_5door_plucker_fov55_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_seed615455575

/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/button_door_5door_keyframe_w3_r3_sample30_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/button_door_5door_plucker_fov55_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/button_door_5door_keyframe_w3_r3_sample30_plucker_fov55_50k_seed615455575

/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eval_5door_plucker_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eval_5door_kf_plucker_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eval_plucker_finetune20k_seed615455575_syncedcfg
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eval_kf_plucker_finetune20k_seed615455575_syncedcfg

/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_eval_pi05_5door_025000_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/firedoor_eval_pi05_5door_025000_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/buttondoor_eval_pi05_5door_025000_seed615455575
```

## 7. Recovery 数据微调实验

### 7.1 实验目的与设置

本实验从同一个 `Plücker ACT FOV55` 50K checkpoint 出发，对比两种 optimizer-reset finetune 设置：

1. **仅原始数据**：继续使用 250 条 5-door 专家轨迹，不加入 recovery 数据。
2. **Recovery 20%**：训练时 80% 从原始专家帧采样，20% 从经过仿真验证的 WC4 recovery 帧采样。

初始 checkpoint：

```text
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000
```

合并数据集包含：

| 数据来源 | Episodes | Frames | 合并数据中的自然帧比例 |
|---|---:|---:|---:|
| 原始 5-door 专家数据 | 250 | 125,000 | 82.74% |
| WC4 verified recovery | 98 | 26,080 | 17.26% |
| 合计 | 348 | 151,080 | 100% |

Recovery 20%实验使用显式采样器，将训练采样比例固定为 `expert=80%`、`recovery=20%`。两个微调实验都加载相同的模型权重，但重新初始化 AdamW optimizer；因此这里比较的是 finetune，而不是恢复原训练器 optimizer 状态后的无缝 resume。

WC4评测参数与本文第2节一致，关键设置如下：

```text
base_seed=615455575
实际 batch seed=615455575, 615455576, 615455577, 615455578
num_envs=16
total_trials=64
steps=1000
dp_action_horizon=25
dp_fps=25
success_metric=abs
pass_open_angle_deg=80
depth_only
depth_noise=off
gaussian_blur=off
camera_randomization=on, pos=0.02m, rot=5deg
robot_pitch=0.0
robot_pitch_randomization=[0.0, 0.0]
```

### 7.2 不同微调步数的 WC4 成功率

| 微调步数 | 仅原始数据 | Recovery 20% |
|---:|---:|---:|
| 0K | 59/64 = 92.19% | 59/64 = 92.19% |
| 5K | 58/64 = 90.62% | 53/64 = 82.81% |
| 20K | 57/64 = 89.06% | 49/64 = 76.56% |
| 50K | 55/64 = 85.94% | **60/64 = 93.75%** |

50K checkpoint 的分 batch 结果：

| 设置 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | Total |
|---|---:|---:|---:|---:|---:|
| 仅原始数据微调 50K | 15/16 | 12/16 | 13/16 | 15/16 | 55/64 |
| Recovery 20% 微调 50K | 16/16 | 14/16 | 15/16 | 15/16 | 60/64 |

### 7.3 结果分析

#### 7.3.1 Recovery 20% 存在明显的先下降、后适应过程

Recovery 20%模型在5K和20K阶段分别下降到 `82.81%` 和 `76.56%`，但在50K时回升到 `93.75%`。这说明加入 recovery 后，模型需要较长时间适应新的状态—动作分布；只评测5K或20K会得到“recovery有害”的过早结论。

早期下降可能来自多个因素共同作用：

1. 原始轨迹中的“继续闭合、转把手、推门”和 recovery 轨迹中的“张开、后退、重新对齐、重新抓取”在相似观测附近形成动作多模态。
2. 当前策略没有显式输入接触状态、恢复模式或任务阶段，需要仅凭双深度和 state10 自行区分正常主线与恢复状态。
3. finetune 使用合并数据集的新 normalization stats，旧模型在训练初期需要重新适应输入和动作归一化的变化。
4. Recovery episode 主要覆盖抓取、转把手和推门附近的状态，其阶段分布与完整专家轨迹不同。

训练到50K后，模型逐渐吸收 recovery行为，同时恢复了正常开门主线的稳定性。

#### 7.3.2 单纯继续训练原始数据会缓慢退化

仅原始数据的对照组从 `59/64` 依次下降到 `58/64`、`57/64`、`55/64`。这说明继续在相同250条轨迹上训练并不会自然提升 closed-loop 成功率，反而可能产生轻微过拟合或策略漂移。Open-loop训练loss继续下降，不代表closed-loop开门性能一定提升。

#### 7.3.3 Recovery 20% 在相同训练预算下取得正收益

50K时的公平对照为：

```text
仅原始数据：55/64 = 85.94%
Recovery 20%：60/64 = 93.75%
```

Recovery 20%比相同初始化、相同finetune步数的原始数据对照多成功5次，提高 `7.81` 个百分点；同时比微调前的Plücker 50K checkpoint多成功1次，提高 `1.56` 个百分点。

相对原始checkpoint的 `60/64 vs 59/64` 差异只有1次成功，暂时不能声称统计显著提升；但相对同训练步数对照的 `60/64 vs 55/64` 更能说明 recovery replay 抵消了持续训练造成的退化。后续仍应增加多组独立seed，并加入强制EE偏移、抓空和接触丢失等 recovery-specific 评测，确认提升是否来自真正的恢复能力。

### 7.4 Recovery 实验日志

```text
# 原始 Plücker 50K，作为 0K 对照
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_recovery_step00000_rerun_seed615455575

# 仅原始数据继续微调
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_original250_continue_step05000_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_original250_continue_step20000_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_original250_continue_step50000_seed615455575

# 原始 80% + Recovery 20%
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_recovery20_step05000_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_recovery20_step20000_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_recovery20_step50000_seed615455575
```
