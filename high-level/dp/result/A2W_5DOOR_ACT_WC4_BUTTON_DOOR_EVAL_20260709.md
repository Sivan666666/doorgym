# A2W 5-door ACT 策略在 WC4 / Button Door / Fire Door 上的评测汇总

日期：2026-07-09，更新：2026-07-13
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

![第3章主要结果：seen/unseen door 成功率与 Fire Door finetune](figures/a2w_chapter3_seen_unseen_and_firedoor_finetune.png)

*图3-1：左图比较不同策略在 WC4、Button Door 和 Fire Door 上的成功率；右图展示 Fire Door 50 条数据 finetune 前后的变化。所有结果均为 64 次评测，ACT horizon 固定为 25。*

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

### 3.2.1 固定专家轨迹 Replay 与 Baseline ACT 对照

使用 WC4 训练数据中 `episode_000000` 的 action 作为固定开环轨迹，在保留环境域随机化的 WC4 评测环境中重复执行 64 次，并与 Baseline ACT 50K 的闭环策略评测结果对比：

| 控制方式 | WC4 成功率 | Batch 0 | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|---:|---:|
| 固定 WC4 episode 0 action replay | **9/64 = 14.06%** | 2/16 | 6/16 | 1/16 | 0/16 |
| Baseline ACT 50K | **58/64 = 90.62%** | 16/16 | 12/16 | 14/16 | 16/16 |

在同样存在环境随机化的条件下，固定专家 action replay 比 Baseline ACT 低 `76.56` 个百分点。这说明单条开环轨迹无法有效适应随机化引起的观测和交互偏差，而 ACT 的闭环重规划是该评测中成功率大幅提升的关键。

### 3.2.2 Baseline ACT Action Horizon 对比

固定 Baseline ACT 50K checkpoint、WC4、64 个 trials、batch seeds `615455575–615455578` 以及第 2 章中的全部域随机化参数，仅改变每次推理后连续执行的 action 数量：

| Action horizon | Batch 0 | Batch 1 | Batch 2 | Batch 3 | 总成功率 | 相对 horizon=25 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5/16 | 5/16 | 8/16 | 5/16 | 23/64 = 35.94% | -54.69 pp |
| 10 | 4/16 | 5/16 | 9/16 | 5/16 | 23/64 = 35.94% | -54.69 pp |
| **25** | **16/16** | **12/16** | **14/16** | **16/16** | **58/64 = 90.62%** | - |
| 50 | 12/16 | 12/16 | 12/16 | 13/16 | 49/64 = 76.56% | -14.06 pp |
| 75 | 13/16 | 9/16 | 10/16 | 12/16 | 44/64 = 68.75% | -21.87 pp |
| 100 | 6/16 | 7/16 | 9/16 | 9/16 | 31/64 = 48.44% | -42.19 pp |

结果不是“horizon 越短越好”，而是在当前已测配置中 `horizon=25` 最佳。horizon 5/10 时，控制器很快丢弃当前 chunk 并从新 observation 重新预测；模型反复执行每个 chunk 最前面的短段，远期动作相位难以稳定推进，41 个失败中分别有 40/41 个 rollout 的最大开门角低于 `5°`。这说明策略多数没有稳定完成抓取和转把手，而不是推门距离不足。

horizon 从 25 增大到 50/75/100 后，成功率又逐步下降。ACT 虽然一次预测 100-step chunk，但后半段是基于旧 observation 的远期预测；连续执行越久，机器人和门的实际状态越容易偏离预测条件，抓取、转把手和推门阶段的误差无法及时由新视觉观测纠正。`horizon=100` 等于完整 chunk 全开环执行，成功率仅剩 `48.44%`。

因此该 checkpoint 存在明显的折中：太短会造成 chunk 前段反复执行和相位推进不足，太长会造成开环误差积累；`horizon=25` 在两者之间取得最佳平衡。

失败仍主要发生在接触建立之前：horizon 25/50/75/100 分别有 `6/14/19/33` 个失败 rollout 的最大开门角低于 `5°`。因此长 horizon 造成的主要问题不是门已经打开但未达到 80°，而是把手定位、闭合夹爪或转把手阶段的早期偏差被持续执行的旧 chunk 放大。

原始日志：

- `high-level/logs/door-policy-success/wc4_baseline_act50k_h5_seed615455575/summary.json`
- `high-level/logs/door-policy-success/wc4_baseline_act50k_h10_seed615455575/summary.json`
- `high-level/logs/door-policy-success/wc4_5door_baseline_act_50k_seed615455575/summary.json`
- `high-level/logs/door-policy-success/wc4_baseline_act50k_h50_seed615455575/summary.json`
- `high-level/logs/door-policy-success/wc4_baseline_act50k_h75_seed615455575/summary.json`
- `high-level/logs/door-policy-success/wc4_baseline_act50k_h100_seed615455575/summary.json`

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

![第7章主要结果：Recovery 20% 微调曲线](figures/a2w_chapter7_recovery_finetune.png)

*图7-1：左图显示仅使用原始数据与加入 Recovery 20% 后的 WC4 成功率变化；右图给出 Recovery 20% 相对同训练步数原始数据对照的成功率差值。*

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

## 8. RGB 双相机策略评测

### 8.1 数据、模型与输入

本节单独记录使用 WC4 RGB 数据训练的 Baseline ACT 和 π0.5。两种策略使用同一份 50-episode 数据集：

```text
local/door_a2w_wc4_rgb_state10_camerarand_50
```

模型的视觉输入实际包含 4 路 `480×640` tensor，而不只是两张 RGB：

```text
observation.images.wrist_handle_mask
observation.images.wrist_rgb
observation.images.front_handle_mask
observation.images.front_rgb
```

此外输入 `observation.state: 10D`，输出 `action: 10D`；`chunk_size=100`，训练配置中的 `n_action_steps=50`。训练时 `image_transforms.enable=false`。

| 策略 | 训练/评测 checkpoint | 训练步数 |
|---|---|---:|
| RGB Baseline ACT | `leroact_a2w_wc4_rgb_state10_camerarand_50_baseline_50k_chunk100_exec50_bs16_0713_0200/checkpoints/050000` | 50K |
| RGB π0.5 | `pi05_a2w_wc4_rgb_state10_camerarand_50_steps30k_chunk100_exec50_bs16_0713_0200/checkpoints/010000` | 10K（训练任务总预算为 30K） |

### 8.2 统一评测参数

两种 RGB 策略使用完全相同的 WC4 评测设置：

```text
door=wc4
base_seed=615455575
实际 batch seed=615455575, 615455576, 615455577, 615455578
num_envs=16
total_trials=64
steps=1000
dp_action_horizon=25
dp_fps=25
success_metric=abs
pass_open_angle_deg=80
rgb=true
depth_only=false
depth_noise=off
camera_randomization=on, pos=0.02m, rot=5deg
```

### 8.3 WC4 成功率

| RGB 策略 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | 总成功率 |
|---|---:|---:|---:|---:|---:|
| Baseline ACT 50K | 2/16 | 1/16 | 4/16 | 1/16 | **8/64 = 12.50%** |
| π0.5 10K | 8/16 | 8/16 | 9/16 | 10/16 | **35/64 = 54.69%** |

在相同 RGB 数据和评测条件下，π0.5 10K 比 Baseline ACT 50K 多成功 27 次，提高 `42.19` 个百分点。结果说明 π0.5 对这批 RGB + handle-mask 输入的利用明显好于当前 ResNet18 ACT。

### 8.4 结果解释与对比限制

1. RGB Baseline ACT 的 `12.50%` 明显偏低，主要可能与训练集只有 50 条轨迹、4 路视觉输入带来的学习难度以及没有启用 RGB image augmentation 有关。训练 loss 收敛并不保证抓取、闭合和转把手阶段的 closed-loop 时序稳定。
2. π0.5 在仅训练 10K 时已经达到 `54.69%`，说明预训练视觉语言模型对 RGB 局部语义、把手区域和双视角信息具有更好的样本效率；但这个成功率仍低于本文 250 条 5-door depth-only Baseline/Plücker ACT 的高成功率区间。
3. 不能把 RGB Baseline ACT 的 `8/64` 与主表中 depth-only 5-door Baseline ACT 的 `58/64` 直接解释为“RGB 不如 depth”。前者只训练于 50 条 WC4 数据，后者训练于 250 条 5-door 数据，训练集规模和门分布都不同。
4. 当前所谓 RGB 模式还包含 front/wrist handle mask，因此也不是纯粹的“两张 RGB”消融。若要严格比较 RGB 与 depth，应从同一批 raw episode 生成两份数据，保持 episode、state/action、模型、训练步数和随机种子一致，只改变视觉模态。

### 8.5 RGB 评测日志

```text
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_rgb_baseline_act_50k_h25_seed615455575_gpu3
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_rgb_pi05_10k_h25_seed615455575_gpu3
```

## 9. 加入辅助目标后的策略退化

### 9.1 实验目的与模型变体

本节记录 End Signal 和 Interaction State 两类辅助目标加入 ACT 后的离线预测精度与 WC4 闭环成功率。辅助目标的设计目的分别是：

- **End Signal**：从 ACT decoder feature 预测任务是否进入 `return_home/hold_home`，用于后续自主结束检测；当前评测为 monitor-only，不用 end probability 修改运动动作或提前停止 rollout。
- **Interaction State**：预测双指把手接触、把手解锁进度和门打开进度，使策略获得显式交互状态表示。

共测试了以下四种辅助目标实现：

| 模型 | 辅助目标对 ACT 主干的梯度 | Interaction 是否反馈给 decoder | 辅助 loss 权重 |
|---|---|---|---:|
| Baseline ACT + End Signal（联合训练） | `L_end` 回传 decoder 和共享主干 | 不适用 | 1.0 |
| Plücker + End Signal Detach | `decoder_feature.detach()`，`L_end` 只更新 End head | 不适用 | 0.1 |
| Plücker + Interaction State（Full） | 三个 interaction loss 回传 encoder | 是 | 0.1 / 0.1 / 0.1 |
| Plücker + Interaction Probe-only | `encoder_out[1:].detach()`，interaction loss 只更新 probe head | 否 | 0.1 / 0.1 / 0.1 |

Probe-only 的预期数据流为：

```text
L_action      -> 正常更新 ACT 主干和 motion head
L_interaction -> 只更新 interaction probe head
interaction prediction 不送入 action decoder
```

因此 Probe-only 用于检查原 ACT 特征是否已经包含可线性/轻量解码的交互状态，同时尽量避免辅助目标直接干扰 motion policy。

### 9.2 统一 WC4 闭环结果

本节闭环评测继续使用与第2节相同的协议：

```text
door=wc4
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

| 模型 | Batch 0 | Batch 1 | Batch 2 | Batch 3 | WC4 成功率 | 相对对应参考模型 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ACT 50K | 16/16 | 12/16 | 14/16 | 16/16 | **58/64 = 90.62%** | 参考 |
| Baseline + End Signal，联合训练 | - | - | - | - | 37/64 = 57.81% | 相对 Baseline `-32.81 pp` |
| Plücker ACT FOV55 50K | 16/16 | 14/16 | 13/16 | 16/16 | **59/64 = 92.19%** | 参考 |
| Plücker + End Signal Detach，weight=0.1 | 12/16 | 8/16 | 12/16 | 9/16 | 41/64 = 64.06% | 相对 Plücker `-28.13 pp` |
| Plücker + Interaction State（Full） | - | - | - | - | 17/64 = 26.56% | 相对 Plücker `-65.63 pp` |
| Plücker + Interaction Probe-only | 9/16 | 6/16 | 6/16 | 7/16 | 28/64 = 43.75% | 相对 Plücker `-48.44 pp` |

Probe-only 的失败门角具有明显双峰：成功环境通常达到约 `88°–90°`，绝大多数失败环境最大门角只有约 `1.6°–2.8°`；仅发现一个失败环境达到 `76.4°`。因此其主要失败发生在建立抓取、转把手或解锁链路，而不是大量样本停在80°阈值附近。这里已经显式使用 `success_metric=abs`，结果不是门开合方向判反造成的。

### 9.3 Offline expert-observation 诊断

离线诊断使用同一组 WC4 专家轨迹：episode `200, 205, ..., 245`，每10帧采样一个当前观测，每个观测比较未来25步动作，共比较12,000个 motion timestep。该测试位于专家观测分布，不能替代闭环评测。

| 模型 | Motion L1 ↓ | 归一化 L1 ↓ | Position L2 ↓ | Quaternion error ↓ | Gripper MAE ↓ | 全10D同时过阈值 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 旧 Baseline ACT | 0.00631 | 0.09880 | 0.01722 m | 1.540° | 0.01101 | 82.38% |
| Baseline + End Signal（联合训练） | 0.00766 | 0.10579 | 0.02220 m | 1.772° | 0.01334 | 72.49% |
| 旧 Plücker ACT FOV55 | **0.00576** | **0.09576** | **0.01641 m** | **1.221°** | 0.01151 | **83.43%** |
| Plücker + Interaction State（Full） | 0.00661 | 0.10109 | 0.01948 m | 1.629° | **0.00960** | 82.38% |

辅助目标本身在专家观测上预测得很准确：

| 辅助目标 | 主要离线指标 |
|---|---:|
| End Signal | BCE `0.00252`，Accuracy `99.925%`，F1 `99.816%` |
| Contact | Accuracy `99.583%`，Precision `100.000%`，Recall `98.450%`，F1 `99.219%` |
| Handle progress | MAE `0.01338`，RMSE `0.02550` |
| Door progress | MAE `0.01387`，RMSE `0.02006` |

这说明 End/Interaction 标签生成和辅助 head 的拟合本身基本正常。旧联合训练模型的 motion 离线误差却同步增大，并且 closed-loop 成功率大幅下降，说明“辅助目标预测准确”不等于“辅助目标能改善连续控制”。

新训练的 End Detach 和 Interaction Probe-only checkpoint 尚未补跑同一套 offline expert-observation 指标，因此当前只能对其闭环结果下结论，不能假定它们的 motion L1 已经恢复到旧 Plücker 水平。

### 9.4 退化分析

#### 9.4.1 完整 Interaction feedback 会放大闭环误差

Full Interaction 模型把预测的 contact、handle progress 和 door progress 重新注入 decoder。即使专家观测上的预测误差很小，策略一旦偏离专家轨迹，interaction prediction 就进入训练数据覆盖不足的状态；预测偏差改变动作，动作又改变下一帧观测，可能形成闭环误差放大。`17/64` 的结果和这一风险一致。

Probe-only 取消 feedback 后从 `17/64` 提升到 `28/64`，多成功11次，说明 decoder feedback 确实可能是 Full Interaction 退化的一部分来源。但是 `28/64` 仍远低于旧 Plücker 的 `59/64`，所以不能把全部退化都归因于 feedback conditioning。

#### 9.4.2 Detach 并没有在当前实验中恢复 closed-loop motion

End Detach 已阻断 `L_end` 到 motion decoder 的直接反向传播，Interaction Probe-only 也阻断了三个辅助 loss 到 encoder 的直接梯度，并且不把预测状态送入 decoder。从计算图设计看，两者都应最大限度保留正常 motion 学习路径。

但当前闭环结果仍分别只有 `41/64` 和 `28/64`。这表明“阻断辅助 loss 的直接梯度”本身还不足以保证重新训练出的策略复现旧 Plücker checkpoint 的性能。

可能因素包括：

1. 新旧模型是分别从头训练的独立 run，不是同一组初始权重逐项开关辅助 head；额外模块初始化可能改变全局 RNG 消耗、DataLoader采样顺序或后续优化轨迹。ACT closed-loop 对小的训练差异可能非常敏感。
2. 新模型使用带 aux feature 的重新转换数据集。理论上原 observation/action 应相同，但仍需逐帧核对 action、state、depth解码结果和 normalization stats，不能只依据 raw 来源相同就视为字节级一致。
3. 即使辅助特征已 `detach`，如果训练器对全部参数做全局 gradient norm/clipping，probe head 的梯度仍可能通过全局缩放间接改变 motion 参数的有效更新；需要结合每步 grad norm 和 clipping阈值验证这一机制是否实际发生。
4. 当前结果可能包含普通的训练随机性，而64次 closed-loop评测会把很小的抓取位置/姿态偏差放大为“门完全没开”和“成功到90°”两类结果。
5. Probe-only 代码的固定输入单元测试只能证明单次 forward 不向 decoder 注入 interaction，也只能证明 interaction loss 不直接更新主干；它不能自动保证两个独立50K训练run得到相同的 motion权重。

#### 9.4.3 当前可以确认与不能确认的结论

可以确认：

- 两类辅助 target 在专家观测上可以被高精度预测。
- 直接联合训练 End Signal 会提高 motion 离线误差，并明显降低闭环成功率。
- Full Interaction feedback 的闭环性能最差；取消 feedback 后有所恢复，但仍未恢复到旧 Plücker。
- 当前失败主要发生在抓取/解锁前段，不是成功阈值或门角符号错误。

当前不能确认：

- 不能仅凭这些独立训练run断言“任何 detach/probe head 都必然损害 ACT”。
- 不能把 Probe-only 的全部下降归因于 interaction loss，因为其对主干没有直接梯度；需要严格的同期 Plücker-only控制组。
- 不能根据辅助状态的高离线精度推断它在策略偏离专家分布后仍然可靠。

### 9.5 下一步严格消融

最关键的对照是用当前同一代码、同一重新转换数据、同一训练 seed 和显式相同配置，同时训练：

```text
A. Plücker-only
B. Plücker + End Detach
C. Plücker + Interaction Probe-only
```

除辅助模块开关外，三组必须固定：模型初始化、DataLoader采样序列、normalization stats、`use_action_loss_weight=false`、batch size、训练步数和保存频率。为了排除额外模块初始化消耗 RNG 的影响，应在创建模型和 DataLoader 时使用独立 generator，或保存并复用完全相同的 Plücker 主干初始 state dict。

同时补充以下检查：

1. 对 End Detach 和 Probe-only 跑同一套 offline expert-observation 诊断。
2. 比较新数据集与原250条数据的 observation/action、episode边界和 normalization stats。
3. 记录主干 gradient norm、辅助 head gradient norm以及是否触发全局 clipping。
4. 保存失败 rollout，统计 contact establishment、handle unlock 和 push-door 三阶段失败数量。

只有同期 Plücker-only控制组仍接近 `59/64`，而 Detach/Probe-only显著更低时，才能把退化可靠地归因于辅助模块或训练器耦合；如果同期控制组也明显下降，则主要问题应优先从代码版本、数据转换和训练随机性中排查。

### 9.6 Checkpoint、离线报告与评测日志

```text
# Baseline + End Signal（联合训练）
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_baseline_end_signal_50k_chunk100_exec50_bs16_0712_2331/checkpoints/050000

# Plücker + Interaction State（Full）
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_plucker_fov55_interaction_state_50k_chunk100_exec50_bs16_0713_0051/checkpoints/050000

# Plücker + End Signal Detach，weight=0.1
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_plucker_fov55_end_signal_detach_w01_noactionweight_50k_chunk100_exec50_bs16_0713_1724/checkpoints/050000

# Plücker + Interaction Probe-only
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/leroact_a2w_5door_plucker_fov55_interaction_probeonly_w01_noactionweight_50k_chunk100_exec50_bs16_0713_1750/checkpoints/050000

# 新版本闭环评测日志
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_end_detach_w01_50k_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_plucker_interaction_probeonly_50k_seed615455575

# 本地离线专家观测诊断
high-level/dp/result/offline_expert_diagnostics_20260713/README.md
high-level/dp/result/offline_expert_diagnostics_20260713/*.json
```

## 10. 从原始 59/64 Plücker checkpoint 分叉的严格 Interaction 辅助目标消融

本节用于回答一个更严格的问题：Interaction State 辅助目标能否改善已经达到
`59/64` 的原始 Plücker ACT，而不是比较两个相互独立、从头训练的随机 run。

### 10.1 实验设计与一致性检查

两条分支均从同一个 checkpoint 初始化：

```text
leroact_a2w_5door_plucker_fov55_50k_chunk100_exec50_bs16_0709_0209/checkpoints/050000
```

分支定义：

```text
A. Action-only：仅使用原 ACT action loss 继续训练
B. Interaction auxiliary-only：action loss + 三个 0.1 权重的 interaction loss
```

Interaction 分支不把预测状态送入 decoder，只允许辅助 loss 更新 encoder 和
Interaction head。两组都重新创建 optimizer，避免 Interaction 新 head 与旧 optimizer
状态不匹配；两组使用相同的新 optimizer 配置。

严格控制项：

```text
dataset = door_a2w_5door250_interaction_state
train seed = 1000
dataloader seed = 424242
batch size = 16
learning rate = 1e-5
optimizer foreach = false
strict determinism = true
Plucker deterministic pooling = true
save steps = 5K, 10K
```

初始化检查结果：

- 两组公共主干共有240个 tensor，逐元素比较 `changed_common=0`。
- 公共主干 SHA-256：`4d1cd97dcae2ffafb2d7e2d6e4a461f900f4b6dc9b2c046f242b78f26b6706d0`。
- 前两个 batch 的全部样本索引完全一致。
- 第一步训练前 motion L1 完全相同：`0.043962676078`。
- Interaction 分支仅多出13个 Interaction head/conditioner tensor。

评测配置与原始59/64结果一致：

```text
door = wc4
trials = 64（4 batches × 16 envs）
seeds = 615455575, 615455576, 615455577, 615455578
horizon = 25
steps = 1000
success = abs(door angle) >= 80 deg
camera randomization = position 0.02 m, rotation 5 deg
depth noise = off
Gaussian blur = off
robot pitch = 0
```

另外增加了一个0K控制：保持旧权重不训练，只把 Plucker pooling 切换为训练分支使用的
确定性实现。该模型仍得到 `59/64`，且四个 batch 仍为 `16/14/13/16`，说明后续下降
不是 pooling 或评测配置变化造成的。

### 10.2 主要结果

| 额外微调步数 | Action-only | Interaction auxiliary-only | Interaction 相对同期 Action-only |
|---:|---:|---:|---:|
| 0K | **59/64 = 92.19%** | 同一初始化 | - |
| 5K | 38/64 = 59.38% | **54/64 = 84.38%** | **+25.00 pp** |
| 10K | 46/64 = 71.88% | **53/64 = 82.81%** | **+10.94 pp** |

分 batch 结果：

| 模型 | Batch 0 | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|---:|
| 0K 原始 Plücker | 16/16 | 14/16 | 13/16 | 16/16 |
| 5K Action-only | 9/16 | 9/16 | 10/16 | 10/16 |
| 5K Interaction | 16/16 | 12/16 | 12/16 | 14/16 |
| 10K Action-only | 13/16 | 12/16 | 9/16 | 12/16 |
| 10K Interaction | 15/16 | 12/16 | 12/16 | 14/16 |

同 trial 配对统计：

| 对比 | Action失败/Interaction成功 | Action成功/Interaction失败 | 两者成功 | 两者失败 | McNemar精确检验 |
|---|---:|---:|---:|---:|---:|
| 5K | 18 | 2 | 36 | 8 | `p=0.000402` |
| 10K | 7 | 0 | 46 | 11 | `p=0.015625` |

门角统计也支持相同结论：

| 模型 | 64次最大门角均值 | 失败trial最大门角均值 |
|---|---:|---:|
| 0K 原始 Plücker | 83.15° | 2.30° |
| 5K Action-only | 54.13° | 1.70° |
| 5K Interaction | 76.55° | 3.92° |
| 10K Action-only | 65.11° | 1.49° |
| 10K Interaction | 74.85° | 1.88° |

### 10.3 系统性结论

1. **Interaction auxiliary-only 相对同期 Action-only 有稳定且显著的提升。** 5K时多成功16次，
   10K时多成功7次；配对检验均显著。这说明在完全相同的初始化、batch和训练RNG下，
   Interaction监督能够减轻继续训练导致的策略漂移。
2. **Interaction并未超过原始最佳checkpoint。** 最好的Interaction结果是5K的`54/64`，仍低于
   0K的`59/64`。因此当前证据支持“辅助目标具有正则化/抗退化作用”，不支持“它已经改善
   原始92.19%策略”。
3. **主要问题是继续微调本身造成灾难性遗忘或策略漂移。** 仅Action微调5K后从`59/64`
   降到`38/64`；到10K虽恢复至`46/64`，仍明显低于起点。训练loss继续下降并不代表闭环
   控制性能提高。
4. **下降不是数据、pooling、seed或评测门角符号错误。** 数据公共字段和视频已逐项验证一致；
   0K确定性pooling控制仍为`59/64`；所有模型使用相同四组评测seed和`abs(angle)>=80°`。
5. **当前最合理的模型选择仍是原始0K Plücker 59/64。** 如果继续研究Interaction，应降低
   主干学习率、短步数微调，或冻结大部分ACT后只逐步解冻视觉/encoder层，而不应直接用
   `1e-5`全参数继续训练5K以上并期待自动提升。

### 10.4 Checkpoint与评测日志

```text
# Action-only训练
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/finetune_old59_plucker_actiononly_strict_10k_seed1000_dl424242_0714/checkpoints/{005000,010000}

# Interaction auxiliary-only训练
/home/ps/workspace/txc/doorgym/high-level/dp/logs/lerobot-train/finetune_old59_plucker_interaction_auxonly_w01_strict_10k_seed1000_dl424242_0714/checkpoints/{005000,010000}

# 0K确定性pooling控制
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_old59_0k_deterministic_pooling_seed615455575

# 5K评测
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_old59_finetune_actiononly_5k_strict_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_old59_finetune_interaction_auxonly_5k_strict_seed615455575

# 10K评测
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_old59_finetune_actiononly_10k_strict_seed615455575
/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success/wc4_old59_finetune_interaction_auxonly_10k_strict_seed615455575
```
