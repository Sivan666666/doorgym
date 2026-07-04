# A2W ACT phase200 W3 loss-only 50k 评测（2026-07-03）

## 1. 被测 checkpoint

训练 run：

```text
high-level/dp/logs/lerobot-train/leroact_a2w_phase200_w3_lossonly_nogating_chunk100_exec50_bs16_0703_1620/checkpoints/050000
```

包装后的 Door policy manifest：

```text
high-level/dp/logs/door-auto-wrapped/leroact_a2w_phase200_w3_lossonly_nogating_chunk100_exec50_bs16_0703_1620/model_latest.pt
```

该模型使用：

- ACT depth-only ResNet18。
- `policy.use_action_loss_weight=true`。
- `keyframe_sampling_ratio=0.0`，即没有关键帧重采样。
- camera gating 关闭。
- 关键帧权重来自数据集 `loss.action_weight`，本轮是 phase200 区间权重 W=3。

## 2. 评测设置

与之前 baseline / keyframe 对比保持一致：

- `num_envs=16`
- `total_trials=64`
- `steps=1000`
- `success_metric=abs`
- `pass_open_angle_deg=80`
- `base_seed=62000`，共 4 个 batch：seed 62000–62003
- 不开 depth noise
- 不开 Gaussian blur
- 开启 depth camera pose/rot randomization：
  - `depth_camera_pos_rand_m=0.02`
  - `depth_camera_rot_rand_deg=5.0`
- robot pitch 固定：
  - `robot_pitch=0.0`
  - `ikpush_robot_pitch_rand_min=0.0`
  - `ikpush_robot_pitch_rand_max=0.0`

## 3. Horizon sweep

| Checkpoint | H10 | H12 | H14 | H15 | 最佳 |
|---|---:|---:|---:|---:|---|
| phase200 W3 loss-only 50k | 26/64（40.62%） | 33/64（51.56%） | **35/64（54.69%）** | 34/64（53.12%） | H14 |

对应日志：

```text
high-level/logs/door-policy-success/phase200_w3_lossonly_50k_h10_abs64_clean_20260703
high-level/logs/door-policy-success/phase200_w3_lossonly_50k_h12_abs64_clean2_20260703
high-level/logs/door-policy-success/phase200_w3_lossonly_50k_h14_abs64_clean2_20260703
high-level/logs/door-policy-success/phase200_w3_lossonly_50k_h15_abs64_clean2_20260703
```

## 4. 和之前结果对比

| 模型 | 关键帧方式 | Gating | 最佳结果 |
|---|---|---:|---:|
| 无关键帧 baseline 50k | 无 keyframe loss / sampler | 关 | **42/64（65.62%）**，H15 |
| 旧 W8/R3 keyframe 50k | chunk-level scalar weight，sample20 | 关 | 29/64（45.31%），H9/H11/H14 |
| per-timestep W8/R3 50k | per-timestep weight，sample20 | 关 | 28/64（43.75%），H12/H15 |
| per-timestep W3/R3 50k | per-timestep weight，sample30 | 关 | 33/64（51.56%），H12/H15 |
| W3/R3 + sample30 50k | chunk-level scalar weight，sample30 | 开 | 39/64（60.94%），H12/H14 |
| 本次 phase200 W3 loss-only 50k | phase200 frames weight=3，无重采样 | 关 | **35/64（54.69%）**，H14 |

## 5. 结论

1. 本次 phase200 W3 loss-only 的最佳成功率是 **35/64（54.69%）**。

2. 它比之前 no-gating 的 keyframe 版本略好：
   - 旧 W8/R3：29/64。
   - per-timestep W3/R3 sample30：33/64。
   - 本次 phase200 W3 loss-only：35/64。

3. 但它仍然没有超过无关键帧 baseline：
   - baseline 最佳 42/64。
   - 本次最佳 35/64。
   - 低 7/64，约低 10.94 个百分点。

4. 从 horizon 依赖看，H14 最好，H12/H15 接近，H10 明显差。这说明该模型仍然依赖 action chunk 执行长度；关键动作时序并没有被 phase200 loss 完全稳定下来。

5. 当前结果支持一个比较清楚的判断：单纯把 `grasp -> close gripper -> rotate` 这段长区间加权到 3，可以比之前一些关键帧 loss 方案稳定一点，但还不足以解决 ACT 在闭环中 gripper/rotate 时序漂移的问题。

