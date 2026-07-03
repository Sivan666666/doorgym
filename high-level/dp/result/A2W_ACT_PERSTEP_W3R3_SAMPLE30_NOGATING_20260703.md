# A2W ACT per-timestep keyframe W3/R3 sample30 no-gating 评测（2026-07-03）

## 1. 模型

训练 run：

```text
high-level/dp/logs/lerobot-train/leroact_a2w_keyframe_perstep_w3_r3_sample30_nogating_chunk100_exec50_bs16_0702_2354
```

训练设置：

- ACT ResNet18，chunk size 100，训练时 `n_action_steps=50`。
- `loss.action_weight` 为 per-timestep action loss weight。
- 关键帧窗口半径 R=3。
- 关键帧 loss weight W=3。
- 关键帧窗口采样比例 30%。
- Camera input gating 关闭。

评测 checkpoint：

```text
checkpoints/050000
checkpoints/100000
```

## 2. 评测设置

- 每组 64 次 rollout。
- 每次 16 env，共 4 批。
- batch seed：62000–62003。
- WC4，1000 steps。
- 门轴绝对角度达到 80° 判定成功。
- depth-only，深度范围 0.2–1.5 m。
- 不加 depth noise，不加 Gaussian blur。
- 开启相机位姿随机化：位置 ±0.02 m，角度 ±5°。
- 固定 base pitch：

```text
--robot_pitch 0.0
--ikpush_robot_pitch_rand_min 0.0
--ikpush_robot_pitch_rand_max 0.0
```

## 3. 结果

| Checkpoint | H12 | H14 | H15 | 最佳 |
|---:|---:|---:|---:|---:|
| 50k | **33/64（51.56%）** | 32/64（50.00%） | **33/64（51.56%）** | H12/H15 |
| 100k | 27/64（42.19%） | **34/64（53.12%）** | 29/64（45.31%） | H14 |

原始日志：

```text
high-level/logs/door-policy-success/w3r3_sample30_nogating_50k_h12_abs64_20260703
high-level/logs/door-policy-success/w3r3_sample30_nogating_50k_h14_abs64_20260703
high-level/logs/door-policy-success/w3r3_sample30_nogating_50k_h15_abs64_20260703
high-level/logs/door-policy-success/w3r3_sample30_nogating_100k_h12_abs64_20260703
high-level/logs/door-policy-success/w3r3_sample30_nogating_100k_h14_abs64_20260703
high-level/logs/door-policy-success/w3r3_sample30_nogating_100k_h15_abs64_20260703
```

## 4. 和之前结果对比

| 模型 | 关键设置 | Gating | 最佳 checkpoint/horizon | 最佳成功率 |
|---|---|---:|---:|---:|
| 无关键帧 baseline | 无 keyframe loss / sampler | 关 | 50k / H15 | **42/64（65.62%）** |
| 旧 keyframe W8/R3 | chunk-level scalar weight，sample20 | 关 | 50k / H9,H11,H14 | 29/64（45.31%） |
| per-timestep W8/R3 | per-timestep weight，sample20 | 关 | 100k / H12 | 34/64（53.12%） |
| W3/R3 + sample30 | chunk-level scalar weight，sample30 | 开 | 50k / H12,H14 | **39/64（60.94%）** |
| 本次 W3/R3 + sample30 | per-timestep weight，sample30 | 关 | 100k / H14 | 34/64（53.12%） |

## 5. 结论

1. 本次 W3/R3 + sample30 + no-gating 没有超过无关键帧 baseline。
   - 最佳 34/64，baseline 最佳 42/64。

2. 本次比旧 W8/R3 keyframe 好，但只恢复到 per-timestep W8/R3 100k 的水平。
   - 旧 W8/R3：29/64。
   - per-timestep W8/R3：34/64。
   - 本次 per-timestep W3/R3：34/64。

3. 50k 和 100k 没有稳定单调提升。
   - 50k：H12/H15 为 33/64。
   - 100k：H14 为 34/64，但 H12/H15 更低。
   - 说明继续训练到 100k 没有带来明显可靠提升，模型对执行 horizon 仍然敏感。

4. 和 W3/R3 + gating 的 39/64 相比，本次 no-gating 低 5/64。
   - 但这不能直接证明 gating 一定有效，因为旧 W3/R3 + gating 仍然使用 chunk-level scalar keyframe loss，而本次使用的是 per-timestep keyframe loss。
   - 严格判断 gating 贡献还需要同一 loss 形式下的 gating on/off 对照。

5. 当前最重要的结论仍然是：关键帧 loss/采样没有带来超过 baseline 的闭环成功率提升。
   - 更像是改善了一部分关键动作拟合，但仍没有解决闭环 rollout 中 grasp/rotate 阶段的动作稳定性问题。
