# A2W ACT per-timestep keyframe W8/R3 horizon 评测（2026-07-02）

## 1. 模型

Checkpoint：

```text
high-level/dp/logs/lerobot-train/leroact_a2w_keyframe_perstep_w8_r3_sample20_nogating_chunk100_exec50_bs16_0702_0106/checkpoints/100000
```

Door-wrapped checkpoint：

```text
high-level/dp/logs/door-auto-wrapped/leroact_a2w_keyframe_perstep_w8_r3_sample20_nogating_chunk100_exec50_bs16_0702_0106/100000/model_latest.pt
```

训练设置：

- ACT ResNet18，chunk size 100。
- 关键帧窗口半径 3，action loss weight 8。
- 关键帧窗口采样 20%。
- 不启用 camera input gating。
- 与旧 W8/R3 keyframe 训练的主要差别：`loss.action_weight` 从 chunk-level scalar 加权改为 chunk 内 per-timestep action loss 加权。

## 2. 评测设置

- 每个 horizon 64 次 rollout。
- 每次 16 env，共 4 批。
- batch seed：62000–62003。
- WC4，1000 steps。
- 门轴绝对角度达到 80° 为成功。
- depth-only，深度范围 0.2–1.5 m。
- 不加 depth noise，不加 Gaussian blur。
- 开启相机位姿随机化：位置 ±0.02 m，角度 ±5°。
- 为了和旧报告对齐，本次显式关闭 base pitch 随机化：

```text
--robot_pitch 0.0
--ikpush_robot_pitch_rand_min 0.0
--ikpush_robot_pitch_rand_max 0.0
```

原始日志：

```text
high-level/logs/door-policy-success/perstep_w8r3_100k_h*_abs64_20260702
```

## 3. Horizon sweep

| Horizon | 成功数 | 成功率 |
|---:|---:|---:|
| 10 | 32/64 | 50.00% |
| 12 | **34/64** | **53.12%** |
| 14 | 29/64 | 45.31% |
| 15 | 28/64 | 43.75% |
| 50 | 32/64 | 50.00% |
| 100 | 23/64 | 35.94% |

当前最佳为 **H12：34/64（53.12%）**。

## 4. 和之前结果对比

| 模型 | 关键设置 | Gating | 最佳 horizon | 最佳成功率 |
|---|---|---:|---:|---:|
| 无关键帧 baseline 50k | 无 keyframe loss / sampler | 关 | H15 | **42/64（65.62%）** |
| 旧 keyframe W8/R3 50k | chunk-level scalar loss weight，sample 20% | 关 | H9/H11/H14 | 29/64（45.31%） |
| W3/R3 + sample30 50k | chunk-level scalar loss weight，sample 30% | 开 | H12/H14 | 39/64（60.94%） |
| 本次 per-timestep W8/R3 100k | per-timestep action loss weight，sample 20% | 关 | H12 | 34/64（53.12%） |

同 horizon 的主要对比：

| Horizon | 旧 W8/R3 no-gating 50k | 本次 per-timestep W8/R3 100k | W3/R3 + gating 50k | baseline 50k |
|---:|---:|---:|---:|---:|
| 10 | 28/64 | **32/64** | 31/64 | 35/64 |
| 12 | 26/64 | **34/64** | 39/64 | 34/64 |
| 14 | 29/64 | 29/64 | 39/64 | 36/64 |
| 15 | 25/64 | **28/64** | 36/64 | 42/64 |
| 50 | - | 32/64 | 34/64 | - |
| 100 | - | 23/64 | 22/64 | - |

## 5. 结论

1. per-timestep keyframe loss 相比旧 W8/R3 keyframe 训练有改善：
   - 旧 W8/R3 最佳 29/64。
   - 本次最佳 34/64。
   - 提升 5/64，但仍没有超过无关键帧 baseline。

2. H12 是当前模型最好的执行 horizon。
   - H10 和 H50 都是 32/64。
   - H12 提升到 34/64。
   - H15、H100 明显更差。

3. per-timestep 加权解决了“关键帧附近 anchor 整个 100-step chunk 被一起放大”的问题，但 W8/R3 + sample20 仍然可能偏强。
   - 当前结果优于旧 W8/R3，但低于 W3/R3 + gating，也低于无关键帧 baseline。
   - 下一步更值得试的是 per-timestep + 更温和权重，例如 W3/R3 或 W4/R3，并保留 sample 20%/30% 的消融。

## 6. 50k checkpoint 补测

用户指出上一轮误用了 100k checkpoint，因此额外补测同一 run 的 50k：

```text
high-level/dp/logs/lerobot-train/leroact_a2w_keyframe_perstep_w8_r3_sample20_nogating_chunk100_exec50_bs16_0702_0106/checkpoints/050000
```

评测设置与 100k 完全一致，只测用户指定的 H12 / H15 / H50 / H100。

| Checkpoint | H12 | H15 | H50 | H100 | 最佳 |
|---:|---:|---:|---:|---:|---:|
| 50k | **28/64** | **28/64** | 25/64 | 15/64 | H12/H15 |
| 100k | **34/64** | 28/64 | 32/64 | 23/64 | H12 |

50k 结果日志：

```text
high-level/logs/door-policy-success/perstep_w8r3_50k_h12_abs64_20260702
high-level/logs/door-policy-success/perstep_w8r3_50k_h15_abs64_20260702
high-level/logs/door-policy-success/perstep_w8r3_50k_h50_abs64_20260702
high-level/logs/door-policy-success/perstep_w8r3_50k_h100_abs64_20260702
```

50k 结论：

- H12/H15 并列最好，都是 28/64。
- H50 和 H100 明显更差，说明这个 checkpoint 不适合长 open-loop 执行。
- 与旧 W8/R3 no-gating 50k 相比，per-timestep 50k 在短 horizon 上只有小幅提升：
  - 旧 H12: 26/64，新 H12: 28/64。
  - 旧 H15: 25/64，新 H15: 28/64。
- 但它仍明显低于无关键帧 baseline 50k 和 W3/R3 + gating 50k。
