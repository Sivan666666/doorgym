# A2W Door ACT 评测总表：WC4 与 unseen doors

Date: 2026-07-06

## 1. 改进方法概览

本轮 A2W-Z1 开门策略围绕 ACT 做了几类改进。所有方法的 action 输出仍然是原来的 door policy action chunk，核心区别在训练监督、视觉几何条件和视觉特征融合方式。

1. **Baseline ACT**
   - 输入：front depth、wrist depth、10D state。
   - 模型：LeRobot ACT + ResNet18 visual backbone。
   - 不使用 keyframe loss、不使用 camera gating、不使用 Plücker ray、不使用 handle-latent auxiliary loss。

2. **Keyframe ACT**
   - 目标：让模型更重视 `pregrasp → grasp → close gripper → rotate handle → push` 这些短关键事件。
   - 尝试过两类 loss：
     - chunk-level scalar keyframe weight：一个样本的整个 100-step action chunk 一起加权。
     - per-timestep action weight：只对 chunk 内靠近关键帧的 timestep 加权。
   - 尝试过不同参数：
     - W8/R3/sample20。
     - W3/R3/sample30。
     - phase200：直接把 grasp/close/rotate 约 200 step 区间设为高权重。

3. **Camera input gating**
   - 在视觉 encoder 后、Transformer 前，对 front/wrist visual tokens 学习两个 gate。
   - 使用 `2 * softmax`，初始化附近等价于 front=1、wrist=1，兼容原始 ACT。
   - 目标：让模型按阶段自动调整 front 与 wrist camera 的相对重要性。

4. **Plücker-conditioned ACT**
   - 为每个 pixel 生成 camera ray 的 6D Plücker 表示 `[d, m]`。
   - Plücker map 在 480×640 原分辨率生成，再由单独 small CNN 编码。
   - depth image 仍然走 pretrained ResNet，不直接把 6D ray 拼进 ResNet 输入。
   - 目标：显式注入相机内参/外参，尤其 wrist camera 的动态空间几何。

5. **DINOv2 Handle-Latent Auxiliary**
   - 训练时从 handle bbox crop depth patch，用 frozen DINOv2-small 生成 front/wrist handle latent。
   - ACT 从完整图像 token 中通过 handle query cross-attention 预测 handle latent。
   - 推理时不需要 DINOv2、bbox、crop，只保留 ACT 主干。
   - 目标：用辅助监督迫使视觉 token 更关注把手区域。

## 2. WC4 主结果汇总

统一评测设置，除非特别注明：

- Door: `wc4`
- 64 trials，16 env × 4 batches
- `base_seed=62000`
- success: door absolute open angle ≥ 80°
- depth range: 0.2–1.5 m
- depth noise off
- Gaussian blur off
- robot pitch fixed at 0
- camera pose randomization on: ±0.02 m, ±5°

| Model / Variant | 主要改动 | Checkpoint | Best horizon | WC4 success | 备注 |
|---|---|---:|---:|---:|---|
| **Baseline ACT** | 无 keyframe / gating / Plücker | 50k | H15 | **42/64 = 65.62%** | 当前 WC4 camera-rand 下最强 baseline |
| Keyframe W8/R3 sample20 | 早期 keyframe，加权较强 | 50k | H9/H11/H14 | 29/64 = 45.31% | keyframe 过强，夹爪/rotate 稳定性下降 |
| Per-timestep keyframe W8/R3 sample20 | per-timestep loss weight | 50k | H12/H15 | 28/64 = 43.75% | 未超过 baseline |
| Per-timestep keyframe W8/R3 sample20 | per-timestep loss weight | 100k | H12 | 34/64 = 53.12% | 100k 好于 50k，但仍低于 baseline |
| Keyframe W3/R3 sample30 + gating | 较温和 keyframe + camera gating | 50k | H12/H14 | 39/64 = 60.94% | 接近 baseline，但夹爪重开更多 |
| Per-timestep keyframe W3/R3 sample30, no gating | 温和 keyframe，关闭 gating | 50k | H12/H15 | 33/64 = 51.56% | 关闭 gating 后下降 |
| Per-timestep keyframe W3/R3 sample30, no gating | 温和 keyframe，关闭 gating | 100k | H14 | 34/64 = 53.12% | 继续训练未带来稳定提升 |
| Phase200 W3 loss-only | grasp/close/rotate 约 200 step 加权 | 50k | H14 | 35/64 = 54.69% | 比 per-timestep W3 小幅好，但仍低于 baseline |
| Plücker FOV55 | Plücker ray，训练数据无 camera-rand | 50k | H12 | 11/64 = 17.19% | 与 camera-rand eval 分布严重不匹配 |
| Plücker FOV55 | Plücker ray，训练数据无 camera-rand | 50k | H12 | 48/64 = 75.00% | **关闭 camera-rand eval** 时结果；说明分布匹配很关键 |
| Plücker FOV55 no-handle-aux | Plücker ray，camera-rand 数据，关闭 handle aux | 50k | H14 | 37/64 = 57.81% | 公平 camera-rand Plücker 对照，仍低于 baseline |
| Plücker + DINOv2 handle-latent aux | Plücker + handle latent auxiliary loss | 50k | H12 | 26/64 = 40.63% | handle aux 在当前设置下明显负收益 |

## 3. Unseen 4door 结果：短 horizon 与 H25 对比

Unseen doors:

- `99650089960001`
- `99650089960006`
- `99655039960001`
- `99655039960006`

统一评测设置：

- 每个 door / policy 16 trials
- `base_seed=63100`
- success: door absolute open angle ≥ 80°
- camera pose randomization on
- depth noise off
- Gaussian blur off
- robot pitch fixed at 0

### 3.1 原先各模型最佳/常用短 horizon

| Policy | Horizon | 99650089960001 | 99650089960006 | 99655039960001 | 99655039960006 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ACT | H15 | 2/16 | 0/16 | 0/16 | 0/16 | 2/64 = 3.12% |
| Plücker no-aux | H14 | 2/16 | 1/16 | 0/16 | 0/16 | 3/64 = 4.69% |
| Keyframe W3 no-gating | H12 | 4/16 | 3/16 | 0/16 | 0/16 | 7/64 = 10.94% |
| Keyframe W3 + gating | H14 | 7/16 | 4/16 | 0/16 | 1/16 | 12/64 = 18.75% |

### 3.2 统一使用 HORIZON = 25

| Policy | Horizon | 99650089960001 | 99650089960006 | 99655039960001 | 99655039960006 | Total | Avg max door angle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline ACT | H25 | 5/16 | 6/16 | 0/16 | 1/16 | 12/64 = 18.75% | 34.4° |
| Plücker no-aux | H25 | 10/16 | 8/16 | 0/16 | 0/16 | 18/64 = 28.12% | 44.7° |
| **Keyframe W3 no-gating** | H25 | **12/16** | **11/16** | **2/16** | 0/16 | **25/64 = 39.06%** | **52.2°** |
| Keyframe W3 + gating | H25 | 12/16 | 11/16 | 0/16 | 0/16 | 23/64 = 35.94% | 47.0° |

## 4. 对比分析

### 4.1 WC4 上，baseline 仍然最强

在 WC4、camera-rand、64 trials 的主设置下，最强仍是 **Baseline ACT 50k H15：42/64 = 65.62%**。

Keyframe、gating、Plücker、handle-latent 都没有在 WC4 主设置下稳定超过 baseline：

- Keyframe W3 + gating 最接近，39/64，但仍低于 baseline。
- Keyframe no-gating 和 phase200 只能到 33–35/64。
- Plücker no-handle-aux 在 camera-rand 数据上达到 37/64，说明几何条件有帮助趋势，但还没超过 baseline。
- Handle-latent auxiliary 明显下降到 26/64，当前辅助任务可能干扰了主 action learning。

### 4.2 Plücker 的核心问题是训练/评测相机分布必须一致

早期 Plücker FOV55 在 **关闭 camera-rand eval** 时达到 48/64，但在 **开启 camera-rand eval** 时只有 11/64。

这说明 Plücker 本身不是没用；更准确地说：

- 如果训练数据没有 camera pose randomization，eval 时打开 camera-rand 会造成 depth pattern 和 Plücker ray-map 同时 OOD。
- 使用 camera-rand 数据训练后的 Plücker no-aux 恢复到 37/64，但仍低于 baseline 42/64。

### 4.3 Keyframe loss 没能解决 WC4 闭环稳定性

Keyframe 的初衷是提高 close gripper / rotate handle 等短关键相位权重，但 WC4 上没有超过 baseline。

主要现象：

- 关键帧模型的 gripper action 更容易出现 close-open-close。
- horizon 较小时，频繁 replanning 会放大 chunk 边界不一致。
- per-timestep weight 比 chunk-level scalar 更合理，但单独使用仍不足以提升闭环成功率。

### 4.4 Unseen doors 上 H25 是最大变量

在 4door unseen 测试中，统一改成 H25 后所有策略明显提升：

- Baseline: 2/64 → 12/64
- Plücker: 3/64 → 18/64
- Keyframe W3 no-gating: 7/64 → 25/64
- Keyframe W3 + gating: 12/64 → 23/64

这说明 unseen door 上最大的收益来自更长 execution horizon，而不是单纯模型结构变化。

更长 horizon 可能减少：

- 每 10–15 step 重新预测导致的 gripper 抖动；
- close/rotate/push 阶段被 chunk 边界打断；
- OOD 几何下频繁 replanning 带来的动作不连续。

### 4.5 Unseen doors 上 keyframe W3 no-gating 最好

在 H25 下，**Keyframe W3 no-gating** 达到 25/64，是 unseen 4door 当前最好结果。

这和 WC4 主结果不同：

- WC4：baseline 最强。
- unseen 4door：keyframe W3 no-gating + H25 最强。

可能原因是 keyframe 训练虽然没有改善 WC4 的平均成功率，但确实增强了一些关键阶段动作模式，在 door geometry 改变时更能帮助跨过 grasp/rotate 阶段。

### 4.6 两扇 unseen door 仍然很难

`99655039960001` 和 `99655039960006` 仍然成功率很低：

- `99655039960001`: H25 下只有 keyframe W3 no-gating 有 2/16。
- `99655039960006`: H25 下只有 baseline 有 1/16。

说明 WC4 训练出的策略对这两扇门仍有明显几何/接触分布偏移，后续如果要做泛化，可能需要：

- 多门训练，而不是只用 WC4；
- door/handle geometry randomization；
- 以 H25 为默认 closed-loop execution horizon；
- 重新检查这些门的 handle pose、墙体、门轴方向和推门路径是否与 WC4 差异过大。

## 5. 当前推荐结论

1. **WC4 单门部署**：优先用 Baseline ACT 50k，H15。
2. **WC4 + Plücker 研究**：使用 camera-rand 数据训练的 Plücker no-aux 作为公平对照，不建议用 handle-latent aux 当前版本。
3. **Unseen doors 泛化**：优先用 Keyframe W3 no-gating，H25。
4. **执行 horizon**：unseen door 上 H25 很关键；短 horizon 会严重低估策略能力。
5. **下一步最值得做**：多门数据训练 + H25 固定评测，而不是继续只在 WC4 上堆 auxiliary loss。

