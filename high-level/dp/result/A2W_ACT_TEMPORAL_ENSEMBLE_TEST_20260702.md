# A2W ACT temporal ensemble 本机仿真测试（2026-07-02）

## 1. 目的

把 NX 真机部署里使用的 ACT chunk overlap temporal ensemble 复用到本机 IsaacGym
closed-loop play / success eval 中，测试是否能提升最新 100K checkpoint 的开门成功率。

## 2. Checkpoint

```text
high-level/dp/logs/door-auto-wrapped/leroact_a2w_keyframe_perstep_w8_r3_sample20_nogating_chunk100_exec50_bs16_0702_0106/100000/model_latest.pt
```

训练结束日志对应 step 100000。

## 3. 实现改动

新增可选参数，默认关闭，不影响旧实验：

```text
--dp_temporal_ensemble
--dp_temporal_prefetch_actions 3
--dp_temporal_old_weight 0.3
--dp_temporal_new_weight 0.7
```

改动文件：

```text
high-level/dp/door_policy_backend.py
high-level/dp/door_policy_worker.py
high-level/dp/door_dp_common.py
high-level/dp/play/play_door_policy.py
high-level/dp/eval/eval_door_policy_success.py
high-level/float_ik/door_common.py
high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py
```

核心逻辑：

- policy controller 新增 `predict_action_chunks_for_envs(env_ids)`，可以从本地 controller 或 subprocess worker 取完整 `(B,T,10)` action chunk。
- `door_common.py` 新增 `FloatDPActionOverlapBuffer`：
  - 新 chunk 的 step 0 对齐当前 policy timestep；
  - 队列剩余 action 数 `<= prefetch_actions` 时提前推理新 chunk；
  - 对重叠 timestep 做 `0.3 old + 0.7 new`；
  - `vx/vyaw/xyz` 线性融合；
  - quat 使用 shortest-path SLERP；
  - gripper 直接使用最新预测，不融合。
- JSONL 日志新增 `temporal_ensemble` 字段，记录 action 来源和 overlap 信息。

## 4. Smoke test

用最新 checkpoint 测试新增 full-chunk 接口：

```text
chunk_shape (2, 100, 10)
action_horizon 10
action_dim 10
```

说明 `predict_action_chunks_for_envs()` 输出符合 temporal buffer 预期。

## 5. Success eval 设置

与已有 latest 100K horizon report 对齐：

- WC4
- 1000 steps
- 16 env / batch
- base seed 62000
- success metric: abs
- threshold: 80 deg
- depth-only
- clip 0.2–1.5 m
- 相机随机化开启
- base pitch 随机化关闭：

```text
--robot_pitch 0.0
--ikpush_robot_pitch_rand_min 0.0
--ikpush_robot_pitch_rand_max 0.0
```

## 6. 结果

### H10 小测

Temporal H10，16 trials：

```text
high-level/logs/door-policy-success/perstep_w8r3_100k_h10_temporal_abs16_20260702_170443
```

结果：

```text
temporal H10: 6/16 = 37.50%
raw H10 同 seed batch0: 7/16 = 43.75%
```

H10 没有提升。

### H12 完整 64 trials

Temporal H12，64 trials：

```text
high-level/logs/door-policy-success/perstep_w8r3_100k_h12_temporal_abs64_20260702_171037
```

结果：

```text
temporal H12: 34/64 = 53.12%
raw H12:      34/64 = 53.12%
```

分 batch：

| Batch | raw H12 | temporal H12 |
|---:|---:|---:|
| 0 | 8/16 | 8/16 |
| 1 | 8/16 | 8/16 |
| 2 | 10/16 | 9/16 |
| 3 | 8/16 | 9/16 |
| Total | 34/64 | 34/64 |

Temporal log 确认确实发生融合：

```text
new_chunk records: 42880
overlap_0.3_old_0.7_new records: 21120
max blend_count: 2
```

也就是说动作序列确实被改变了，但总体成功率没有提升。

## 7. 结论

对这个最新 100K checkpoint，本机仿真里的 NX-style temporal ensemble 没有提升成功率：

- H10 从 7/16 下降到 6/16；
- H12 完整 64 trials 与 raw 完全持平，都是 34/64。

它改变了个别 env 的成败，但只是互相抵消，没有形成净收益。

当前建议：

1. 本机仿真默认继续使用 raw chunk/horizon sweep，不默认启用 temporal ensemble。
2. temporal ensemble 可以保留为真机部署一致性/消融选项。
3. 如果想继续优化成功率，优先尝试训练侧更温和的 per-timestep keyframe 权重或 camera gating，而不是指望 overlap smoothing 单独提升。

