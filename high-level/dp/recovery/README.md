# ACT 失败恢复数据流程 v1

本目录实现了第一版“智能体提出候选，仿真器负责验证”的恢复数据生成流程。

核心原则：

- 智能体或规则系统只负责诊断失败并提出恢复候选。
- Isaac Gym 从失败轨迹的中间快照恢复状态并实际执行候选。
- 只有局部恢复成功且最终任务成功的轨迹才进入训练数据。
- 诊断结果只用于搜索和统计，不直接作为策略训练真值。

当前版本主要支持：

```text
A2W + ikpush + 门把手开门任务
```

## 1. 评测时导出失败轨迹

运行 `high-level/dp/eval/eval_door_policy_success.py` 时增加：

```bash
--save_failure_rollouts \
--failure_rollout_root high-level/data/door_dp_failure_rollouts/<运行名称> \
--failure_snapshot_interval 1 \
--failure_save_camera_obs \
--failure_save_privileged_state \
--failure_save_events
```

评测器会为每个失败环境导出：

```text
failure_rollout.npz
metadata.json
diagnosis.json
```

`failure_rollout.npz` 包含 ACT observation/action、机器人与门的仿真状态、EE 位姿、门轴和把手角度、接触事件，以及用于精确恢复的 `sim_snapshot_*` 字段。

旧日志如果没有 `sim_snapshot`，仍可进行诊断，但不能从中间状态精确恢复仿真。

## 2. 生成结构化失败诊断

```bash
python high-level/dp/recovery/diagnose_failure_rollouts.py \
  --failure_root high-level/data/door_dp_failure_rollouts/<运行名称> \
  --success_threshold_deg 80 \
  --write \
  --summary_json high-level/data/door_dp_failure_rollouts/<运行名称>/diagnosis_summary.json
```

诊断结果包括：

```text
failure_type
t_dev
t_fail
recoverability
evidence
candidate_recovery_family
```

- `t_dev`：最早出现明显偏离、但仍容易修复的时刻。
- `t_fail`：能够确认当前行为已经失败的时刻。
- `candidate_recovery_family`：建议搜索的恢复控制器类别。

当前诊断由规则系统完成，只用于指导候选搜索。

## 3. 生成恢复候选

```bash
python high-level/dp/recovery/generate_recovery_candidates.py \
  --failure_root high-level/data/door_dp_failure_rollouts/<运行名称> \
  --out_json high-level/data/door_dp_failure_rollouts/<运行名称>/recovery_candidates.json \
  --max_candidates_per_branch 5 \
  --branch_offsets=-5,0,5,fail \
  --local_contact_min_frames 5 \
  --final_open_angle_deg 80
```

每条失败轨迹默认从四个时刻分叉：

```text
t_dev - 5
t_dev
t_dev + 5
t_fail
```

候选会搜索是否保持抓取、后退距离、重新接近的 EE offset 和恢复控制器类型。

当前固定约束：

```text
grasp_hold_steps = 0
base_lateral_adjust_m = 0
handle_rotate_steps = 100
door_push_steps = 300
```

候选文件只是待验证方案，不能直接作为训练数据。

## 4. 在 Isaac Gym 中验证候选

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/dp/recovery/verify_recovery_candidates.py \
  --candidate_manifest high-level/data/door_dp_failure_rollouts/<运行名称>/recovery_candidates.json \
  --run_root high-level/logs/door-recovery/<运行名称> \
  --verified_raw_root high-level/data/door_dp_raw/a2w_recovery_verified_<运行名称> \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_name wc4 \
  --batch_size 16 \
  --seed 615455575 \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --pass_open_angle_deg 80 \
  --contact_min_frames 5 \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5
```

验证流程：

```text
恢复 sim_snapshot[t_branch]
→ 执行 scripted recovery controller
→ 检查局部恢复条件
→ 继续转把手和推门
→ 检查最终开门条件
```

只有同时满足以下条件才会保存为 raw episode：

```text
局部恢复成功
且最终任务成功
且没有碰撞或不安全状态
且没有提前解锁
```

不同恢复类型按照自身长度独立结束：

- 重新抓取类约 280 帧。
- `maintain_grasp_then_push` 约 200 帧。
- 短轨迹不会为了对齐并行 batch 而重复最后一帧。

## 5. 可视化和回放

```bash
python high-level/dp/recovery/visualize_recovery_demonstrations.py \
  --raw_root high-level/data/door_dp_raw/a2w_recovery_verified_<运行名称> \
  --candidate_ids 0,20,40,60,80,102,1,22,42,62 \
  --out_dir high-level/dp/result/recovery_visualization/<运行名称>
```

生成：

```text
recovery_front_depth_stages.png
recovery_wrist_depth_stages.png
recovery_metrics.png
selected_recovery_demonstrations.json
replay_selected_recovery.sh
```

依次回放：

```bash
bash high-level/dp/result/recovery_visualization/<运行名称>/replay_selected_recovery.sh
```

## 6. 合并原始专家数据与恢复数据

转换器可以直接追加多个兼容的 raw root，无需提前复制深度文件或重新编号：

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root high-level/data/door_dp_raw/<原始专家数据> \
  --additional_raw_root high-level/data/door_dp_raw/<验证成功的恢复数据> \
  --root high-level/data/lerobot \
  --repo_id local/<混合数据集名称> \
  --depth_only \
  --include_recovery_indicator \
  --num_workers 4
```

转换后：

```text
原始专家帧：aux.is_recovery = 0
恢复轨迹帧：aux.is_recovery = 1
```

未启用 `--add_handle_latent` 时，允许原始数据有 handle bbox、恢复数据没有 handle bbox。state/action schema、action frame、controller mode、vision mode 和 camera pose 仍必须兼容。

不同 episode 保持各自边界，不会把原始轨迹和恢复轨迹首尾拼成一条轨迹。

## 7. 按20%原始专家、80%恢复数据训练

在正常 `lerobot-train` 命令中增加：

```bash
--recovery_sampling_ratio=0.8 \
--recovery_sampling_feature=aux.is_recovery \
--recovery_sampling_threshold=0.5 \
--policy.use_action_loss_weight=false
```

含义：

```text
20% 概率质量分配给原始专家 chunk anchor
80% 概率质量分配给 recovery chunk anchor
```

这是有放回加权采样：

- 不复制深度视频。
- 不修改 action loss。
- 比例是长期抽样概率，不保证每个 batch 恰好达到20/80。
- 采样单位是 ACT action chunk 的起始帧，不是完整 episode。

Recovery sampler 与 keyframe sampler 当前不能同时开启。启用 recovery sampling 时不要设置 `--keyframe_sampling_ratio`。

## 8. 失败类型

- `geometric_misalignment`：几何位置或姿态偏差。
- `contact_establishment_failure`：未建立目标接触。
- `contact_maintenance_failure`：建立接触后又丢失。
- `insufficient_interaction`：执行动作后门或把手运动不足。
- `kinematic_infeasibility`：机械臂接近极限或目标不可达。
- `collision_or_clearance_failure`：发生碰撞或安全间隙不足。
- `temporal_coordination_failure`：底盘、机械臂和夹爪时序不协调。
- `perception_induced_deviation`：观测误差导致动作方向偏离。
- `progress_stagnation`：任务进度长时间停滞。

## 9. 完整数据流

```text
ACT 策略评测
→ 导出失败轨迹和仿真快照
→ 结构化失败诊断
→ 生成恢复候选
→ Isaac Gym 从中间状态执行候选
→ 仿真器验证局部恢复和最终任务
→ 只保存验证成功的恢复轨迹
→ 与原始专家数据统一转换
→ 按指定比例进行 ACT 训练
```
