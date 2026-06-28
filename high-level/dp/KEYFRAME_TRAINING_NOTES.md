# Door DP Keyframe Extraction / Post-processing / ACT Training Notes

本文记录当前 A2W Door DP 里“关键帧提取、关键帧后处理、ACT loss 加权、ACT 采样重采样”的实现方式。

## 1. 目标

普通 ACT 训练是从轨迹所有帧里近似均匀采样 chunk 起点。开门任务里很多帧是匀速移动或平滑过渡，信息密度低；真正容易出错的是阶段切换、接触、把手旋转、门开始转动等关键时刻。

因此现在做两件事：

1. 在 raw episode 里提取关键帧，并给关键帧附近窗口写入 `action_loss_weight`。
2. ACT 训练时可以额外提高关键帧附近窗口的采样概率。

## 2. 关键帧来源

关键帧分两类：

1. `manual:*`
   - 由 scripted phase 切换规则得到。
   - 保证至少包含原来手动定义的几个阶段边界。

2. 自动检测事件
   - 从 raw replay 信号里按动态变化检测。
   - 例如 base 速度变化、机械臂关节快速变化、门把手/门轴角速度变化、夹爪接触瞬间。

最终 keyframes 是两者的并集，再经过后处理去重。

实现入口：

```text
high-level/dp/door_dp_common.py
extract_motion_keyframes_from_raw_arrays(...)
```

## 3. Manual keyframes

manual keyframes 来自 phase 切换边界：

```text
manual:start             = 第 0 帧
manual:stop_before_door  = 第一个 initial_hold 帧
manual:pregrasp          = 第一个 grasp 帧
manual:grasp             = 第一个 close_gripper 或 grasp_hold 帧
manual:rotate            = 第一个 push_door 帧
```

注意：`manual:rotate` 不是 `rotate_handle` 的第一帧，而是 rotate 已经完成后的第一帧，也就是 `push_door` 的第一帧。

## 4. 自动关键帧判断条件

所有滑动变化量都用：

```text
δ_t = 1 / w * Σ_i ||q_{t-i} - q_{t-i-1}||_2
```

代码实现：

```text
sliding_mean_displacement(values, window=5)
```

当前默认配置在：

```text
high-level/dp/door_dp_common.py
DEFAULT_MOTION_KEYFRAME_CONFIG
```

### 4.1 狗/base 速度变化快

信号：

```text
q_t = [vx, vy, yaw_rate]
```

来自 raw：

```text
replay_root_state[:, 7]
replay_root_state[:, 8]
replay_root_state[:, 12]
```

判断：

```text
metric = sliding_mean_delta([vx, vy, yaw_rate])
threshold = 当前 episode 的 p95
最多取 4 个
间隔至少 20 帧
```

注意：可视化里 `base speed change` 不是 base 速度本身，而是速度变化量，近似加速度强度。

### 4.2 机械臂关节角变化快

信号：

```text
q_t = joint1..joint6
```

来自 raw：

```text
replay_dof_pos[:, -7:-1]
```

即最后 7 个 Z1 DOF 里排除 `jointGripper`。

判断：

```text
metric = sliding_mean_delta(joint1..joint6)
threshold = 当前 episode 的 p85
最多取 6 个
间隔至少 20 帧
```

### 4.3 门把手转动速度变化快

信号：

```text
q_t = handle_angular_velocity
```

来自 raw：

```text
replay_door_dof_vel[:, 1]
```

判断：

```text
metric = sliding_mean_delta(handle_angular_velocity)
threshold = 当前 episode 的 p95
最多取 4 个
间隔至少 15 帧
```

### 4.4 门轴转动速度变化快

信号：

```text
q_t = door_angular_velocity
```

来自 raw：

```text
replay_door_dof_vel[:, 0]
```

判断：

```text
metric = sliding_mean_delta(door_angular_velocity)
threshold = 当前 episode 的 p95
最多取 2 个
间隔至少 15 帧
```

### 4.5 夹爪与把手接触瞬间

优先用 phase：

```text
第一个 close_gripper frame
```

如果没有 phase 信息，则退化成：

```text
第一个 abs(delta_gripper) > 1e-4 的帧
```

## 5. 关键帧后处理

提取完所有 manual / 自动关键帧后，会做一次相邻去重：

```text
相邻 <= 10 frames 的关键帧归为一组
每组只保留一个
优先保留自动检测事件
如果这一组全是 manual，再保留 manual
```

例如：

```text
manual:start at 0
base_speed_change at 5
```

合并后保留：

```text
5  base_speed_change+manual:start
```

再比如：

```text
manual:rotate at 236
door_hinge_speed_change at 241
```

合并后保留：

```text
241  door_hinge_speed_change+manual:rotate
```

当前 episode_000000 示例：

```text
[5, 61, 91, 136, 161, 184, 202, 241, 257, 275, 328, 390, 408, 428]
```

这批 100 条数据统计：

```text
min = 11
max = 16
mean = 13.24
median = 13
```

## 6. Raw 数据保存

新录制 raw episode 时，`RawDoorDPRecorder.save_episode()` 会保存：

```text
keyframe_indices
keyframe_names
keyframe_target_phase_names
keyframe_extraction_rules
keyframe_mask
action_loss_weight
```

其中 `action_loss_weight` 是每一帧一个 scalar，用于之后转换成 LeRobot feature：

```text
loss.action_weight
```

## 7. 旧 raw 转换兼容

旧 raw episode 可能没有 `keyframe_indices` / `action_loss_weight`。

现在 `convert_door_raw_to_lerobot.py` 会自动 fallback：

1. 从 replay 信号重新提取动态关键帧。
2. 根据关键帧生成 `action_loss_weight`。
3. 写入 LeRobot 的 `loss.action_weight` feature。

如果旧 raw 已经保存了旧权重，可以在转换时强制用新窗口重新计算，而不改关键帧索引：

```bash
--keyframe_loss_weight 8 \
--keyframe_loss_radius 3
```

如果 LeRobot 数据集已经转换完成，可直接更新 Parquet 权重列及统计，不重新编码视频：

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/update_lerobot_keyframe_weights.py \
  --raw_root <raw_root> \
  --dataset_root <lerobot_dataset_root> \
  --weight 8 \
  --radius 3
```

对应位置：

```text
high-level/dp/convert_door_raw_to_lerobot.py
extract_motion_keyframes_from_raw_arrays(data)
high-level/dp/update_lerobot_keyframe_weights.py
```

## 8. Keyframe ACT loss

LeRobot ACT 已经支持读取：

```text
loss.action_weight
```

现在 action loss 是 chunk-level weighted mean：

```text
L_action = Σ_t w_t * l_t / Σ_t w_t
```

其中：

```text
l_t = 以当前帧 t 为起点的整个 action chunk 的平均 L1 loss
w_t = loss.action_weight[t]
```

默认：

```text
keyframe_loss_weight = 8.0
keyframe_loss_radius = 3
```

也就是距离关键帧 `<= 3 frames` 的 chunk 起点帧，整段 action chunk loss 会乘以 8。

注意：当前权重只加在 action L1 loss 上，KL loss 仍然普通平均。

## 9. Keyframe sampler

原版 LeRobot train 在没有 sampler 时是：

```text
shuffle=True
```

也就是近似从所有帧均匀采样 chunk 起点。

现在新增：

```bash
--keyframe_sampling_ratio=0.2
```

含义：

```text
约 20% 的 chunk 起点从 loss.action_weight > 1 的关键帧窗口采样
约 80% 从普通帧采样
```

关闭方式：

```bash
--keyframe_sampling_ratio=0.0
```

如果数据里没有 `loss.action_weight`，会自动 fallback 到原始均匀采样。

实现位置：

```text
high-level/lerobot/src/lerobot/datasets/sampler.py
make_keyframe_window_sampler(...)

high-level/lerobot/src/lerobot/scripts/lerobot_train.py
```

## 10. 可视化

生成某条 raw episode 的关键帧可视化：

```bash
python high-level/dp/visualize_door_raw_keyframes.py \
  --raw_root high-level/data/door_dp_raw/a2w_state10_wc4_newfov_randomized_stop015_graspx015_100 \
  --episode 0 \
  --cols 3 \
  --thumb_width 240
```

输出：

```text
<raw_root>/keyframe_viz/episode_000000/keyframe_timeline.png
<raw_root>/keyframe_viz/episode_000000/keyframe_contact_sheet.png
<raw_root>/keyframe_viz/episode_000000/keyframes.csv
```

其中：

```text
keyframe_timeline.png       画 base/arm/handle/door/gripper 动态指标，并标出 keyframe index
keyframe_contact_sheet.png  每个 keyframe 的 wrist/front 深度图缩略图墙
keyframes.csv               index / phase / name / rule 表格
```

只看文本统计：

```bash
python high-level/dp/analyze_door_raw_keyframes.py \
  --raw_root high-level/data/door_dp_raw/a2w_state10_wc4_newfov_randomized_stop015_graspx015_100 \
  --max_episodes 5 \
  --summary_all
```

## 11. 训练命令参数

ACT 训练命令里建议加：

```bash
--keyframe_sampling_ratio=0.2
```

并确保数据转换后包含：

```text
loss.action_weight
```

`high-level/dp/command_lerobot.txt` 里的 ACT / ACT-DINOv2 / ACT-DeFM 命令已经加入该参数。
