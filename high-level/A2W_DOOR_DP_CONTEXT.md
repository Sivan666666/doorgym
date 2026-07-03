# A2W + Z1 Door DP 项目上下文

更新时间：2026-06-24  
工作目录：`/home/sivan/whole_body/visual_whole_body`

这个文件是 A2W+Z1 开门仿真、数据录制、LeRobot 转换、训练、play/eval 的项目记忆。后续如果继续修改这些脚本，也要同步更新本文件，避免后来忘记为什么这么写。

## 1. 当前主线脚本

1. A2W scripted ikpush / policy play 底层脚本：

   `high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py`

2. 通用门、相机、DP recorder、状态/action helper：

   `high-level/float_ik/door_common.py`

3. A2W raw 数据录制：

   `high-level/dp/record/record_door_dp_dataset_a2w_state10.py`

4. raw 转 LeRobot：

   `high-level/dp/convert_door_raw_to_lerobot.py`

5. LeRobot 官方训练命令参考：

   `high-level/dp/command_lerobot.txt`

6. 常用 Door DP 命令参考：

   `high-level/dp/commands.txt`

7. policy play / success eval：

   `high-level/dp/play/play_door_policy.py`

   `high-level/dp/eval/eval_door_policy_success.py`

## 2. A2W 相机当前默认配置

配置在 `high-level/float_ik/door_common.py`：

```python
DEFAULT_WRIST_CAMERA_CFG = {
    "horizontal_fov": 55,
    "resolution": DEPTH_CAMERA_RESOLUTION,
    "position": [0.093, -0.031, 0.22],
    "rotation_deg": [0.0, 60.0, 0.0],
}

DEFAULT_FRONT_CAMERA_CFG = {
    "horizontal_fov": 55,
    "resolution": DEPTH_CAMERA_RESOLUTION,
    "position": [0.29, 0.031, 0.165],
    "rotation_deg": [0.0, -45.0, 0.0],
}
```

当前挂载方式：

1. wrist camera：挂在机械臂 `link06` 上，位置改到夹爪正上方，而不是侧面。
2. front camera：挂在 actor root rigid body 上，不再优先找 `base_link`/`trunk` 的旧方式。
3. 相机角度参数用度数表达：yaw / pitch / roll，默认来自 `DEFAULT_*_CAMERA_CFG["rotation_deg"]`。
4. 深度图默认 clip 范围：`0.2m - 1.5m`。
5. raw 里深度可以紧凑存单通道；转 LeRobot 时会通过 `image_to_three_channel_uint8` 复制成三通道，给 LeRobot 的 image backend 使用。

### 2.1 A2W+Z1 相机支架 URDF

新复制的带相机支架资产在：

```text
high-level/data/asset/a2wz1_mount/urdf/a2wz1.urdf
```

支架 mesh 在：

```text
high-level/data/asset/a2wz1_mount/meshes/wrist_up.STL
high-level/data/asset/a2wz1_mount/meshes/wrist_down.STL
```

当前 URDF 修改：

1. 新增 `wrist_up_mount` link，通过 fixed joint 接到机械臂 `link06`。
2. 新增 `wrist_down_mount` link，通过 fixed joint 接到机械臂 `link06`。
3. 当前默认加载的是原始 A2W+Z1 资产：

   ```text
   high-level/data/asset/a2wz1/urdf/a2wz1.urdf
   ```

   新版 float 脚本会把这份 URDF split 成 base actor 和 arm actor；`wrist_up_mount` / `wrist_down_mount` 已加入 `A2W_ARM_LINKS`，所以 split 后会进入 `z1_arm_visual_flip.urdf`。

4. 当前两个 wrist mount 都通过 fixed joint 接在 `link06` 下，并绕 X 轴 roll -90°：

   ```xml
   rpy="-1.5707963267948966 0 0"
   ```

5. 当前支架有 visual，也加了 STL mesh collision。collision 直接使用对应 STL，而不是简化 box：

   ```text
   wrist_up_mount   collision mesh: ../meshes/wrist_mount_down.STL, collision origin rpy=[+pi/2, 0, 0]
   wrist_down_mount collision mesh: ../meshes/wrist_mount_up.STL,   collision origin rpy=[+pi/2, 0, 0]
   ```

   注意：STL mesh collision 比 bbox box collision 更重，如果之后出现 PhysX 不稳定、接触异常或加载变慢，可以再退回简化 box / convex hull / VHACD。link 里还有极小 inertial，主要用于保持 URDF/Isaac Gym 加载稳定。

## 3. A2W scripted trajectory 是怎么写的

核心函数在 `isaacgym_float_ik_a2w_basearn_push_door_parallel.py`：

```python
trajectory_targets(...)
```

每个 step 生成：

- base 目标位置 `base_xy`
- base yaw
- EE target position
- EE target quaternion
- gripper target
- 当前 phase 名字

wc4 这扇门的把手目标点额外做了门级别 z 修正：

```bash
--wc4_pregrasp_z_offset 0.0
--wc4_grasp_z_offset 0.0
```

`wc4_pregrasp_z_offset` 只加在 `pregrasp[2]` 上；`wc4_grasp_z_offset` 加在 `grasp[2]` 上。由于 `rotate` / `push` 点是从 `grasp` 推导出来的，所以 wc4 的后续 rotate/push 轨迹点也会跟着 `grasp` 上移。

当前 phase 顺序：

1. `walk`
2. `initial_hold`
3. `grasp`
4. `close_gripper`
5. `rotate_handle`
6. `push_door`
7. `return_home`
8. `hold_home`

默认 phase steps：

```text
walk_steps                 260，但默认启用 dynamic walk steps
walk_min_speed             0.20 m/s
initial_hold_steps         150
initial_hold_move_steps    100
grasp_steps                50
grasp_hold_steps           0
gripper_close_steps        50
handle_rotate_steps        100
door_push_steps            300
return_home_steps          150
hold_steps                 300
```

仿真控制频率是 50Hz，`dt=0.02s`。所以：

- 50 steps = 1.0s
- 100 steps = 2.0s
- 150 steps = 3.0s
- 300 steps = 6.0s

注意：`walk_steps` 如果没有显式从 CLI 指定，会根据出生点到 stop distance 的距离动态算，保证最小移动速度约 `0.2m/s`。这样不同 env 因为出生距离不同，可以不同步进入 `initial_hold`，不会出现远的 env 被迫走得特别快、近的 env 走得特别慢。

### 3.1 Door Digital Twin V1

新增模块：

```text
high-level/float_ik/door_twin/
```

当前 V1 做的是确定性结构提取 + skill program 解释执行 + rollout report，不直接让 VLM 改 Python。主脚本新增参数：

```bash
--skill_program_json /path/to/skill.json   # 或 auto，用当前 scripted 参数生成默认 skill
--door_twin_log_dir /path/to/run_dir
--save_failed_rollouts
--dump_keyframe_images
--door_twin_camera_views wrist,front,observer_left,observer_right
```

不传 `--skill_program_json` 时，`trajectory_targets(...)` 仍走原来的手写 offset 轨迹。`--skill_program_json auto` 会生成带 `execution_mode=legacy_replay` 的结构化程序，第一轮仍逐步执行原 scripted waypoint、smoothstep、phase timing 和 per-env randomization，用于建立严格 baseline。只有用户传入新 skill，或者 optimizer/VLM 产生参数补丁后，程序才切换到 `skill_interpreter`：

- `MoveEEToHandle` / `RotateHandle` 的 EE 关键点按局部 frame `[approach, lateral, z]` 生成，不再直接依赖 world-X offset。
- `MoveTo` 是显式底盘原语，可重复出现并通过 `stage=approach/push/traverse` 区分，参数包含 `vx`、`vyaw`、`distance` / `stop_distance`、`duration_steps`。`skill_interpreter` 中该原语使用线性时间参数化，实际底盘命令会贴近指定的 `vx/vyaw`；`legacy_replay` 仍使用原 smoothstep。
- rollout trace 会记录实际执行得到的 `base_command={vx, vyaw}`、`max_abs_base_vx` 和 `max_abs_base_vyaw`。

示例底盘原语：

```json
{"name": "MoveTo", "stage": "approach", "vx": 0.20, "vyaw": 0.0, "stop_distance": 0.20}
{"name": "MoveTo", "stage": "push", "vx": 0.05, "vyaw": 0.0, "distance": 0.30, "duration_steps": 300}
```

`--dump_keyframe_images` 在 headless 模式下也会强制创建相机。默认视角是 wrist、front、observer_left、observer_right，其中 observer 是对准 handle 区域的世界坐标相机，适合 VLM 同时观察机械臂接触、门板角度和底盘碰撞。每个关键 phase 同时保存 RGB、depth、handle mask 和多视角 montage。

`--door_twin_log_dir` 会为每个 env 输出：

```text
env_0000_report.json
rollout_summary.json
keyframes/env_0000/*.png             # RGB/depth/mask
keyframes/env_0000/*_montage.png     # 多视角拼图
expert_raw/episode_*.npz             # 仅成功且启用 --record_dp_dataset
```

`RolloutReport` 里包含 success、failure_stage、secondary_failures、door_open_deg、handle_rotation_deg、ee_handle_dist、ee_tracking_error、base_collision、body_passed、camera_available、handle_unlocked、底盘命令统计和 artifact 路径。主失败按因果阶段排序：解锁前 IK / grasp / handle failure 优先于之后继续 push 导致的碰撞。`rollout_summary.json` 的 `expert_trajectories` 会直接索引本轮保存成功的 raw NPZ。失败分类当前包括：

```text
grasp_miss
handle_not_unlocked
contact_lost
door_push_insufficient
arm_joint_limit_or_ik_bad
base_collision
body_blocked
camera_unavailable
asset_invalid
timeout
```

成功 raw DP episode 仍然沿用原来的 recorder 逻辑保存；失败 rollout 不写入 expert 数据，只写 DoorTwin report / trace / keyframe artifacts。

已验证的 wc4 skill 示例：

```text
high-level/float_ik/door_twin/examples/wc4_smoke_skill.json
```

2026-07-02 实际 smoke 结果：

- Unit + compile：`9 passed`，同时在默认 Python 和 `b1z1` Python 3.8 做过 `py_compile`。
- legacy A/B：wc4 原 scripted 与 `auto legacy_replay` 各跑 930 steps；45 个 trace 采样点的 phase、base XY、EE target、EE pose、handle goal 全部最大差值 `0.0`，两边都是 `1/1 success`、无碰撞。
- wc4：`1/1 success`，door `90.0 deg`、handle `45.0 deg`、无 base collision、4 个相机有效；trace 中 approach `vx=0.2016 m/s`、push `vx=0.0500 m/s`。
- expert recorder：保存 186 帧 `episode_000000.npz`；state `(186, 73)`、action `(186, 10)`、wrist/front depth `(186, 480, 640)`。
- 每个 rollout 保存 78 张关键帧 PNG，其中 6 张是 phase montage。
- record_materialization door index 0：原 scripted 与 `auto legacy_replay` 的轨迹也逐点一致。两者都会先出现解锁前 IK 误差/关节极限，把手未解锁、门保持关闭；脚本继续执行 push 后，front bumper 才与关闭的 door panel 产生真实 PhysX contact。因此主失败是 `arm_joint_limit_or_ik_bad`，`handle_not_unlocked` / `base_collision` / `door_push_insufficient` 是 secondary failures，不应先修改原来的 `stop_distance=0.15`、`push_base_distance=0.35`。该 generated door 尚未优化成功。

完整成功 smoke 的产物：

```text
high-level/experiments/door_twin_runs/smoke_wc4_multiview_v8_final_expert/
```

## 4. A2W 默认 base / door 参数

在 `isaacgym_float_ik_a2w_basearn_push_door_parallel.py` 的 custom parameters 里。现在这些默认值同时集中到了 YAML，方便之后直接改配置：

```text
high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml
```

脚本默认会加载这份 YAML；也可以用下面两种方式换另一份：

```bash
--a2w_float_ik_config /path/to/a2w_float_ik_push_door_parallel.yaml
export A2W_FLOAT_IK_CONFIG=/path/to/a2w_float_ik_push_door_parallel.yaml
```

命令行参数优先级最高：如果同一个参数既在 YAML 里写了，又在终端里传了，终端值会覆盖 YAML。

```text
door_x                  2.5
door_y                  0.0
door_z_offset           0.01
door_actor_scale        1.2
robot_x                 4.1
robot_y                 0.0
robot_y_alignment       handle
robot_z                 0.50
robot_pitch             0.0
robot_yaw               pi
robot_front_offset      0.55
robot_rear_offset       0.65
stop_distance           0.15
push_base_distance      0.35
door_push_distance      1.10
```

`robot_y_alignment=handle` 很重要：机器狗出生时默认对齐门把手，而不是整个资产中心。之前四扇 rec 门也按这个逻辑处理过，避免狗正对门中心导致把手偏到一边。

## 5. 随机化怎么加的

每个 env 会在 `make_env_args(args, env_index)` 里复制一份 `args`，再按 `env_index` 的 seed 做 per-env randomization。

关闭随机化：

```bash
--no_ikpush_env_randomization
```

默认开启时，当前主要随机项：

```text
door_x                         ±0.03m
door_y                         ±0.03m
door_wall_x_offset             ±0.03m
robot_x                        -0.70m 到 0.0m
robot_y                        -0.10m 到 +0.04m
robot_z                        ±0.03m
robot_pitch                     0° 到 +5°
robot_yaw                      ±0.03rad
door_joint_friction            ±0.08，下限 0
door_joint_damping             ±0.04，下限 0
handle_joint_friction          ±0.005，下限 0
handle_joint_damping           ±0.005，下限 0
handle_spring_stiffness        ±0.05，下限 0
handle_spring_damping          ±0.01，下限 0
```

当前这些是固定不随机的：

```text
pregrasp_offset
grasp_x_offset
grasp_z_offset
wc4_pregrasp_z_offset
wc4_grasp_z_offset
handle_rotate_angle
door_push_distance
```

门回弹力：

```text
ikpush_door_open_resistance_prob = 0.50
ikpush_door_open_resistance_min  = 0.10
ikpush_door_open_resistance_max  = 0.30
```

含义是固定选中约 50% 的 env，这些 env 的 `door_open_resistance` 从 `[0.10, 0.30]` 均匀采样；另外 50% 是 0。不是每帧随机，而是 env 级别固定。

深度噪声：

深度噪声默认参数现在集中在共享配置文件：

```text
high-level/dp/config/depth_camera_aug_default.yaml
```

`isaacgym_float_ik_a2w_basearn_push_door_parallel.py` 和 `record_door_dp_dataset_a2w_state10.py` 都通过 `high-level/dp/depth_camera_aug.py` 从这里取默认值。命令行参数仍然可以临时覆盖 config。若要临时换另一份 config，可以在启动前设置：

```bash
export DOOR_DEPTH_AUG_CONFIG=/path/to/your_depth_camera_aug.yaml
```

然后再运行仿真/录制命令。

`record_door_dp_dataset_a2w_state10.py` 默认可用 `--enable_depth_noise` 开启。当前语义是固定比例 env 加噪，默认 50%：

```text
depth_noise_prob             0.5
gaussian_std_m               0.005
gaussian_distance_factor     0.05
edge_noise_prob              0.1
edge_gradient_threshold_m    0.05
edge_dilation_kernel_size    3
hole_noise_prob              0.005
hole_block_size_min          3
hole_block_size              15  # 最大 block size，实际每次从 3..15 随机抽
hole_white_prob              0.50
dropout_prob                 0.002
salt_pepper_prob             0.002
enable_gaussian_blur         false
gaussian_blur_ksize          15
gaussian_blur_sigma          4.0
near_clip_m                  0.2
far_clip_m                   1.5
```

选中的 env 每一帧 wrist/front depth 都加噪；没选中的 env 全程不加噪。

其中 blocky hole 的 `depth_hole_block_size_min` / `depth_hole_block_size` 表示实际块大小的最小/最大采样范围，不是固定块大小。每次应用 blocky hole 噪声时，会先从 `[depth_hole_block_size_min, depth_hole_block_size]` 均匀随机抽一个整数作为实际 block size。默认就是从 `[3, 15]` 抽。`depth_hole_noise_prob` 选中一个 block 后，会按 `depth_hole_white_prob` 决定这个 block 填白还是填黑：

```text
depth_hole_white_prob = 0.50 -> 50% hole block 填 far_clip/白，50% 填 0/黑
depth_hole_white_prob = 1.00 -> 全部 hole block 填白
depth_hole_white_prob = 0.00 -> 全部 hole block 填黑
```

Gaussian blur 当前默认参数参考 NX RealSense 调试记录里的 `cv2.GaussianBlur 15x15 sigma=4.0`，但默认关闭：

```bash
--enable_depth_gaussian_blur
--depth_gaussian_blur_ksize 15
--depth_gaussian_blur_sigma 4.0
```

执行顺序是先完成 metric gaussian / edge / block hole / dropout / salt-pepper 噪声，再把 depth 映射到 policy-depth 归一化空间做 Gaussian blur，最后映射回 meter depth 给后续显示/录制。

深度相机位姿随机化：

```bash
--enable_depth_camera_randomization
--depth_camera_pos_rand_m 0.01
--depth_camera_rot_rand_deg 2.0
```

如果不传 `--enable_depth_camera_randomization`，默认不做相机位姿随机化。

## 6. 门墙体、flip、资产方向相关踩坑

1. `wc4` 的墙体和门方向是正常的。

2. 原有 PartNet 数字门，例如 `99650069960021` 这类，之前墙体生成在 X 轴方向，看起来不对。现在逻辑在 `door_common.py`：

   - `door_wall_opening_axis(door)` 默认返回 `"y"`。
   - `door_wall_bounds_yaw_offset(door)` 对 `partnet_numeric` 返回 `pi/2`，只用于墙体 bbox 方向修正。

   这样 legacy numeric door 的视觉 actor yaw 不乱改，但墙体会按正确方向生成。

3. 门/把手打开方向如果和策略/评估方向相反，不要盲目改资产，优先检查：

   - `--flip_door_motion_sign`
   - `--door_auto_open_sign`
   - `--handle_rotate_direction_sign`
   - eval 里的 `--success_metric` 和 `--door_motion_sign`

4. A2W 的 wc4 实际开门角度可能是正方向。以前 eval 使用 `success_metric=auto` 或 signed + 默认 `door_motion_sign=-1` 时，可能 play 看起来能开，但 eval 统计成失败。A2W wc4 评测更稳的是：

   ```bash
   --success_metric abs
   ```

   或明确：

   ```bash
   --success_metric signed --door_motion_sign 1.0
   ```

## 7. 数据 schema：10D EE 版和 9D joint 版

`record_door_dp_dataset_a2w_state10.py` 现在支持两种 schema：

### 7.1 10D EE state/action

参数：

```bash
--state_action_mode ee_state10
```

raw observation.state：

```text
[last_command_vx, last_command_vyaw,
 ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw,
 gripper]
```

raw action：

```text
[vx, yaw,
 ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw,
 gripper]
```

这个是原主线 A2W 10D state，前两维用 last command，因为实机四足上不能稳定拿到真实 vx/vyaw，但 command 一般能很快达到。

### 7.2 9D joint state/action

参数：

```bash
--state_action_mode joint_state9
```

raw observation.state：

```text
[last_command_vx, last_command_vyaw,
 joint1, joint2, joint3, joint4, joint5, joint6, jointGripper]
```

raw action：

```text
[vx, yaw,
 joint1, joint2, joint3, joint4, joint5, joint6, jointGripper]
```

这个模式是后来加的，目的是把 EE target 换成 Z1 7 个机械臂关节目标。policy play 时，如果 checkpoint sidecar/action_names 是 joint9，会在 A2W 脚本里直接给关节 target，绕开 EE IK target 覆盖。

不要把 10D 和 9D raw episode 混在同一个 `raw_root` 里。录制脚本会检查 `door_dp_feature_names.json` 防止 schema 混用。

### 7.3 关键帧提取和 weighted ACT action loss

详细说明单独记录在：

```text
high-level/dp/KEYFRAME_TRAINING_NOTES.md
```

A2W raw 录制现在会基于每帧的 `subtask_index/phase_id` 自动提取关键帧，并写进每个 raw episode：

```text
keyframe_indices
keyframe_names
keyframe_target_phase_names
keyframe_mask
action_loss_weight
```

默认关键帧语义：

```text
start              第一帧
stop_before_door   第一个 initial_hold 帧，也就是 base 停在门前
pregrasp           第一个 grasp 帧，也就是开始从 pregrasp 往 grasp 走
grasp              第一个 grasp_hold/close_gripper 帧，也就是到达 grasp 点
rotate             第一个 push_door 帧，也就是 rotate_handle 结束、到达 rotate 点
```

默认 loss 权重是：

```text
lambda = 8.0
delta  = 3 frames
```

也就是对距离任一关键帧 `<= 3` 帧的 chunk 起点帧，写入：

```text
action_loss_weight = 8.0
```

其他位置是：

```text
action_loss_weight = 1.0
```

可以从录制 wrapper 或底层 A2W 脚本直接调：

```bash
--keyframe_loss_weight 8.0
--keyframe_loss_radius 3
--no_keyframe_loss_weights
```

raw 转 LeRobot 时，`action_loss_weight` 会变成 LeRobot feature：

```text
loss.action_weight
```

本地 LeRobot ACT 已经支持读取这个 feature：如果 batch 里存在 `loss.action_weight`，dataset 会像 `action` 一样取未来 chunk：

```text
action[t : t + H]
loss.action_weight[t : t + H]
```

ACT loss 会对 chunk 内每个未来 timestep 单独加权：

```text
L = Σ_{b,h,d} w_{b,h} valid_{b,h} |a_hat_{b,h,d} - a_{b,h,d}|
    / Σ_{b,h,d} w_{b,h} valid_{b,h}
```

其中 `h` 是当前 observation 之后的第 `h` 个 action step。也就是说，如果 `t+h` 靠近关键帧，只会给 chunk 里第 `h` 个动作加权，不再把整个 100-step chunk 一起乘同一个权重。老数据/老 loader 如果只提供 `(B,)` 或 `(B,1)` scalar 权重，ACT 仍会兼容地 broadcast 到整段 chunk；如果没有这个 feature，则保持原来的普通平均 loss。

ACT 训练 sampler 也支持关键帧窗口重采样。原版 LeRobot train 在没有 sampler 时是 `shuffle=True` 的均匀随机采样；现在可通过：

```bash
--keyframe_sampling_ratio=0.2
```

让大约 20% 的 chunk 起点从 `loss.action_weight > 1` 的关键帧窗口采样，80% 从普通帧采样。设成：

```bash
--keyframe_sampling_ratio=0.0
```

即可回到原来的均匀随机采样。如果数据里没有 `loss.action_weight`，训练会自动 fallback 到原采样方式。

当前动态关键帧默认规则：

```text
base_speed_change         p95，最多 4 个
arm_joint_motion_fast     p85，最多 6 个
handle_speed_change       p95，最多 4 个
door_hinge_speed_change   p95，最多 2 个
gripper_handle_contact    第一个 close_gripper/grasp_hold，或 gripper delta > 1e-4
```

提取后还会做一次相邻关键帧整理：

```text
相邻 <= 10 frames 的关键帧只保留一个；
同一组里优先保留自动检测事件，其次保留 manual phase keyframe。
```

关键帧可视化脚本：

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

## 8. A2W 10D 数据录制命令：新版 FOV，相机 0.2-1.5m

16 个 env 一组，录制成功 150 条，开启随机化，所有门用 wc4：

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
  --num_episodes 150 \
  --num_envs 16 \
  --raw_root high-level/data/door_dp_raw/a2w_state10_wc4_newfov_randomized_150 \
  --state_action_mode ee_state10 \
  --steps 1000 \
  --seed -1 \
  --headless \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --depth_only \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --enable_depth_noise \
  -- \
  --door_name wc4
```

如果想随机不同门，把最后的 `--door_name wc4` 换成：

```bash
  --door_selection diverse
```

录制脚本行为：

1. 每批最多 `--num_envs` 个 env。
2. 只保存成功 env rollout。
3. 如果最后还差不到 16 条，下一批会自动缩小 env 数到 remaining。
4. `--max_quota_rollouts` 是安全上限，防止无限跑。
5. 当前实现每个成功 episode 写成独立 raw `.npz`，不是把全部数据攒到最后一次性写；如果电脑卡，优先把 `num_envs` 从 16 降到 8 或关闭额外显示。

## 9. A2W 9D joint 数据录制命令

16 个 env 一组，录制成功 100 条：

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
  --num_episodes 100 \
  --num_envs 16 \
  --raw_root high-level/data/door_dp_raw/a2w_joint_state9_wc4_randomized_100 \
  --state_action_mode joint_state9 \
  --steps 1000 \
  --seed -1 \
  --headless \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --depth_only \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --enable_depth_noise \
  -- \
  --door_name wc4
```

## 10. raw 转 LeRobot

### 10.1 A2W 10D EE 数据转换

注意一定要加 `--depth_only`，否则会报：

```text
Raw data vision_mode='depth_only', but converter was run with depth mode.
```

命令：

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root high-level/data/door_dp_raw/a2w_state10_wc4_newfov_randomized_150 \
  --root high-level/data/lerobot \
  --repo_id local/door_a2w_state10_wc4_newfov_randomized_150 \
  --overwrite \
  --fps 25 \
  --depth_only \
  --image_storage video \
  --video_codec h264 \
  --num_workers 4 \
  --state_preprocess none \
  --action_preprocess none \
  --keyframe_loss_weight 8 \
  --keyframe_loss_radius 3 \
  --near_zero_rate_eps 1e-5
```

### 10.2 A2W 9D joint 数据转换

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root high-level/data/door_dp_raw/a2w_joint_state9_wc4_randomized_100 \
  --root high-level/data/lerobot \
  --repo_id local/door_a2w_joint_state9 \
  --overwrite \
  --fps 25 \
  --depth_only \
  --image_storage video \
  --video_codec h264 \
  --num_workers 4 \
  --state_preprocess none \
  --action_preprocess none \
  --keyframe_loss_weight 8 \
  --keyframe_loss_radius 3 \
  --near_zero_rate_eps 1e-5
```

转换后会生成：

```text
high-level/data/lerobot/local/<repo_id>/door_dp_feature_names.json
```

这个 sidecar 很重要，记录 state/action 名字、vision mode、depth noise、depth clip、action_format 等。policy play / export / eval 要靠它判断 10D 还是 9D。

已经转换好的数据可以只更新 `loss.action_weight` 和统计信息，不重新编码视频：

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/update_lerobot_keyframe_weights.py \
  --raw_root <raw_root> \
  --dataset_root <lerobot_dataset_root> \
  --weight 8 \
  --radius 3
```

## 11. LeRobot ACT 训练命令

训练习惯来自 `high-level/dp/command_lerobot.txt`：

- 仿真/control：50Hz
- dataset：25Hz
- depth_only 两路图：
  - `observation.images.wrist_masked_depth`
  - `observation.images.front_masked_depth`
- depth 图不使用 ImageNet stats：

  ```bash
  --dataset.use_imagenet_stats=false
  ```

- ACT：
  - `chunk_size=100`：25Hz 下预测 4s。
  - `n_action_steps=50`：25Hz 下每次推理执行 2s。训练时也会进入 policy config，但主要影响 rollout/inference 的执行步数；训练监督主要由 chunk/action 序列决定。
  - `batch_size=16`
  - `steps=100000`

### 11.1 A2W 10D EE ACT

```bash
conda run --no-capture-output -n b1z1_lerobot bash -lc '
cd /home/sivan/whole_body/visual_whole_body
export PYTHONPATH="$PWD/high-level/lerobot/src:${PYTHONPATH:-}"

DATASET_REPO_ID=local/door_a2w_state10_wc4_newfov_randomized_150
DATASET_ROOT="$PWD/high-level/data/lerobot/local/door_a2w_state10_wc4_newfov_randomized_150"
OUTPUT_ROOT="$PWD/high-level/dp/logs/lerobot-train"
ACT_RUN_NAME="leroact_a2w_state10_wc4_newfov_depthonly_chunk100_exec50_bs16_$(date +%m%d_%H%M)"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num-processes=1 \
  --mixed_precision=no \
  "$(which lerobot-train)" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.video_backend=torchcodec \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --batch_size=16 \
  --steps=100000 \
  --num_workers=4 \
  --save_freq=10000 \
  --log_freq=10 \
  --keyframe_sampling_ratio=0.2 \
  --job_name="$ACT_RUN_NAME" \
  --output_dir="$OUTPUT_ROOT/$ACT_RUN_NAME" \
  --wandb.enable=true \
  --wandb.project=door-act \
  --wandb.disable_artifact=true
'
```

### 11.2 A2W 9D joint ACT

```bash
conda run --no-capture-output -n b1z1_lerobot bash -lc '
cd /home/sivan/whole_body/visual_whole_body
export PYTHONPATH="$PWD/high-level/lerobot/src:${PYTHONPATH:-}"

DATASET_REPO_ID=local/door_a2w_joint_state9
DATASET_ROOT="$PWD/high-level/data/lerobot/local/door_a2w_joint_state9"
OUTPUT_ROOT="$PWD/high-level/dp/logs/lerobot-train"
ACT_RUN_NAME="leroact_a2w_joint_state9_wc4_randomized_depthonly_chunk100_exec50_bs16_$(date +%m%d_%H%M)"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num-processes=1 \
  --mixed_precision=no \
  "$(which lerobot-train)" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.video_backend=torchcodec \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --batch_size=16 \
  --steps=100000 \
  --num_workers=4 \
  --save_freq=10000 \
  --log_freq=10 \
  --keyframe_sampling_ratio=0.2 \
  --job_name="$ACT_RUN_NAME" \
  --output_dir="$OUTPUT_ROOT/$ACT_RUN_NAME" \
  --wandb.enable=true \
  --wandb.project=door-act \
  --wandb.disable_artifact=true
'
```

### 11.3 可选的 front/wrist camera input gating

Camera gating 默认关闭，因此旧 ACT 配置、旧 checkpoint 和原始视觉 token 路径保持不变。显式开启时，在共享 CNN 和 `encoder_img_feat_input_proj` 之后、Transformer encoder 之前执行：

```text
h_front = AvgPool(front_feature_map)
h_wrist = AvgPool(wrist_feature_map)
logits = MLP([h_front, h_wrist, observation.state])
[g_front, g_wrist] = 2 * softmax(logits / temperature)
```

MLP 最后一层权重和 bias 都初始化为 0，因此初始严格为：

```text
g_front = 1
g_wrist = 1
```

只缩放视觉 feature，不缩放相机位置编码。启用参数：

```bash
--policy.camera_input_gating=true \
--policy.camera_input_gating_hidden_dim=128 \
--policy.camera_input_gating_temperature=1.0
```

当前 A2W 的 `front_masked_depth` / `wrist_masked_depth` 会按 feature key 自动识别；非标准名称可通过下面两个参数显式指定：

```bash
--policy.camera_input_gating_front_key=observation.images.front_masked_depth \
--policy.camera_input_gating_wrist_key=observation.images.wrist_masked_depth
```

训练日志会额外包含：

```text
camera_gate_front
camera_gate_wrist
```

Official LeRobot checkpoint 转 Door checkpoint 时会同步保存这些 gating 配置，Door backend 在 play/eval 时按相同结构严格加载。

小注：

- `--num-processes=1` 是 accelerate 只启动一个训练进程，适合单卡。
- `--wandb.disable_artifact=true` 是不把 checkpoint/dataset 作为 wandb artifact 上传，减少网络和存储开销。

## 12. Play / Eval 命令模板

### 12.1 直接 play A2W policy，4 env

当前确认正确 play 训练好的 A2W 10D state policy 的方式是：先把 `DOOR_CKPT` 指向已经 wrap 好的 Door checkpoint，再直接调用底层 A2W float IK 脚本。注意不要把 official LeRobot checkpoint 目录直接传给底层 `isaacgym_float_ik_a2w_basearn_push_door_parallel.py`；底层脚本需要 `door_policy_meta.json`，也就是 Door checkpoint 的 `model_latest.pt`。

```bash
export DOOR_CKPT=high-level/dp/logs/door-auto-wrapped/leroact_a2w_state10_wc4_newfov_randomized_200_depthonly_chunk100_exec50_bs16_0625_0212/100000/model_latest.pt

conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 4 \
  --steps 1000 \
  --seed -1 \
  --door_name wc4 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --enable_wrist_camera \
  --enable_front_camera \
  --camera_depth \
  --depth_only \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --dp_policy_checkpoint "$DOOR_CKPT" \
  --dp_control_all_envs \
  --dp_action_horizon 10 \
  --dp_fps 25 \
  --no_preview_trajectory_at_spawn \
  --no_draw_ik_target \
  --no_draw_camera_axes \
  --no_show_seg
```

### 12.2 用 play_door_policy.py 入口

`play_door_policy.py` 可以自动 wrap official LeRobot checkpoint，但当前 A2W 新 FOV / wc4 randomized 200 的 play 调试里，稳定确认正确的是上面的 12.1 底层脚本 + `DOOR_CKPT=model_latest.pt` 方式。

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/dp/play/play_door_policy.py \
  --checkpoint high-level/dp/logs/lerobot-train/<RUN>/checkpoints/100000 \
  --mode ikpush \
  --robot_body a2wz1 \
  --num_envs 4 \
  --steps 1000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --depth_only \
  --dp_action_horizon 10 \
  --dp_fps 25 \
  -- \
  --door_name wc4 \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5
```

### 12.3 success eval，16 env 一批，跑 3 组

3 组 × 16 env = 48 trials：

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/dp/eval/eval_door_policy_success.py \
  --checkpoint high-level/dp/logs/lerobot-train/<RUN>/checkpoints/100000 \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --mode ikpush \
  --robot_body a2wz1 \
  --num_envs 16 \
  --total_trials 48 \
  --steps 1000 \
  --pass_open_angle_deg 80 \
  --success_metric abs \
  --headless \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --depth_only \
  --dp_action_horizon 10 \
  --dp_fps 25 \
  --base_seed -1 \
  --run_root high-level/logs/door-policy-success/a2w_eval \
  -- \
  --door_name wc4 \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5
```

## 13. 之前犯过/遇到过的问题

1. 转换 raw 到 LeRobot 时忘记加 `--depth_only`：

   会报 `Raw data vision_mode='depth_only', but converter was run with depth mode.`  
   解决：转换命令加 `--depth_only`。

2. shell 变量没设置导致 eval expert obs 读到了当前目录：

   现象是 `IsADirectoryError: ... '/home/sivan/whole_body/visual_whole_body'`，因为 `$RAW_EP` 或 `$OFFICIAL_CKPT` 为空。  
   解决：直接写全路径，或先 `echo "$RAW_EP"` 确认。

3. eval success 只有一半或 0，但 play 看着能开：

   常见原因不是策略，而是成功角度符号统计错。A2W wc4 优先用 `--success_metric abs`。

4. `num_envs=20` 录数据容易卡死：

   原因通常是 Isaac Gym + cameras + raw/video IO + GPU/CPU 内存压力叠加。当前建议一批 16；如果还卡，降到 8。转换时 `--num_workers 4` 也可以降到 2。

5. Gym CUDA illegal memory access：

   之前并行 env 多、资产/相机/PhysX 压力大时出现过。不要因此长期改成 CPU；主线仍是 `--sim_device cuda:0 --rl_device cuda:0`，必要时降低 env 数、headless、重启 Python/Isaac Gym 进程。

6. wrist camera 角度改了但看起来没变：

   原因通常是 A2W 脚本里有自己的默认/挂载覆盖，或者改的是 `door_common.py` 但命令行参数显式覆盖了 yaw/pitch/roll。现在 A2W 默认直接读 `dc.DEFAULT_WRIST_CAMERA_CFG["rotation_deg"]`。

7. front camera 45 度：

   当前是 `DEFAULT_FRONT_CAMERA_CFG["rotation_deg"] = [0.0, -45.0, 0.0]`，不是旧的单独 30 度参数。

8. `--policy.n_action_steps=50`：

   它在 policy config 里，主要决定推理/rollout 每次执行多少 action。ACT 训练监督核心还是 `chunk_size=100` 的 action chunk。

9. joint9 policy play：

   如果 checkpoint sidecar/action_names 是 `[vx, yaw, joint1..jointGripper]`，A2W 脚本会识别为 joint9 并直接给 Z1 joint targets；不要再当 EE target 解。

10. A2W wrist mount 明明在原始 URDF 里，但仿真里没显示：

   新版 `isaacgym_float_ik_a2w_basearn_push_door_parallel.py` 不是直接加载整份 `a2wz1.urdf`，而是调用 `isaacgym_a2w_ik_push_door_parallel.py` 里的 `build_a2wz1_split_asset_root(...)`，把 A2W base 和 Z1 arm 拆成两个临时 URDF：

   ```text
   /tmp/.../a2wz1_split_assets/urdf/a2w_base_visual_only.urdf
   /tmp/.../a2wz1_split_assets/urdf/z1_arm_visual_flip.urdf
   ```

   只有 `A2W_ARM_LINKS` 白名单里的 link 会进入 arm actor。因此如果原始 URDF 新增了 `link06` 的 fixed 子 link，例如：

   ```text
   wrist_up_mount
   wrist_down_mount
   ```

   必须同时把这些 link 加进 `A2W_ARM_LINKS`，否则 split 后的 `z1_arm_visual_flip.urdf` 会把 mount 过滤掉，看起来就像“URDF 写了但没有加载”。

## 14. 当前 git 工作区里和这条链路相关的未提交改动

当前相对 HEAD 已修改的关键文件包括：

1. `high-level/float_ik/door_common.py`
   - 相机默认 FOV/位姿。
   - 门墙体方向修复。
   - DP state/action 10D/9D helper。
   - depth clip/noise metadata。

2. `high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py`
   - A2W robot 默认参数。
   - 默认参数从 `high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml` 读取，CLI 仍然优先。
   - scripted trajectory phase/timing。
   - per-env randomization。
   - A2W camera mount。
   - SPACE 暂停仿真。
   - joint9 policy action 执行入口。

3. `high-level/dp/record/record_door_dp_dataset_a2w_state10.py`
   - A2W 数据录制入口。
   - success target loop。
   - `ee_state10` / `joint_state9` schema。

4. `high-level/dp/convert_door_raw_to_lerobot.py`
   - 动态识别 state/action feature names。
   - depth_only / video 转换。
   - 9D joint action 支持。

5. `high-level/dp/door_dp_common.py`
   - 10D/9D action names。
   - raw/LeRobot recorder metadata。

6. `high-level/dp/play/play_door_policy.py`
   - `--robot_body a2wz1` 入口。

7. `high-level/dp/eval/eval_door_policy_success.py`
   - `--robot_body a2wz1` 入口。

## 15. 维护规则

以后如果修改下面任意内容，需要同步更新本文件：

1. A2W 相机 FOV / position / rotation / mount body。
2. A2W scripted trajectory phase、默认 step、gripper/handle/push 参数。
3. 随机化范围。
4. `high-level/float_ik/config/a2w_float_ik_push_door_parallel.yaml` 里的默认参数分类或字段名。
5. raw schema，尤其 10D / 9D state/action 名字。
6. depth clip、depth noise、camera randomization。
7. raw 转 LeRobot 命令。
8. LeRobot train 命令或 checkpoint 目录命名。
9. play/eval 成功率命令。
10. 任何已经踩过的坑，例如门方向、墙方向、success metric、depth_only mismatch。
11. ACT camera input gating 的结构、默认开关和训练参数。

## 16. ACT camera gating 推理曲线记录

ACT 的 camera input gating 在训练时会把 batch 平均的 `camera_gate_front` / `camera_gate_wrist`
写到 loss dict / W&B；这只是训练 batch 的平均值，不等价于一次 closed-loop play 过程中的时间曲线。

现在 A2W play 推理时也会把最近一次 action chunk forward 得到的 gate 写入 `--dp_log_path` 的 jsonl：

```json
"camera_gates": {
  "front": 0.94,
  "wrist": 1.06,
  "sum": 2.0
}
```

注意：

1. gate 只有启用了 `--policy.camera_input_gating=true` 的 ACT checkpoint 才会有。
2. gate 使用 `2 * softmax`，因此 `front + wrist ≈ 2`。
3. play 中一次 forward 会预测一个 100-step action chunk，但实际只执行 `--dp_action_horizon` 个 action；所以 gate 会在重新采样 action chunk 时更新。比如 `--dp_action_horizon 12` 时，曲线通常是阶梯状，而不是每个 sim step 都重新计算。
4. 如果没有 camera gating，jsonl 里不会出现 `camera_gates` 字段。

画曲线：

```bash
python high-level/dp/plot_camera_gates_from_dp_log.py \
  --log high-level/logs/gating_debug/a2w_gate_trace.jsonl \
  --env_id 0 \
  --out high-level/logs/gating_debug/a2w_gate_trace_env0.png
```

常用 play 示例：

```bash
export DOOR_CKPT=/home/sivan/whole_body/visual_whole_body/high-level/dp/logs/door-auto-wrapped/leroact_a2w_gating_keyframe_w3_r3_sample30_chunk100_exec50_bs16_0630_2234/050000/model_latest.pt

conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 4 \
  --steps 1000 \
  --seed 62000 \
  --door_name wc4 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --enable_wrist_camera \
  --enable_front_camera \
  --camera_depth \
  --depth_only \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --dp_policy_checkpoint "$DOOR_CKPT" \
  --dp_control_all_envs \
  --dp_action_horizon 12 \
  --dp_fps 25 \
  --no_enable_depth_noise \
  --no_enable_depth_gaussian_blur \
  --enable_depth_camera_randomization \
  --depth_camera_pos_rand_m 0.02 \
  --depth_camera_rot_rand_deg 5.0 \
  --dp_log_path high-level/logs/gating_debug/a2w_gate_trace.jsonl \
  --no_preview_trajectory_at_spawn \
  --no_draw_ik_target \
  --no_draw_camera_axes \
  --no_show_seg
```

## 17. 本机 ACT temporal ensemble / NX chunk smoothing

2026-07-02 增加了一个本机仿真可选开关，用来复用 NX 真机部署里的 ACT action chunk overlap temporal ensemble。

新增参数：

```text
--dp_temporal_ensemble
--dp_temporal_prefetch_actions 3
--dp_temporal_old_weight 0.3
--dp_temporal_new_weight 0.7
```

行为：

1. 默认关闭，不影响旧 play/eval。
2. 开启后，policy queue 剩余 action 数 `<= prefetch_actions` 时提前推理新 chunk。
3. 新 chunk 的 step0 对齐当前 policy timestep。
4. 重叠 timestep 按 NX 逻辑融合：
   - `vx/vyaw/xyz`: `0.3 old + 0.7 new`
   - quat: shortest-path SLERP
   - gripper: 直接使用最新 chunk，不融合
5. `--dp_log_path` JSONL 中会写入 `temporal_ensemble` 字段，记录 action 来源、blend_count 和 overlap meta。

相关实现：

```text
high-level/dp/door_policy_backend.py
high-level/dp/door_policy_worker.py
high-level/dp/door_dp_common.py
high-level/dp/play/play_door_policy.py
high-level/dp/eval/eval_door_policy_success.py
high-level/float_ik/door_common.py
high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py
```

最新 100K checkpoint 测试报告：

```text
high-level/dp/result/A2W_ACT_TEMPORAL_ENSEMBLE_TEST_20260702.md
```

关键结果：

```text
raw H12:      34/64 = 53.12%
temporal H12: 34/64 = 53.12%
```

结论：对 `leroact_a2w_keyframe_perstep_w8_r3_sample20_nogating_chunk100_exec50_bs16_0702_0106/100000`
这份 checkpoint，本机仿真里启用 NX-style temporal ensemble 没有提升成功率。它会改变个别 env 的成败，但总体互相抵消。因此本机 eval 默认仍建议用 raw chunk/horizon sweep，只把 temporal ensemble 作为部署一致性或消融选项。

## 18. Plücker-conditioned dual-depth ACT

2026-07-03 增加了一个默认关闭的新 ACT 输入变体：`Plücker-conditioned ACT`。

目标是让 front / wrist 两路 depth token 显式知道每个 pixel 在 robot base 坐标系下对应的 3D 观察射线。实现上不把 Plücker map 拼进 pretrained ResNet 输入，而是：

```text
depth image -> pretrained ResNet -> image feature
camera pose -> full-res 6D Plücker ray-map -> small CNN -> geometry feature
concat(image feature, geometry feature) -> 1x1 projection -> ACT transformer
```

新增训练参数：

```text
--policy.plucker_conditioning=true
--policy.plucker_front_pose_key=observation.camera_pose.front
--policy.plucker_wrist_pose_key=observation.camera_pose.wrist
--policy.plucker_encoder_channels=32,64
--policy.plucker_image_width=640
--policy.plucker_image_height=480
--policy.plucker_horizontal_fov_deg=69.0
```

默认 `plucker_conditioning=false`，旧 ACT / 旧 checkpoint / 旧 play 路径不受影响。开启后 v1 只支持 ResNet backbone；DINOv2 / DeFM 会直接报错。

录制新增参数：

```text
--record_camera_pose
```

启用后 raw episode 会保存：

```text
front_camera_pose_base: T x 7
wrist_camera_pose_base: T x 7
```

并在 sidecar / LeRobot metadata 中记录：

```text
camera_intrinsics
camera_pose_frame = robot_base
camera_pose_convention = optical_frame
camera_pose_features = [
  observation.camera_pose.front,
  observation.camera_pose.wrist,
]
```

注意：

1. Plücker ray-map 在原始 `480x640` 全分辨率生成，再由 Plücker encoder 下采样到 ResNet feature map 尺寸。
2. wrist camera pose 在仿真里由实际 link06 + camera local pose 计算，能跟随机械臂运动变化。
3. camera pose 是几何量，ACT preprocessor 会跳过它的 STATE 归一化，保持真实 robot-base 坐标。
4. 旧 raw dataset 没有每帧 camera pose，不能用于正式 Plücker 训练；正式实验需要重新录制 `--record_camera_pose` 数据。

相关实现：

```text
high-level/lerobot/src/lerobot/policies/act/configuration_act.py
high-level/lerobot/src/lerobot/policies/act/modeling_act.py
high-level/lerobot/src/lerobot/policies/act/processor_act.py
high-level/dp/door_dp_common.py
high-level/dp/convert_door_raw_to_lerobot.py
high-level/dp/door_policy_backend.py
high-level/dp/door_policy_worker.py
high-level/dp/export_official_lerobot_act_to_door_checkpoint.py
high-level/dp/eval/eval_door_dp_on_expert_obs.py
high-level/float_ik/door_common.py
high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py
high-level/dp/record/record_door_dp_dataset_a2w_state10.py
```

已做 smoke test：

```text
Plücker map shape: (1, 6, 480, 640)
max ||d|| error: ~1e-7
max |d·m|: ~2e-8
ACT forward shape: (1, chunk, 10)
plucker_conditioning=false 时 state_dict 无 plucker 参数
```
