# VBC low-level 与 low_level_WBC 差异对比

对比时间：2026-05-12  
VBC 当前仓库：`/home/sivan/whole_body/visual_whole_body/low-level`  
B2Z1 WBC 仓库：`/home/sivan/whole_body/Interactive-Navigation-for-legged-manipulator/low_level_WBC`

本文主要对比 `legged_gym` 的 low-level 行为差异。B2Z1 仓库里使用 B2Z1 机器人、URDF、关节命名和部分 asset 字段，这是预期差异；下面重点列出除机器人型号外会影响训练/仿真的差异。

## 总结

外部 `low_level_WBC` 不只是把 B1Z1 换成 B2Z1，还明显改了训练设置：

- Sample command 更保守：线速度/yaw 范围从 `[-0.8, 0.8]` / `[-1.0, 1.0]` 缩到 `[-0.6, 0.6]` / `[-0.6, 0.6]`，clip 从 `0.2/0.5` 缩到 `0.1/0.1`，且额外加入 5% 全零命令和 10% 纯转向命令。
- 腿部 PD 更硬：leg `Kp/Kd` 从 `80/2.0` 改为 `360/5.0`；Z1 arm 仍是 `5/0.5`。
- Reward scale 大改：B2Z1 版本开启 `only_positive_rewards`，接触力阈值更高，base height 从负惩罚变成正向 exp 奖励，新增多个站立/摆腿/关节姿态约束项。
- Observation/noise 也变了：B2Z1 开启 obs noise，但把 lin vel、gravity、height noise 置 0；同时 observation 中 foot contact 被强行置 0。
- 环境逻辑也改了：B2Z1 reset dof 不再随机扰动，gait clock 无论 `observe_gait_commands` 是否开启都会更新，并加入 headless camera sensor。

## 主要配置差异

| 模块 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 | 影响 |
|---|---:|---:|---|
| EE sphere center x offset | `0.3` | `0.2` | 手臂目标球心相对 base 更靠后 |
| EE pos_l | `[0.4, 0.95]` | `[0.35, 0.90]` | EE 半径范围更小 |
| EE pos_p | `[-pi/2.5, pi/3]` | `[-1.2*pi/2, 1.2*pi/2]` | pitch 采样范围更大 |
| EE pos_y | `[-1.2, 1.2]` | `[-1.6, 1.6]` | yaw 采样范围更大 |
| noise.add_noise | `False` | `True` | B2Z1 默认加 obs noise |
| noise dof_vel | `1.5` | `1.0` | dof velocity noise 变小 |
| noise lin_vel / gravity / height | `0.1 / 0.05 / 0.1` | `0 / 0 / 0` | B2Z1 这些量不加噪 |
| obs scale lin_vel / ang_vel | `1.0 / 1.0` | `2.0 / 0.25` | command/velocity 归一化权重不同 |
| num_envs | `6144` | `1` | B2Z1 配置当前像调试/play 设置，不是大规模训练默认值 |
| init base z | `0.5` | `0.6` | 初始机身高度不同 |
| hip default angle | `FL/RL=0.2, FR/RR=-0.2` | `FL/RL=0.1, FR/RR=-0.1` | 默认站姿更窄 |
| gripper default | `-0.785` | `0.0` | gripper 初始角不同 |
| termination z_threshold | `0.1` | `0.2` | B2Z1 低高度终止阈值更高 |

## Sample Command 差异

| 项 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---:|---:|
| resampling_time | `3.0s` | `3.0s` |
| lin_vel_x range | `[-0.8, 0.8]` | `[-0.6, 0.6]` |
| ang_vel_yaw range | `[-1.0, 1.0]` | `[-0.6, 0.6]` |
| lin_vel_x_clip | `0.2` | `0.1` |
| ang_vel_yaw_clip | `0.5` | `0.1` |
| lin_vel_y_clip | 无 | `0.1` |
| 采样策略 | 前 `5000*24` steps 只采样正向 vx `[0, max]`，之后采样完整 vx 范围 | 从一开始采完整 vx/yaw 范围 |
| 额外命令分布 | 无 | 5% 命令强制全零；10% 命令强制 `vx=0`，形成纯转向 |
| y 方向命令 | 始终置 0 | 始终置 0 |

对应代码：

- B1Z1 config: `low-level/legged_gym/envs/manip_loco/b1z1_config.py`
- B2Z1 config: `/home/sivan/whole_body/Interactive-Navigation-for-legged-manipulator/low_level_WBC/legged_gym/envs/manip_loco/b2z1_config.py`
- B1Z1 sampling: `ManipLoco._resample_commands`
- B2Z1 sampling: `ManipLoco._resample_commands`

## KP/KD 与控制

| 项 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---:|---:|
| leg stiffness / Kp | `{'joint': 80}` | `{'joint': 360.0}` |
| leg damping / Kd | `{'joint': 2.0}` | `{'joint': 5.0}` |
| arm stiffness / Kp | `{'z1': 5}` | `{'z1': 5}` |
| arm damping / Kd | `{'z1': 0.5}` | `{'z1': 0.5}` |
| action_scale | 相同：leg `[0.4,0.45,0.45]*4`，arm `[2.1,0.6,0.6,0,0,0]` | 相同 |
| decimation | `4` | `4` |
| torque_supervision | `False` | `False` |

结论：B2Z1 主要把腿部 PD 大幅加硬，arm 的 OSC 参数和 `z1` PD 没变。

## Domain Randomization 差异

| 项 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---:|---:|
| friction_range | `[0.3, 3.0]` | `[0.4, 3.0]` |
| base added_mass_range | `[0, 15]` | `[-5, 10]` |
| base COM x/y/z range | `[-0.15, 0.15]` | `[-0.3, 0.3]` |
| gripper_added_mass_range | `[0, 0.1]` | `[-3.0, 3.0]` |
| motor strength range | leg/arm 都 `[0.7, 1.3]` | 相同 |
| push interval / max vel | `8s / 0.5` | 相同 |

B2Z1 对 COM 和末端质量扰动放大很多，base mass 还允许负向扰动。

## Reward 基础参数差异

| 项 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---:|---:|
| only_positive_rewards | `False` | `True` |
| soft_torque_limit | `0.4` | `0.9` |
| max_contact_force | `40` | `200` |
| min_contact_force | 无 | `10` |
| gait_vel_sigma | `0.5` | `1` |
| gait_force_sigma | `0.5` | `100` |
| feet_height_target | `0.3` | `0.2` |
| base_height_sigma | 无 | `1.0` |
| swing_ratio / stance_ratio | 无 | `0.375 / 0.625` |
| clearance_height_target | 无 | `-0.3` |

## Reward Scale 差异

注意：B1Z1 config 中 `delta_torques` 和 `work` 有重复赋值，Python 实际会采用后一次赋值，所以 B1Z1 实际 scale 是 `delta_torques=-1e-7`、`work=-0.003`。

| reward | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---:|---:|
| tracking_contacts_shaped_force | `-2.0` | `1.0` |
| tracking_contacts_shaped_vel | `-2.0` | `1.0` |
| tracking_contacts_shaped_force_2 | 无 | `0.0` |
| tracking_contacts_shaped_vel_2 | 无 | `0.0` |
| feet_air_time | `2.0` | `0.8` |
| feet_height | `1.0` | `1.0` |
| feet_height_standing | 无 | `1.0` |
| feet_height_turning | 无 | `3.0` |
| tracking_lin_vel_max / max_x | `tracking_lin_vel_max=2.0` | `tracking_lin_vel_max_x=1.0` |
| tracking_lin_vel_x_l1 | `0.0` | `0.0` |
| tracking_lin_vel_x_exp | `0` | `0.0` |
| tracking_ang_vel | `0.5` | `0.5` |
| torques | `-2.5e-5` | `-2.5e-7` |
| stand_still | `1.0` | `1.0` |
| walking_dof | `1.5` | `1.5` |
| dof_default_pos | `0.0` | `0.05` |
| alive | `1.0` | `0.0` |
| lin_vel_z | `-1.5` | `-3.0` |
| roll | `-2.0` | `-1.5` |
| base_height | `-5.0` | `3.0` |
| ang_vel_xy | `-0.2` | `-0.02` |
| dof_acc | `-7.5e-7` | `-5e-7` |
| dof_vel | 无 | `-0.0008` |
| collision | `-10.0` | `-5.0` |
| action_rate | `-0.015` | `-0.05` |
| dof_pos_limits | `-10.0` | `-3.0` |
| delta_torques | `-1e-7` | `-1e-7/4` |
| hip_pos | `-0.3` | `-0.5` |
| hip_pos_standing | 无 | `-0.1` |
| thigh_pos | 无 | `-0.1` |
| thigh_pos_back | 无 | `-1.5` |
| calf_pos | 无 | `-1.0` |
| work | `-0.003` | `0` |
| feet_jerk | `-0.0002` | `-0.02` |
| feet_drag | `-0.08` | `-0.08` |
| feet_contact_forces | `-0.001` | `-0.01` |
| feet_contact_forces_standing | 无 | `-0.001` |
| orientation | `0.0` | `-0.2` |
| orientation_walking | `0.0` | `-0.0` |
| orientation_standing | `0.0` | `0.0` |
| penalty_lin_vel_y | `0.0` | `2.0` |
| arm tracking_ee_world | `0.8` | `2.0` |
| arm tracking_ee_orn | `0.4` | `0.0` |

## Reward 函数实现差异

B2Z1 版本不只是改 scale，还改了 reward 函数本身：

- `_reward_tracking_ee_orn`：B1Z1 从 `ee_goal_orn_quat` 转 Euler；B2Z1 直接用 `ee_goal_orn_euler`。
- `_reward_tracking_ang_vel`：B1Z1 metric 返回 yaw error；B2Z1 metric 返回 reward 本身。
- 新增 `_reward_ang_vel_standing`、`_reward_dof_vel`、`_reward_action_jerk`。
- `_reward_base_height`：B1Z1 返回 `abs(base_height-target)`，配 `base_height=-5` 做惩罚；B2Z1 返回 `exp(-height_error^2/base_height_sigma)`，配 `base_height=3` 做正奖励。
- `_reward_dof_default_pos`：B1Z1 是 `exp(-abs_error*0.05)`；B2Z1 是 `exp(-abs_error^2/tracking_position_sigma)`。
- `_reward_tracking_lin_vel_max`：B2Z1 改名为 `_reward_tracking_lin_vel_max_x`，并新增 `_reward_tracking_lin_vel_max_y`。
- `_reward_penalty_lin_vel_y`：B1Z1 返回 `abs(vy)` 并在转向时置 0；B2Z1 返回 `-vy^2`。由于 B2Z1 scale 是正的 `2.0`，这个项实际仍是惩罚。
- 新增站立接触力惩罚 `_reward_feet_contact_forces_standing`，对站立时低于 `min_contact_force` 的足端力做惩罚。
- 新增关节姿态项 `_reward_hip_pos_standing`、`_reward_thigh_pos`、`_reward_thigh_pos_back`、`_reward_calf_pos`。
- `_reward_feet_height`：B1Z1 使用 `clamp(norm(feet_height)-target, max=0)`；B2Z1 改为前脚高度相对 target 的负平方误差，并新增站立/转向版本。
- `_reward_feet_air_time`：B1Z1 阈值 `0.5` 且用 `foot_contacts_from_sensor`；B2Z1 阈值 `0.4` 且用 `contact_filt`。

## Observation 与环境逻辑差异

| 项 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---|---|
| foot contact observation | 直接加入 `_reindex_feet(self.foot_contacts_from_sensor)` | 强制 `0 * _reindex_feet(...)`，等于不给策略真实 foot contact |
| arm_base_offset | `[0.3, 0, 0.09]` | `[0.2, 0, 0.09]` |
| reset dof | `default_dof_pos * rand(0.8, 1.2)` | 直接 `default_dof_pos`，不随机 |
| gait clock update | 只有 `observe_gait_commands=True` 时更新 | 无条件更新 |
| turning mask | 无 | 新增 `_get_turning_cmd_mask` |
| terrain creation | 默认直接创建 `Terrain + trimesh` | 支持 `plane/heightfield/trimesh` 分支 |
| headless/render | 无 floating camera sensor | 新增 `FloatingCameraSensor`，每步写 `extras["vis"]` |
| base_task headless | headless 时 graphics device 设为 `-1` | 相关逻辑被注释，headless 也保留 graphics device |

## PPO / 训练参数差异

| 项 | VBC low-level / B1Z1 | low_level_WBC / B2Z1 |
|---|---:|---:|
| actor_hidden_dims | `[128]` | `[512, 256, 128]` |
| critic_hidden_dims | `[128]` | `[512, 256, 128]` |
| init_noise_std | 无显式字段 | `1.0` |
| entropy_coef | `0.0` | `0.001` |
| max_iterations | `80000` | `100000` |
| save_interval | `200` | `100` |
| experiment_name | `b1z1_v2` | `B2Z1_v2` |

脚本层面还有这些不同：

- B2Z1 `train.py` 强行优先导入本仓库的 `legged_gym`，避免拿到系统里安装的旧包。
- B2Z1 `train.py` wandb 保存的是 `b2z1_config.py`；VBC 保存的是 `b1z1_config.py`。
- B2Z1 `helpers.py` 默认 `--resume=True`，VBC 默认 `False`。
- B2Z1 `helpers.py` 默认 `--headless=False`，VBC 默认 `True`。
- B2Z1 `play.py` 写死了一个 B2Z1 log path，并默认 teleop、plane terrain、关闭 noise 和大部分 domain randomization；VBC `play.py` 主要按传入参数决定路径，并支持 `--fixed_vx/--fixed_yaw`。

## 机器人/asset 相关差异

这部分按你的说明属于预期的 B1Z1 vs B2Z1 差异，但也会影响训练行为：

- URDF 从 `resources/robots/b1z1/urdf/b1z1.urdf` 换成 `resources/robots/b2_z1/urdf/b2_plus_z1.urdf`。
- gripper body name 从 `ee_gripper_link` 换成 `gripperMover`。
- penalized contacts 从 `["thigh", "trunk", "calf"]` 改成 `["thigh", "base", "calf"]`。
- terminate contacts 从空列表改成 `["base"]`。
- B2Z1 asset 额外设置 `default_dof_drive_mode=3`、`replace_cylinder_with_capsule=True`、`flip_visual_attachments=True`。

## 结论建议

如果要把外部 `low_level_WBC` 的策略/训练行为迁回 VBC，不能只替换 URDF。至少需要同步确认以下开关：

1. 是否采用 B2Z1 的 conservative command distribution：更小 command range、5% stop、10% yaw-only。
2. 是否采用 B2Z1 的 leg PD：`Kp=360, Kd=5`。
3. 是否采用 B2Z1 reward set：尤其是 `only_positive_rewards=True`、`base_height=+3` 的 exp reward、站立足端力/脚高/大腿小腿姿态项。
4. 是否让策略失去 foot contact observation：B2Z1 当前实际传入全零 contact obs。
5. 是否保留 B2Z1 的 reset dof 不随机、gait clock 无条件更新、headless camera sensor。

