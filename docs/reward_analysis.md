# 奖励函数分析

这份文档总结了这个项目里两套主要奖励设计：

- low-level 全身控制器：`low-level/legged_gym/envs/manip_loco`
- high-level 抓取策略：`high-level/envs/b1z1_pickmulti.py`

这份文档主要回答三个问题：

1. 默认真正启用的是哪些奖励
2. 每个奖励的大意是什么
3. 每个奖励的公式或计算形式是什么

## 为什么我把公式改写了

上一版文档里的很多公式用了 LaTeX 语法，比如：

- `\[ ... \]`
- `\(...\)`

很多 IDE 的 Markdown 预览器默认**不支持数学公式渲染**，或者需要额外插件，所以你会看到公式“显示不出来”。

为了让它在更多 Markdown 查看器里正常显示，我把公式统一改成了：

- 行内代码：`reward = exp(-error / sigma)`
- 多行代码块

这种写法虽然没有 LaTeX 漂亮，但兼容性最好。

## 1. 奖励的汇总方式

## 1.1 Low-level `ManipLoco`

代码入口：

- 奖励函数实现：`low-level/legged_gym/envs/rewards/maniploco_rewards.py`
- 奖励装配：`low-level/legged_gym/envs/manip_loco/manip_loco.py`
- 默认权重：`low-level/legged_gym/envs/manip_loco/b1z1_config.py`

`ManipLoco` 不是只有一个总奖励，而是有两条奖励流：

- `rew_buf`：身体 / locomotion 奖励
- `arm_rew_buf`：机械臂 / 末端奖励

它们在 `compute_reward()` 里分别计算：

```text
R_body = (sum_i w_body_i * r_body_i + termination_term) / 100
R_arm  = (sum_j w_arm_j  * r_arm_j  + arm_termination_term) / 100
```

要点：

- 只有 `scale != 0` 且 `scale is not None` 的奖励会被注册
- 默认 `only_positive_rewards = False`，所以不会把负奖励硬裁成 0
- body reward 和 arm reward 最后都会再除以 `100`
- 默认配置下，机械臂这边实际只开了 `tracking_ee_world`

## 1.2 High-level `B1Z1PickMulti`

代码入口：

- 通用奖励基类：`high-level/envs/reward_vec_task.py`
- 任务特化：`high-level/envs/b1z1_pickmulti.py`
- 奖励装配：`high-level/envs/b1z1_base.py`
- 默认权重：`high-level/data/cfg/b1z1_pickmulti.yaml`

high-level 是单一总奖励：

```text
R_high = sum_k w_k * r_k + termination_term
```

要点：

- `_prepare_reward_function()` 会去掉 `scale == 0` 的项
- high-level 末尾**不会**再除以 `100`
- 默认 `only_positive_rewards = False`
- 默认奖励主要围绕：靠近物体、抬起物体、抓取成功、底盘停止与对齐、动作平滑

## 1.3 记号说明

下面会用到一些常见记号：

- `v_cmd`：底盘速度指令
- `v_base`：底盘实际线速度
- `w_cmd`：底盘 yaw 角速度指令
- `w_base`：底盘实际 yaw 角速度
- `p_ee`：末端当前位置
- `p_ee_goal`：末端目标位置
- `q`：关节位置
- `tau`：关节力矩
- `qdot`：关节速度
- `a`：策略动作

注意：代码里很多“惩罚项”函数本身返回的是**正误差**，真正变成惩罚，是靠配置里的负权重实现的。

例如：

```text
r_torque = sum(tau^2)
w_torque < 0
```

所以真正加到总奖励里的是负值。

## 2. Low-level 奖励分析

## 2.1 默认启用的身体奖励

默认权重来自 `B1Z1RoughCfg.rewards.scales`。

| 名称 | 默认权重 | 作用简介 | 公式 / 计算形式 |
| --- | ---: | --- | --- |
| `tracking_contacts_shaped_force` | `-2.0` | 摆动相不该着地时，惩罚脚部接触力 | `reward = -(1 - desired_contact) * (1 - exp(-foot_force^2 / gait_force_sigma))`，4 条腿取平均 |
| `tracking_contacts_shaped_vel` | `-2.0` | 支撑相不该乱动时，惩罚足端速度 | `reward = -desired_contact * (1 - exp(-foot_vel^2 / gait_vel_sigma))`，4 条腿取平均 |
| `feet_air_time` | `2.0` | 鼓励更长的腾空步态 | `reward = sum((feet_air_time - 0.5) * first_contact)` |
| `feet_height` | `1.0` | 鼓励前脚抬脚高度 | `reward = clamp(norm(feet_height) - feet_height_target, max=0)` |
| `tracking_lin_vel_max` | `2.0` | 奖励前向速度接近指令值 | 分段比值形式，见下文 |
| `tracking_ang_vel` | `0.5` | 奖励 yaw 角速度跟踪 | `reward = exp(-(w_cmd - w_base)^2 / tracking_sigma)` |
| `torques` | `-2.5e-5` | 惩罚总力矩过大 | `reward = sum(tau^2)` |
| `stand_still` | `1.0` | 没有走路指令时，鼓励腿保持默认姿态 | `reward = exp(-0.05 * sum(abs(q - q_default)))` |
| `walking_dof` | `1.5` | 有走路指令时，也鼓励腿保持合理姿态 | `reward = exp(-0.05 * sum(abs(q - q_default)))` |
| `alive` | `1.0` | 存活奖励 | `reward = 1` |
| `lin_vel_z` | `-1.5` | 惩罚机身 z 方向速度 | `reward = v_base_z^2` |
| `roll` | `-2.0` | 惩罚机身 roll 过大 | `reward = abs(roll)` |
| `ang_vel_xy` | `-0.2` | 惩罚机身 x/y 角速度 | `reward = wx^2 + wy^2` |
| `dof_acc` | `-7.5e-7` | 惩罚腿部关节加速度 | `reward = sum(((qdot_last - qdot_now) / dt)^2)` |
| `collision` | `-10.0` | 惩罚不该碰撞的刚体发生碰撞 | `reward = sum(contact_force_norm > 0.1)` |
| `action_rate` | `-0.015` | 惩罚动作变化过快 | `reward = sum((a_t - a_t-1)^2)` |
| `dof_pos_limits` | `-10.0` | 惩罚关节越限 | `reward = sum(limit_violation)` |
| `delta_torques` | `-1.0e-7` | 惩罚力矩跳变过大 | `reward = sum((tau_t - tau_t-1)^2)` |
| `hip_pos` | `-0.3` | 惩罚 hip 偏离默认位姿 | `reward = sum((q_hip - q_hip_default)^2)` |
| `work` | `-0.003` | 惩罚腿部机械功 | `reward = abs(sum(tau * qdot over leg joints))` |
| `feet_jerk` | `-0.0002` | 惩罚接触力突变 | `reward = sum(norm(F_t - F_t-1))` |
| `feet_drag` | `-0.08` | 惩罚着地时还在拖脚 | `reward = sum(contact_mask * abs(foot_vel_xyz).sum())` |
| `feet_contact_forces` | `-0.001` | 惩罚接触力超过阈值 | `reward = sum(max(norm(F) - max_contact_force, 0))` |
| `base_height` | `-5.0` | 惩罚机身高度偏离目标值 | `reward = abs(base_height - base_height_target)` |

### `tracking_lin_vel_max` 的具体形式

这个奖励不是常见的平方误差，而是“朝着正确方向达到目标速度就给奖励”。

当 `cmd_x > 0` 时：

```text
reward = min(v_base_x, cmd_x) / (cmd_x + 1e-5)
```

当 `cmd_x < 0` 时：

```text
reward = min(-v_base_x, -cmd_x) / (-cmd_x + 1e-5)
```

当命令速度几乎为 0 时：

```text
reward = exp(-abs(v_base_x))
```

所以它更像“达到命令速度上限前持续奖励”，而不是标准 tracking loss。

## 2.2 默认启用的机械臂奖励

默认权重来自 `B1Z1RoughCfg.rewards.arm_scales`。

| 名称 | 默认权重 | 作用简介 | 公式 / 计算形式 |
| --- | ---: | --- | --- |
| `tracking_ee_world` | `0.8` | 奖励末端跟踪世界坐标系目标点 | `error = sum(abs(p_ee - p_ee_goal_world))`，`reward = exp(-2 * error / tracking_ee_sigma)` |

这也是默认配置里唯一真正启用的 arm reward。

## 2.3 Low-level 奖励可以怎么理解

从设计上看，low-level 奖励大致可以分成四组：

### A. 底盘运动跟踪

- `tracking_lin_vel_max`
- `tracking_ang_vel`
- `lin_vel_z`
- `ang_vel_xy`
- `roll`
- `base_height`

这些奖励告诉四足底盘：

- 应该怎么跟速度指令
- 怎样保持身体稳定
- 怎样控制机身高度和姿态

### B. 腿部平滑性与安全性

- `torques`
- `dof_acc`
- `action_rate`
- `delta_torques`
- `dof_pos_limits`
- `hip_pos`
- `work`
- `collision`
- `feet_contact_forces`
- `feet_jerk`
- `feet_drag`

这些大多是正则项，用来减少：

- 抖动
- 力矩过大
- 冲击过强
- 越限
- 拖脚

### C. 步态塑形

- `tracking_contacts_shaped_force`
- `tracking_contacts_shaped_vel`
- `feet_air_time`
- `feet_height`
- `stand_still`
- `walking_dof`
- `alive`

这些奖励会明显影响学出来的是哪一种 gait。

### D. 机械臂末端跟踪

- `tracking_ee_world`

默认 low-level 的 arm reward 很简单，本质就是让 gripper 去跟踪动态目标点。

## 2.4 已实现但默认没开的 low-level 奖励

`maniploco_rewards.py` 里实现的奖励比默认配置启用的更多。

### 机械臂侧可选奖励

| 名称 | 作用简介 | 公式 / 计算形式 |
| --- | --- | --- |
| `tracking_ee_sphere` | 在球坐标系里跟踪末端目标 | `error = sum(abs(cart2sphere(p_ee_local) - ee_goal_sphere) * sphere_error_scale)`，`reward = exp(-error / tracking_ee_sigma)` |
| `tracking_ee_sphere_walking` | 只在 walking 时启用球坐标跟踪 | `tracking_ee_sphere` 加上 walking mask |
| `tracking_ee_sphere_standing` | 只在 standing 时启用球坐标跟踪 | `tracking_ee_sphere` 加上 standing mask |
| `tracking_ee_cart` | 用球坐标目标反推笛卡尔点再跟踪 | `reward = exp(-sum(abs(p_ee - p_target)) / tracking_ee_sigma)` |
| `tracking_ee_orn` | 跟踪末端姿态 | `orn_err = sum(abs(wrap_to_pi(goal_rpy - ee_rpy)) * orn_error_scale)`，`reward = exp(-orn_err / tracking_ee_sigma)` |
| `tracking_ee_orn_ry` | 只跟踪姿态中的 roll 和 yaw | 同上，但只取 `[roll, yaw]` |
| `arm_energy_abs_sum` | 惩罚机械臂功耗 | `reward = sum(abs(tau_arm * qdot_arm))` |

### 底盘侧可选奖励

| 名称 | 作用简介 | 公式 / 计算形式 |
| --- | --- | --- |
| `tracking_lin_vel` | 标准 xy 速度跟踪 | `reward = exp(-sum((v_cmd_xy - v_base_xy)^2) / tracking_sigma)` |
| `tracking_lin_vel_x_l1` | 前向速度 L1 风格跟踪 | `reward = normalized(-abs(cmd_x - v_x) + abs(cmd_x))` |
| `tracking_lin_vel_x_exp` | 前向速度指数跟踪 | `reward = exp(-abs(cmd_x - v_x) / tracking_sigma)` |
| `tracking_ang_vel_yaw_l1` | yaw 角速度 L1 风格跟踪 | `reward = -abs(cmd_yaw - w_yaw) + abs(cmd_yaw)` |
| `tracking_ang_vel_yaw_exp` | yaw 角速度指数跟踪 | `reward = exp(-abs(cmd_yaw - w_yaw) / tracking_sigma)` |
| `tracking_lin_vel_y_l2` | 惩罚 y 方向速度误差 | `reward = (cmd_y - v_y)^2` |
| `tracking_lin_vel_z_l2` | 惩罚 z 方向速度误差 | `reward = (cmd_z - v_z)^2` |
| `survive` | 常数存活奖励 | `reward = 1` |
| `foot_contacts_z` | 惩罚垂直接触力 | `reward = sum(Fz^2)` |
| `energy_square` | 惩罚腿部功率平方 | `reward = sum((tau * qdot)^2)` |
| `tracking_lin_vel_y` | 侧向速度指数跟踪 | `reward = exp(-(cmd_y - v_y)^2 / tracking_sigma)` |
| `orientation` | 惩罚机身倾斜 | `reward = sum(projected_gravity_xy^2)` |
| `orientation_walking` | walking 时的机身倾斜惩罚 | `orientation + walking mask` |
| `orientation_standing` | standing 时的机身倾斜惩罚 | `orientation + standing mask` |
| `torques_walking` | walking 时的力矩惩罚 | `torques + walking mask` |
| `torques_standing` | standing 时的力矩惩罚 | `torques + standing mask` |
| `energy_square_walking` | walking 时的功率平方惩罚 | `energy_square + walking mask` |
| `energy_square_standing` | standing 时的功率平方惩罚 | `energy_square + standing mask` |
| `base_height_walking` | walking 时的 base height 奖励 | `base_height + walking mask` |
| `base_height_standing` | standing 时的 base height 奖励 | `base_height + standing mask` |
| `dof_default_pos` | 奖励腿关节接近默认姿态 | `reward = exp(-0.05 * sum(abs(q - q_default)))` |
| `dof_error` | 惩罚腿关节偏离默认姿态 | `reward = sum((q - q_default)^2)` |
| `penalty_lin_vel_y` | 非转弯时惩罚横向漂移 | `reward = abs(v_y)`，大 yaw 命令时置 0 |

## 2.5 Low-level 设计上的直观理解

low-level 奖励的总体风格很明确：

- 底盘侧：以稳定 locomotion 为主
- 机械臂侧：以末端跟踪为主
- 正则项很多：防止学出“能完成任务但很难看”的动作
- 步态项很多：说明作者对 gait 形态是有明显偏好的

另外要特别注意：

- `rew_buf` 和 `arm_rew_buf` 是分开的
- 默认 arm reward 非常稀疏，只管位置跟踪，不直接管 arm energy，也不直接管姿态

## 3. High-level 奖励分析

## 3.1 默认启用的 high-level 奖励

默认权重来自 `high-level/data/cfg/b1z1_pickmulti.yaml`。

| 名称 | 默认权重 | 作用简介 | 公式 / 计算形式 |
| --- | ---: | --- | --- |
| `approaching` | `0.5` | 奖励 gripper 比之前更接近物体 | `dist_delta = clip(closest_dist - curr_dist, 0, 10)`，`reward = tanh(10 * dist_delta)` |
| `lifting` | `1.0` | 奖励物体比之前抬得更高 | `height_delta = clip(curr_height - highest_object, 0, 10)`，`reward = tanh(10 * height_delta)` |
| `pick_up` | `3.5` | 抓起成功给稀疏奖励 | `reward = 1 if lifted_object else 0` |
| `acc_penalty` | `-0.001` | 惩罚机械臂加速度过大 | `penalty = norm(qdot_arm_now - qdot_arm_last) / dt`，`reward = 1 - exp(-penalty)` |
| `command_penalty` | `-1.0` | 当底盘已经靠近物体时，惩罚继续给大前进指令 | `penalty = norm(commands[:, :1]) if base_obj_dist < 0.6 else 0` |
| `command_reward` | `0.25` | 靠近物体后，奖励底盘命令接近 0 | `reward = exp(-abs(cmd_x)) if base_obj_dist < 0.6 else 0` |
| `standpick` | `0.25` | 奖励“停稳了再抓” | `reward = 1 if (base_obj_dist < threshold and cmd_x < 0.15) else 0` |
| `action_rate` | `-0.001` | 惩罚动作变化过快 | `reward = norm(actions[:, 7:9] - last_actions[:, 7:9])` |
| `ee_orn` | `0.01` | 奖励 gripper 朝向物体 | `reward = cosine_similarity(ee_x_dir_world, obj_dir_unit)` |
| `base_dir` | `0.25` | 名义上想奖励底盘朝向物体，但当前实现疑似有坐标 bug | 当前实现见下文注意事项 |
| `base_approaching` | `0.01` | 奖励底盘与物体保持在合适半径附近 | `delta = abs(base_obj_dist - base_object_distance_threshold)`，`reward = tanh(-10 * delta) + 1` |
| `grasp_base_height` | `0.5` | 物体被抓住时，奖励底盘保持目标高度 | `reward = exp(-abs(base_height - target_height)) * lifted_now` |

默认未启用：

- `gripper_rate = 0.0`
- `rad_penalty = 0.0`
- `base_ang_pen = 0.0`

## 3.2 High-level 里几个关键状态量

high-level 的奖励不是只看当前帧，它还维护了一些“历史最优值”：

- `curr_dist`：当前末端到物体的距离
- `closest_dist`：本回合历史最小距离
- `curr_height`：当前物体相对初始参考高度的抬升量
- `highest_object`：本回合历史最高抬升量
- `lifted_now`：当前这一刻物体是否被抬起且还靠近 gripper
- `lifted_object`：是否已经达到抓起成功阈值

所以它属于比较典型的“progress shaping”：

- 离得更近了给奖励
- 抬得更高了给奖励
- 真正抬起来再给一个成功奖励

## 3.3 High-level 奖励的直观分组

### A. 操作进度奖励

- `approaching`
- `lifting`
- `pick_up`

这些奖励直接对应抓取任务的三个阶段：

1. 先靠近
2. 再抬起
3. 最后成功

### B. 底盘行为塑形

- `command_reward`
- `command_penalty`
- `standpick`
- `base_dir`
- `base_approaching`
- `grasp_base_height`

这组奖励的目标是：

- 底盘先走到合适位置
- 到位后别继续乱冲
- 姿态和高度尽量合适

### C. 正则和平滑项

- `acc_penalty`
- `action_rate`
- `ee_orn`

这些项主要负责：

- 减少抖动
- 减少动作突变
- 让夹爪姿态更合理

## 3.4 已实现但默认没开的 high-level 奖励

| 名称 | 作用简介 | 公式 / 计算形式 |
| --- | --- | --- |
| `reach` | 当末端进入物体附近 8cm 时给稀疏奖励 | `reward = 1 if norm(p_ee - p_obj) < 0.08 else 0` |
| `gripper_rate` | 惩罚 gripper 动作变化过快 | `reward = norm(actions[:, 6:7] - last_actions[:, 6:7])` |
| `rad_penalty` | 奖励末端目标半径接近 0.9m | `reward = exp(-abs(norm(curr_ee_goal_cart) - 0.9) / 0.15)` |
| `base_ang_pen` | 惩罚底盘角速度过大 | `reward = norm(base_ang_vel_local)` |
| `base_height` | 奖励底盘高度接近目标值 | `reward = exp(-abs(base_height - base_height_target))` |

## 3.5 High-level 的整体风格

和 low-level 相比，high-level 更偏任务驱动：

- 不怎么关心 gait 细节
- 更关心“靠近了没、抬起来没、成功了没”
- 同时通过底盘停稳、朝向、距离环来塑造抓取策略

这和整个系统分层结构是匹配的：

- low-level 负责稳定的全身运动
- high-level 负责任务策略和物体交互逻辑

## 4. Low-level 和 High-level 的对比

| 维度 | Low-level | High-level |
| --- | --- | --- |
| 核心目标 | 稳定行走 + 末端跟踪 | 接近物体、抬起物体、抓取成功 |
| 奖励结构 | 两条奖励流：body + arm | 单个任务奖励 |
| 主要 shaping 来源 | 速度跟踪、步态、稳定性 | 接近进度、抬升进度、成功信号 |
| 主要正则来源 | 力矩、加速度、动作变化、接触惩罚 | 动作平滑、臂加速度、底盘指令抑制 |
| 是否有显式成功奖励 | 基本没有 | 有，`pick_up` |
| 是否强依赖步态设计 | 很强 | 很弱 |

## 5. 实际调参时最值得先看的奖励

如果你后面要调这个项目，我建议优先看这些项。

### 想改 low-level 行走质量

- `tracking_lin_vel_max`
- `tracking_ang_vel`
- `feet_air_time`
- `tracking_contacts_shaped_force`
- `tracking_contacts_shaped_vel`
- `base_height`

### 想改 low-level 末端跟踪

- `tracking_ee_world`

### 想改 high-level 抓取成功率

- `approaching`
- `lifting`
- `pick_up`
- `command_penalty`
- `command_reward`
- `base_approaching`
- `ee_orn`

## 6. 代码实现里的注意事项

## 6.1 low-level 配置里有重复赋值

在 `low-level/legged_gym/envs/manip_loco/b1z1_config.py` 里，有些奖励权重名字在同一个 class 里被写了两次，比如：

- `delta_torques`
- `work`
- `energy_square`

Python class attribute 的规则是：**后面的赋值覆盖前面的赋值**。  
所以真正生效的是后面那个值。

## 6.2 很多“惩罚函数”本体返回的是正数

例如下面这些函数本身都返回正误差：

- torque norm
- collision count
- joint limit violation
- acceleration norm

它们之所以是惩罚，是因为 scale 配成了负数。

## 6.3 low-level 的 body reward 和 arm reward 没有在环境里直接相加

这一点很容易看漏。

`ManipLoco.step()` 返回的是：

- `rew_buf`
- `arm_rew_buf`

也就是说，训练脚本可能是分开消费这两个奖励的，而不是直接拿一个统一标量。

## 6.4 walking / standing mask 会影响一批奖励

代码里 walking 的判定大致是：

```text
walking =
    (abs(cmd_x)   > lin_vel_x_clip) or
    (abs(cmd_y)   > lin_vel_x_clip) or
    (abs(cmd_yaw) > ang_vel_yaw_clip)
```

这个 mask 会影响：

- `stand_still`
- `walking_dof`
- `orientation_walking`
- `orientation_standing`
- `torques_walking`
- `torques_standing`
- `energy_square_walking`
- `energy_square_standing`
- `base_height_walking`
- `base_height_standing`

## 6.5 一些实现上值得留意的坑

### `base_dir` 很可能有坐标 bug

在 `high-level/envs/reward_vec_task.py` 里，`base_dir` 的实现是：

```text
obj_dir = obj_pos - robot_root_pos
obj_dir[:, :2] = 0
```

这意味着它把 `x/y` 清零了，只留下 z 分量。  
但从名字看，这个奖励本来应该是“让底盘朝向物体”，那理论上更合理的写法通常应该是把 `z` 清零，而不是把 `x/y` 清零。

所以目前这个奖励的实现，和它的名字可能不一致。

### 一些 low-level 可选奖励不能直接放心打开

有些可选奖励写法是 `self.env._reward_xxx()`，这会带来两类问题：

1. `ManipLoco` 根本没有这个方法，打开后可能直接报错
2. 有些名字会落到父类 `LeggedRobot` 的实现上，导致公式和你以为的不一样

比较典型的例子：

- `tracking_ee_sphere_walking`
- `tracking_ee_sphere_standing`
- `energy_square_walking`
- `energy_square_standing`

这些项如果后面你想启用，最好先单独检查一遍调用路径。

### `base_height_walking` / `base_height_standing` 可能不是你以为的那个 `base_height`

这两个 masked 版本调用的是 `self.env._reward_base_height()`。  
如果最终解析到了父类 `LeggedRobot` 里的实现，那么它用的就不是 `maniploco_rewards.py` 这一版 `base_height`。

所以：

- `base_height`
- `base_height_walking`
- `base_height_standing`

三者未必完全一致。

### 末端姿态奖励可能会用到旧的欧拉角目标

`tracking_ee_orn` 和 `tracking_ee_orn_ry` 依赖的是：

```text
self.env.ee_goal_orn_euler
```

但在 `manip_loco.py` 里：

- `ee_goal_orn_euler` 初始化过一次
- 后续更新主要是在改 `ee_goal_orn_quat`
- 没看到同步更新 `ee_goal_orn_euler`

所以如果以后你打开姿态奖励，这里值得再确认一下，否则可能会出现：

- quaternion 目标是新的
- euler 目标还是旧的

## 7. 相关源码位置

- `low-level/legged_gym/envs/manip_loco/manip_loco.py`
- `low-level/legged_gym/envs/rewards/maniploco_rewards.py`
- `low-level/legged_gym/envs/manip_loco/b1z1_config.py`
- `high-level/envs/reward_vec_task.py`
- `high-level/envs/b1z1_base.py`
- `high-level/envs/b1z1_pickmulti.py`
- `high-level/data/cfg/b1z1_pickmulti.yaml`
