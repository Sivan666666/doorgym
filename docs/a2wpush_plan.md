# A2WPush Mode 集成方案

## Summary

目标是新增 `a2wpush` mode，用 A2W+Z1 替代当前 `ikpush` 的 float-base+Z1 开门流程，并完整接入 scripted、record、convert、train、play、eval/replay pipeline。

第一版保持 high-level policy I/O 不变：

- 输入：`73D state + 四张图`
- 输出：`10D action`
- action 语义仍是：`[vx, yaw_rate, ee_xyz_base, ee_quat_base, gripper]`

第一版只做 Isaac Gym 仿真闭环，不接真机 SDK。由于当前可能只有单独 A2W URDF，没有 A2W+Z1 总 URDF/YAML，实施时第一步必须是资产检查和最小可运行资产/config 构建，不能假设配置已经完整存在。

## Important Stop Conditions

实现过程中遇到以下情况必须立即停止并向用户确认，不能擅自决定：

- A2W+Z1 总 DOF 或需要进入 policy obs 的 DOF 超过现有 `19 dof_pos + 19 dof_vel` 槽位。
- 需要改变 policy state 维度，不再是 73D。
- 需要改变 policy action 维度，不再是 10D。
- 想把 freeze 做成 policy action 的额外维度。
- 想接入真机 SDK 或真实 A2W 控制接口。
- A2W URDF 里的 wheel DOF、Z1 DOF、gripper DOF 无法可靠识别。

如果 DOF 超过 19，不允许擅自截断、不允许按 URDF 前 19 个取、不允许随便丢弃某些关节。必须让用户在以下方向中确认：

- 保持 73D，只选择 19 个关键 DOF 进入 obs。
- 扩展 state 维度，重新训练新 schema。
- 只记录 Z1+wheel，leg joints 不进 policy obs。

## Asset And Config Preparation

第一阶段先检查用户提供的 A2W URDF：

- 是否能被 Isaac Gym 加载。
- 有哪些 DOF，打印 DOF 名称、顺序、limit、drive mode。
- 哪些 DOF 是 wheel，哪些是 leg，哪些可能需要固定。
- 是否已有 Z1 mount，若没有，需要构建 A2W+Z1 组合资产方案。

若没有完整 A2W+Z1 URDF/YAML，则需要新建最小可运行配置，至少包含：

- A2W 或 A2W+Z1 asset 路径。
- wheel DOF 名称。
- Z1 arm/gripper DOF 名称。
- 初始姿态、stiffness/damping、velocity/torque limits。
- 相机挂载位置。
- 进入 73D state 的 `state_dof_names`，长度必须明确为 19。

## A2W Base Control

当前 `ikpush` 是直接改 root pose；`a2wpush` 不能继续这样做。

`a2wpush` 中 action 的 `vx/yaw_rate` 先进入 A2W 底盘控制器，再转换成 wheel velocity targets：

- `vx`：机体系前向速度。
- `yaw_rate`：机体系 yaw 角速度。
- 第一版不支持 `vy`。
- 第一版按仿真 wheel velocity target 驱动，不使用 Unitree SDK。

如果 wheel 几何参数不明确，不能硬猜，需要从 URDF/YAML 或用户确认中确定：

- wheel DOF 名称。
- wheel radius。
- 左右轮距或等效转向参数。
- wheel velocity/acceleration/torque limit。

## State And Action Schema

第一版保持 73D state 硬兼容：

- `base_roll`
- `base_pitch`
- `base_ang_vel_x/y/z`
- 19 个显式配置的 DOF position
- 同一 19 个显式配置的 DOF velocity
- 18D last action，沿用当前 padding 规则
- 4D foot contact，第一版继续置零
- EE base-frame xyz
- EE quaternion
- gripper position

第一版保持 10D action：

- `vx`
- `yaw_rate`
- `ee_x/y/z` in base frame
- `ee_qx/qy/qz/qw` in base frame
- `gripper`

raw metadata 需要新增或区分：

- `mode=a2wpush`
- `a2wpush_state_version`
- A2W asset metadata
- `state_dof_names`
- wheel control metadata

不要把 `a2wpush_state_version` 和旧 `ikpush_state_version` 混用。

## Freeze Definition

当前不把 freeze 加进 policy action，仍保持 10D。

这里讨论的 freeze 含义是：在某些阶段让底盘关节/轮子被锁住或强制零速度，使 Z1 IK 开门时底盘不抖，从而让 EE tracking 更准。

第一版可以只做调试型 freeze，不进入数据集 action：

- 通过 CLI 参数控制全程 freeze、按 scripted phase freeze、或按固定 step window freeze。
- freeze 生效时：`vx=0`，`yaw_rate=0`，wheel velocity target 为 0，并尽量保持相关底盘/轮腿关节目标稳定。
- freeze 只用于消融和调试，不改变 raw action 维度，不改变 checkpoint action_dim。

如果以后决定让 policy 学 freeze，则必须另开 A2W 专用 11D action schema：

- 例如 `[vx, yaw_rate, ee_xyz, ee_quat, gripper, freeze_base]`
- record 需要生成 freeze label
- convert/train/play/eval 都要识别新 action_dim
- 旧 10D checkpoint 不兼容
- 实现前必须再次向用户确认

## Pipeline Integration

需要接入以下入口：

- `record_door_dp_dataset.py` 增加 `--mode a2wpush`，调用 A2W scripted scene 录 raw `.npz`。
- `play_door_policy.py` 增加 `--mode a2wpush`，加载 DP/ACT/pi0.5 checkpoint 后在 A2W scene 闭环执行。
- `eval_door_policy_success.py` 增加 `--mode a2wpush`，继续批量跑成功率。
- `replay_door_dp_raw_in_isaacgym.py` 增加 A2W raw replay 路径。
- `eval_door_dp_on_expert_obs.py` 不依赖仿真，理论上只需接受 `a2wpush_state_version` metadata 并保持 73D/10D 校验。
- `convert/train` 尽量不改模型结构；只要 raw 数据仍是 73D/10D，就沿用现有 LeRobot DP/ACT/pi0.5 训练脚本。

## Test Plan

资产 smoke：

- A2W URDF 能加载。
- A2W+Z1 组合资产能加载。
- 打印 DOF 名称、数量、limits。
- 若 DOF/obs 需求超过 19，立即停止并请求确认。

仿真 smoke：

- `a2wpush` scripted 短步数运行。
- wheel velocity target 能推动底盘。
- Z1 IK target 能正常更新。
- 相机图像能正常渲染。

数据 smoke：

- raw episode 的 state shape 为 73。
- action shape 为 10。
- metadata 标记 `mode=a2wpush`。
- `state_feature_names` 长度为 73。
- `state_dof_names` 与实际写入 DOF 槽一致。

Policy smoke：

- convert 成 LeRobotDataset。
- 短训练 DP/ACT。
- `play --mode a2wpush` 能加载 checkpoint。
- `eval_door_policy_success --mode a2wpush` 能统计成功率。

回归：

- 旧 `ikpush` record/play/eval 不受影响。
- 旧 10D checkpoint 不会被误当作 11D freeze checkpoint。
- DP/ACT/pi0.5 后端无需因为 `a2wpush` 改模型内部结构。

## Assumptions

- 用户至少能提供 A2W URDF。
- A2W+Z1 总 URDF/YAML 不一定已存在，实施者需要先检查资产并构建最小可运行配置。
- 第一版只做仿真闭环，不接真机 SDK。
- 第一版不改变 policy action 维度，不加入 learnable freeze。
- 第一版保持 73D state，除非 DOF 检查后用户明确允许改变 schema。
- 任何 state/action 维度变化、freeze 进入 action、或真机 SDK 接入，都必须先停下来问用户。
