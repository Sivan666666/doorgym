# NX + Z1 + A2W ACT 真机部署调试纪要

Last updated: 2026-06-24 16:27 CST

这个文档是 NX 真机部署的“黑匣子记录”。以后凡是修改下面这些部署相关代码，都同步更新本文的“代码修改记录”和“当前状态”：

- `high-level/real_deploy/door_act_shadow.py`
- `high-level/real_deploy/z1_act_ee_bridge.py`
- `high-level/real_deploy/depth_inpaint.py`
- `high-level/real_deploy/run_nx_shadow.sh`
- PC2 上的 `robot_control` 节点
- 任何影响 ACT action/state、Z1 IK/LOWCMD、RealSense 输入、ROS 话题的代码

## 0. 一页速查：每次开机启动顺序

> 注意：本节含真机 SSH 账号/密码，只保存在本地部署文档里，不要提交到公开仓库。

### 0.1 SSH / IP / 密码

| 设备 | SSH | 密码/说明 | 用途 |
| --- | --- | --- | --- |
| NX | `ssh anx@192.168.1.154` | 当前未记录 NX 密码；目前按已有 SSH 配置/密钥使用 | ACT 推理、RealSense、Z1 bridge |
| NX 124 网段 | `192.168.124.25/24` | 有线网，连交换机 | 连接 Z1 和 PC2 |
| Z1 机械臂 | `192.168.124.110` | 非 SSH；由 `z1_ctrl` 通过网口连接 | Z1 SDK 控制器 |
| PC2 / 板载 i7 | `ssh unitree@192.168.124.162` | `Unitree#24226` | 运行 A2 底层 helper + `robot_control_node` |

### 0.2 每次完整启动：推荐 4 个终端

#### 终端 1：PC2 启动底层通信 helper + robot_control

```bash
ssh unitree@192.168.124.162
source ~/whole_body/install/setup.bash
ros2 launch robot_control robot_control_node.launch.py
```

这个 launch 会同时启动两部分：

1. `a2_sport_udp_helper`：负责通过 `unitree_sdk2` 和 A2 底层通信。
2. `robot_control_node`：订阅 `/cmd_vel_safe`，调用 helper 转发到底盘，并以 50 Hz 发布 `/vel_state`。

2026-06-25 注意：PC2 上 `source ~/whole_body/install/setup.bash` 后，ROS Humble 的 `LD_LIBRARY_PATH`
会优先找到 ROS 自带的 `libddsc`，而 `a2_sport_udp_helper` 需要 Unitree SDK 的
`/usr/local/lib/libddsc`。否则 helper 会在 launch 中报 `free(): invalid pointer` 后退出。
已在 PC2 的 source 与 install 两份 launch 文件里给 helper 单独加了：

```python
additional_env={
    "LD_LIBRARY_PATH": "/usr/local/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
}
```

当前 launch 默认网卡：

```text
network_interface = eth0
```

这里的 `eth0` 是 PC2 上连接 A2/狗底层通信网络的网卡，不是 NX 的网卡。如果 PC2 实际连狗的网卡不是 `eth0`，需要改 PC2 上的 launch 文件：

```bash
~/whole_body/src/robot_control/launch/robot_control_node.launch.py
```

或安装后的对应文件：

```bash
~/whole_body/install/robot_control/share/robot_control/launch/robot_control_node.launch.py
```

检查：

```bash
ros2 topic echo /vel_state
```

底盘实际运动测试：

```bash
ros2 topic pub -r 10 /cmd_vel_safe geometry_msgs/msg/Twist \
"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
```

如果 `/cmd_vel_safe` 有消息但狗不动，优先检查 `a2_sport_udp_helper` 是否已经由 launch 启动、`network_interface` 是否是 PC2 连狗的网卡。

#### 终端 2：NX 重启 z1_ctrl + z1 bridge

在本地 PC 直接执行：

```bash
ssh anx@192.168.1.154 'bash -s' <<'REMOTE'
tmux kill-session -t z1_bridge_live 2>/dev/null || true
pkill -KILL -f '^python3 .*/z1_act_ee_bridge.py' 2>/dev/null || true
sleep 1
tmux kill-session -t z1_ctrl_live 2>/dev/null || true
pkill -KILL -x z1_ctrl 2>/dev/null || true
sleep 4

: > /tmp/door_act_services/z1_ctrl.log
: > /tmp/door_act_services/z1_bridge.log
: > /tmp/door_act_services/z1_bridge.jsonl

tmux new-session -d -s z1_ctrl_live \
  'cd /home/anx/door_act_deploy/z1_controller/build && exec ./z1_ctrl >>/tmp/door_act_services/z1_ctrl.log 2>&1'
sleep 5

tmux new-session -d -s z1_bridge_live \
  'cd /home/anx/door_act_deploy/visual_whole_body && exec high-level/real_deploy/run_z1_act_ee_bridge.sh --enable_arm --max_joint_speed 3.0 --max_joint_acceleration 15.0 --joint_trajectory_duration_s 0.04 --max_gripper_speed 3.14 --max_gripper_acceleration 120.0 --state_tx_host 127.0.0.1 --state_tx_port 15013 --log_path /tmp/door_act_services/z1_bridge.jsonl >>/tmp/door_act_services/z1_bridge.log 2>&1'
REMOTE
```

#### 终端 3：检查 NX/Z1 bridge ready

```bash
ssh anx@192.168.1.154 'python3 - <<'"'"'PY'"'"'
import json, sys
r = None
for line in open("/tmp/door_act_services/z1_bridge.jsonl"):
    try:
        r = json.loads(line)
    except Exception:
        pass
print(json.dumps(r, ensure_ascii=False) if r else "NO_STATE")
if not r or r.get("startup_zero_done") is not True:
    sys.exit(2)
PY
pgrep -a -x z1_ctrl
pgrep -a -f "^python3 .*/z1_act_ee_bridge.py"'
```

ready 判断：

- `z1_ctrl` 在运行。
- `z1_act_ee_bridge.py` 在运行。
- `startup_zero_done=true`。
- `last_error=no_action` 可以接受，表示等待 ACT command。

#### 终端 4：NX 启动 ACT

```bash
ssh anx@192.168.1.154
cd /home/anx/door_act_deploy/visual_whole_body

RUN_DIR=/tmp/door_act_realsense_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

high-level/real_deploy/run_nx_shadow.sh \
  --checkpoint /home/anx/door_act_deploy/checkpoints/leroact_a2w_wc4_randomized_state10_depthonly_chunk100_exec50_bs16_0621_1742/080000/model_latest.pt \
  --camera_mode realsense \
  --depth_inpaint_mode realsense \
  --steps 1000 \
  --hz 25 \
  --device cuda:0 \
  --record_policy_depth_video_dir "$RUN_DIR/policy_depth_video" \
  --log_path "$RUN_DIR/door_act_realsense_live.jsonl"
```

ACT 默认行为：

- 推理前 warmup 2 次，不发布命令。
- 默认每个 ACT chunk 只执行前 10 个 action。
- 发布底盘 `vx/vyaw` 到 `/cmd_vel_safe`。
- 发布 Z1 EE + gripper 到 `z1_act_ee_bridge.py`。
- 保存 policy depth 视频。
- 同时保存 raw unfiltered depth 视频。
- Ctrl+C 后自动：
  - 发布底盘零速度。
  - 请求 Z1 bridge 执行 `backToStart()`。
  - bridge 回零后退出。

### 0.3 每次实验结束回传日志

把 NX 的 run 目录拉回本地归档目录：

```bash
RUN_NAME=door_act_realsense_YYYYMMDD_HHMMSS
mkdir -p /home/sivan/whole_body/visual_whole_body/high-level/real_deploy/nx_runs/"$RUN_NAME"
rsync -avz anx@192.168.1.154:/tmp/"$RUN_NAME"/ \
  /home/sivan/whole_body/visual_whole_body/high-level/real_deploy/nx_runs/"$RUN_NAME"/
```

当前统一本地保存目录：

```bash
/home/sivan/whole_body/visual_whole_body/high-level/real_deploy/nx_runs/
```

## 1. 当前机器和网络约定

- NX：`anx@192.168.1.154`
- NX 密码：未记录；目前使用已有 SSH 配置/密钥
- NX 124 网段有线地址：`192.168.124.25/24`
- Z1 机械臂：`192.168.124.110`
- PC2 / 板载 i7：`unitree@192.168.124.162`
- PC2 密码：`Unitree#24226`
- Z1 bridge UDP action：`127.0.0.1:15011`
- Z1 bridge UDP state：`127.0.0.1:15013`
- ROS2 base command topic：`/cmd_vel_safe`
- ROS2 base state topic：`/vel_state`

## 2. 当前本地归档目录

真机 run、日志、视频统一保存到：

```bash
high-level/real_deploy/nx_runs/
```

最新一次用户手动 ACT run：

```bash
high-level/real_deploy/nx_runs/door_act_realsense_20260623_231708/
```

其中包含：

- `door_act_realsense_live.jsonl`
- `policy_depth_video/policy_depth_u8_side_by_side.mp4`
- `raw_unfiltered_depth_video/raw_unfiltered_depth_u8_side_by_side.mp4`
- `gripper_trace.csv`
- `gripper_trace.png`

这次 run 的视频确认：

- policy depth video：`616` 帧，`25 fps`，`1280x480`，`24.64 s`
- raw unfiltered depth video：`616` 帧，`25 fps`，`1280x480`，`24.64 s`

## 3. ACT 推理启动命令

在用户 PC 上：

```bash
ssh anx@192.168.1.154
cd /home/anx/door_act_deploy/visual_whole_body

RUN_DIR=/tmp/door_act_realsense_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

high-level/real_deploy/run_nx_shadow.sh \
  --checkpoint /home/anx/door_act_deploy/checkpoints/leroact_a2w_wc4_randomized_state10_depthonly_chunk100_exec50_bs16_0621_1742/080000/model_latest.pt \
  --camera_mode realsense \
  --depth_inpaint_mode realsense \
  --steps 1000 \
  --hz 25 \
  --device cuda:0 \
  --record_policy_depth_video_dir "$RUN_DIR/policy_depth_video" \
  --log_path "$RUN_DIR/door_act_realsense_live.jsonl"
```

说明：

- 设置 `--record_policy_depth_video_dir` 后，会保存送进 policy 前的 `policy_depth_u8` 视频。
- 当前代码默认同时保存一份原始未滤波深度图视频到：

```bash
$RUN_DIR/raw_unfiltered_depth_video/raw_unfiltered_depth_u8_side_by_side.mp4
```

- Ctrl+C 后，ACT 会先发底盘零速度，然后通知 Z1 bridge 执行 `backToStart()` 回零。
- Z1 bridge 在收到 shutdown 回零后会退出；下一次 ACT 前需要重新启动 bridge。

## 4. 后台服务启动和检查

### 4.1 重启 z1_ctrl + z1 bridge

```bash
ssh anx@192.168.1.154 'bash -s' <<'REMOTE'
tmux kill-session -t z1_bridge_live 2>/dev/null || true
pkill -KILL -f '^python3 .*/z1_act_ee_bridge.py' 2>/dev/null || true
sleep 1
tmux kill-session -t z1_ctrl_live 2>/dev/null || true
pkill -KILL -x z1_ctrl 2>/dev/null || true
sleep 4

: > /tmp/door_act_services/z1_ctrl.log
: > /tmp/door_act_services/z1_bridge.log
: > /tmp/door_act_services/z1_bridge.jsonl

tmux new-session -d -s z1_ctrl_live \
  'cd /home/anx/door_act_deploy/z1_controller/build && exec ./z1_ctrl >>/tmp/door_act_services/z1_ctrl.log 2>&1'
sleep 5

tmux new-session -d -s z1_bridge_live \
  'cd /home/anx/door_act_deploy/visual_whole_body && exec high-level/real_deploy/run_z1_act_ee_bridge.sh --enable_arm --max_joint_speed 3.0 --max_joint_acceleration 15.0 --joint_trajectory_duration_s 0.04 --max_gripper_speed 3.14 --max_gripper_acceleration 120.0 --state_tx_host 127.0.0.1 --state_tx_port 15013 --log_path /tmp/door_act_services/z1_bridge.jsonl >>/tmp/door_act_services/z1_bridge.log 2>&1'
REMOTE
```

### 4.2 检查 Z1 是否 ready

```bash
ssh anx@192.168.1.154 'python3 - <<'"'"'PY'"'"'
import json, sys
r = None
for line in open("/tmp/door_act_services/z1_bridge.jsonl"):
    try:
        r = json.loads(line)
    except Exception:
        pass
print(json.dumps(r, ensure_ascii=False) if r else "NO_STATE")
if not r or r.get("startup_zero_done") is not True:
    sys.exit(2)
PY
pgrep -a -x z1_ctrl
pgrep -a -f "^python3 .*/z1_act_ee_bridge.py"'
```

ready 条件：

- `z1_ctrl` 进程存在
- `z1_act_ee_bridge.py` 进程存在
- `startup_zero_done=true`
- `last_error` 允许是 `no_action`，这表示 bridge ready 但还没有 ACT command

## 5. PC2 底盘通信和 robot_control

PC2 上不要只运行 `ros2 run robot_control robot_control_node`。单独启动 node 只能收 ROS 话题和发布 `/vel_state`，但不会自动启动与 A2 底层通信的 helper，狗可能不会动。

推荐始终运行 launch：

```bash
ssh unitree@192.168.124.162
source ~/whole_body/install/setup.bash
ros2 launch robot_control robot_control_node.launch.py
```

该 launch 当前包含：

```text
a2_sport_udp_helper --network_interface eth0 --bind_host 127.0.0.1 --port 15021
robot_control_node:
  cmd_vel_safe_topic=/cmd_vel_safe
  vel_state_topic=/vel_state
  vel_state_publish_hz=50.0
  vel_state_timeout_sec=0.0
  enable_robot_move=True
  network_interface=eth0
  a2_helper_host=127.0.0.1
  a2_helper_port=15021
```

PC2 launch 的 helper 进程需要单独使用 `/usr/local/lib` 优先的 `LD_LIBRARY_PATH`，
否则在 ROS 环境中可能误加载 ROS 的 `libddsc`，表现为：

```text
a2_sport_udp_helper starting network_interface=eth0 bind=127.0.0.1:15021
free(): invalid pointer
```

当前已修复到：

```bash
~/whole_body/robot_control/launch/robot_control_node.launch.py
~/whole_body/install/robot_control/share/robot_control/launch/robot_control_node.launch.py
```

`network_interface=eth0` 的含义：

- 这是 PC2 上连 A2/狗底层通信的网卡。
- 不是 NX 的网卡。
- 如果 PC2 上 A2 网卡名字不是 `eth0`，狗不会动，需要改 launch 文件里的 `eth0`。

接口约定：

- NX 发布 `/cmd_vel_safe`，类型 `geometry_msgs/msg/Twist`
- `action[0]` 对应 `linear.x`
- `action[1]` 对应 `angular.z`
- PC2 发布 `/vel_state`
- `/vel_state` 发布的是上一个时刻收到的 command：`last_vx` 和 `last_vyaw`
- ACT state 前两维对齐录制脚本：`[last command vx, last command vyaw]`

## 5.1 ACT Ctrl-C 与 Z1 bridge 行为

2026-06-25 已回滚连续测试实验改动：

- 不再使用 `startup_open_gripper` bridge 命令。
- 不再让 bridge 在 Ctrl-C/backToStart 后继续留在 LOWCMD 等下一轮 ACT。
- 不再在 Ctrl-C 后额外 pulse 夹爪闭合目标。
- 当前恢复为之前的保守流程：ACT 收到 Ctrl-C 时发送 `shutdown_back_to_start`；bridge 收到后跳出控制循环，并在退出 `finally` 中调用 Z1 SDK `backToStart()` / `passive`。
- bridge 启动回零阶段：`startup_gripper_open_once=True`，参考 Z1 SDK `example_lowcmd.py` 的 `duration=1000` 方式，从当前夹爪反馈插值到默认张开目标 `-pi/2`，共 `startup_gripper_open_duration_steps=1000` 个 LOWCMD 周期；完成后不再发送夹爪当前位置。

如果 ACT 报：

```text
Timed out waiting for Z1 startup zero before ACT inference; last_meta count=0
```

说明 ACT 没收到 bridge 的 15013 状态，优先检查：

```bash
ssh anx@192.168.1.154
pgrep -af 'z1_ctrl|z1_act_ee_bridge'
ss -lunp | grep -E '15011|15012|15013'
tail -n 100 /tmp/door_act_services/z1_bridge.log
```

## 6. RealSense 深度图当前约定

- 双 D435，异步进程读取。
- depth aligned to color：`rs.align(rs.stream.color)`
- 目前不再 crop：`left=0, right=0, top=0, bottom=0`
- clip：`0.2–1.5 m`
- policy 输入：`0.2–1.5 m -> uint8 0–255`
- 当前默认 `--depth_inpaint_mode realsense`
- `realsense` 模式使用 RealSense spatial/temporal/hole filling 相关滤波。
- 新增 raw 录制：aligned 后、RealSense filter 前的原始 depth，同样按项目显示规则转成 uint8 视频，仅用于 debug，不送入 policy。

相机区分：

- wrist serial：`261222075130`
- front serial：`261222075566`

## 7. 仿真 / 真机 action-state 对齐

录制脚本基准：

```bash
high-level/dp/record/record_door_dp_dataset_a2w_state10.py
```

action 10 维约定：

```text
[vx, vyaw, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]
```

state 10 维约定：

```text
[last_vx, last_vyaw, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]
```

gripper 约定：

- 张开约 `-pi/2 = -1.5708`
- 闭合约 `0`
- 当前 bridge 不反号：`map_act_gripper_to_z1(value) = value * scale + offset`
- 当前默认：`scale=1.0, offset=0.0`

### 7.1 RealSense 和仿真视觉对齐

当前真机深度输入：

- 两个 D435：front + wrist。
- `640x480 @ 30 Hz`。
- `depth -> color` 对齐：`rs.align(rs.stream.color)`。
- align 后不 crop。
- 深度 clip：`0.2–1.5 m`。
- policy 前归一化：`0.2–1.5 m -> uint8 0–255`。
- 默认滤波模式：`realsense`。
- raw debug 视频保存的是 align 后、滤波前的原始 depth。

当前仿真深度相机对齐：

- 仿真相机分辨率按训练/部署一致：`640x480`。
- 仿真水平 FOV 已按 D435 aligned color 近似改为 `55 deg`。
- 因为画幅同为 `4:3`，水平 FOV 对齐后垂直 FOV 基本一致。
- 仿真和真机都使用相同 depth clip / uint8 显示规则。

如果未来改相机：

1. 先在 NX 上读取 aligned depth intrinsics。
2. 用 `fov_x = 2 * atan(width / (2 * fx))` 算真实水平 FOV。
3. 再修改 IsaacGym 相机水平 FOV。
4. 保持 crop、clip、resize、归一化全部一致。

### 7.2 Z1 EE 坐标系和仿真对齐

仿真训练使用的是 `ee_gripper_link`。

真机 Z1 SDK 的 `forwardKinematics(q, 6)` 末端点不是夹爪实际 `ee_gripper_link`，零位时 x 方向差约 `8.6 cm`。

当前 bridge 对齐方式：

- 在 SDK EE frame 到 ACT/sim EE frame 之间加 tool offset。
- 默认 offset：local x `+0.086 m`。
- offset 会随着当前末端姿态旋转后再加到 base/arm frame，不是简单固定加 base x。
- IK 时会把 ACT/sim 的 `ee_gripper_link` target 转回 SDK EE target。
- FK/state 上报时会把 SDK EE 加 tool offset 后作为 ACT state 的 EE。

回零/home 对齐：

- Z1 真机不再强行认为六关节全零是 home。
- `backToStart()` 后的实际关节角作为 `home_q`。
- 仿真如果要严格比较，应比较同一 EE target 下 IK 解/反馈，而不是强行比较 `q=0`。

## 8. Z1 bridge 当前控制策略

核心文件：

```bash
high-level/real_deploy/z1_act_ee_bridge.py
```

当前策略：

- `backToStart()` 作为启动 home。
- 不再发送六关节全零作为回零。
- `backToStart()` 完成后的实际关节角记录为 `home_q`。
- 启动检查只看关节反馈有限、速度足够小、稳定约 `0.3 s`，不要求 `home_q == 0`。
- 启动时夹爪参考 Z1 SDK `example_lowcmd.py`，用 `duration=1000` 个 LOWCMD 周期从当前反馈插值到最大张开 `-pi/2`，完成后不再发夹爪当前位置。
- ACT EE action 进入 bridge 后：
  - 精确 6D IK 优先。
  - 失败后使用 soft IK，优先位置，姿态低权重。
  - seed 使用当前轨迹状态 `q_cmd`，避免用遥远 `q_goal`。
  - 限制关节跳变、速度、加速度。
  - 若 IK 失败，继续当前轨迹并平滑减速。
  - 回程如果是 home，直接采用关节空间目标，不重新做 EE IK。
- 500 Hz LOWCMD 输出连续 `q_cmd / qd_cmd / qdd_cmd`。
- 每个 ACT 25 Hz EE waypoint 对应一个约 `40 ms` 的关节空间 quintic 轨迹段。
- 真机当前默认：
  - `--max_joint_speed 3.0`
  - `--max_joint_acceleration 15.0`
  - `--joint_trajectory_duration_s 0.04`
  - `--max_gripper_speed 3.14`
  - `--max_gripper_acceleration 120.0`

## 9. 已解决的重要问题

### 9.1 机械臂保险丝烧掉后的安全修改

曾经危险点：

- 直接发远 EE / 远 joint target。
- 回零曾经尝试发六关节全零，和 Z1 SDK 标定后的真实 startFlat 不一致。
- 大位置差配合较硬控制可能导致瞬时力矩过大。

当前修复：

- 回零只用 `backToStart()`。
- 启动 home 用 SDK 返回后的实际 `home_q`。
- ACT 退出时自动请求 bridge `shutdown_back_to_start`。
- bridge 收到 shutdown 后执行 `backToStart()`，然后退出。
- 关节轨迹采用连续 quintic，避免每 40 ms 路点终点速度强制归零导致抖动。

### 9.2 仿真 EE 和 Z1 SDK EE 对齐

发现：

- Z1 SDK `forwardKinematics(q, 6)` 的末端不等于仿真 `ee_gripper_link`。
- 零位 x 方向约差 `8.6 cm`。

当前处理：

- 增加 tool offset，默认 local x `+0.086 m`。
- offset 按当前末端姿态旋转后应用，不是在 base x 固定加。

### 9.3 ACT 推理 chunk

当前：

- 默认只执行每个 chunk 前 10 个 action。
- warmup：启动时用当前观测推理 2 次，但不发布命令。
- 异步推理和执行重叠。
- 新 chunk 完成后立即融合旧轨迹剩余项：
  - `vx / vyaw / xyz`：`0.3 old + 0.7 new`
  - quaternion：shortest-path SLERP
  - gripper：直接用最新预测，不融合
- 根据推理完成时实际 step 跳过过期 action。
- step 对齐：step `k` 观测来自执行完 step `k-1` 后的状态；新 chunk 的 action 0 对齐 ingest 时刻的执行 step。

### 9.4 深度图录制

当前 ACT 运行可同时录制：

- policy 输入深度：`policy_depth_u8_side_by_side.mp4`
- raw 未滤波深度：`raw_unfiltered_depth_u8_side_by_side.mp4`

录制是异步写入，不应阻塞 policy 主循环。最近一次 `231708` run 显示：

- policy video：`enqueued=616, written=616, dropped=0`
- raw video：`enqueued=616, written=616, dropped=0`

## 10. 最新 run：20260623_231708

路径：

```bash
high-level/real_deploy/nx_runs/door_act_realsense_20260623_231708/
```

ACT 结束输出：

```text
z1_shutdown_back_to_start_requested {"host": "127.0.0.1", "port": 15011, "bridge_command": "shutdown_back_to_start", "sent": 3}
policy_depth_video_done {"enabled": true, "dir": "/tmp/door_act_realsense_20260623_231708/policy_depth_video", "fps": 25.0, "enqueued": 616, "written": 616, "dropped": 0, "error": null, "thread_alive": false}
raw_unfiltered_depth_video_done {"enabled": true, "dir": "/tmp/door_act_realsense_20260623_231708/raw_unfiltered_depth_video", "fps": 25.0, "enqueued": 616, "written": 616, "dropped": 0, "error": null, "thread_alive": false}
z1_shutdown_back_to_start_done done=True error=
shadow_done log=/tmp/door_act_realsense_20260623_231708/door_act_realsense_live.jsonl steps=616 elapsed_s=30.375 steady_hz=23.348 startup_inclusive_hz=20.280 interrupted=true
```

### 10.1 夹爪分析

本次发现：真机夹爪看起来没有明显闭合。

日志结论：

- 不是 action/cmd 没有闭合指令。
- `action[9]` 和 `z1_action[9]` 一致，说明 ACT 输出已经发到 bridge。
- `action[9]` 范围：`-1.572 -> -0.320`
- 最闭合目标出现在 step `261`：target `-0.319824`
- 但反馈 actual 只有约 `-0.714328`
- step `500`：target `-0.540527`，actual 约 `-1.212299`

统计：

```text
action9 min -1.572266
action9 max -0.319824
action9 mean -1.129448
action9 p50 -1.150391
actual gripper min -1.566165
actual gripper max -0.020254
actual gripper mean -0.987073
```

ACT gripper 输出问题：

- gripper target 快速抖动，不是稳定的开/合阶段。
- 单步最大跳变约 `0.955 rad`
- 方向切换约 `175` 次

当前判断：

- 接口正负号大概率正确。
- bridge 确实收到了闭合方向 target。
- 真机夹爪没有明显闭合，主要是 target 抖动且持续时间不足，叠加 bridge 对夹爪做了平滑限速/限加速度，导致反馈没有跟上。

#### 10.1.1 gripper “正值”问题

物理 gripper action 的约定仍然是：

- `-pi/2 ≈ -1.5708`：张开
- `0`：闭合

训练 raw 数据中 gripper 物理标签全是负值：

```text
a2w_state10_wc4_randomized_100 action[9]:
  n=50000
  positive=0, zero=0, negative=50000
  min=-1.570796, max=-0.314159

a2w_state10_wc4_randomized_100 state[9]:
  n=50000
  positive=0, zero=0, negative=50000
  min=-1.503018, max=-0.314333
```

NX 已归档 run 里的物理 ACT `action[9]` 也全是负值：

```text
all nx_runs action[9]:
  runs=4
  steps=3763
  positive=0, zero=0, negative=3763
  min=-1.585938, max=-0.319824
```

如果看到 gripper 这一维出现正值，通常不是物理 gripper 角度，而是 LeRobot/ACT 内部归一化后的 action 值。当前 checkpoint 使用：

```text
normalization_mapping.ACTION = MEAN_STD
gripper action mean = -1.2277608
gripper action std  =  0.4652333
normalized = (physical - mean) / std
```

因此：

```text
physical -1.5708  -> normalized -0.7373   # 张开
physical -1.2278  -> normalized  0.0000   # 数据均值
physical -0.3142  -> normalized +1.9637   # 训练里较闭合
physical  0.0000  -> normalized +2.6390   # 完全闭合，但训练标签没到 0
```

所以 normalized gripper 为正，只代表“比数据均值更闭合”，并不代表物理角度为正。

最新 run `20260623_231708`：

```text
physical action[9]:
  positive=0, zero=0, negative=616
  min=-1.572266, max=-0.319824

normalized action[9] estimated from stats:
  positive=332, zero=0, negative=284
  min=-0.7405, max=+1.9516
```

这解释了为什么可能看到“ACT gripper 输出正值”：如果看的是 normalized/raw policy output，它可以为正；但真正发给 Z1 bridge 的物理 action 仍然是负的。

#### 10.1.2 SDK 原生 gripper 限制 vs bridge 外层限制

Z1 SDK 低层接口：

```cpp
void setGripperCmd(double gripperPos, double gripperW, double gripperTau = 0.)
{
    lowcmd->setGripperQ(gripperPos);
    lowcmd->setGripperQd(gripperW);
    lowcmd->setGripperTau(gripperTau);
}
```

这只是把夹爪位置、速度、力矩写进 lowcmd；低层接口本身没有显式加速度限制参数。

SDK / z1_controller 中能看到的原生 gripper 限制主要是：

```text
gripper position range: [-pi/2, 0]
Unitree_Gripper::MAX_SPEED = pi rad/s
config.xml grasp_max_torque = 10.0 Nm
lowcmd default gripper gain: kp=20, kd=2000
```

高层 `MoveJ/MoveL/MoveC` 轨迹接口带 gripper 时，也只暴露 gripper target 和轨迹 speed；trajectory header 里 `setGripper(..., speed=M_PI)`，没有看到类似 `max acceleration` 的公开参数。

所以当前：

- `max_gripper_speed=3.14 rad/s`：是 bridge 自己设的速度上限，接近 SDK `pi rad/s`。
- `max_gripper_acceleration=120 rad/s^2`：是 bridge 自己额外加的外层平滑限制，不是 SDK 原生接口要求。

如果要让夹爪再更稳，核心不再是单纯放宽速度/加速度，而是 ACT gripper target 抖动，建议做 gripper latch/hysteresis。

建议下一步：

- 不要逐帧直接信任 gripper 连续回归值。
- 增加夹爪 latch/hysteresis：
  - 例如 `action[9] > -0.8` 判定为闭合意图，latched close。
  - 保持闭合至少一段时间，例如 `0.3–0.5 s`。
  - push/release 阶段或 `action[9] < -1.3` 持续若干帧后再张开。
- 或者将策略 gripper 离散化成 open/close 两态后再发 Z1。

## 11. 夹爪 direct replay 实验

### 11.1 Direct gripper replay：20260624_0022

路径：

```bash
high-level/real_deploy/nx_runs/gripper_replay_20260624_0022/
```

目的：

- 只测夹爪，不走 `z1_act_ee_bridge` 的 EE IK。
- 停止 `z1_bridge`，只保留 `z1_ctrl`。
- 直接用 Z1 SDK LOWCMD：
  - 读取当前 6 关节角作为 `q_hold`。
  - 全程发送 `setArmCmd(q_hold, 0, tau_hold)` 锁住关节。
  - 重放上一次 ACT run 的 `action[9]`。
  - 使用新的夹爪平滑参数：`max_gripper_speed=3.0 rad/s`，`max_gripper_acceleration=80 rad/s^2`。

执行脚本：

```bash
high-level/real_deploy/direct_gripper_replay_lowcmd.py
```

本次 replay 输入：

```bash
/tmp/z1_gripper_replay_20260624_0022/source_door_act_realsense_live.jsonl
```

本地回传文件：

```bash
high-level/real_deploy/nx_runs/gripper_replay_20260624_0022/direct_gripper_replay_3p0_80.jsonl
high-level/real_deploy/nx_runs/gripper_replay_20260624_0022/direct_gripper_replay_3p0_80_trace.png
```

结果：

```text
steps: 616
control_count: 12801
target range: -1.572266 -> -0.319824
cmd range:    -1.570796 -> -0.000320
actual range: -1.570130 -> +0.007841

target-vs-actual abs error:
  mean 0.2127 rad
  p50  0.1345 rad
  p95  0.6832 rad
  max  1.5801 rad

smoothed-cmd-vs-actual abs error:
  mean 0.1108 rad
  p50  0.0701 rad
  p95  0.3449 rad
  max  0.6094 rad

joint drift while holding q_hold:
  max single-joint drift: 0.0258 rad
  per-joint abs max:
    [0.000041, 0.02578, 0.00347, 0.000949, 0.000029, 0.000053]
```

与旧 ACT full run `20260623_231708` 对比：

```text
old full ACT target-vs-actual gripper abs error:
  mean 0.4185 rad
  p50  0.3795 rad
  p95  0.8670 rad
  max  1.5520 rad

new direct replay 3.0/80 target-vs-actual gripper abs error:
  mean 0.2127 rad
  p50  0.1345 rad
  p95  0.6832 rad
  max  1.5801 rad
```

结论：

- 放宽夹爪速度/加速度后，夹爪跟踪明显改善。
- 这次 direct 测试基本锁住了机械臂关节，最大关节漂移约 `0.026 rad`，不是 EE IK 导致的大幅运动。
- 但 `target-vs-actual` 的 p95 仍有 `0.68 rad`，说明 ACT gripper target 抖动/频繁开合切换仍然是主要瓶颈。
- `cmd-vs-actual` p50 约 `0.07 rad`，说明真机对平滑后的 gripper command 跟踪比对原始 ACT target 好很多。
- 后续仍建议给 gripper 加 latch/hysteresis，而不是逐帧追 ACT 连续值。

### 11.2 Direct gripper replay：20260624_004502

路径：

```bash
high-level/real_deploy/nx_runs/gripper_replay_20260624_004502/
```

目的：

- 继续只测夹爪，不走 `z1_act_ee_bridge` 的 EE IK。
- 参数改为：
  - `max_gripper_speed=3.14 rad/s`
  - `max_gripper_acceleration=120 rad/s^2`
- 仍然使用 `direct_gripper_replay_lowcmd.py`：
  - 捕获当前 6 关节角作为 `q_hold`。
  - 全程发送 `setArmCmd(q_hold, 0, tau_hold)` 锁关节。
  - 只重放上一次 ACT run 的 `action[9]`。

备注：

- 第一次尝试前 `z1_ctrl` 日志出现 `[ERROR] Motor 2 windings overheat`，当时已停止 replay 和 `z1_ctrl`。
- 机械臂重启后重新测试，本次未再触发 overheat。

本地回传文件：

```bash
high-level/real_deploy/nx_runs/gripper_replay_20260624_004502/direct_gripper_replay_3p14_120.jsonl
high-level/real_deploy/nx_runs/gripper_replay_20260624_004502/direct_gripper_replay_3p14_120.stdout.log
high-level/real_deploy/nx_runs/gripper_replay_20260624_004502/direct_gripper_replay_3p14_120_trace.png
```

结果：

```text
steps: 616
control_count: 12804
target range: -1.572266 -> -0.319824
cmd range:    -1.570796 -> -0.000480
actual range: -1.569345 -> +0.009428

target-vs-actual abs error:
  mean 0.1476 rad
  p50  0.0415 rad
  p95  0.6115 rad
  max  1.5817 rad

smoothed-cmd-vs-actual abs error:
  mean 0.0183 rad
  p50  0.0069 rad
  p95  0.0534 rad
  max  0.0582 rad

joint drift while holding q_hold:
  max single-joint drift: 0.00855 rad
```

与 `3.0 / 80` 对比：

```text
3.0 / 80:
  target-vs-actual p50/p95: 0.1437 / 0.6802 rad
  cmd-vs-actual    p50/p95: 0.0730 / 0.5370 rad
  max joint drift:          0.0258 rad

3.14 / 120:
  target-vs-actual p50/p95: 0.0415 / 0.6115 rad
  cmd-vs-actual    p50/p95: 0.0069 / 0.0534 rad
  max joint drift:          0.00855 rad
```

结论：

- `3.14 / 120` 对夹爪跟踪改善非常明显，尤其是平滑后的 `cmd` 几乎能被真实夹爪跟上。
- ACT 原始 gripper target 仍然有阶段抖动，所以 `target-vs-actual` 的 max 仍然可能很大；这不是低层跟不上，而是上层 target 本身突变。
- 当前更推荐保留 `3.14 / 120`，然后再加 gripper latch/hysteresis，避免策略 gripper 连续值逐帧抖动。

## 12. 已知问题 / 下一步

### 12.1 夹爪跟踪

当前最需要解决。

候选方案：

1. gripper latch/hysteresis。
2. 对 gripper 独立低通或中值滤波。
3. 根据 scripted 阶段或 EE 位置启发式辅助 close/open。
4. 检查 Z1 `setGripperCmd(q, qd, tau)` 是否需要更长持续 command 或不同 `tau`。

### 12.2 ACT 频率

最近 run：

- `steady_hz=23.348`
- 低于目标 `25 Hz`

可能原因：

- ACT forward 仍有周期性峰值。
- camera / video / ROS / Z1 状态读写存在抖动。
- Jetson NX 上 CUDA 推理峰值需要继续 profile。

### 12.3 机械臂 tracking

日志中 `arm_tracking` 已记录：

- feedback vs previous command
- position error
- orientation error
- gripper error
- IK source
- IK fail count

后续每次真实执行都应看 `arm_tracking_summary`。

### 12.4 D435 raw bag 录制与离线滤波 sweep

目标：

- 录一组真正可复处理的 RealSense 原始深度数据，而不是仅保存 u8 可视化视频。
- 录制时让 Z1 按 scripted A2W 轨迹运动，制造真实运动边缘/拖影场景。
- 离线重放 `.bag`，对比 RealSense filter chain：
  - A. no filters
  - B. spatial only
  - C. spatial + temporal
  - D. spatial + hole filling
  - E. spatial + hole filling + temporal

录制脚本：

```bash
high-level/real_deploy/record_dual_d435_raw_bag_with_scripted_z1.py
```

离线滤波脚本：

```bash
high-level/real_deploy/offline_realsense_filter_sweep.py
```

推荐 NX 录制命令：

```bash
ssh anx@192.168.1.154
cd /home/anx/door_act_deploy/visual_whole_body

RUN_DIR=/tmp/d435_raw_scripted_$(date +%Y%m%d_%H%M%S)

python3 high-level/real_deploy/record_dual_d435_raw_bag_with_scripted_z1.py \
  --out_dir "$RUN_DIR" \
  --reference_action_npz /tmp/a2w_full_reference.npz \
  --pre_record_s 2.0 \
  --post_record_s 2.0 \
  --trajectory_hz 25 \
  --send_home_joint_targets_from_step 364 \
  --send_shutdown_back_to_start
```

输出：

```text
$RUN_DIR/wrist_raw.bag
$RUN_DIR/front_raw.bag
$RUN_DIR/record_manifest.json
```

注意：

- 这个脚本只发 Z1 action UDP，不发布狗的底盘速度。
- 需要 `z1_ctrl` 和 `z1_act_ee_bridge.py` 已经运行。
- `action[0:2]` 会被强制置零，避免底盘运动。
- 默认 reference 支持 `/tmp/a2w_full_reference.npz` 这种格式：
  - `action: [T, 10]`
  - 可选 `sim_q: [T, 6]`

推荐离线滤波命令：

```bash
python3 high-level/real_deploy/offline_realsense_filter_sweep.py \
  --wrist_bag "$RUN_DIR/wrist_raw.bag" \
  --front_bag "$RUN_DIR/front_raw.bag" \
  --out_dir "$RUN_DIR/filter_sweep" \
  --align_to_color \
  --depth_lower_m 0.2 \
  --depth_far_m 1.5 \
  --output_width 640 \
  --output_height 480 \
  --output_fps 30 \
  --snapshot_steps 0,30,60,120,240,360,480
```

输出：

```text
$RUN_DIR/filter_sweep/videos/A_no_filters_side_by_side.mp4
$RUN_DIR/filter_sweep/videos/B_spatial_only_side_by_side.mp4
$RUN_DIR/filter_sweep/videos/C_spatial_temporal_side_by_side.mp4
$RUN_DIR/filter_sweep/videos/D_spatial_hole_filling_side_by_side.mp4
$RUN_DIR/filter_sweep/videos/E_spatial_hole_filling_temporal_side_by_side.mp4
$RUN_DIR/filter_sweep/snapshots/filter_sweep_step_*.png
$RUN_DIR/filter_sweep/filter_sweep_manifest.json
```

判断拖影建议：

- `A/B/D` 基本无拖影，但 `C/E` 出现拖影：主要是 `temporal_filter`。
- `B/D` 也明显边缘糊或跨物体扩散：主要是 `spatial_filter` 或 hole filling。
- `D/E` 出现块状假深度：主要是 `hole_filling_filter` 或 spatial `holes_fill`。

#### 12.4.1 已录制 run：20260624_154141

NX 原始路径：

```bash
/tmp/d435_raw_scripted_20260624_154141
```

U 盘路径：

```bash
/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141
/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141.tar.zst
/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141_light.tar.zst
```

本地已回传轻量结果：

```bash
high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/
```

录制内容：

```text
wrist_raw.bag: 725M, 494 frames
front_raw.bag: 1.5G, 1008 frames
full directory: 2.3G
compressed tar.zst: 771M
```

离线 filter sweep：

```text
A_no_filters_side_by_side.mp4: 490 frames
B_spatial_only_side_by_side.mp4: 490 frames
C_spatial_temporal_side_by_side.mp4: 490 frames
D_spatial_hole_filling_side_by_side.mp4: 490 frames
E_spatial_hole_filling_temporal_side_by_side.mp4: 490 frames
snapshots: 7 PNG grids
```

备注：

- 本次网络 SSH 只有约 `0.02 MB/s`，所以大数据不再通过 SSH 回传。
- 完整数据已拷贝到 U 盘 label `Tang`。
- 本地只保留轻量日志和 snapshots，方便快速查看。

#### 12.4.2 Selective hole fill 测试：20260624_1624

新增离线模式：

```text
F_spatial_temporal_selective_fill
```

处理链：

```text
align(color)
spatial_filter(holes_fill=0)
temporal_filter
selective hole fill
clip 0.2–1.5m
resize / normalize
```

目标：

- 补平面/物体表面内部的小黑洞。
- 不像 RealSense `hole_filling_filter()` 那样把夹爪大块 invalid 区域补成背景。
- 要求夹爪黑色区域相对 `A_no_filters` 不要消失太多。

选择性补洞规则：

- 只处理 `depth == 0` 或非有限值的连通域。
- 大面积洞不补。
- 跨度太大的洞不补。
- 接触图像边界的洞不补。
- 周围 ring 有效深度不足不补。
- 周围深度方差/范围太大不补。
- 较近的大洞受 `protect_near_depth_m` 保护，不补。

当前参数：

```text
max_component_area: 1200 px
max_component_span_px: 90 px
ring_radius_px: 5 px
min_valid_ring_ratio: 0.35
min_valid_ring_px: 12
depth_range_thresh_m: 0.14
depth_std_thresh_m: 0.07
large_component_area: 32 px
border_margin_px: 1
protect_near_depth_m: 0.35
protect_near_area: 80 px
```

NX 输出：

```bash
/tmp/d435_raw_scripted_20260624_154141/filter_sweep_selective
```

U 盘输出：

```bash
/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141/filter_sweep_selective
/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141/offline_realsense_filter_sweep_with_selective.py
```

结果：

```text
A_no_filters: 490 frames
F_spatial_temporal_selective_fill: 490 frames
F filled_ratio_of_invalid: 0.01946
```

解释：

- F 只补了约 `1.95%` 的 invalid 像素，明显比 RealSense hole filling 保守。
- 这种比例符合“补小洞，但不大幅吞掉夹爪黑色区域”的方向。
- 离线全量 490 帧耗时约 `292 s`，当前 Python 连通域版本适合离线验证；若要实时部署，需要改成更快的 GPU/ROI 实现或降低处理区域。

## 13. 代码修改记录

### 2026-06-23

#### `door_act_shadow.py`

- 增加 raw unfiltered depth video recorder。
- `--record_policy_depth_video_dir` 存在时，默认同时保存 raw unfiltered depth video。
- Ctrl+C 时：
  - 发布底盘零速度。
  - 发送 Z1 bridge `shutdown_back_to_start`。
  - 等待 bridge 完成 `backToStart()`。
  - 清理相机和视频线程。
- 增加 arm tracking 日志：
  - `position_error_mm`
  - `orientation_error_deg`
  - `gripper_abs_error_rad`
  - `ik_source`
  - `ik_fail_count`
- ACT chunk 改为异步 prefetch + overlap blending。
- 默认执行 chunk 前 `10` 个 action。
- 增加 warmup 两次 ACT forward，不发布控制。

#### `z1_act_ee_bridge.py`

- 回零只使用 `backToStart()`。
- 删除启动最后发送六关节全零行为。
- 以 `backToStart()` 后实际关节角作为 `home_q`。
- 启动完成判据改为反馈有限、速度低、稳定保持。
- 增加 shutdown command：
  - UDP 收到 `{"bridge_command": "shutdown_back_to_start"}`
  - 执行 `backToStart()`
  - 设置 passive
  - 退出 bridge
- 增加 tool offset：local x `+0.086 m`，随当前 EE 姿态旋转。
- IK fallback：
  - 精确 6D IK
  - soft IK，位置优先，姿态低权重
  - 失败时继续当前轨迹并平滑减速
- 关节控制改为 500 Hz 连续 quintic 轨迹输出。
- 夹爪：
  - 启动最大张开改为 `duration=1000` LOWCMD 插值到 `-pi/2`。
  - ACT 阶段 gripper target 来自 action[9]。
  - 当前限速 `3.14 rad/s`，限加速度 `120 rad/s^2`。
  - 增加 close latch：
    - 默认开启：`--gripper_close_latch`
    - 阈值：`--gripper_close_latch_threshold -0.5`
    - 确认步数：`--gripper_close_latch_confirm_steps 3`
    - 闭合目标：`--gripper_close_latch_target 0.1`
    - 逻辑：mapped gripper target 连续 3 个 ACT action packet `>= -0.5 rad` 后，强制发送 `0.1 rad` 闭合。
    - 无 release hysteresis：后续任意新 action packet `< -0.5 rad` 会立即退出 force-closed，继续跟随 ACT 原始夹爪目标。
    - latch 计数按新 ACT action packet 计，不按 500Hz LOWCMD 周期计。

#### `depth_inpaint.py`

- 增加 `rgb_guided` 实时近似补全模式。
- 保留 `off / realsense / rgb_guided` 三种模式。
- 默认仍为 `realsense`，避免改变 checkpoint 输入分布。

### 2026-06-24

#### `z1_act_ee_bridge.py`

- 放宽夹爪外层平滑限制：
  - `--max_gripper_speed` 默认从 `2.5 rad/s` 改为 `3.0 rad/s`。
  - `--max_gripper_acceleration` 默认从 `20 rad/s^2` 改为 `80 rad/s^2`。
- 同步更新本文启动命令，显式传入 `--max_gripper_speed 3.0 --max_gripper_acceleration 80.0`。
- 继续放宽夹爪外层平滑限制：
  - `--max_gripper_speed` 默认从 `3.0 rad/s` 改为 `3.14 rad/s`。
  - `--max_gripper_acceleration` 默认从 `80 rad/s^2` 改为 `120 rad/s^2`。
- 同步更新本文启动命令，显式传入 `--max_gripper_speed 3.14 --max_gripper_acceleration 120.0`。
- 增加 close latch，解决 ACT gripper 连续值没有直接输出 0.0 导致夹爪闭合不足的问题：
  - `--gripper_close_latch` 默认开启。
  - `--gripper_close_latch_threshold -0.5`
  - `--gripper_close_latch_confirm_steps 3`
  - `--gripper_close_latch_target 0.1`
  - 普通 ACT gripper 目标仍按 `--gripper_max 0.0` 裁剪；只有 force-closed latch 生效时，平滑器临时允许目标到 `0.1 rad`。
  - shared snapshot 新增 `gripper_action_raw / gripper_goal_before_latch / gripper_goal_after_latch / gripper_close_latch_count / gripper_force_closed`，便于日志确认 latch 是否触发。

#### `direct_gripper_replay_lowcmd.py`

- 新增纯夹爪 direct LOWCMD replay 工具。
- 功能：
  - 读取旧 ACT jsonl 的 `action[9]`。
  - 捕获当前 6 关节角作为 `q_hold`。
  - 全程 `setArmCmd(q_hold, 0, tau_hold)` 锁关节。
  - 用同样的 gripper 平滑逻辑重放夹爪。
  - 记录 target/cmd/actual gripper error 和每个关节 drift。
- 已用于 `gripper_replay_20260624_0022`。

#### Direct gripper replay + close latch：20260624_194115

- 目的：用上次 ACT 推理日志，只控制夹爪，验证 close latch。
- 源日志：
  - `high-level/real_deploy/nx_runs/door_act_realsense_20260623_231708/door_act_realsense_live.jsonl`
- 输出：
  - NX：`/tmp/z1_gripper_replay_close_latch_20260624_194115/direct_gripper_replay_close_latch.jsonl`
  - 本地：`high-level/real_deploy/nx_runs/z1_gripper_replay_close_latch_20260624_194115/direct_gripper_replay_close_latch.jsonl`
- 配置：
  - `--max_gripper_speed 3.14`
  - `--max_gripper_acceleration 120`
  - `--gripper_close_latch_threshold -0.5`
  - `--gripper_close_latch_confirm_steps 3`
  - `--gripper_close_latch_target 0.1`
- 结果：
  - replay source steps：`616`
  - close latch forced records：`17`
  - close latch forced runs：`99`, `108-111`, `204-207`, `258-261`, `546-549`
  - `cmd_abs_error_rad`：mean `0.0197`，p50 `0.0081`，p95 `0.0508`，max `0.0575`
  - 臂关节 drift max abs：max `0.00495 rad`，说明 direct replay 基本只动夹爪。
  - target 被 latch 到 `0.1 rad`，但由于每段闭合窗口较短且夹爪仍有速度/加速度平滑，cmd 最高约到 `0.0 rad`，实际反馈最高约 `+0.0256 rad`。
  - 结论：close latch 触发正确；如果后续希望更明显地顶紧夹爪，可以考虑减小 confirm steps 或在 latch 后延长保持时间。

#### `replay_z1_gripper_from_log.py`

- 加保护：默认拒绝运行。
- 原因：该脚本虽然只变化 gripper，但仍通过 `z1_act_ee_bridge` 发送完整 EE action，会触发 IK 并导致机械臂关节运动。
- 只有显式加 `--allow_ee_hold` 才允许运行。

#### `record_dual_d435_raw_bag_with_scripted_z1.py`

- 新增双 D435 raw `.bag` 录制脚本。
- 同时录：
  - `wrist_raw.bag`
  - `front_raw.bag`
  - `record_manifest.json`
- 录制 stream：
  - depth `z16`
  - color `rgb8`
- 可选通过 reference `.npz` 向 Z1 bridge 发布 scripted A2W trajectory。
- 发布前强制 `action[0:2]=0`，不控制狗底盘。
- 可选结束时发送 `shutdown_back_to_start`。

#### `offline_realsense_filter_sweep.py`

- 新增离线 RealSense filter sweep 脚本。
- 从 `wrist_raw.bag/front_raw.bag` 重放，生成：
  - A. no filters
  - B. spatial only
  - C. spatial + temporal
  - D. spatial + hole filling
  - E. spatial + hole filling + temporal
  - F. spatial + temporal + selective hole fill
- 默认处理链对齐当前真机部署：
  - `align(color)`
  - RealSense filters
  - no crop
  - resize to `640x480`
  - `0.2–1.5m -> uint8`
- 输出每种模式 side-by-side 视频和抽帧 grid PNG。
- F 模式使用连通域 + ring 深度一致性过滤，只补小/中等平滑表面空洞，保护夹爪/机械臂大黑块不被背景补掉。

#### `offline_opencv_inpaint_step_compare.py`

- 新增单帧 OpenCV inpaint 对比脚本。
- 当前用途：用已录制 raw bag 的 `step=360` 快速验证小连通域补洞策略，不接真机。
- 基础输入链路：
  - `align(color)`
  - `spatial_filter(holes_fill=0)`
  - `temporal_filter`
  - resize `640x480`
  - `0.2–1.5m -> uint8`
- 对比模式：
  - G1：`cv2.inpaint` TELEA，mask=小黑连通域，radius=3
  - G2：`cv2.inpaint` TELEA，mask=小黑连通域，radius=5
  - G3：`cv2.inpaint` Navier-Stokes，mask=小黑连通域，radius=3
  - G4：小黑连通域 dilate 1 次后 TELEA radius=3
- 当前 step 360 参数：
  - `max_component_area=5000`
  - `max_component_span_px=140`
  - `border_margin_px=1`
- 结果路径：
  - NX：`/tmp/d435_raw_scripted_20260624_154141/opencv_inpaint_step360/`
  - 本地：`high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_inpaint_step360/`
- 本次结果统计：
  - wrist：黑像素 `108797`，选中小连通域 `4280 px`，约 `3.93%`
  - front：黑像素 `48029`，选中小连通域 `7859 px`，约 `16.36%`
  - G1/G2/G3 只改 mask 内像素；G4 dilation 后改动更多。
- 观察：
  - G1-G4 明显比 RealSense D/E hole filling 保守，夹爪大黑块基本保留。
  - 小洞填补幅度受小连通域阈值控制；如果仍觉得太弱，下一步优先调大 `max_component_area/max_component_span_px`，而不是打开 RealSense 全局 hole filling。
- 备注：本次尝试拷贝到 U 盘时，NX 上 `/media/anx/Tang` 出现 stale mount / I/O error，`/dev/sdb1` 随后消失；结果暂时只保证在 NX `/tmp` 和本地 `nx_runs` 中。

##### 追加：更激进 OpenCV inpaint 参数

- H 版：
  - `max_component_area=10000`
  - `max_component_span_px=260`
  - `border_margin_px=1`
  - wrist 选中黑洞 `12760 px`，约为原 G 版 `4280 px` 的 3 倍。
  - front 选中黑洞 `10616 px`。
  - 本地结果：`high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_inpaint_step360_aggressive/realsense_AE_plus_opencv_H1H4_step_000360.jpg`
- I 版：
  - `max_component_area=15000`
  - `max_component_span_px=280`
  - `border_margin_px=-1`，允许补非巨大边界黑洞。
  - wrist 选中黑洞 `23784 px`，约占黑像素 `21.86%`。
  - front 选中黑洞 `21479 px`，约占黑像素 `44.72%`。
  - 夹爪主体连通域约 `85013 px / 590x205`，仍因面积超过阈值未被补。
  - 本地结果：`high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_inpaint_step360_more_aggressive/realsense_AE_plus_opencv_I1I4_step_000360.jpg`
- 当前肉眼观察：
  - H：比 G 明显多补 wrist，但仍偏保守。
  - I：更接近“补掉小/中黑洞，同时保留夹爪主体”的目标；优先考虑 I1/ I2，I4 最激进。

#### `offline_opencv_front_fringe_step_compare.py`

- 新增 front 专用 OpenCV inpaint 离线对比脚本。
- 背景：用户确认“不够激进”的主要是 front 相机，不是 wrist。
- 处理策略：
  - 先使用 I 版小/中连通域 mask：
    - `small_max_area=15000`
    - `small_max_span_px=280`
    - `small_border_margin_px=-1`
  - 对剩余最大的黑连通域，不整块填充，而是只取其内部靠近有效深度边界的一圈 fringe。
  - fringe 通过 `cv2.distanceTransform` 得到，避免把大块未知/前景 silhouette 全部抹掉。
- step 360 对比：
  - I：仅小/中连通域，mask `21479 px`
  - J1：I + fringe `8 px`，mask `31928 px`
  - J2：I + fringe `12 px`，mask `35233 px`
  - J3：I + fringe `20 px`，mask `40368 px`
  - J4：I + fringe `30 px`，mask `44753 px`
- 本地结果：
  - `high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_front_fringe_step360/front_AE_plus_I_J_fringe_step_000360.jpg`
- 观察：
  - J1/J2 比 I 明显更积极，front 黑洞减少。
  - J3/J4 更接近 RealSense hole filling 的强度，但仍保留大黑块核心。
  - 如果要上线实时，建议先尝试 J2 或 J3；J4 可能过度吃边缘。

#### `offline_opencv_dual_aggressive_step_compare.py`

- 新增双相机、清晰标签版 aggressive OpenCV inpaint 离线对比脚本。
- 输出每个 tile 顶部有黑底白字标签，包含滤波模式和 mask/changed 像素数。
- step 360 对比列：
  - A：`align(color), raw`
  - C：`spatial + temporal, holes_fill=0`
  - E：`RealSense spatial holes_fill=5 + hole_filling + temporal`
  - I：OpenCV TELEA，小/中黑连通域
  - K1：I + 大黑块内部 fringe `40 px`
  - K2：I + fringe `60 px`
  - K3：I + fringe `80 px`
  - K4：I + fringe `120 px`
- 参数：
  - `small_max_area=15000`
  - `small_max_span_px=280`
  - `small_border_margin_px=-1`
  - `fringe_px=[40,60,80,120]`
- 结果：
  - wrist：
    - C 黑像素 `108797`
    - I mask `23784`
    - K1/K2/K3/K4 mask `58235 / 70019 / 80260 / 96370`
  - front：
    - C 黑像素 `48029`
    - I mask `21479`
    - K1 mask `47455`
    - K2/K3/K4 mask `48029`，即 front 黑像素全部进入补洞 mask
- 本地结果：
  - `high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_dual_aggressive_step360/dual_A_C_E_I_K_aggressive_step_000360.jpg`
- 观察：
  - front K1 已经非常接近全补。
  - front K2 之后就是全黑洞补齐，属于极激进。
  - wrist K2-K4 会明显吃掉大黑块边缘，需要谨慎；如果只想 front 激进、wrist 保守，应在实时逻辑中给两个相机使用不同参数。

#### `offline_opencv_selected_k_multistep_compare.py`

- 新增多 step 选定参数离线对比脚本。
- 当前用于验证非对称相机参数：
  - wrist：K1，`fringe_px=40`
  - front：K2，`fringe_px=60`
- 测试 step：
  - `120`
  - `240`
  - `360`
  - `480`
- 每个 step 输出：
  - C baseline：`spatial + temporal, holes_fill=0`
  - 对应 K 输出：wrist K1 / front K2
- 本地结果：
  - `high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_selected_k_multistep/dual_selected_wristK1_frontK2_steps_120_240_360_480.jpg`
- 结果统计：
  - wrist K1：
    - step 120：C 黑像素 `17757`，mask `15164`，changed `9480`
    - step 240：C 黑像素 `87795`，mask `34362`，changed `22278`
    - step 360：C 黑像素 `108797`，mask `58235`，changed `46036`
    - step 480：C 黑像素 `64822`，mask `36042`，changed `21411`
  - front K2：
    - step 120：C 黑像素 `60821`，mask `47968`，changed `34974`
    - step 240：C 黑像素 `41877`，mask `41877`，changed `41877`
    - step 360：C 黑像素 `48029`，mask `48029`，changed `48029`
    - step 480：C 黑像素 `11352`，mask `11352`，changed `11352`
- 观察：
  - front K2 在多个 step 上基本达到强补洞目标。
  - wrist K1 保留更多黑色主体，适合保护腕部/夹爪轮廓。
  - 如果后续实时化，建议使用 per-camera 参数，而不是两个相机共用同一个 aggressive 级别。

##### 追加：小白洞补齐

- 新增 `make_small_white_component_mask()`。
- 目标：只补类似用户截图黄色框中的小白洞，不补大面积白背景。
- 当前规则：
  - 候选白洞：`depth_u8 >= 250`
  - 面积上限：`white_hole_max_area=2500`
  - 宽高上限：`white_hole_max_span_px=90`
  - 不贴边：`white_hole_border_margin_px=2`
  - 周围 ring 检查：
    - `ring_radius_px=3`
    - 至少 `8` 个灰色邻域像素
    - 灰色邻域比例至少 `0.35`
  - 灰色邻域定义：`0 < depth_u8 < 250`
- 已接入：
  - `offline_opencv_selected_k_video.py`
  - `offline_opencv_selected_k_multistep_compare.py`
- 默认开启，可通过 `--no_fill_small_white_holes` 关闭。
- 验证：
  - step 311 front：检测到 `white_selected_px=625`，`white_selected_components=1`
  - 大白背景未整体进入 white mask。
  - 本地结果：
    - `high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_selected_k_white_holes_step311/dual_selected_wristK1_frontK2_steps_311.jpg`

##### 追加：wrist fringe 10px 对比

- 背景：wrist K1 `fringe=40px` 会把夹爪右上黑色边缘补成白色，因为 TELEA 从周围白背景向大黑块边界扩散。
- step 311 对比：
  - C baseline
  - E RealSense hole filling
  - I：只补小/中连通域
  - K fringe `10px`
  - K fringe `40px`
- 结果：
  - wrist I：mask `17578`，changed `17578`
  - wrist fringe 10px：mask `27824`，changed `24798`
  - wrist fringe 40px：mask `48823`，changed `36731`
- 观察：
  - wrist fringe 10px 明显比 40px 更保护夹爪轮廓。
  - 40px 会吃掉夹爪右上边缘，当前不建议用于 wrist 实时输入。
  - 如果必须在 wrist 上使用 large-component fringe，优先试 `10px`，或者直接关掉 wrist fringe、只用 I。
- 本地结果：
  - `high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_wrist_fringe10_compare_step311/dual_A_C_E_I_K_aggressive_step_000311.jpg`

### 实时部署：`opencv_k_no_rs` 模式

- `door_act_shadow.py` 新增正式实时模式：
  - `--depth_inpaint_mode opencv_k`
  - `--depth_inpaint_mode opencv_k_no_rs`
- 当前默认模式已改为 `opencv_k_no_rs`：
  - 不再需要手动传 `--depth_inpaint_mode opencv_k --no_rs_filters`。
  - `opencv_k_no_rs` 固定表示 full-res OpenCV K 补洞，并强制 `rs_filters=false`。
  - 原有 `realsense` 模式保留，但需要显式传 `--depth_inpaint_mode realsense`。
  - `test_realsense_opencv_k_hz.py` 的默认也已同步为 `opencv_k_no_rs / rs_filters=false`。
- 实时 `opencv_k_no_rs` 参数：
  - wrist：K1，large black component internal fringe `10 px`
  - front：K2，large black component internal fringe `40 px`
  - 小/中黑连通域：
    - `max_area=15000`
    - `max_span_px=280`
    - `border_margin_px=-1`
  - 小白洞：
    - `depth_u8 >= 250`
    - `max_area=2500`
    - `max_span_px=90`
    - `border_margin_px=2`
    - ring gray check enabled
  - 当前实时版恢复为 full resolution `process_scale=1.0`，不再缩小到低分辨率处理。
  - 备注：曾测试 `process_scale=0.25/0.35/0.5` 来追求 30Hz，但视觉效果不满足要求，已废弃。
- `test_realsense_opencv_k_hz.py` 新增双 D435 camera-only 频率测试脚本。
- NX 实测 10 秒（2026-06-24，确认恢复 full-res `process_scale=1.0`，无低分辨率缩放）：
  - `opencv_k + rs_spatial_magnitude=2`：
    - wrist `~16.0 Hz`
    - front `~8.0 Hz`
    - front K2 full-res 大 mask 的 OpenCV inpaint 是主要瓶颈。
  - `opencv_k + rs_spatial_magnitude=1`：
    - wrist `~18.1 Hz`
    - front `~8.6 Hz`
    - 降低 spatial magnitude 只能小幅提升，不能解决 full-res front K2 瓶颈。
  - `opencv_k + --no_rs_filters`：
    - wrist `~20.1 Hz`
    - front `~13.2 Hz`
    - 说明纯 full-res opencv_k 也达不到 25/30Hz。
- 建议：
  - 目前按视觉优先，默认使用 full-res `opencv_k_no_rs`。
  - full-res `opencv_k_no_rs` 的相机 worker 新帧率无法保证 ACT 25Hz 或相机 30Hz；如果后续重新追频率，需要重新做视觉验收后再启用降采样/ROI/更轻量的补洞版本。
- 默认模式验证：
  - 命令：`python3 high-level/real_deploy/door_act_shadow.py --camera_mode realsense --camera_benchmark_s 10 --camera_worker_mode auto`
  - 结果：`depth_inpaint_mode=opencv_k_no_rs`，`filters=false`，`worker_mode=process`
  - benchmark read loop：`~24.91 Hz`
  - camera worker 新帧率：wrist `~20.53 Hz`，front `~13.75 Hz`

#### 原始 bag 生成 runtime opencv_k 视频

- 用原始 raw bag：
  - `/tmp/d435_raw_scripted_20260624_154141/wrist_raw.bag`
  - `/tmp/d435_raw_scripted_20260624_154141/front_raw.bag`
- 使用配置：
  - `--use_runtime_opencv_k`
  - `--rs_spatial_magnitude 1`
  - wrist K1：fringe `10px`
  - front K2：fringe `40px`
  - 小白洞补齐开启
  - realtime opencv_k 当时测试版本曾使用 `process_scale=0.25`；该低分辨率方案现已废弃，当前部署恢复 `process_scale=1.0`。
- 输出视频：
  - NX：`/tmp/d435_raw_scripted_20260624_154141/opencv_k_runtime_mag1_full_video/selected_wristK1_frontK2_full.mp4`
  - U 盘：`/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141/opencv_k_runtime_mag1_full_video/selected_wristK1_frontK2_full.mp4`
- 输出内容：
  - 上排：wrist C baseline / wrist runtime opencv_k
  - 下排：front C baseline / front runtime opencv_k
  - 每帧标注 mask、white、changed 像素数。
- 同目录已保存：
  - `selected_wristK1_frontK2_full_manifest.json`
  - `offline_opencv_selected_k_video.py`
  - `door_act_shadow_runtime_opencv_k.py`

#### 原始 bag 生成 full-res opencv_k + no RS filters 视频

- 目的：检查不经过 RealSense spatial/temporal，仅 `align(color) -> clip/normalize -> runtime opencv_k` 的视觉效果。
- 代码变更：
  - `offline_opencv_selected_k_video.py` 增加 `--rs_filters / --no_rs_filters`。
  - `--no_rs_filters` 时 baseline 标注为 `no_rs_filters`，manifest 中 `rs_filters=false`。
- 使用配置：
  - `--use_runtime_opencv_k`
  - `--no_rs_filters`
  - full-res `process_scale=1.0`
  - wrist K1：fringe `10px`
  - front K2：fringe `40px`
  - 小白洞补齐开启
- 输出：
  - NX：`/tmp/d435_raw_scripted_20260624_154141/opencv_k_runtime_no_rs_filters_full_video/selected_wristK1_frontK2_full.mp4`
  - U 盘：`/media/anx/Tang/door_act_debug/d435_raw_scripted_20260624_154141/opencv_k_runtime_no_rs_filters_full_video/selected_wristK1_frontK2_full.mp4`
  - manifest：`selected_wristK1_frontK2_full_manifest.json`
- 结果：
  - frames_written：`490`
  - output_fps：`30.0`
  - output_resolution：`960 x 836`
  - elapsed_s：`62.79`
  - 视频大小约 `22.3 MB`

#### full-res opencv_k_no_rs 后追加 Gaussian blur：step 313

- 目的：测试当前默认深度输入 `full-res opencv_k_no_rs` 后，再追加不同 OpenCV Gaussian blur 的视觉效果。
- Pipeline：
  - `align(color)`
  - `no_rs_filters`
  - `clip 0.2-1.5m / normalize`
  - `full-res opencv_k`
  - `cv2.GaussianBlur`
- 新增脚本：
  - `high-level/real_deploy/offline_opencv_k_gaussian_blur_step_compare.py`
- 测试 frame：
  - step `313`
- 参数：
  - `none`
  - `3x3 sigma=0`
  - `5x5 sigma=0`
  - `5x5 sigma=1.0`
  - `7x7 sigma=1.5`
  - `9x9 sigma=2.0`
  - `11x11 sigma=3.0`
  - `15x15 sigma=4.0`
- 输出：
  - NX：`/tmp/d435_raw_scripted_20260624_154141/opencv_k_no_rs_gaussian_blur_step313/dual_opencv_k_no_rs_gaussian_blur_step_0313.png`
  - 本地：`high-level/real_deploy/nx_runs/d435_raw_scripted_20260624_154141/opencv_k_no_rs_gaussian_blur_step313/dual_opencv_k_no_rs_gaussian_blur_step_0313.png`
  - manifest：`dual_opencv_k_no_rs_gaussian_blur_step_0313_manifest.json`
- 备注：
  - 该 blur 是在 policy depth u8 上直接做 Gaussian，会平滑边缘，也会轻微扩散黑/白区域边界。
  - wrist 的 mean abs delta 从 `0.41` 到 `2.75`，front 从 `0.27` 到 `1.74`，随 kernel/sigma 增大逐渐变强。

#### 实时 Gaussian blur FPS 测试

- 代码：
  - `door_act_shadow.py` 新增可选参数：
    - `--depth_gaussian_blur_ksize`
    - `--depth_gaussian_blur_sigma`
  - 默认 `ksize=0`，即不加 blur，当前默认输入不变。
  - `test_realsense_opencv_k_hz.py` 同步支持这两个参数。
- 测试命令：
  - baseline：
    - `python3 high-level/real_deploy/test_realsense_opencv_k_hz.py --duration_s 10 --camera_worker_mode process --poll_hz 30`
  - Gaussian：
    - `python3 high-level/real_deploy/test_realsense_opencv_k_hz.py --duration_s 10 --camera_worker_mode process --poll_hz 30 --depth_gaussian_blur_ksize 15 --depth_gaussian_blur_sigma 4.0`
- NX 实测：
  - baseline `opencv_k_no_rs` no blur：
    - wrist `~20.60 Hz`
    - front `~15.90 Hz`
  - `GaussianBlur 15x15 sigma=4.0`：
    - wrist `~18.20 Hz`
    - front `~14.50 Hz`
  - 影响：
    - wrist 下降约 `2.4 Hz`
    - front 下降约 `1.4 Hz`
- 结论：
  - blur 有额外 CPU 开销，但相比 full-res OpenCV inpaint 主瓶颈不算最大项。
  - 当前默认仍保持不加 Gaussian blur；如需测试/上线 blur，显式加 `--depth_gaussian_blur_ksize 15 --depth_gaussian_blur_sigma 4.0`。

## 14. 维护规则

以后每次改真机部署代码时，同步做三件事：

1. 在本文 `代码修改记录` 下加日期和文件级变更。
2. 如果改变默认行为，更新 `当前状态`、`启动命令` 或 `接口约定`。
3. 如果跑了新真机实验，把 run 目录放到：

```bash
high-level/real_deploy/nx_runs/<run_name>/
```

并在本文增加简短实验结论。

## 15. 2026-07-04 NX / PC2 实机代码与环境备份

- NX Wi-Fi DHCP 地址已由旧的 `192.168.1.154` 变为
  `192.168.1.173`；NX 有线机器人网仍是 `192.168.124.25/24`。
- PC2 `192.168.124.162` 可由 NX 稳定访问。
- 本机备份根目录：

```text
/home/sivan/whole_body/backups/nx_pc2_20260704_183407
```

- NX 已备份：
  - 实机 `/home/anx/door_act_deploy/z1_controller`
  - 实机 `/home/anx/door_act_deploy/z1_sdk`
  - 实机 `/home/anx/door_act_deploy/visual_whole_body`
  - ARM/aarch64 编译产物、启动脚本和 CycloneDDS 配置
  - `pip freeze`、dpkg/apt、ROS 包、JetPack/CUDA、RealSense、网络和文件清单
- PC2 已备份：
  - `/home/unitree/whole_body/robot_control`
  - `/home/unitree/whole_body/build/robot_control`
  - `/home/unitree/whole_body/install/robot_control`
  - PC2 Python、dpkg/apt、ROS 和文件清单
- 两个压缩包均已用远端生成的 SHA256 在本机验证通过。
- 与本机旧副本对比后确认：
  - NX `z1_controller/config/config.xml` 与本机旧配置不同。
  - NX `z1_sdk` 多出实际使用的 aarch64 Python 扩展。
  - PC2 launch 多出 `a2_sport_udp_helper` 所需的
    `/usr/local/lib` `LD_LIBRARY_PATH` 修复。
  - NX 主要真机部署 Python 文件与本机一致；差异主要是两个测试和历史
    `.bak` 文件。
- 该备份是“源码 + build/install + 配置 + 环境版本清单”，不是完整 NVMe
  镜像；未重复复制 checkpoint、数据集、日志和 NX 约 1.7 GB 的完整
  `site-packages`。裸机恢复仍应使用 JetPack R36.5 基础系统或额外制作磁盘镜像。

### 15.1 CycloneDDS / x86 小主机迁移快照

- NX 当前 DDS 配置已单独归档到：

```text
high-level/real_deploy/environment_snapshots/nx_20260704/nx_dds_backup_20260704/
```

- 归档内容：
  - `/home/anx/cyclonedds.xml`
  - `/etc/sysctl.d/60-cyclonedds.conf`
  - `/etc/sysctl.d/99-cyclonedds-buffers.conf`
  - CycloneDDS、`rmw_cyclonedds`、`unitree_ros2` 源码快照与提交号
  - NX/PC2 Python、ROS、apt/dpkg 和系统环境清单
- 当前 XML 固定使用 NX 接口 `eno1`，并将 PC2
  `192.168.124.162` 配为 discovery peer。迁移到 x86 后必须替换为新主机
  124 网段接口名。
- x86 完整部署、重编译 Z1 Python 3.10 扩展和通信验证步骤见：

```text
high-level/real_deploy/X86_MINIPC_DEPLOYMENT.md
```

## 16. 2026-07-05 RTX 4090 x86 小主机迁移

- 主机：`robo@192.168.1.124`，本机 SSH 别名：`ssh 4090`
- 部署目录：`/home/robo/txc/door_act_deploy`
- Python venv：`/home/robo/txc/venvs/door_act`
- 已迁移 NX 真机代码、DDS 快照、Z1 SDK/controller 和 100000 checkpoint。
- 已完成：
  - torch `2.11.0+cu130` / torchvision `0.26.0+cu130` CUDA 验证；
  - checkpoint SHA256 对账；
  - 真实 checkpoint dummy observation 100 步推理；
  - 稳定 ACT forward 约 `11.3 ms`，控制循环 `25.078 Hz`；
  - Z1 SDK CPython 3.10 x86 扩展和 `z1_ctrl` 重编译；
  - Z1 FK/IK 往返与 bridge UDP dry-run；
  - ROS Humble CycloneDDS 本机 publisher/echo 测试。
- 机器人网络后续已接入 `enp5s0`：
  - 持久地址 `192.168.124.25/24`；
  - PC2 ping 0% 丢包，约 `0.1 ms`；
  - 跨机器 CycloneDDS 已发现 `/cmd_vel_safe` 和 `/vel_state`；
  - PC2 `robot_control` launch 已在后台启动；
  - 4090 只发布过一次全零 Twist，PC2 返回全零 `/vel_state`。
- 尚未做真实 Z1/相机测试：
  - Z1 `192.168.124.110` ARP `FAILED`、ping 不通；
  - 4090 当前未连接 D435；
  - 没有发送 LOWCMD、`backToStart()` 或真实机械臂运动命令。
- 4090 的最终环境、DDS XML 和验收摘要已回存本机：

```text
high-level/real_deploy/environment_snapshots/x86_4090_20260705/
```

- 完整状态与接线后命令：

```text
high-level/real_deploy/X86_4090_DEPLOYMENT_STATUS.md
```

## 16. 2026-07-05 robo ZED 2i 交互式 RGBD 采集

- `capture_zed2i_aligned_frames.py` 新增 `--interactive`：
  - 实时显示 ZED 左目 RGB 与对齐到左目坐标系的 depth。
  - 每按一次空格保存一组 RGBD；默认保存满 3 组后退出。
  - `Q` 或 `Esc` 可提前退出。
  - 每组仍保存 RGB、毫米 PNG、米制 NPY、深度可视化、拼图及
    `metadata.json` 相机内参。
- `robo` 运行命令：

```bash
cd ~/txc/rgbd
RUN_DIR=~/txc/rgbd/zed2i_manual_$(date +%Y%m%d_%H%M%S)
python3 capture_zed2i_aligned_frames.py \
  --interactive \
  --frames 3 \
  --resolution HD2K \
  --fps 15 \
  --depth_mode NEURAL_PLUS \
  --out_dir "$RUN_DIR"
```

## 17. 2026-07-09 Z1 EE 键盘控制 / 录制 / 回放

- 新增脚本：

```text
high-level/real_deploy/keyboard_z1_ee_teleop.py
```

- 控制路径：
  - 脚本只通过 UDP 给 `z1_act_ee_bridge.py` 发 Door-ACT 10D EE action；
  - 不直接调用 Z1 SDK、不直接发 LOWCMD；
  - IK、关节限速、online quintic 平滑、夹爪平滑仍全部由 `z1_act_ee_bridge.py`
    统一处理。
- 运行前必须先启动：
  1. `z1_ctrl`
  2. `z1_act_ee_bridge.py`
- NX 默认端口：
  - action UDP：`127.0.0.1:15011`
  - bridge state UDP：`127.0.0.1:15013`

### 17.1 键盘实时控制并录制

```bash
cd /home/anx/door_act_deploy/visual_whole_body

RUN_DIR=/tmp/z1_keyboard_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

python3 high-level/real_deploy/keyboard_z1_ee_teleop.py \
  --record_path "$RUN_DIR/z1_keyboard_record.jsonl" \
  --hz 25 \
  --position_step 0.005 \
  --rotation_step_deg 2.0 \
  --gripper_step 0.05
```

键位：

```text
w/s : ACT-frame x +/-
a/d : ACT-frame y +/-
r/f : ACT-frame z +/-
u/o : roll +/-
i/k : pitch +/-
j/l : yaw +/-
[   : 夹爪张开一步
]   : 夹爪闭合一步
g   : 夹爪全开 -pi/2
c   : 夹爪闭合 0
h   : target 重置为当前反馈 EE
p   : 暂停/继续发布
space: 录制 mark
q/Esc: 退出
```

### 17.2 回放录制轨迹

```bash
cd /home/anx/door_act_deploy/visual_whole_body

python3 high-level/real_deploy/keyboard_z1_ee_teleop.py \
  --replay_path /tmp/z1_keyboard_YYYYmmdd_HHMMSS/z1_keyboard_record.jsonl \
  --record_path /tmp/z1_keyboard_replay_$(date +%Y%m%d_%H%M%S).jsonl \
  --replay_speed 1.0 \
  --hz 25
```

### 17.3 安全检查

- 脚本启动后会先等待 bridge state，初始 target 取当前真实反馈 EE，
  不会凭空跳到固定姿态。
- `--dry_run` 可检查键盘、录制和回放流程，但不发送 UDP action。
- `p` 暂停发布后，bridge 会因为 action stale 而按自身逻辑平滑刹停。
- 如果需要限制工作空间，可加：

```bash
--min_xyz XMIN YMIN ZMIN --max_xyz XMAX YMAX ZMAX
```

## 18. 2026-07-20 Joint9 ACT / Z1 bridge

新模型的 state/action 都是 9 维关节格式：

```text
state  = [last_command_vx, last_command_vyaw, q1, q2, q3, q4, q5, q6, gripper]
action = [vx,              vyaw,              q1, q2, q3, q4, q5, q6, gripper]
```

bridge 新增显式模式：

```bash
--act_state_action_mode joint9
```

该模式的控制路径：

1. 50 Hz 异步读取 Z1 实际 `q/qd/gripper`，直接组装 joint9 state。
2. 收到 ACT joint9 action 后，直接取 `action[2:8]` 作为关节路点；不做 EE IK。
3. 保留关节范围和跳变检查。
4. 保留相邻实时路点终点速度估计。
5. 保留在线 quintic 重规划，以及最大关节速度/加速度约束。
6. 保留 500 Hz LOWCMD、命令超时平滑刹车、夹爪限速/限加速度和 close latch。
7. 保留启动 `backToStart()`、实际 `home_q` 稳定检测和退出回零流程。

ACT runner 默认 `--z1_state_action_mode auto`：checkpoint 的 state/action 都为
9 维时自动选 `joint9`；都为 10 维时保持旧 `ee10`。chunk overlap 对
`vx/vyaw/q1..q6` 使用 `0.3 old + 0.7 new`，夹爪采用最新预测，不融合。

joint9 bridge 启动参数：

```bash
cd /home/anx/door_act_deploy/visual_whole_body

high-level/real_deploy/run_z1_act_ee_bridge.sh \
  --act_state_action_mode joint9 \
  --enable_arm \
  --max_joint_speed 3.0 \
  --max_joint_acceleration 15.0 \
  --joint_trajectory_duration_s 0.02 \
  --max_gripper_speed 3.14 \
  --max_gripper_acceleration 120.0 \
  --state_tx_host 127.0.0.1 \
  --state_tx_port 15013 \
  --log_path /tmp/door_act_services/z1_bridge_joint9.jsonl
```

验证记录：协议/维度/旧模式兼容测试共 41 项通过；joint9 UDP dry-run 已确认
state 返回 9 维并跟随输入关节目标，测试全程 `--no_enable_arm`，没有发送 LOWCMD。

Plücker camera pose 已接入：front pose 固定使用仿真默认安装位姿；wrist pose 使用
`forwardKinematics(q, 6)`，先右乘现有 `+0.086 m` tool offset 得到 ACT EE，再右乘
EE-to-camera 相对变换。两者都以 robot base 表达为
`[x,y,z,qx,qy,qz,qw]`。这里的 FK 只服务于移动相机 extrinsic，不参与 joint9 state
构造或 joint command 求解。

注意：Z1 SDK `forwardKinematics(q, 6)` 的输出原点不是 Isaac Gym 的 `link06`
原点，而是沿末端局部 x 轴前移约 `0.100 m`。不能直接在 SDK FK 后加仿真的
link06 camera offset，否则 wrist pose 会多出约 10 cm。当前实现使用等价且旋转正确的
ACT-EE 局部变换：

仿真一致的默认值：

```text
front @ robot base: xyz=[0.29, 0.031, 0.165], ypr_deg=[0, -45, 0]
wrist @ ACT EE:     xyz=[-0.093, 0.031, 0.22], ypr_deg=[0, 60, 0]
```

等价仿真定义仍是 `wrist @ Isaac link06 = [0.093, 0.031, 0.22]`。使用同一条
episode 的 5 组大范围关节姿态反推 SDK-FK-to-camera 固定变换，平移各轴极差均小于
`1e-6 m`，确认该差异是固定坐标系定义，不是跟踪误差。

NX 离线前向验证（50K checkpoint，500 帧 joint9/Plücker episode 中按 25 step
抽样，18 个 query、共 180 个 action）：

```text
motion_l1=0.01118
normalized_motion_l1=0.02441
threshold_accuracy=99.58%
all_action_accuracy=98.33%
vx_mae=0.0052
joint_l2_mean=0.0451 rad
joint_max_abs_mean=0.0316 rad
gripper_mae=0.0034 rad
interaction contact precision/recall/F1=1.0/1.0/1.0
```

### 2026-07-21 joint9 raw episode 0 真机实时重放

新增 `high-level/real_deploy/replay_joint9_episode_z1.py`，专门将 joint9 raw
episode 按原始采样频率送入 Z1 bridge，并同步接收 `q/qd/gripper` 反馈。脚本会将
`action[0:2]` 强制清零且不创建 ROS publisher，因此不会控制底盘；结束时请求 bridge
执行 `backToStart()`。

本次在 NX 上重放：

```text
episode: episode_000000_joint9_eval.npz
frames: 500
frequency: 25 Hz
duration: 20 s trajectory + settle/backToStart
joint bridge limits: 3.0 rad/s, 15.0 rad/s^2
valid feedback: 498/500
dog command published: false
shutdown backToStart confirmed: true
```

跟踪结果：

```text
raw target vs feedback abs error: p50=0.0269 rad, p95=0.1561 rad, max=0.2922 rad
best causal lag: 6 frames = 240 ms
lag-compensated median abs error: 0.00485 rad (about 0.28 deg)
measured joint speed: p95=0.6310 rad/s, max=1.0455 rad/s
gripper target vs feedback: p50=0.00373 rad, p95=0.2481 rad, max=0.4229 rad
```

原始 episode 少数帧的离散目标速度达到约 `2.77 rad/s`、离散加速度达到约
`88.5 rad/s^2`。bridge 没有原样下发这些尖峰，而是通过在线 quintic 和
`15 rad/s^2` 加速度上限自动延长轨迹段。因此主要问题是约 240 ms 的轨迹滞后，
不是关节稳态精度；去除估计滞后后中位误差约 0.28°。

完整日志：

```text
high-level/real_deploy/nx_runs/joint9_episode0_replay_20260721_010213/
```

### 2026-07-22 interaction decoder chunk 部署日志

ACT 部署日志已补齐 interaction decoder 输出。每次新策略 chunk 到达时，
`policy_chunk_ingests[*].interaction_state_chunk` 保存未经 action horizon 截断的
完整 `H x 3`：

```text
[contact_probability, handle_progress, door_progress]
```

每个控制 step 的 `executed_interaction_state` 保存与当步实际发布 action 严格按
全局 timestep 对齐的三维预测，并记录 `chunk_ids/action_source/blend_count`。异步
推理迟到时，过期 action 与对应 interaction 行会一起丢弃；新旧 chunk 重叠时，
interaction 与运动 action 一样执行 `0.3 old + 0.7 new`。新增复数记录字段
`policy_chunk_ingests`，避免同一个控制周期摄取两次推理结果时覆盖第一份完整 chunk；
原有 `policy_chunk_ingest` 保留兼容旧分析脚本。

本地验证：静态编译通过，action/interaction 时间对齐、迟到跳过、重叠融合和无
interaction 旧模型兼容测试共 6 项通过。此次只修改并离线验证日志链路，没有启动
Z1、底盘或任何真机控制进程。
