# x86 RTX 4090 小主机迁移状态

更新日期：2026-07-05

## 1. 主机与路径

- SSH：`ssh 4090`
- 实际地址：`robo@192.168.1.124`
- 硬件：FEVM FN70G，RTX 4090 Laptop GPU 16 GB
- 系统：Ubuntu 22.04.5，x86_64，kernel `6.8.0-124-generic`
- 部署根目录：`/home/robo/txc/door_act_deploy`
- Python venv：`/home/robo/txc/venvs/door_act`
- 测试输出：`/home/robo/txc/tests`

本机 `~/.ssh/config` 已增加：

```sshconfig
Host 4090
  HostName 192.168.1.124
  User robo
  IdentityFile ~/.ssh/id_rsa
  IdentitiesOnly yes
```

## 2. 已迁移内容

- NX 实机版 `visual_whole_body`
- NX 实机版 `z1_controller`
- NX 实机版 `z1_sdk`
- CycloneDDS XML、sysctl、DDS 源码与 NX/PC2 环境快照
- checkpoint：

```text
/home/robo/txc/door_act_deploy/checkpoints/
  leroact_a2w_state10_wc4_newfov_randomized_200_depthonly_chunk100_exec50_bs16_0625_0212/
  100000/model_latest.pt
```

checkpoint 的 `model.safetensors` 已与训练机做 SHA256 对账，一致：

```text
dda1adc2102e97fe4a4722aebb3ecd492116af841b24c588221cb13c1ef25671
```

## 3. 已安装环境

- Python 3.10.12
- ROS 2 Humble
- CycloneDDS `0.10.5`
- `rmw_cyclonedds_cpp 1.3.4`
- torch `2.11.0+cu130`
- torchvision `0.26.0+cu130`
- numpy `1.24.0`
- OpenCV `4.11.0`
- pyrealsense2 `2.57.7.10387`
- LeRobot 本仓库版本 `0.4.4`，使用部署专用 minimal ACT import

NVIDIA driver 为 `595.71.05`，PyTorch wheel 自带 CUDA 13.0 runtime，并已识别
4090。系统遗留的 `nvcc 11.5` 不参与当前 PyTorch 推理。

实际安装后的环境清单位于：

```text
/home/robo/txc/door_act_deploy/environment_snapshot/x86_4090_20260705/
```

同一份快照已备份回本机仓库：

```text
high-level/real_deploy/environment_snapshots/x86_4090_20260705/
```

其中 `validation_summary.txt` 记录最终网络、DDS、ACT、Z1 和相机验收结果，
`cyclonedds.xml` 是当前 4090 实际使用的 DDS 配置。

## 4. x86 Z1 编译结果

`z1_sdk` 已针对 Python 3.10 / x86_64 重编译：

```text
/home/robo/txc/door_act_deploy/z1_sdk/lib/
  unitree_arm_interface.cpython-310-x86_64-linux-gnu.so
```

构建时需要显式加入 Eigen：

```bash
cmake -S . -B build_x86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_CXX_FLAGS="-I/usr/include/eigen3"
```

`z1_controller/build/z1_ctrl` 也已重编译为 x86-64 ELF。

无机械臂网络时启动 `z1_ctrl`，程序能正常运行并按预期报告
`connect with z1_arm wait time out`，说明二进制和动态库加载正常。

纯运动学验证：

- SDK module 确认加载 CPython 3.10 x86 扩展；
- q=0 FK 正常；
- q=0 FK→IK 成功；
- 最大关节往返误差约 `4.4e-16 rad`；
- SDK 报告六关节速度上限均为 `pi rad/s`。

## 5. ACT 推理验证

使用真实 100000 checkpoint、dummy 双深度图、AMP、异步推理和
`action_horizon=10` 运行 100 步：

- 第一次 CUDA warmup：约 `979 ms`
- warmup 后 forward：约 `11.3 ms`
- 稳态控制循环：`25.078 Hz`
- 完成 100/100 步，无 CUDA/模型/NaN 错误
- ROS、Z1 发布均显式关闭，没有发送实机命令

日志：

```text
/home/robo/txc/tests/act_dummy_20260705_162303/
```

## 6. Z1 bridge dry-run

已使用新的 x86 wrapper 启动：

```bash
/home/robo/txc/door_act_deploy/visual_whole_body/high-level/real_deploy/
  run_x86_z1_bridge.sh
```

在 `--no_enable_arm` 下发送 5 个模拟 10D ACT action：

- UDP 15011 接收正常；
- 10D action/state 格式正常；
- dry-run IK 状态正常；
- 明确打印 `no LOWCMD is sent`；
- 没有控制真实机械臂。

日志：

```text
/home/robo/txc/tests/z1_bridge_dry/
```

## 7. CycloneDDS 与 PC2 验证

ROS Humble + `rmw_cyclonedds_cpp` 已安装。使用 Wi-Fi 接口做本机收发测试：

```text
/migration_dds_test
linear.x  =  0.123
angular.z = -0.456
```

publisher/echo 收发内容完全一致，证明 x86 上 RMW/CycloneDDS 可用。

测试记录：

```text
/home/robo/txc/tests/dds/
```

随后机器人交换机接入 `enp5s0`，已创建持久 NetworkManager profile：

```text
profile: door-robot-124
enp5s0: 192.168.124.25/24
default route: disabled
```

生成的实机 DDS 配置：

```text
/home/robo/txc/door_act_deploy/config/cyclonedds.xml
```

它绑定 `enp5s0`，并配置 PC2 `192.168.124.162` 为 discovery peer。

跨机器实测：

- PC2 ping：3/3，0% 丢包，平均约 `0.105 ms`
- 可发现 PC2 ROS 图和 `/cmd_vel_safe`、`/vel_state`
- `/cmd_vel_safe` 明确显示一个 `robot_control_node` subscriber
- `/vel_state` 能正常读取全零 Twist
- `/vel_state` 实测稳定发布约 `50.000 Hz`
- 4090 发布过一次全零 `/cmd_vel_safe`，PC2 返回全零 `/vel_state`
- 没有发布非零底盘速度

PC2 当前已在后台运行：

```bash
source ~/whole_body/install/setup.bash
ros2 launch robot_control robot_control_node.launch.py
```

其中 `a2_sport_udp_helper` 与 `robot_control_node` 都已启动。

## 8. 当前剩余物理阻塞

截至最终检查：

- `enp5s0 carrier=1`，PC2 正常；
- Z1 `192.168.124.110` ARP 状态为 `FAILED`，ping 不通；
- RealSense device count 为 0

因此还不能完成以下物理测试：

1. 与 Z1 `192.168.124.110` 的真实 UDP/LOWCMD 通信；
2. 双 D435 实时输入。

Z1 x86 动态库、FK/IK、`z1_ctrl` 启动和 bridge dry-run 均已通过。当前阻塞发生
在以太网 ARP 层，通常表示机械臂未上电、未连接交换机或实际地址不对。

## 9. Z1/相机接线后的下一步

机器人网络已经配置完成，无需重复运行网络脚本。先恢复 Z1 网络：

```bash
ssh 4090

ROOT=/home/robo/txc/door_act_deploy
ping -c 3 192.168.124.110
ping -c 3 192.168.124.162
```

若以后更换物理网口，可重新运行：

```bash
$ROOT/visual_whole_body/high-level/real_deploy/setup_x86_robot_network.sh <interface>
```

检查 PC2 状态：

```bash
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/robo/txc/door_act_deploy/config/cyclonedds.xml
ros2 topic echo /vel_state
```

新增加的 x86 启动脚本：

```text
high-level/real_deploy/setup_x86_robot_network.sh
high-level/real_deploy/run_x86_shadow.sh
high-level/real_deploy/run_x86_z1_ctrl.sh
high-level/real_deploy/run_x86_z1_bridge.sh
```

真实 Z1 测试必须在确认机械臂周围无碰撞风险、保险丝与急停正常后进行。当前
阶段没有执行 `backToStart()`、LOWCMD 或任何真实机械臂运动。
