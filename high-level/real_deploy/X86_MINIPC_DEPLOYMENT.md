# Door ACT：NX 到 x86 小主机迁移环境

这份文档记录 2026-07-04 正在工作的 NX / PC2 环境，并说明如何把 NX
职责迁移到一台 x86_64 小主机。完整版本清单和 CycloneDDS 源码快照位于：

```text
high-level/real_deploy/environment_snapshots/nx_20260704/
```

历史实机源码/build/install 的独立备份位于：

```text
/home/sivan/whole_body/backups/nx_pc2_20260704_183407/
```

## 1. 当前实机拓扑

| 设备 | 地址 | 职责 |
| --- | --- | --- |
| NX | Wi-Fi DHCP 当前为 `192.168.1.173` | ACT、双 D435、Z1 bridge |
| NX 机器人网口 | `192.168.124.25/24`，接口 `eno1` | Z1、PC2、ROS 2 DDS |
| Z1 | `192.168.124.110` | 机械臂控制器 |
| PC2 | `192.168.124.162` | A2 helper、`robot_control_node` |

x86 小主机替换 NX 后，应接入同一个交换机，并把机器人网口设为
`192.168.124.25/24`。该网口不需要默认网关；互联网/Wi-Fi 使用另一块网卡。

## 2. NX 当前软件基线

- Jetson Orin NX，Ubuntu 22.04.5，Jetson Linux R36.5
- CUDA 12.6
- system Python 3.10.12；实际部署没有使用 conda
- torch 2.11.0、torchvision 0.26.0
- numpy 1.24.0、OpenCV 4.11.0
- pyrealsense2 2.57.7.10387
- librealsense native library 2.58
- ROS 2 Humble
- `rmw_cyclonedds_cpp`

完整包清单：

```text
environment_snapshots/nx_20260704/environment/python3_pip_freeze.txt
environment_snapshots/nx_20260704/environment/dpkg_versions.tsv
environment_snapshots/nx_20260704/environment/apt_manual.txt
environment_snapshots/nx_20260704/environment/ros2_packages.txt
environment_snapshots/nx_20260704/environment/system_summary.txt
```

NX 上遗留的 `/home/anx/miniforge3` 不是当前运行环境，其 `conda` 启动器仍指向
已经不存在的 `/home/nx/miniforge3/bin/python`。迁移时不要以它为基准，也不要
为了“满足 LeRobot 3.11”改变当前已经验证过的 Python 3.10 部署路径。

## 3. x86 基础环境

推荐先安装 Ubuntu 22.04 x86_64、ROS 2 Humble 和 Python 3.10。若小主机带
NVIDIA GPU，再安装与该机器驱动匹配的 CUDA/PyTorch；NX 的 aarch64 CUDA
wheel 不能复制到 x86。

基础 ROS/DDS 包：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git python3-pip python3-venv python3-colcon-common-extensions \
  ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp
```

建议创建能看见 ROS system packages 的 Python 3.10 venv：

```bash
python3 -m venv --system-site-packages ~/venvs/door_act
source ~/venvs/door_act/bin/activate
python -m pip install --upgrade pip
```

不要直接对完整 `pip freeze` 执行一次性安装：其中包含 Jetson wheel 和 apt
提供的包。应先安装 x86 对应的 PyTorch，再按
`environment/python3_pip_freeze.txt` 核对其余版本，最后安装本仓库 LeRobot：

```bash
python -m pip install -e high-level/lerobot
```

## 4. CycloneDDS 配置

NX 当前实际使用的文件：

```text
environment_snapshots/nx_20260704/nx_dds_backup_20260704/cyclonedds.xml
```

关键配置是：

```xml
<NetworkInterface name="eno1" multicast="true"/>
<Peer address="192.168.124.162"/>
```

复制到新主机后，必须将 `eno1` 改成新主机连接 124 网段的真实接口名：

```bash
ip -br addr
```

然后配置：

```bash
mkdir -p ~/door_act_deploy/config
cp high-level/real_deploy/environment_snapshots/nx_20260704/nx_dds_backup_20260704/cyclonedds.xml \
  ~/door_act_deploy/config/cyclonedds.xml

export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=$HOME/door_act_deploy/config/cyclonedds.xml
source /opt/ros/humble/setup.bash
```

NX 的 UDP 缓冲区配置也已保存：

```text
environment_snapshots/nx_20260704/nx_dds_backup_20260704/sysctl/
```

安装到新主机：

```bash
sudo cp high-level/real_deploy/environment_snapshots/nx_20260704/nx_dds_backup_20260704/sysctl/*.conf \
  /etc/sysctl.d/
sudo sysctl --system
```

通常 x86 上直接使用 Ubuntu/ROS 的 `ros-humble-rmw-cyclonedds-cpp` 即可。如果
必须完全复现 NX 的源码版本，解压：

```bash
mkdir -p ~/cyclonedds_ws
tar -xzf \
  high-level/real_deploy/environment_snapshots/nx_20260704/nx_dds_backup_20260704/cyclonedds_ws_src.tar.gz \
  -C ~/cyclonedds_ws
```

保存的源码提交为：

- CycloneDDS：`5041f3560c088c99e5088b2b8520b69169621196`
- rmw_cyclonedds：`fa8831b9f331d669c3ea408fd00606a223c6bbd9`
- unitree_ros2：`3ff13ea08ec619496c2651fd21b172f7958dd5a5`

必须在 x86 上重新 `colcon build`，不能使用 NX 备份里的 aarch64 `.so`。

## 5. RealSense

双 D435 必须在新主机上重新安装 x86_64 librealsense/pyrealsense2，并验证：

```bash
rs-enumerate-devices
python3 -c 'import pyrealsense2 as rs; print(rs.context().query_devices())'
```

部署脚本依赖两台相机序列号区分 wrist/front。换机器不会改变相机序列号，但
USB 端口和设备权限可能改变。先运行：

```bash
high-level/real_deploy/run_nx_shadow.sh --list_realsense
```

## 6. Z1 SDK 与 controller

备份同时保留了 NX 的 aarch64 和 SDK 自带的 x86 库。新主机只能加载：

```text
libZ1_SDK_x86_64.so
```

SDK 自带的 x86 Python 扩展是 CPython 3.8，不能由 Python 3.10 直接导入。
因此必须在新主机上用 Python 3.10 和 pybind11 重编译：

```bash
cd ~/door_act_deploy/z1_sdk
python3 -m pip install pybind11
mkdir -p build && cd build
cmake .. \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build . -j"$(nproc)"
```

编译后应生成类似：

```text
unitree_arm_interface.cpython-310-x86_64-linux-gnu.so
```

Z1 controller 也需要在 x86 上重新编译：

```bash
cd ~/door_act_deploy/z1_controller
mkdir -p build && cd build
cmake ..
cmake --build . -j"$(nproc)"
```

确认 `z1_controller/config/config.xml` 使用：

```text
192.168.124.110
```

## 7. PC2 robot_control

PC2 不随 NX 更换。它继续运行：

```bash
source ~/whole_body/install/setup.bash
ros2 launch robot_control robot_control_node.launch.py
```

PC2 当前实际 source/build/install 和环境清单已保存在：

```text
/home/sivan/whole_body/backups/nx_pc2_20260704_183407/pc2/
```

不要丢掉 PC2 launch 中给 `a2_sport_udp_helper` 添加的
`LD_LIBRARY_PATH=/usr/local/lib:...`，否则可能加载 ROS 自带的错误 `libddsc`。

## 8. 通信验证顺序

先验证网络：

```bash
ping -c 3 192.168.124.110
ping -c 3 192.168.124.162
```

再验证 DDS：

```bash
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=$HOME/door_act_deploy/config/cyclonedds.xml

ros2 topic list
ros2 topic echo /vel_state
```

最后只发布零速度检查 PC2 是否收到：

```bash
ros2 topic pub --once /cmd_vel_safe geometry_msgs/msg/Twist \
  '{linear: {x: 0.0}, angular: {z: 0.0}}'
```

只有 ROS、相机、Z1 状态和 dry-run 全部通过后，才允许启用真实机械臂和底盘
命令。

## 9. 备份边界

当前归档包含源码、编译产物、DDS 配置、sysctl 和环境版本清单，但不是 NX
NVMe 的逐块镜像。要做到裸机一键恢复，还应额外制作：

- checkpoint/data/log 的独立归档；
- x86 主机安装完成后的系统镜像；
- 新主机网络接口名、相机序列号和 CUDA/PyTorch 实测记录。

## 10. 2026-07-05 RTX 4090 实际迁移

环境已经迁移到 `robo@192.168.1.124`，本机 SSH 别名为：

```bash
ssh 4090
```

4090 上的部署根目录为：

```text
/home/robo/txc/door_act_deploy
```

实际安装版本、编译命令、ACT 速度、Z1 dry-run、DDS 测试结果和当前接线状态见：

```text
high-level/real_deploy/X86_4090_DEPLOYMENT_STATUS.md
```
