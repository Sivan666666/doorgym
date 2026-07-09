# Door ACT real-deploy notes

NX 环境快照、CycloneDDS 备份以及迁移到 x86 小主机的完整说明见：

```text
high-level/real_deploy/X86_MINIPC_DEPLOYMENT.md
high-level/real_deploy/X86_4090_DEPLOYMENT_STATUS.md
high-level/real_deploy/environment_snapshots/x86_4090_20260705/
```

Target tested on NX（2026-07-04 Wi-Fi DHCP 地址为
`anx@192.168.1.173`，机器人有线网地址固定为 `192.168.124.25/24`）：

- Jetson Orin NX Super, Ubuntu 22.04 / L4T R36.5, CUDA 12.6
- Python 3.10.12 with Jetson CUDA PyTorch 2.11.0
- RealSense SDK installed, but D435 cameras must be connected before serial binding

This deployment intentionally uses Python 3.10.  The local LeRobot source in this
repository declares `requires-python >=3.10`; using Python 3.11 on Jetson would
lose the currently working NVIDIA CUDA PyTorch stack.

## Shadow inference

Dummy safety test:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_nx_shadow.sh \
  --camera_mode dummy \
  --steps 100 \
  --log_path /home/anx/door_act_deploy/logs/dummy_shadow.jsonl
```

List connected RealSense cameras:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_nx_shadow.sh \
  --list_realsense
```

Run two-D435 shadow mode after assigning serials:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_nx_shadow.sh \
  --camera_mode realsense \
  --wrist_serial WRIST_D435_SERIAL \
  --front_serial FRONT_D435_SERIAL \
  --steps 1500 \
  --log_path /home/anx/door_act_deploy/logs/realsense_shadow.jsonl
```

Single-D435 snapshot test.  This reads one asynchronous RealSense depth stream at
640x480@30Hz, preprocesses it into the ACT depth input format, duplicates it to
wrist/front for testing, saves PNGs, and exits without loading the ACT model:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_nx_shadow.sh \
  --camera_mode realsense \
  --wrist_serial D435_SERIAL \
  --allow_single_realsense_duplicate \
  --camera_warmup_frames 15 \
  --depth_snapshot_only \
  --save_depth_debug_dir /home/anx/door_act_deploy/logs/d435_depth_snapshot
```

Single-D435 ACT shadow test:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_nx_shadow.sh \
  --camera_mode realsense \
  --wrist_serial D435_SERIAL \
  --allow_single_realsense_duplicate \
  --steps 100 \
  --camera_warmup_frames 15 \
  --save_depth_debug_dir /home/anx/door_act_deploy/logs/d435_depth_debug \
  --log_path /home/anx/door_act_deploy/logs/d435_shadow.jsonl
```

The shadow runner does not publish `/cmd_vel`, does not instantiate Z1 control,
and does not call any motion command.  It only logs observations, ACT actions,
latency, and camera timestamp metadata.

## Runtime paths

- Deploy root: `/home/anx/door_act_deploy`
- Model: `/home/anx/door_act_deploy/checkpoints/door_act_model_latest`
- Logs: `/home/anx/door_act_deploy/logs`
- Z1 SDK Python module: `/home/anx/door_act_deploy/z1_sdk/lib/unitree_arm_interface.cpython-310-aarch64-linux-gnu.so`

## Z1 ACT EE bridge

The Z1 bridge converts Door-ACT 10D actions to Z1 end-effector LOWCMD control.
It uses exactly two worker threads: one arm command thread and one UDP IO thread.

ACT action/state layout:

```text
[vx, yaw_rate, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]
```

Dry-run first; this does not instantiate or command the Z1:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_z1_act_ee_bridge.sh \
  --action_udp_port 15011 \
  --vel_state_udp_port 15012 \
  --log_path /home/anx/door_act_deploy/logs/z1_bridge_dryrun.jsonl
```

After Z1 power/network/safety are verified, enable real LOWCMD:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_z1_act_ee_bridge.sh \
  --enable_arm \
  --action_udp_port 15011 \
  --vel_state_udp_port 15012 \
  --max_joint_speed 0.4 \
  --max_gripper_speed 0.5 \
  --log_path /home/anx/door_act_deploy/logs/z1_bridge_live.jsonl
```

Default UDP inputs:

- ACT action: `0.0.0.0:15011`
- i7 robot-dog `vel_state`: `0.0.0.0:15012`

Accepted ACT action packet examples:

```json
{"action": [0.12, -0.03, 0.42, -0.08, 0.31, 0, 0, 0, 1, -0.7]}
```

```json
{"vx": 0.12, "yaw_rate": -0.03, "ee_pos": [0.42, -0.08, 0.31], "ee_quat_xyzw": [0, 0, 0, 1], "gripper": -0.7}
```

Accepted `vel_state` packet examples:

```json
{"vel_state": [0.11, -0.025]}
```

```json
{"vx": 0.11, "yaw_rate": -0.025}
```

The bridge can also parse little-endian float32 UDP payloads: 10 floats for ACT
action and 2 floats for `vel_state`.

If `vel_state` is stale for more than `--vel_timeout_s` (default 0.5 s), the
published ACT state uses zero base velocity.  If ACT action is stale for more
than `--command_timeout_s` (default 0.25 s), the arm command thread freezes at
the last safe joint target.

Use `--arm_from_act_xyz x y z --arm_from_act_rpy r p y` after measuring the real
A2-W/Z1 mount transform.  Until then, the bridge assumes the ACT base frame is
the Z1 arm base frame.

## Important safety boundary

The current checkpoint was trained from the B1Z1 scripted data.  The real base is
Unitree A2-W, so live motion must wait until A2-W frame transforms, Z1 mount
calibration, command limits, and a safety mux are in place.
