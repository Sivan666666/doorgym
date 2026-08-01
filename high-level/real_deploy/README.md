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

## Z1 ACT EE/joint bridge

The Z1 bridge supports legacy 10D EE actions and new 9D joint-state actions.
It uses exactly two worker threads: one arm command thread and one UDP IO thread.

ACT action/state layout:

```text
[vx, yaw_rate, ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw, gripper]
```

Joint checkpoint layout (`--act_state_action_mode joint9`):

```text
[last_vx, last_yaw_rate, q1, q2, q3, q4, q5, q6, gripper]  # state
[vx,      yaw_rate,      q1, q2, q3, q4, q5, q6, gripper]  # action
```

In `joint9` mode the bridge does not call arm FK or IK for the ACT state/action
conversion. Measured joint feedback is published directly, and each predicted
`q1..q6` target enters the existing joint-jump guard, online quintic trajectory,
joint speed/acceleration caps, 500 Hz LOWCMD output, timeout braking, and gripper
smoothing. The legacy default remains `ee10` so old checkpoints are unchanged.

For Plücker-conditioned checkpoints, FK is still evaluated asynchronously for
one separate purpose: computing the moving wrist-camera optical-frame pose.
This FK result is never used to convert the joint9 state or joint9 command.
The default camera transforms match the simulator:

```text
front in robot base: position [0.29, 0.031, 0.165], yaw/pitch/roll [0, -45, 0] deg
wrist in link06:     position [0.093, 0.031, 0.22], yaw/pitch/roll [0, 60, 0] deg
```

The bridge publishes both poses as `[x,y,z,qx,qy,qz,qw]`; the ACT runner passes
them directly to the Plücker inputs. A missing/non-finite pose stops inference
before any new action is published.

Joint-mode bridge example:

```bash
/home/anx/door_act_deploy/visual_whole_body/high-level/real_deploy/run_z1_act_ee_bridge.sh \
  --act_state_action_mode joint9 \
  --enable_arm \
  --max_joint_speed 3.0 \
  --max_joint_acceleration 15.0 \
  --joint_trajectory_duration_s 0.02 \
  --state_tx_host 127.0.0.1 \
  --state_tx_port 15013
```

`door_act_shadow.py` accepts `--z1_state_action_mode auto` (default) and checks
that checkpoint state/action dimensions are both 9 or both 10 before starting.
For a joint checkpoint, chunk overlap blends `vx/vyaw/q1..q6` as
`0.3 old + 0.7 new`; gripper always uses the newest prediction.

### Interaction-state deployment logging

For ACT checkpoints with the interaction decoder head, every JSONL control
record now preserves both chunk-level and executed-step predictions:

- `policy_chunk_ingests[*].interaction_state_chunk`: the complete raw `H x 3`
  prediction produced by each newly ingested chunk, before the executable
  action horizon is truncated. Columns are `contact_probability`,
  `handle_progress`, and `door_progress`.
- `executed_interaction_state`: the three values aligned to the action actually
  published at this control step, together with its global action timestep,
  chunk IDs, and overlap count.

If old and new chunks overlap at one global timestep, the three interaction
values use the same `0.3 old + 0.7 new` aggregation as the motion action. The
legacy singular `policy_chunk_ingest` field remains for existing log readers;
`policy_chunk_ingests` is the lossless list when more than one result is
ingested during one control cycle. Checkpoints without an interaction head log
`null` for the executed value and remain compatible.

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

For joint9 Plücker checkpoints, the bridge publishes simulator-aligned camera
poses together with the 9D state.  Front is fixed in the ACT robot base at
`xyz=[0.29, 0.031, 0.165], ypr_deg=[0, -45, 0]`.  Wrist is computed online as
`ACT_from_arm @ SDK_FK(q,6) @ SDK_EE_to_ACT_EE @ ACT_EE_to_camera`, where the
default EE-local camera transform is
`xyz=[-0.093, 0.031, 0.22], ypr_deg=[0, 60, 0]`.  This is equivalent to the
simulator's `link06` camera offset `[0.093, 0.031, 0.22]`; the different x value
accounts for the SDK FK frame being 0.100 m ahead of Isaac Gym `link06` and the
ACT EE being another 0.086 m ahead of the SDK frame.

## Important safety boundary

The current checkpoint was trained from the B1Z1 scripted data.  The real base is
Unitree A2-W, so live motion must wait until A2-W frame transforms, Z1 mount
calibration, command limits, and a safety mux are in place.
