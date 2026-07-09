# DoorTwin Debug Notes: glass_door

Date: 2026-07-09

This note records the adaptation path for:

`high-level/data/asset/door_set/glass_door/model.urdf`

The final reusable skill lives in:

`high-level/float_ik/door_twin/examples/glass_door_push_traverse_skill.json`

The accepted setup treats the glass door as a push door, rotates the gripper by
90 degrees so it can horizontally clamp the vertical fixed handle, adds a
`handle_closeup` view for inspection, and increases hinge resistance so the door
does not swing open from a tiny touch.

## Final Command

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 1 --steps 1100 --seed 62000 \
  --door_name glass_door \
  --skill_program_json high-level/float_ik/door_twin/examples/glass_door_push_traverse_skill.json \
  --door_twin_log_dir high-level/float_ik/door_twin/experiments/runs/glass_door_resistance_check_v2 \
  --save_failed_rollouts --dump_keyframe_images \
  --door_twin_camera_views wrist,front,handle_closeup,observer_left,observer_right \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --no_ikpush_env_randomization \
  --no_collision_geom_check \
  --draw_scripted_trajectory --draw_ik_target \
  --no_show_camera_images \
  --no_enable_depth_camera_randomization
```

Final result:

```text
success: true
door_open_deg: 89.99996
body_passed: true
base_collision: false
handle_unlocked: true
final_phase: return_home
ee_handle_dist: 0.04458
```

The accepted report is:

`high-level/float_ik/door_twin/experiments/runs/glass_door_resistance_check_v2/env_0000_report.json`

## Asset Loading

URDF structure:

- Door body: `glass_door`
- Handle body: `glass_door`
- Door DOF: `wall_to_glass_door`
- Handle DOF: none
- Door hinge axis: `0 0 1`

This asset is a single-DOF door. The vertical handle is fixed geometry on the
moving door panel rather than a separate articulated lever. DoorTwin therefore
sets:

```yaml
handle_dof_name: ""
handle_unlock_angle: 0.0
```

The analyzer treats `handle_unlocked` as true for this no-handle-DOF asset, and
`handle_rotation_deg: 0.0` is expected rather than a failure.

DoorTwin metadata was added:

- `high-level/data/asset/door_set/glass_door/bounding_box.json`
- `high-level/data/asset/door_set/glass_door/handle_bounding.json`

Current metadata:

```json
{
  "handle_min": [-0.92, -0.06, 0.82],
  "handle_max": [-0.88, -0.02, 1.08],
  "goal_pos": [-0.9, -0.04, 0.95]
}
```

Config entry highlights:

```yaml
name: "glass_door"
path: "glass_door/model.urdf"
actor_yaw_offset: -1.5707963267948966
bounds_yaw_offset_override: 0.0
observer_target_local: [-0.44, -0.058, 0.95]
handle_closeup_target_local: [-0.44, -0.058, 0.95]
robot_alignment_y_offset: -0.24
door_motion_sign_multiplier: 1.0
door_body_name: "glass_door"
handle_body_name: "glass_door"
door_dof_name: "wall_to_glass_door"
handle_dof_name: ""
```

## Final Skill Behavior

The final example skill:

- Approaches from a slightly handle-side stance.
- Moves the EE to the lower half of the vertical handle.
- Applies a 90 degree roll offset so the gripper clamps the vertical handle
  horizontally.
- Closes the gripper.
- Runs a short semantic `RotateHandle` stage with near-zero angle because there
  is no real handle DOF.
- Pushes while the base moves forward at `0.2 m/s`.
- Traverses straight through at the same `0.2 m/s`.
- Holds the EE near the handle until `ReleaseAndRetract`.
- Smoothly retracts to home.

Important skill fields:

```json
{
  "name": "MoveEEToHandle",
  "pregrasp_offset": [0.16, 0.0, -0.02],
  "grasp_offset": [0.0, 0.0, -0.02],
  "handle_goal_bias_world": [-0.035, -0.02, -0.015],
  "ee_roll_offset": 1.5707963267948966,
  "duration_steps": 50
}
```

Base motion:

- Push: `1.2 m / 300 steps / 0.02 s = 0.2 m/s`
- Traverse: `0.74 m / 185 steps / 0.02 s = 0.2 m/s`

## User Feedback And Fixes

### Feedback 1: gripper should turn 90 degrees

The first visible adaptation used the usual lever-door gripper orientation. That
worked numerically, but it was not a reasonable grasp for this vertical fixed
handle. The user pointed out that the gripper should turn sideways by 90 degrees
to clamp the vertical handle.

Fix:

```json
"ee_roll_offset": 1.5707963267948966
```

The runner now applies this skill-level roll offset to the EE target
orientation. The `handle_closeup` montage confirms that the gripper approaches
the vertical handle horizontally.

### Feedback 2: door was pull-style, but should be push-style

The early setup opened in the wrong visual direction for the task. The user
pointed out that the door was behaving like a pull door and should instead be
configured as a push door.

Fix:

```yaml
door_motion_sign_multiplier: 1.0
```

In the final run, the printed raw hinge angle is negative during opening:

```text
[0540] phases=['push_door'] door_deg=[-21.4]
[0780] phases=['push_door'] door_deg=[-71.3]
[0900] phases=['traverse_door'] door_deg=[-89.9]
```

The report converts this with the door motion sign and records
`door_open_deg: 89.99996`, which is the intended push-open direction for this
asset.

### Feedback 3: door should have more opening resistance

After the skill worked, the user pointed out that the door was too light and
opened from a tiny touch. The door needed more realistic opening resistance.

First attempt:

```yaml
door_joint_friction: 0.25
door_joint_damping: 0.18
door_open_resistance: 0.30
door_open_damping: 0.08
```

Result:

```text
run: glass_door_resistance_check
success: false
door_open_deg: 75.8112
failure_stage: arm_joint_limit_or_ik_bad
```

The important lesson is that `door_open_resistance` behaves like a restoring
hinge torque in `compute_door_efforts`, so making it too large pulls the door
back and prevents the final target angle.

Final accepted resistance:

```yaml
controller_overrides:
  door_joint_friction: 0.25
  door_joint_damping: 0.18
  door_open_resistance: 0.08
  door_open_damping: 0.05
  door_auto_open_force: 0.0
```

This made the door visibly less free while still allowing the same skill to open
it to about 90 degrees and traverse.

## Repair Timeline

| Round | Change | Result | Lesson |
| --- | --- | --- | --- |
| `glass_door_round_00_initial_push_only` | Initial metadata and skill from existing push/traverse examples. | Rollout reached success, but the visual grasp/orientation was not final. | Binary success is not enough for expert-quality trajectories. |
| `glass_door_round_01_handle_bias` | Added handle goal bias to better target the vertical handle. | Rollout remained successful. | Fixed-handle glass doors need explicit grasp-point bias. |
| `glass_door_round_02_push_direction_gripper_roll` | Added 90 degree EE roll for the vertical handle. | Rollout succeeded and closeup showed horizontal gripper posture. | `handle_closeup` is essential for seeing gripper-handle quality. |
| `glass_door_round_03_push_sign_fixed` | Fixed door motion sign for push-door behavior. | Rollout succeeded; signed open angle reached 90 degrees. | Push/pull direction must be verified visually and by raw hinge sign. |
| `glass_door_round_04_final_push_gripper_roll` | Promoted the stable push-door skill. | Full open and traverse success. | Skill was stable before physics resistance tuning. |
| `glass_door_resistance_check` | Increased resistance too aggressively. | Failed final success; door opened only 75.8 degrees. | Too much restoring hinge resistance fights the task. |
| `glass_door_resistance_check_v2` | Kept hinge friction/damping high, reduced restoring resistance. | Final accepted success; door opened to 90 degrees and body passed. | Use friction/damping for "heavier door" feel; keep restoring torque modest. |

## Mistakes And User Corrections

Mistake 1: I accepted early scalar success before checking whether the grasp
orientation made physical sense. The user caught that the vertical fixed handle
should be grasped with a 90 degree gripper roll.

Mistake 2: I initially did not clearly separate "door opens" from "door opens in
the correct push direction." The user pointed out the pull-door behavior, and the
fix was the per-door `door_motion_sign_multiplier`.

Mistake 3: I made the first resistance patch too strong by increasing
`door_open_resistance` to `0.30`. That made the door heavier, but it also fought
the final open angle. The accepted version uses high hinge friction/damping and
smaller restoring resistance.

Useful user modifications/feedback:

- Turn the gripper sideways for the vertical handle.
- Make the task a push door, not a pull door.
- Add realistic resistance so the door does not swing open from tiny contact.
- Keep using `handle_closeup` and observer montages, not only rollout success.

## Reference Runs

Useful run directories:

- `glass_door_round_00_initial_push_only`: first working load and rollout.
- `glass_door_round_01_handle_bias`: handle target bias experiment.
- `glass_door_round_02_push_direction_gripper_roll`: 90 degree gripper roll test.
- `glass_door_round_03_push_sign_fixed`: push-direction sign fixed.
- `glass_door_round_04_final_push_gripper_roll`: stable skill before resistance tuning.
- `glass_door_resistance_check`: too much resistance, failed at about 75.8 degrees.
- `glass_door_resistance_check_v2`: accepted final physics and trajectory.
- `glass_door_manual_play`: manual play command using the accepted skill before
  resistance retune.

The final accepted run is:

`high-level/float_ik/door_twin/experiments/runs/glass_door_resistance_check_v2/`
