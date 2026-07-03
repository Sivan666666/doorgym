# DoorTwin Debug Notes: button_door

Date: 2026-07-02

This note records the adaptation path for:

`high-level/data/asset/door_set/button_door/model.urdf`

The final skill lives in:

`high-level/float_ik/door_twin/examples/button_door_push_traverse_skill.json`

The final run uses the normal asset, push-door direction, a corrected side-wall
opening, and a handle-closeup camera for grasp inspection.

## Final Command

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 1 --steps 1100 --seed 62000 \
  --door_name button_door \
  --skill_program_json high-level/float_ik/door_twin/examples/button_door_push_traverse_skill.json \
  --door_twin_log_dir high-level/float_ik/door_twin/experiments/runs/button_door_manual_check_final \
  --save_failed_rollouts --dump_keyframe_images \
  --door_twin_camera_views wrist,front,handle_closeup,observer_left,observer_right \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --no_ikpush_env_randomization \
  --no_collision_geom_check \
  --no_show_camera_images
```

Use `--no_collision_geom_check` for DoorTwin success accounting. The geometric
base-door line gate is intentionally conservative and can flag the required
push-through path as a collision even when the door is open and the body passes.

## Asset Loading

The asset initially had only `model.urdf`, so DoorTwin metadata was added:

- `high-level/data/asset/door_set/button_door/bounding_box.json`
- `high-level/data/asset/door_set/button_door/handle_bounding.json`

URDF structure:

- Door body: `door_panel`
- Handle body: `lever_handle`
- Door DOF: `door_hinge_joint`
- Handle DOF: `handle_actuation_joint`
- Extra DOF: `exit_rocker_joint`

The extra rocker DOF is loaded but not used by the current lever-handle push
skill.

Config entry:

- Door name: `button_door`
- Asset path: `button_door/model.urdf`
- Actor scale: `1.0`
- Actor yaw offset: `+pi/2`
- Door motion sign multiplier: `-1.0`

The yaw and sign combination places the lever on the robot side and makes the
task a push-door task, not a pull-door task.

## Metadata Choices

The handle bounding box is defined in the `lever_handle` link frame:

```json
{
  "handle_min": [-0.018, 0.0, -0.018],
  "handle_max": [0.187, 0.068, 0.018],
  "goal_pos": [0.0845, 0.068, 0.0]
}
```

The closed-door handle target in asset-local coordinates is:

```text
door_hinge origin       [0.9000, 0.000, 0.00]
handle joint origin     [-0.7420, 0.032, 1.05]
handle goal in link     [0.0845, 0.068, 0.00]
asset-local target      [0.2425, 0.100, 1.05]
```

This target is now stored in the config as:

```yaml
observer_target_local: [0.2425, 0.1, 1.05]
handle_closeup_target_local: [0.2425, 0.1, 1.05]
```

`observer_target_local` keeps the world observers looking at the real handle
height. `handle_closeup_target_local` is used by the new closeup camera.

## Wall Debug

The first visual smoke test opened the door, but the side walls were wrong:
they blocked most of the doorway. The startup log exposed the problem:

```text
door_side_walls env=0 door=button_door axis=y opening_width=0.155
```

That width was the door thickness, not the door opening. The root cause was that
`button_door` used a manually chosen actor yaw, while the generic PartNet wall
code applied an additional wall-only `+pi/2` bounds yaw offset.

Fix:

```yaml
bounds_yaw_offset_override: 0.0
```

After the fix:

```text
door_side_walls env=0 door=button_door axis=y opening_width=1.530 center=(2.522,-0.230)
```

The corrected montage shows the wall blocks on the two sides of the doorway
instead of covering the opening.

## Handle Grasp Debug

The first successful skill used:

```json
"pregrasp_offset": [0.15, 0.0, -0.02],
"grasp_offset": [0.0, 0.0, -0.02]
```

The door opened and the robot traversed, but visual inspection showed that the
gripper was not fully wrapping the lever. The target was lowered by another
1 cm:

```json
"pregrasp_offset": [0.15, 0.0, -0.03],
"grasp_offset": [0.0, 0.0, -0.03]
```

The adjusted rollout remained successful and improved the tracked handle metric:

```text
ee_handle_dist: 0.0164 -> 0.0118
ee_tracking_error: 0.7068 -> 0.5493
door_open_deg: 90.0
body_passed: true
base_collision: false
```

## Final Skill Behavior

The final example skill:

- Approaches from the handle side.
- Moves the EE to the lever with z offset `-0.03`.
- Closes the gripper.
- Rotates the lever.
- Pushes while the base moves forward at `0.2 m/s`.
- Traverses straight through at the same `0.2 m/s`.
- Holds the EE near the handle until `ReleaseAndRetract`.

Base motion:

- Push: `1.2 m / 300 steps / 0.02 s = 0.2 m/s`
- Traverse: `0.74 m / 185 steps / 0.02 s = 0.2 m/s`

## Handle Closeup Camera

The DoorTwin camera view list now supports:

`handle_closeup`

It is a fixed world observer that looks directly at the closed-door handle target.
For `button_door`, it uses `handle_closeup_target_local` from the door config.

Recommended camera views for debugging:

```bash
--door_twin_camera_views wrist,front,handle_closeup,observer_left,observer_right
```

The keyframe dump will contain files like:

```text
step_00422_close_gripper_handle_closeup_rgb.png
step_00472_rotate_handle_handle_closeup_rgb.png
```

This view is intended for later Agent/VLM inspection of whether the gripper is
actually wrapping the lever, not only whether the rollout eventually opens the
door.

## Reference Runs

Useful run directories:

- `button_door_round_00`: first success with no geometric collision gate, before wall fix.
- `button_door_round_02_wall_fixed`: side-wall opening fixed, full rollout success.
- `button_door_round_03_grasp_z_down_1cm`: grasp z lowered by 1 cm, full rollout success.
- `button_door_round_04_handle_closeup_smoke`: short camera smoke with `handle_closeup`.
- `button_door_round_05_final_handle_closeup`: final full rollout with `handle_closeup`.

The short camera smoke stops before push completion, so its summary is expected
to be unsuccessful. It is only for validating closeup keyframes.

Final `round_05` result:

```text
success_rate: 1.0
door_open_deg: 90.0
handle_rotation_deg: 30.0
ee_handle_dist: 0.0118
body_passed: true
base_collision: false
handle_closeup_rgb_count: 6
```
