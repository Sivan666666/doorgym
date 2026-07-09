# DoorTwin Debug Notes: fire_door

Date: 2026-07-09

This note records the adaptation path for:

`high-level/data/asset/door_set/fire_door/model.urdf`

The final skill lives in:

`high-level/float_ik/door_twin/examples/fire_door_push_traverse_skill.json`

The final run loads the normal `model.urdf`, treats the door as a push door, and
produces multi-view keyframes plus a keyframe trajectory video.

## Final Command

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 1 --steps 1100 --seed 62000 \
  --door_name fire_door \
  --skill_program_json high-level/float_ik/door_twin/examples/fire_door_push_traverse_skill.json \
  --door_twin_log_dir high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset \
  --save_failed_rollouts --dump_keyframe_images \
  --door_twin_camera_views wrist,front,handle_closeup,observer_left,observer_right \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --no_ikpush_env_randomization \
  --no_collision_geom_check \
  --no_show_camera_images
```

Keyframe video:

`high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset/fire_door_wc4_like_y_offset_keyframes.mp4`

The mp4 is built from the phase montage images:

```bash
ffmpeg -y -framerate 1 -pattern_type glob \
  -i 'high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset/keyframes/env_0000/*_montage.png' \
  -vf 'fps=12,format=yuv420p' \
  high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset/fire_door_wc4_like_y_offset_keyframes.mp4
```

## Asset Loading

URDF structure:

- Door body: `fire_door`
- Handle body: `lever_handle`
- Door DOF: `frame_to_fire_door`
- Handle DOF: `door_to_lever_handle`
- Door hinge axis: `0 0 -1`
- Handle axis: `0 -1 0`

DoorTwin metadata was added:

- `high-level/data/asset/door_set/fire_door/bounding_box.json`
- `high-level/data/asset/door_set/fire_door/handle_bounding.json`

Config entry:

- Door name: `fire_door`
- Asset path: `fire_door/model.urdf`
- Actor scale: `1.0`
- Actor yaw offset: `-pi/2`
- Door motion sign multiplier: `1.0`

The actor yaw places the lever on the robot side. Because this URDF uses a
negative Z hinge axis, the default DoorTwin negative door motion sign already
corresponds to pushing the door open from the robot side.

## Metadata Choices

The lever handle is defined in the `lever_handle` link frame:

```json
{
  "handle_min": [-0.125, -0.07, -0.012],
  "handle_max": [0.01, -0.005, 0.012],
  "goal_pos": [-0.0575, -0.06, 0.0]
}
```

The closed-door handle target in asset-local coordinates is:

```text
door hinge origin       [-0.4750, 0.0000, 0.000]
handle joint origin     [ 0.9100,-0.0325, 0.945]
handle goal in link     [-0.0575,-0.0600, 0.000]
asset-local target      [ 0.3775,-0.0925, 0.945]
```

This target is stored in the config as:

```yaml
observer_target_local: [0.3775, -0.0925, 0.945]
handle_closeup_target_local: [0.3775, -0.0925, 0.945]
robot_alignment_y_offset: 0.18
```

`handle_closeup` is important for this door because the handle is lower and
shorter than the previous examples.

The first successful run aligned the robot with the handle at world `y=0.3775`.
Visual playback showed that this made the dog start too far to the right side of
the doorway. A centerline test with `robot_alignment_y_offset: 0.0` also worked,
but was more centered than the wc4 scripted stance. The final config uses
`robot_alignment_y_offset: 0.18`, matching wc4's slight handle-side offset while
still staying well inside the doorway.

## Wall Debug

Round 00 opened and traversed successfully, but the side-wall placement was
wrong:

```text
door_side_walls env=0 door=fire_door axis=y opening_width=0.155
```

The wall code classified the asset as generic numeric/PartNet-like and applied
the extra wall-only bbox yaw offset. That turned the door thickness into the
opening width.

Fix:

```yaml
bounds_yaw_offset_override: 0.0
```

After the fix:

```text
door_side_walls env=0 door=fire_door axis=y opening_width=1.070 center=(2.527,0.000)
```

The corrected montage shows the walls on the two sides of the doorway instead
of covering the door.

## Repair Timeline

| Round | Change | Result | Lesson |
| --- | --- | --- | --- |
| `round_00_push_traverse` | First push-door skill and metadata. Base was aligned with handle at `y=0.3775`. | Opened and traversed, but stance was visually too far right. Side-wall opening width was also wrong at `0.155 m`. | Success metrics are not enough; inspect observer/closeup views for stance and wall placement. |
| `round_01_wall_fixed` | Added `bounds_yaw_offset_override: 0.0`. | Opening width became `1.070 m`; rollout still succeeded. | Generated/manual assets with custom actor yaw may need wall-bbox yaw override. |
| `round_02_centered_base_grasp_down` | Set base line to doorway center `y=0.0`; lowered grasp/pregrasp z from `-0.03` to `-0.04`. | Rollout succeeded; `ee_handle_dist` improved to `0.0130 m`. | User visual feedback caught that the gripper was still high on the handle. |
| `round_03_wc4_like_y_offset` | Moved base line to wc4-like handle-side offset `y=0.18`. | Final rollout succeeded; `ee_handle_dist` improved to `0.0104 m`. | Fire door should not be exactly centerline or fully handle-aligned; wc4-style slight right offset is a better stance. |

## Mistakes And User Corrections

Mistake 1: I initially treated handle alignment as the right base alignment.

The first config used:

```yaml
robot_alignment_y_offset: 0.3775
```

That made the robot nearly line up with the handle rather than the doorway. The
door still opened, but visually the dog was too far right. The user pointed out
that the robot does not need to face the door perfectly, but should only shift a
little toward the handle side, like wc4. Final fix:

```yaml
robot_alignment_y_offset: 0.18
```

Mistake 2: I did not catch the high grasp from the first screenshots.

The first skill used z offset `-0.03`. The user noticed that the gripper was not
fully wrapping the handle and asked to lower grasp z by about `1 cm`. Final fix:

```json
"pregrasp_offset": [0.15, 0.0, -0.04],
"grasp_offset": [0.0, 0.0, -0.04]
```

This improved closest EE-handle distance from about `0.0258 m` to `0.0104 m`.

Mistake 3: I initially accepted a successful rollout before cleaning up side-wall
placement.

Round 00 had success metrics, but the side-wall opening width was only `0.155 m`.
That was a visual/scene-layout problem, not a trajectory success problem. The fix
was:

```yaml
bounds_yaw_offset_override: 0.0
```

After that, wall opening width was `1.070 m`.

The useful user modifications were therefore:

- make the base stance visually reasonable, not just handle-centered,
- lower grasp z by `1 cm`,
- compare stance against wc4 instead of only checking binary rollout success,
- keep using multi-view keyframes, especially `handle_closeup`, for qualitative
  checks.

## Final Skill Behavior

The final skill uses the same base-speed convention as the wc4 and generated
door examples:

- Approach at `0.2 m/s`.
- Move EE to the lever with z offset `-0.04`.
- Close the gripper.
- Rotate/unlock the lever.
- Push while the base moves forward at `0.2 m/s`.
- Traverse straight through at the same `0.2 m/s`.
- Keep the EE command held near the handle during traverse.
- Smoothly return home in `ReleaseAndRetract`.

Base motion:

- Push: `1.2 m / 300 steps / 0.02 s = 0.2 m/s`
- Traverse: `0.74 m / 185 steps / 0.02 s = 0.2 m/s`

## Final Result

Final run directory:

`high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset`

Result:

```text
success_rate: 1.0
door_open_deg: 90.0
handle_rotation_deg: 35.5
ee_handle_dist: 0.0104
body_passed: true
base_collision: false
camera_available: true
keyframe_artifacts: 96
```

The final user-feedback patch made two changes:

- moved the robot spawn/approach line from handle alignment to a wc4-like
  handle-side offset of `0.18 m`,
- lowered `MoveEEToHandle.pregrasp_offset[2]` and `grasp_offset[2]` from `-0.03`
  to `-0.04`.

After the z adjustment, the closest EE-handle distance improved from about
`0.0258 m` to `0.0104 m`.

The console prints raw door angle as negative during push/traverse, for example
`door_deg=[-90.0]`. With this door's configured sign, that is the successful
push-open direction.

The max EE tracking error grows during traverse because the current DoorTwin
skill intentionally holds the EE command near the push/handle target while the
base continues through the doorway. This matches the current data-recording
convention and avoids a new EE command jump before `ReleaseAndRetract`.

## Experiment Artifacts

Repair history:

`high-level/float_ik/door_twin/experiments/runs/fire_door_repair_history.md`

Final run README:

`high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset/README.md`

Final keyframe video:

`high-level/float_ik/door_twin/experiments/runs/fire_door_round_03_wc4_like_y_offset/fire_door_wc4_like_y_offset_keyframes.mp4`
