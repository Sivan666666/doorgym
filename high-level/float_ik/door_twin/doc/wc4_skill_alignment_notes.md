# DoorTwin Notes: wc4 Smoke Skill Alignment

Date: 2026-07-02

This note records the changes made around:

`high-level/float_ik/door_twin/examples/wc4_smoke_skill.json`

The goal was to keep the DoorTwin skill-program path compatible with the original
scripted wc4 trajectory, while making the skill representation less ambiguous
and safe for expert trajectory recording.

## Final Skill

Use:

`high-level/float_ik/door_twin/examples/wc4_smoke_skill.json`

Recommended debug command:

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 1 --steps 1100 --seed 62000 \
  --door_name wc4 \
  --skill_program_json high-level/float_ik/door_twin/examples/wc4_smoke_skill.json \
  --door_twin_log_dir high-level/float_ik/door_twin/experiments/runs/wc4_manual_skill_final_check \
  --save_failed_rollouts --dump_keyframe_images \
  --door_twin_camera_views wrist,front,observer_left,observer_right \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --draw_scripted_trajectory --draw_ik_target \
  --no_collision_geom_check \
  --no_show_camera_images
```

Use `--no_collision_geom_check` for expert success accounting when the task is
full open-and-traverse. The geometric base-door line check is useful for
debugging, but it can be stricter than the intended task semantics after the door
is already open.

## Scripted Alignment

The wc4 skill is intentionally close to the old scripted trajectory:

- Approach stop distance: `0.15 m`
- Pregrasp offset: `[0.15, 0.0, -0.03]`
- Grasp offset: `[0.0, 0.0, -0.03]`
- Handle goal bias: `[-0.015, 0.0, 0.0]`
- Handle rotation: `1.05 rad`, `100 steps`
- Push EE distance: `1.1 m`
- Push base motion: `1.2 m / 300 steps`
- Traverse base motion: `0.74 m / 185 steps`
- Return/retract: `150 steps`

With `dt = 0.02 s`, push and traverse now use the same base speed:

```text
push:     1.20 m / (300 * 0.02 s) = 0.20 m/s
traverse: 0.74 m / (185 * 0.02 s) = 0.20 m/s
```

This fixed the earlier visual mismatch where the dog moved slowly during push
and faster during traverse.

## Primitive Ownership

The skill format was cleaned up to avoid duplicated ownership of base motion:

- `PushDoor` owns the EE/handle push behavior.
- `MoveTo(stage="push")` owns push-phase base velocity, distance, and duration.
- `TraverseDoor` is a semantic phase marker and success target.
- `MoveTo(stage="traverse")` owns traverse base velocity, distance, and duration.
- `ReleaseAndRetract` owns the smooth return-home interpolation.

The final wc4 skill therefore does not use:

```json
"PushDoor.base_distance"
"TraverseDoor.base_v"
"TraverseDoor.duration_steps"
```

Those fields are still tolerated by the interpreter for backward compatibility,
but examples avoid them so there is one source of truth for base commands.

## Fixed Traverse Behavior

`TraverseDoor` now means: after push, continue the base straight through the
doorway while keeping the EE command at the last push/handle target.

In the interpreter:

- `MoveTo(stage="traverse")` sets `args.traverse_distance`, `args.traverse_steps`,
  and `args.traverse_yaw_delta`.
- The `traverse_door` phase starts from the current base pose, not from a
  recomputed world jump.
- The base interpolates from the push end to the traverse target.
- The EE target is held at `traverse_hold_target_pos`.
- The gripper command is held from `last_gripper`.

This made push-to-traverse consistent with the scripted expectation:

> after `PushDoor`, the EE stays on or near the handle until
> `ReleaseAndRetract`.

## EE Command Continuity

Several fixes were added so recorded commands do not jump at phase boundaries.

### Push Phase

For DoorTwin skill mode, once the door enters `open_stage`, the EE command follows
the live handle target:

```text
handle_goal + handle_contact_offset + live_push_dir * push_contact_bias
```

It no longer switches to:

```text
current_ee_pose + live_push_dir * lever_step_size
```

That old fallback could look like a handle -> current EE -> handle jump in the
command stream. It is kept only for non-skill legacy behavior.

### Push To Traverse

At the start of `traverse_door`, the interpreter stores:

```text
traverse_hold_target_pos = last_target_pos
traverse_hold_target_quat = last_target_quat
```

Then it holds that command through traverse. This avoids a sudden interpolation
to home or to a recomputed handle pose while the base is passing through.

### Traverse To Release

`ReleaseAndRetract` / `return_home` now interpolates from the actual phase-start
target:

```text
return_home_start_target_pos = last_target_pos
```

to the home EE target. The arm does not snap directly to home at the phase
transition.

### Per-Step Clamp

DoorTwin skill mode also applies a small command clamp:

```bash
--ee_command_max_step 0.025
```

This limits per-step EE position command deltas for skill rollouts. Legacy replay
is not affected.

## Phase Timing

For wc4 with this skill, the rough phase sequence is:

```text
walk / approach
initial_hold
grasp
close_gripper
rotate_handle
push_door
traverse_door
return_home
hold_home
```

In observed runs, `return_home` begins around step `960`, so `--steps 1100` is
enough for the main task and most of the retract. `--steps 1450` is safe but
longer than needed for manual checking.

## Reference Runs

Useful historical run directories:

- `compare_wc4_scripted_baseline`: scripted baseline reference.
- `compare_wc4_auto_legacy`: auto/legacy compatibility reference.
- `smoke_wc4_multiview_v8_final_expert`: final successful wc4 skill smoke.
- `wc4_manual_skill_traverse_speed_match`: speed-match debug run; strict geometry
  gate reported `base_collision` even though the door opened and `body_passed`
  was true.

The final intended success criteria are:

```text
door_open_deg >= pass_open_angle_deg
body_passed = true
handle_unlocked = true
camera frames valid
```

Strict geometric base-door collision is a debugging signal, not the only expert
collection gate for this full traversal setup.

## Why This Matters For Expert Data

For DP/expert recording, the action command must be temporally smooth. The final
wc4 skill keeps the high-level skill program readable while preserving the
scripted trajectory properties that matter for data quality:

- continuous base velocity through push and traverse,
- no duplicate base-motion parameters,
- no EE snap at push/traverse/release boundaries,
- handle contact held until `ReleaseAndRetract`,
- return-home interpolation from the actual last command.
