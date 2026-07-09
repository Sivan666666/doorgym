# Agent DoorTwin

Date: 2026-07-02

This directory contains the first working version of the DoorTwin loop for A2W +
Z1 push-door tasks. The goal is to take a new generated articulated door asset,
extract or provide its structure, instantiate a skill program, run Isaac Gym
rollouts, analyze failures from logs and images, patch the program, and finally
save successful expert trajectories.

Current focus:

- Hinged push doors with a lever handle.
- Full task: approach, grasp, unlock, push open, traverse through, release.
- Skill-program control rather than direct Python code generation.
- Logs, reports, and keyframe images as the feedback surface for an Agent/VLM.

## Directory Layout

```text
door_twin/
  agent_door_twin.md
  spec.py
  skill.py
  analyzer.py
  optimizer.py
  orchestrator.py
  examples/
  doc/
  experiments/
```

## Main Components

### `spec.py`

`spec.py` defines `DoorTwinSpec`, a stable JSON-serializable description of a
door asset.

It records:

- asset path and root,
- bounding box and handle bounding metadata,
- door body and handle body names,
- door DOF and handle DOF names,
- actor scale, yaw, offsets, and robot alignment hints,
- door motion sign,
- whether the asset is supported by the current DoorTwin assumptions.

This is the structural interface between a generated door asset and the skill
interpreter. The Agent should prefer metadata/URDF-derived coordinates over VLM
guesses.

### `skill.py`

`skill.py` defines the skill-program schema:

- `MoveTo`
- `ApproachDoor`
- `MoveEEToHandle`
- `CloseGripper`
- `RotateHandle`
- `PushDoor`
- `TraverseDoor`
- `ReleaseAndRetract`

The current convention is:

- `PushDoor` owns EE/handle push behavior.
- `MoveTo(stage="push")` owns push-phase base motion.
- `TraverseDoor` is a semantic marker and success target.
- `MoveTo(stage="traverse")` owns traverse base motion.
- `ReleaseAndRetract` owns the smooth return-home interpolation.

The examples avoid duplicated fields such as `PushDoor.base_distance` and
`TraverseDoor.base_v`, so the base command has one source of truth.

`ProgramPatch` restricts what the optimizer/Agent may edit. It clamps changes to
an allowlist of continuous parameters, such as grasp offsets, rotate angle,
push distance, and base distances. The first version intentionally does not let
the VLM rewrite arbitrary Python.

### `analyzer.py`

`analyzer.py` collects rollout diagnostics into `RolloutReport` and
`rollout_summary.json`.

Important metrics include:

- `success`
- `failure_stage`
- `door_open_deg`
- `handle_rotation_deg`
- `ee_handle_dist`
- `ee_tracking_error`
- `base_collision`
- `body_passed`
- `camera_available`
- `handle_unlocked`
- keyframe artifact paths

Failure categories include:

- `grasp_miss`
- `handle_not_unlocked`
- `contact_lost`
- `door_push_insufficient`
- `arm_joint_limit_or_ik_bad`
- `base_collision`
- `body_blocked`
- `camera_unavailable`
- `asset_invalid`
- `timeout`

These reports are the main input for automatic repair.

### `optimizer.py`

`optimizer.py` currently provides small deterministic and stochastic parameter
search utilities. It is meant to fine-tune continuous skill parameters after a
coarse Agent/VLM patch.

The intended use is:

1. Run a batch of rollouts.
2. Classify failures from reports.
3. Generate candidate patches.
4. Search locally over safe parameters.
5. Keep the best skill program.

### `orchestrator.py`

`orchestrator.py` is the glue layer for reading rollout summaries and writing the
next candidate skill program. It currently supports simple heuristic repair via
the patch machinery.

Future versions should connect this layer to a VLM observer that reads:

- `rollout_summary.json`,
- per-env reports,
- `handle_closeup` and observer keyframes,
- optional trace summaries.

The VLM should return a JSON patch plus diagnostic text, not arbitrary code.

## Main Runner Integration

The Isaac Gym runner is:

`high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py`

DoorTwin additions include:

- `--skill_program_json`
- `--door_twin_log_dir`
- `--save_failed_rollouts`
- `--dump_keyframe_images`
- `--door_twin_camera_views`
- `--door_asset_path_override`
- `--no_collision_geom_check`
- `--ee_command_max_step`

If no skill program is passed, the old scripted behavior remains available. If a
skill program is passed, the runner interprets the skill primitives and writes
DoorTwin reports/artifacts.

## Camera Views

Current DoorTwin keyframe views:

- `wrist`
- `front`
- `front_left`
- `front_right`
- `observer_left`
- `observer_right`
- `overhead`
- `handle_closeup`

`handle_closeup` was added specifically for Agent/VLM inspection of gripper-handle
contact. For assets with a configured `handle_closeup_target_local`, it looks at
the closed-door handle target instead of relying on a distant observer.

Recommended debug views:

```bash
--door_twin_camera_views wrist,front,handle_closeup,observer_left,observer_right
```

## Examples

Reusable skill JSON files live in `examples/`.

### `wc4_smoke_skill.json`

Reference wc4 skill aligned with the original scripted trajectory.

Important properties:

- approach stop distance `0.15 m`,
- wc4-style grasp offsets,
- push and traverse both at `0.2 m/s`,
- EE target held near the handle during traverse,
- smooth `ReleaseAndRetract`.

See:

`doc/wc4_skill_alignment_notes.md`

### `99692809960048_traverse_scripted_speed_skill.json`

Generated-door example adapted from the wc4/scripted-speed behavior.

Important properties:

- uses the repaired asset override
  `99692809960048/mobility_no_frame_collision.urdf`,
- push and traverse base speeds match,
- full open-and-traverse success with geometric collision gate disabled.

See:

`doc/99692809960048_debug_to_final.md`

### `button_door_push_traverse_skill.json`

New button/lever style door example.

Important properties:

- normal asset path `button_door/model.urdf`,
- push-door orientation and door motion sign configured in the door cfg,
- corrected side-wall opening,
- grasp z lowered by `1 cm`,
- `handle_closeup` camera target configured for visual grasp checks.

See:

`doc/button_door_debug_to_final.md`

### `fire_door_push_traverse_skill.json`

Gray metal fire-door example adapted as a push door.

Important properties:

- normal asset path `fire_door/model.urdf`,
- actor yaw places the short lever on the robot side,
- default negative door motion sign is kept because the hinge axis is `0 0 -1`,
- side-wall bbox yaw override fixes the doorway opening,
- final base stance uses a wc4-like handle-side offset of `0.18 m`,
- final grasp/pregrasp z offset is `-0.04`,
- multi-view keyframes are converted into a keyframe trajectory mp4.

See:

`doc/fire_door_debug_to_video.md`

Repair history:

`experiments/runs/fire_door_repair_history.md`

### `glass_door_push_traverse_skill.json`

Single-DOF glass-door example adapted as a push door.

Important properties:

- normal asset path `glass_door/model.urdf`,
- no separate handle DOF; the vertical handle is fixed on the moving panel,
- `ee_roll_offset = pi/2` turns the gripper sideways for the vertical handle,
- door motion sign is configured for push-open behavior,
- `handle_closeup` target is configured for visual grasp checks,
- final config increases hinge resistance so the door does not open from a tiny
  touch while still reaching about 90 degrees.

See:

`doc/glass_door_debug_to_final.md`

## Debug Documents

Detailed per-door notes live in `doc/`.

Currently documented:

- `wc4_skill_alignment_notes.md`
- `99692809960048_debug_to_final.md`
- `button_door_debug_to_final.md`
- `fire_door_debug_to_video.md`
- `glass_door_debug_to_final.md`

These files record the actual debugging path: what failed, what was changed, and
which command/run confirmed the final behavior. New generated doors should get a
similar note once a stable skill is found.

## Experiments

Local rollout outputs live under:

`experiments/runs/`

Typical contents:

- `rollout_summary.json`
- `env_0000_report.json`
- `keyframes/env_0000/*.png`
- `manifest.jsonl`
- repair candidates or manual debug outputs

This is generated data. The repo `.gitignore` ignores `**/runs`, so examples and
docs should contain reusable knowledge, while experiments contain local evidence.

## Current DoorTwin Workflow

For a new generated door:

1. Put the asset under `high-level/data/asset/door_set/<door_name>/`.
2. Ensure the URDF has at least a door DOF and a handle DOF.
3. Add or generate:
   - `bounding_box.json`
   - `handle_bounding.json`
   - a config entry in `high-level/data/cfg/b1z1_opendoor.yaml`
4. Create an initial skill JSON from a similar example.
5. Run a single-env rollout with keyframes and reports.
6. Inspect:
   - `rollout_summary.json`
   - `env_0000_report.json`
   - `handle_closeup` keyframes for grasp quality
   - observer views for wall/door/body geometry
7. Patch skill/config parameters.
8. Repeat until:
   - door opens past the target angle,
   - handle unlocks,
   - body traverses,
   - camera frames are valid,
   - trajectory commands are smooth enough for expert data.
9. Save the final reusable skill in `examples/`.
10. Record the debug path in `doc/`.

## Typical Command

```bash
conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 1 --steps 1100 --seed 62000 \
  --door_name <door_name> \
  --skill_program_json high-level/float_ik/door_twin/examples/<skill>.json \
  --door_twin_log_dir high-level/float_ik/door_twin/experiments/runs/<run_name> \
  --save_failed_rollouts --dump_keyframe_images \
  --door_twin_camera_views wrist,front,handle_closeup,observer_left,observer_right \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 \
  --no_ikpush_env_randomization \
  --no_collision_geom_check \
  --no_show_camera_images
```

For strict collision debugging, remove `--no_collision_geom_check` or add PhysX
contact checks. For expert collection, the conservative geometric door-line gate
should not be treated as the only success criterion for full traversal, because
the robot can intentionally pass close to or over door-frame geometry.

## Completed Work

Implemented so far:

- DoorTwin asset spec extraction/loading helpers.
- Skill-program parser and interpreter integration.
- Base-motion skill primitive via `MoveTo(vx, vyaw, distance, duration_steps)`.
- Rollout reports and failure classification.
- Keyframe image dumping with multi-view montages.
- `handle_closeup` camera view for grasp inspection.
- Program patch allowlist and clamp logic.
- Default DoorTwin experiment root under `door_twin/experiments/runs`.
- wc4 skill aligned with scripted behavior.
- Generated door `99692809960048` adapted with repaired frame-collision asset.
- `button_door` adapted from raw `model.urdf` with metadata, wall fix, grasp fix,
  and closeup camera target.
- `fire_door` adapted as a push-door video/data example with wc4-like base
  stance and documented user-corrected grasp/stance fixes.
- `glass_door` adapted as a single-DOF push door with 90 degree gripper roll,
  push-direction sign fix, and tuned door resistance.

## Known Limitations

- The first version focuses on hinged lever push doors.
- Sliding doors, folding doors with multiple moving panels, and multi-DOF door
  mechanisms are not yet fully supported.
- Some door metadata is still hand-authored when generated assets do not provide
  reliable bounding boxes or handle goals.
- VLM/Agent repair has been validated manually through Codex-in-the-loop
  debugging, but it is not yet a fully automated service.
- Collision success semantics need task-aware policy: strict geometry checks are
  useful diagnostics, but can reject valid traversal behavior.
- The Agent should inspect `handle_closeup` images for grasp quality; scalar
  success alone is not enough for expert data.

## Current Agent Loop

The current loop has already been validated with Codex/Agent in the loop. In the
three documented doors, the Agent effectively acted as the VLM observer:

1. Read `rollout_summary.json` and per-env reports.
2. Inspected observer, wrist, front, and `handle_closeup` keyframes.
3. Classified the failure or quality issue.
4. Patched bounded skill/config parameters instead of rewriting arbitrary code.
5. Re-ran the rollout and compared metrics/images.
6. Promoted the stable skill to `examples/`.
7. Recorded the debug path in `doc/`.

Concrete examples:

- wc4: aligned the skill path with the scripted trajectory, matched push/traverse
  speed, and smoothed EE commands across phase transitions.
- 99692809960048: diagnosed generated-frame collision issues, created a
  no-frame-collision asset variant, and matched scripted-speed traversal.
- button_door: loaded a raw new asset, fixed side-wall opening geometry, lowered
  grasp z by 1 cm after visual inspection, and added `handle_closeup` for grasp
  quality checks.

This is the intended DoorTwin repair behavior; it is currently executed manually
by Codex rather than by a persistent automated service.

## Automation Next

The next engineering step is to turn the proven manual loop into a repeatable
orchestrator:

1. Automatically parse generated-door metadata from URDF and asset exports.
2. Produce an initial skill program by matching against existing examples.
3. Run N rollouts.
4. Summarize logs and keyframes for the Agent/VLM observer.
5. Ask the Agent/VLM for a bounded JSON patch and diagnosis.
6. Run a small local optimizer over continuous parameters.
7. Stop when success rate or expert trajectory count reaches the target.

The automated Agent should treat the examples and debug docs as prior experience:

- wc4: scripted reference and smooth-command baseline.
- 99692809960048: generated-door adaptation and frame-collision repair.
- button_door: new raw asset adaptation, wall geometry fix, grasp-quality camera.
- fire_door: stance correction, wall correction, and 1 cm grasp lowering from
  user visual feedback.
- glass_door: fixed-handle vertical grasp, push/pull sign correction, and
  realistic hinge resistance tuning.
