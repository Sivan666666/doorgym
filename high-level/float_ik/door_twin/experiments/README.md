# DoorTwin Experiments

DoorTwin agent run artifacts live under `runs/`.

The reproducible paired Agent ablation runner and protocol are documented in
`../benchmark/README.md`. Formal fresh-door manifests should be generated at run
time so test-door identities do not leak into examples or repair history.

Tool-driven Codex sessions additionally contain `session.json`,
`tool_trace.jsonl`, `prior_snapshot.json`, `candidate_graph.json`,
`best_candidate/`, per-candidate static/probe/rollout reports, append-only repair
records, and `final_report.md`. Re-run the same `run_agent.py` command to resume
an interrupted session.

Use this directory for rollout logs, keyframe images, repair histories, and local
optimization/debug outputs. Reusable skill programs should stay in
`../examples/`, and writeups or debug notes should stay in `../doc/`.

The repository-level `.gitignore` ignores `**/runs`, so this directory is meant
for local/generated experiment data rather than source-controlled code.

Current useful local histories:

- `runs/asset_99692809960048_rounds/repair_history.md`
- `runs/fire_door_repair_history.md`

For `fire_door`, the accepted run is:

`runs/fire_door_round_03_wc4_like_y_offset/`

For `glass_door`, the accepted run after gripper-roll, push-direction, and
resistance tuning is:

`runs/glass_door_resistance_check_v2/`

The earlier `runs/glass_door_resistance_check/` run is intentionally kept as a
negative reference: its restoring hinge resistance was too high, so the door only
opened to about 75.8 degrees.
