# DoorTwin Agent Ablation Benchmark

This package implements the paired benchmark for automatic DoorTwin
asset/skill generation and repair.

## Methods

| Internal name | Report name | Development feedback | Repair |
|---|---|---|---|
| `rule_based` | Rule-based | none | deterministic URDF/name + handle-bbox initialization, evaluated directly |
| `ours` | Ours full | logs + probe + fixed phase montages | retrieval-augmented bounded Agent patch |
| `without_visual_feedback` | Ours w/ VLM Fix - Log | logs + probe only | retrieval-augmented bounded Agent patch |
| `without_vlm_fix` | Ours w/o VLM Fix | logs only | deterministic heuristic patch |
| `validation_only` | Ours Validation Only | logs + montages | diagnosis only, candidate unchanged |
| `without_simulation_rollout` | Ours w/o Simulation Rollout | none | Rule-based candidate + one retrieval-augmented residual patch |

The deterministic Rule-based candidate is the common base of the whole paired
experiment. `rule_based` evaluates that base directly. Every Ours ablation then
receives the same single retrieval-augmented residual patch on that base and
shares the resulting immutable initial candidate. The residual observer is
explicitly instructed to preserve Rule-based fields unless a retrieved prior and
public URDF/bbox evidence justify changing them. `rule_based` itself receives
neither prior retrieval nor development rollout. The final
held-out rollouts for `without_simulation_rollout` are scoring only and are never
fed back to the observer.

The current `run_agent_ablation.py` remains compatible with historical
patch-per-round results. New engineering runs should use `../run_agent.py`,
which lets Codex autonomously select tools, enforces static/probe/rollout gates,
and shares the regression-safe candidate graph. Both paths use the same frozen
experience catalog and final-best selection policy.

The shared Rule-based + residual candidate is also scored once on the held-out seed set for
the paired "first generation" metric. Those artifacts live under
`initial_candidates/*/heldout_scoring_only` and are never read by a repair
observer. The one-shot branch reuses this score instead of running it again.

## 1. Build a leakage-checked fresh-door manifest

The selector excludes doors mentioned under `door_twin/examples`, `doc`, or
`experiments`, excludes the known regression doors, and excludes config entries
with hand-tuned fields. It then samples ten hinged AIGC doors with a movable
door joint and handle metadata, allowing either a movable or fixed handle, using
a recorded selection seed.

```bash
cd /home/sivan/whole_body/visual_whole_body

python high-level/float_ik/door_twin/benchmark/make_fresh_manifest.py \
  --output high-level/float_ik/door_twin/experiments/fresh10_agent_ablation.yaml \
  --run_root high-level/float_ik/door_twin/experiments/runs/agent_ablation \
  --python_executable /home/sivan/miniconda3/envs/b1z1/bin/python \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0
```

The manifest contains hidden annotations used only by the scorer. They are not
included in VLM requests. The candidate starts from a public bbox-center handle
estimate rather than the hidden `goal_pos`.

## 2. Freeze retrieval prior and configure the observer

The implementation uses the OpenAI Responses API through Python's standard
library, so no additional Python package is required.

```bash
export OPENAI_API_KEY=...
```

The formal manifest fixes model `gpt-5`, temperature `0`, prompt schema, repair
bounds, development seeds `41001..41016`, and held-out seeds `42001..42016`.
Initial generation and repair receive top-3 records from
`../experience/catalog.yaml`; the target door is excluded and the snapshot hash
is persisted. `rule_based` receives no records.
`OPENAI_BASE_URL` may point to an OpenAI-compatible endpoint, but it must remain
fixed for the whole experiment and be recorded with the run.

Asset validity is measured separately with seed `43001`: a benchmark-only
100-step probe bypasses the handle lock and applies a fixed hinge torque. It
requires actual PhysX hinge motion of at least 30 degrees, in addition to URDF
range, semantic-name, handle-goal, stability, and ground-clearance checks. This
probe is scoring-only and is never shown to the repair observer.

## 3. Run

First run one candidate as a smoke test:

```bash
python high-level/float_ik/door_twin/benchmark/run_agent_ablation.py \
  --manifest high-level/float_ik/door_twin/experiments/fresh10_agent_ablation.yaml \
  --doors DOOR_ID_FROM_MANIFEST \
  --candidate_indices 0 \
  --methods rule_based,ours,without_visual_feedback,without_vlm_fix,validation_only,without_simulation_rollout \
  --stream_output
```

Then run the complete paired benchmark:

```bash
python high-level/float_ik/door_twin/benchmark/run_agent_ablation.py \
  --manifest high-level/float_ik/door_twin/experiments/fresh10_agent_ablation.yaml \
  --stream_output
```

Runs are resumable. Existing seed summaries and completed branches are reused;
`--force` explicitly reruns them. `--dry_run --initial_generation metadata`
checks artifact layout and commands without launching Isaac Gym or calling a
VLM. `--mock_vlm_response FILE` supports deterministic integration tests.

To let the current Codex session act as the observer without an API key, use a
local file-exchange directory:

```bash
python high-level/float_ik/door_twin/benchmark/run_agent_ablation.py \
  --manifest high-level/float_ik/door_twin/experiments/fresh10_agent_ablation.yaml \
  --doors DOOR_ID_FROM_MANIFEST \
  --candidate_indices 0 \
  --methods ours \
  --manual_vlm_dir high-level/float_ik/door_twin/experiments/runs/manual_vlm_exchange \
  --stream_output
```

For every call the runner atomically writes `request_NNNN.json`, prints the
expected `response_NNNN.json` path, and waits. The response can be the bounded
patch object directly. The request contains the sanitized prompt and the local
paths of montage images, but never an API key or embedded base64 data. This mode
is intended for Codex/human-in-the-loop smoke tests; it is not the fixed
`gpt-5` formal benchmark condition unless the responder identity is separately
controlled and recorded.

For multiple GPUs, shard by disjoint `--doors` (or disjoint candidate indices),
keep all five methods for one candidate in the same process, and assign a unique
`--shard_id`. Do not shard methods of an unprepared candidate across processes,
because the shared initial VLM candidate must be generated exactly once.

Each visual round sends at most five phase montages. Each montage is generated
by the existing DoorTwin runner from `front`, `wrist`, `handle_closeup`, and
`observer_left`. Log-only runs never embed image content.

## 4. Report

```bash
python high-level/float_ik/door_twin/benchmark/report_benchmark.py \
  --manifest high-level/float_ik/door_twin/experiments/fresh10_agent_ablation.yaml \
  --strict
```

Outputs are placed under `<run_root>/<benchmark_name>/report/`:

- `candidate_results.csv`
- `summary.json`
- `REPORT.md`
- `diagnosis_labels_template.csv`

After two blinded reviewers and adjudication fill the diagnosis CSV, regenerate
the report with:

```bash
python high-level/float_ik/door_twin/benchmark/report_benchmark.py \
  --manifest high-level/float_ik/door_twin/experiments/fresh10_agent_ablation.yaml \
  --diagnosis_labels /path/to/completed_diagnosis_labels.csv
```

The report includes asset load/structure success, first-attempt rollout success,
held-out stage rates, candidate generation success (`>=12/16` and valid asset),
bootstrap confidence intervals, paired exact McNemar tests, and paired Wilcoxon
tests for repair rounds.

## Safety and reproducibility invariants

- Asset patches are restricted to an allowlist and numerical bounds.
- Body/link and DOF names must exist in the candidate URDF.
- Skill patches reuse the existing `ProgramPatch` allowlist.
- The observer cannot modify Python, meshes, URDF files, or arbitrary config keys.
- Every candidate and parent is SHA-256 fingerprinted.
- Hidden annotations never enter observer prompts.
- API keys and base64 image payloads are never persisted.
- `validation_only` must keep the candidate fingerprint unchanged.
- `without_vlm_fix` has zero repair-time VLM calls; the shared initial VLM call
  is reported separately as `vlm_initial_calls`.
