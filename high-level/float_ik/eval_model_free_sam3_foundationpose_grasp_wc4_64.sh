#!/usr/bin/env bash
set -euo pipefail

cd /home/sivan/whole_body/visual_whole_body

ROOT="${1:-high-level/logs/foundationpose/wc4_model_free_sam3_fp_grasp_track_64_seed615455575}"
MESH="${2:-high-level/logs/foundationpose/model_free_wc4_refs_sam3_16/model/model.obj}"
GRASP_POINT="${3:-}"
MODEL_GRASP_ARGS=()
if [[ -n "$GRASP_POINT" ]]; then
  read -r -a GRASP_XYZ <<< "$GRASP_POINT"
  if [[ "${#GRASP_XYZ[@]}" -ne 3 ]]; then
    echo "Third argument must contain exactly three mesh-frame coordinates" >&2
    exit 2
  fi
  MODEL_GRASP_ARGS=(--foundationpose_model_grasp_point "${GRASP_XYZ[@]}")
fi
mkdir -p "$ROOT"

SEEDS=(615455575 615455576 615455577 615455578)
OFFSETS=(0 2 4 6 8 10 12 14)

for seed in "${SEEDS[@]}"; do
  for offset in "${OFFSETS[@]}"; do
    run_dir="$ROOT/seed_${seed}_offset_${offset}"
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/summary.json" ]]; then
      echo "===== Skip completed seed=${seed}, offset=${offset} ====="
      continue
    fi
    echo "===== Model-free SAM3 + FoundationPose seed=${seed}, offset=${offset}, envs=2 ====="
    conda run --no-capture-output -n b1z1 python \
    high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel_sam3_foundationpose_grasp.py \
    --num_envs 2 \
    --steps 1000 \
    --seed "$seed" \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --door_name wc4 \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --graphics_device_id 0 \
    --headless \
    --camera_depth \
    --depth_only \
    --camera_depth_clip_lower 0.2 \
    --camera_depth_clip_far 1.5 \
    --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --robot_pitch 0.0 \
    --ikpush_robot_pitch_rand_min 0.0 \
    --ikpush_robot_pitch_rand_max 0.0 \
    --no_preview_trajectory_at_spawn \
    --no_draw_ik_target \
    --no_draw_camera_axes \
    --no_show_seg \
    --foundationpose_virtual_env_offset "$offset" \
    --foundationpose_virtual_num_envs 16 \
    --foundationpose_mesh "$MESH" \
    "${MODEL_GRASP_ARGS[@]}" \
    --foundationpose_track_during_walk \
    --foundationpose_walk_track_interval 5 \
    --foundationpose_output_dir "$run_dir/foundationpose" \
    --foundationpose_grasp_summary "$run_dir/summary.json" \
    --foundationpose_grasp_success_angle_deg 80 \
    --foundationpose_grasp_sim_safety_max_error_m -1 \
    --foundationpose_debug 0 \
    --sam3_output_dir "$run_dir/sam3" \
    --sam3_prompt "door handle" \
    --sam3_confidence_threshold 0.005 \
    --sam3_min_selected_score 0.1 \
    --sam3_selection wc4_initial_handle_roi \
    2>&1 | tee "$run_dir/run.log"
  done
done

python - "$ROOT" "$MESH" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
mesh = Path(sys.argv[2])
summaries = []
for path in sorted(root.glob("seed_*_offset_*/summary.json")):
    summaries.append((path, json.loads(path.read_text(encoding="utf-8"))))

success = sum(int(payload["success_count"]) for _, payload in summaries)
trials = sum(int(payload["num_envs"]) for _, payload in summaries)
final_reports = []
registration_reports = []
trial_records = []
for _, payload in summaries:
    by_env = {}
    registration_by_env = {}
    for report in payload.get("pose_reports", []):
        env_id = int(report["env"])
        by_env[env_id] = report
        if report.get("mode") == "register":
            registration_reports.append(report)
            registration_by_env[env_id] = report
    final_reports.extend(by_env.values())
    for env_id in range(int(payload["num_envs"])):
        trial_records.append(
            {
                "success": bool(payload["success_by_env"][str(env_id)]),
                "accepted": bool(payload["perception_accepted_by_env"][str(env_id)]),
                "registration": registration_by_env.get(env_id),
                "final": by_env.get(env_id),
            }
        )

goal_errors_mm = [
    1000.0 * float(report["grasp_goal_error_m"])
    for report in final_reports
    if report.get("grasp_goal_error_m") is not None
]
registration_steps = [int(report.get("sim_step", 0)) for report in registration_reports]
no_visual_initialization = sum(not record["accepted"] for record in trial_records)
registered_failures = [
    record for record in trial_records if record["accepted"] and not record["success"]
]
wrong_mask_failures = sum(
    float(record["registration"].get("sam3", {}).get("mask_iou_eval_only", 0.0)) < 0.5
    for record in registered_failures
    if record["registration"] is not None
)
aggregate = {
    "method": (
        "16-view RGB-D + SAM3 masks + camera intrinsics/relative poses -> BundleSDF mesh; "
        "SAM3 first visible frame -> FoundationPose register; RGB-D tracking during walk; "
        "estimated grasp XYZ replaces manual grasp XYZ"
    ),
    "mesh": str(mesh.resolve()),
    "seeds": [615455575, 615455576, 615455577, 615455578],
    "success_count": success,
    "total_trials": trials,
    "success_rate": success / max(1, trials),
    "success_angle_deg": 80.0,
    "gt_used_for_control": False,
    "sam3_min_selected_score": 0.1,
    "foundationpose_walk_track_interval": 5,
    "failure_breakdown": {
        "total_failures": int(trials - success),
        "no_visual_initialization": int(no_visual_initialization),
        "registered_but_failed": int(len(registered_failures)),
        "registered_failure_with_sam3_iou_lt_0_5_eval_only": int(wrong_mask_failures),
    },
    "final_grasp_goal_error_mm_eval_only": {
        "count": len(goal_errors_mm),
        "mean": statistics.fmean(goal_errors_mm) if goal_errors_mm else None,
        "median": statistics.median(goal_errors_mm) if goal_errors_mm else None,
        "max": max(goal_errors_mm) if goal_errors_mm else None,
    },
    "registration_step": {
        "count": len(registration_steps),
        "mean": statistics.fmean(registration_steps) if registration_steps else None,
        "max": max(registration_steps) if registration_steps else None,
    },
    "runs": [
        {
            "summary": str(path),
            "success_count": int(payload["success_count"]),
            "num_envs": int(payload["num_envs"]),
        }
        for path, payload in summaries
    ],
}
output = root / "aggregate_summary.json"
output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
print(f"MODEL_FREE_SAM3_FOUNDATIONPOSE_GRASP_SUCCESS {success}/{trials} = {success / max(1, trials):.6f}")
print(f"Aggregate summary: {output}")
PY
