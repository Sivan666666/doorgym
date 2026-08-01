#!/usr/bin/env bash
set -euo pipefail

cd /home/sivan/whole_body/visual_whole_body

ROOT="${1:-high-level/logs/foundationpose/wc4_bundlesdf_no_gt_sam3_fp_first_valid_once_16_seed615455575}"
MESH="${2:-high-level/logs/foundationpose/bundlesdf_wc4_front_sam3_no_gt_60_v2_output/textured_mesh.obj}"
GRASP_POINT=(0.00570725 0.00250802 0.01294013)
SEED=615455575
mkdir -p "$ROOT"

for offset in 0 2 4 6 8 10 12 14; do
  run_dir="$ROOT/seed_${SEED}_offset_${offset}"
  mkdir -p "$run_dir"
  if [[ -f "$run_dir/summary.json" ]]; then
    echo "===== Skip completed offset=${offset} ====="
    continue
  fi
  echo "===== First-valid one-shot pose: seed=${SEED}, virtual envs ${offset}..$((offset + 1)) ====="
  conda run --no-capture-output -n b1z1 python \
    high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel_sam3_foundationpose_grasp.py \
    --num_envs 2 \
    --steps 1000 \
    --seed "$SEED" \
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
    --foundationpose_model_grasp_point "${GRASP_POINT[@]}" \
    --foundationpose_initial_pose_only \
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

python - "$ROOT" "$MESH" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
mesh = Path(sys.argv[2])
runs = []
for path in sorted(root.glob("seed_*_offset_*/summary.json")):
    runs.append((path, json.loads(path.read_text(encoding="utf-8"))))

success = sum(int(payload["success_count"]) for _, payload in runs)
trials = sum(int(payload["num_envs"]) for _, payload in runs)
registers = [
    report
    for _, payload in runs
    for report in payload.get("pose_reports", [])
    if report.get("mode") == "first_valid_register"
]
errors_mm = [1000.0 * float(report["grasp_goal_error_m"]) for report in registers]
steps = [int(report["sim_step"]) for report in registers]
aggregate = {
    "method": "SAM3 first reliable mask -> one FoundationPose registration -> fixed world grasp XYZ; no tracking",
    "mesh": str(mesh.resolve()),
    "seed": 615455575,
    "virtual_env_indices": list(range(16)),
    "success_count": success,
    "total_trials": trials,
    "success_rate": success / max(1, trials),
    "registration_count": len(registers),
    "foundationpose_tracking_calls_after_registration": 0,
    "registration_step": {
        "min": min(steps) if steps else None,
        "median": statistics.median(steps) if steps else None,
        "max": max(steps) if steps else None,
    },
    "grasp_goal_error_mm_eval_only": {
        "mean": statistics.fmean(errors_mm) if errors_mm else None,
        "median": statistics.median(errors_mm) if errors_mm else None,
        "max": max(errors_mm) if errors_mm else None,
    },
    "runs": [
        {
            "summary": str(path),
            "success_count": int(payload["success_count"]),
            "num_envs": int(payload["num_envs"]),
        }
        for path, payload in runs
    ],
}
output = root / "aggregate_summary.json"
output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
print(f"INITIAL_ONCE_SUCCESS {success}/{trials} = {success / max(1, trials):.6f}")
print(f"Aggregate summary: {output}")
PY
