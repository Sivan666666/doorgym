#!/usr/bin/env bash
set -euo pipefail

cd /home/sivan/whole_body/visual_whole_body

ROOT="${1:-high-level/logs/foundationpose/fire_door_sam3_fp_wc4traj_64_seed615455575}"
MESH="${2:-high-level/data/asset/door_set/fire_door/assets/meshes/lever_handle_foundationpose.obj}"
mkdir -p "$ROOT"

SEEDS=(615455575 615455576 615455577 615455578)
OFFSETS=(0 2 4 6 8 10 12 14)

for seed in "${SEEDS[@]}"; do
  for offset in "${OFFSETS[@]}"; do
    run_dir="$ROOT/seed_${seed}_offset_${offset}"
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/summary.json" ]]; then
      echo "===== Skip completed fire_door seed=${seed} offset=${offset} ====="
      continue
    fi
    echo "===== Fire door: SAM3 + FoundationPose seed=${seed} offset=${offset} ====="
    conda run --no-capture-output -n b1z1 python -u \
      high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel_sam3_foundationpose_grasp.py \
      --num_envs 2 \
      --steps 1000 \
      --seed "$seed" \
      --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
      --door_name fire_door \
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
      --foundationpose_mesh "$MESH" \
      --foundationpose_virtual_env_offset "$offset" \
      --foundationpose_virtual_num_envs 16 \
      --foundationpose_output_dir "$run_dir/foundationpose" \
      --foundationpose_grasp_summary "$run_dir/summary.json" \
      --foundationpose_grasp_success_angle_deg 80 \
      --foundationpose_grasp_sim_safety_max_error_m 0.015 \
      --foundationpose_debug 0 \
      --sam3_output_dir "$run_dir/sam3" \
      --sam3_prompt "door handle" \
      --sam3_confidence_threshold 0.005 \
      --sam3_selection wc4_front_roi \
      2>&1 | tee "$run_dir/run.log"
  done
done

python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = []
for path in sorted(root.glob("seed_*_offset_*/summary.json")):
    summaries.append((path, json.loads(path.read_text(encoding="utf-8"))))

success = sum(int(payload["success_count"]) for _, payload in summaries)
trials = sum(int(payload["num_envs"]) for _, payload in summaries)
accepted = sum(int(payload["perception_accepted_count"]) for _, payload in summaries)
sam_reports = [
    report.get("sam3", {})
    for _, payload in summaries
    for report in payload.get("pose_reports", [])
    if report.get("sam3")
]
ious = [float(report["mask_iou_eval_only"]) for report in sam_reports]
aggregate = {
    "door_name": "fire_door",
    "trajectory": "default WC4-style scripted trajectory",
    "mask_source": "SAM3 front RGB",
    "mesh_source": "fire_door URDF lever_handle GT geometry",
    "success_count": success,
    "total_trials": trials,
    "success_rate": success / max(1, trials),
    "perception_accepted_count": accepted,
    "sam3_report_count": len(sam_reports),
    "sam3_mask_iou_mean_eval_only": sum(ious) / max(1, len(ious)),
    "sam3_mask_iou_ge_0_5_count_eval_only": sum(iou >= 0.5 for iou in ious),
    "runs": [
        {
            "summary": str(path),
            "success_count": int(payload["success_count"]),
            "num_envs": int(payload["num_envs"]),
            "perception_accepted_count": int(payload["perception_accepted_count"]),
        }
        for path, payload in summaries
    ],
}
(root / "aggregate_summary.json").write_text(
    json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
)
print(
    f"FIRE_DOOR_SAM3_FOUNDATIONPOSE_GRASP_SUCCESS {success}/{trials} "
    f"= {success / max(1, trials):.6f}"
)
print(f"Aggregate summary: {root / 'aggregate_summary.json'}")
PY
