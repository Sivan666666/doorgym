#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
BASE_SEED=${BASE_SEED:-615455575}
TAG=${TAG:-0813_wc_nomenkuang_fixed_release_scripted64}
EXPERIMENT_MODE=${EXPERIMENT_MODE:-fixed_angles}
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-3}
ROOT="$REPO/high-level/logs/pull-scripted-validation/$TAG"

cd "$REPO"
mkdir -p "$ROOT"

COMMON=(
  --num_envs 16 --steps 2600 --headless
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml
  --door_name wc_nomenkuang --stop_distance 0.293
  --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0
  --ikpush_door_wall_x_offset_rand 0
  --handle_rotate_steps 100
  --rotate_pull_overlap_steps 20
  --rotate_pull_overlap_ee_distance 0.01
  --pull_then_panel_push
  --pull_release_angle_tolerance_deg 0
  --pull_panel_adaptive_release_retreat
  --pull_panel_low_angle_base_retreat_distance 0.36
  --pull_panel_low_angle_right_arc_radius 0.12
  --pull_panel_low_angle_edge_clearance 0.12
  --pull_panel_low_angle_outward_clearance 0.20
  --pull_panel_low_angle_reposition_steps 260
  --pull_vx_only_base --pull_panel_base_mode no_vy
  --pull_base_retreat_distance 0.48
  --pull_base_progress_reference_angle_deg 42
  --pull_late_retreat_start_angle_deg 22
  --pull_late_retreat_distance 0.12
  --pull_late_retreat_steps 110
  --pull_base_follow_max_speed 0.36
  --pull_base_follow_position_gain 2.5
  --pull_vx_monotonic_retreat
  --pull_vx_monotonic_max_accel_mps2 0.8
  --release_handle_steps 90 --release_handle_motion_steps 90
  --release_to_clearance_pose
  --release_clearance_retreat_distance 0.10
  --release_clearance_right_distance 0.18
  --release_clearance_lift_distance 0.03
  --release_clearance_retreat_fraction 0.30
  --release_clearance_low_angle_base_retreat_extra 0.15
  --release_clearance_base_retreat_extra 0.30
  --release_clearance_arc_radius_extra 0.09
  --release_clearance_edge_extra 0.08
  --release_clearance_outward_extra 0.08
  --panel_reposition_path right_arc
  --panel_right_arc_continuous_spline
  --panel_reposition_steps 170
  --panel_reposition_completion_gate
  --panel_reposition_completion_backside_margin_m 0.01
  --panel_reposition_completion_position_tolerance_m 0.12
  --panel_reposition_completion_max_extra_steps 300
  --panel_reposition_base_forward_distance 0
  --panel_right_arc_joint1_hold_rad -1.0
  --panel_right_arc_radius 0.27
  --panel_right_arc_outward_clearance 0.27
  --panel_right_arc_lift 0
  --panel_edge_clearance 0.22
  --panel_contact_steps 80
  --panel_push_goal_angle_deg 80
  --panel_push_motion_mode hinge_arc
  --panel_push_planned_arc_steps 320
  --panel_push_arc_lead_angle_deg 4
  --panel_contact_initial_penetration 0.03
  --panel_contact_penetration 0.12
  --panel_target_max_step_m 0.03
  --panel_base_follow_ratio 0
  --panel_base_follow_max_speed 0.15
  --panel_postpush_home_steps 50
  --wc4_disable_door_open_resistance --door_auto_open_force 0
  --pass_open_angle_deg 80 --pull_record_traversal_distance_m 0.8
  --robot_pitch 0 --ikpush_robot_pitch_rand_min 0
  --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full
  --log_interval 200
)

run_case() {
  local gpu=$1 label=$2 angle=$3
  shift 3
  local extra=("$@")
  local batch seed result log
  for batch in 0 1 2 3; do
    seed=$((BASE_SEED + batch))
    result="$ROOT/${label}_batch${batch}_seed${seed}.json"
    log="$ROOT/${label}_batch${batch}_seed${seed}.log"
    if [[ -s "$result" ]]; then
      continue
    fi
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
      high-level/float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py \
      "${COMMON[@]}" --seed "$seed" \
      --pull_panel_release_angle_deg "$angle" \
      "${extra[@]}" \
      --pull_validation_result_json "$result" >"$log" 2>&1; then
      # Isaac Gym can SIGSEGV while destroying a completed headless sim. The
      # result is durable before teardown, so only a missing JSON is a failed run.
      if [[ ! -s "$result" ]]; then
        return 1
      fi
    fi
  done
}

if [[ "$EXPERIMENT_MODE" == "adaptive_defer" ]]; then
  CASES=(fixed20_defer fixed20_contact_defer)
  run_case "$GPU_A" fixed20_defer 20 \
    --pull_defer_base_retreat_until_panel_push & p0=$!
  run_case "$GPU_B" fixed20_contact_defer 20 \
    --pull_panel_contact_loss_release \
    --pull_panel_contact_loss_min_angle_deg 10 \
    --pull_panel_contact_loss_min_frames 3 \
    --pull_contact_loss_defer_base_retreat \
    --pull_defer_base_retreat_until_panel_push & p1=$!
elif [[ "$EXPERIMENT_MODE" == "adaptive_reachable" ]]; then
  CASES=(fixed20_reachable fixed20_contact_reachable)
  run_case "$GPU_A" fixed20_reachable 20 \
    --release_clearance_base_retreat_extra_scale 0 & p0=$!
  run_case "$GPU_B" fixed20_contact_reachable 20 \
    --release_clearance_base_retreat_extra_scale 0 \
    --pull_panel_contact_loss_release \
    --pull_panel_contact_loss_min_angle_deg 10 \
    --pull_panel_contact_loss_min_frames 3 & p1=$!
elif [[ "$EXPERIMENT_MODE" == "fixed20_reachable_fallback" ]]; then
  CASES=(fixed20_extra0 fixed20_extra05_fallback)
  run_case "$GPU_A" fixed20_extra0 20 \
    --release_clearance_base_retreat_extra_scale 0 & p0=$!
  run_case "$GPU_B" fixed20_extra05_fallback 20 \
    --release_clearance_base_retreat_extra_scale 0.5 \
    --pull_panel_contact_loss_release \
    --pull_panel_contact_loss_min_angle_deg 5 \
    --pull_panel_contact_loss_min_frames 3 \
    --pull_contact_loss_keep_release_geometry & p1=$!
elif [[ "$EXPERIMENT_MODE" == "fixed_low_angles" ]]; then
  CASES=(release10 release15)
  run_case "$GPU_A" release10 10 & p0=$!
  run_case "$GPU_B" release15 15 & p1=$!
else
  CASES=(release20 release25)
  run_case "$GPU_A" release20 20 & p0=$!
  run_case "$GPU_B" release25 25 & p1=$!
fi
wait "$p0"
wait "$p1"

/home/ps/miniconda3/bin/python - "$ROOT" "${CASES[@]}" <<'PY' | tee "$ROOT/summary.tsv"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("case\trelease_deg\topen\ttraversal\tjoint\ttotal\tjoint_rate")
for label in sys.argv[2:]:
    rows = [json.loads(path.read_text()) for path in sorted(root.glob(f"{label}_batch*.json"))]
    if len(rows) != 4:
        raise RuntimeError(f"{label}: expected 4 batches, got {len(rows)}")
    total = sum(int(row["num_envs"]) for row in rows)
    opened = sum(int(row["open_successes"]) for row in rows)
    traversed = sum(int(row["traversal_successes"]) for row in rows)
    joint = sum(int(row["joint_successes"]) for row in rows)
    release = float(rows[0]["release_target_deg"])
    print(f"{label}\t{release:g}\t{opened}\t{traversed}\t{joint}\t{total}\t{joint/total:.8f}")
PY
