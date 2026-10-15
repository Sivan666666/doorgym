#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
GPU=${GPU:-3}
SEED=${SEED:-615455575}
TAG=0811_wc_nomenkuang_stop0293_clearance36_v2
PIPE="$REPO/high-level/logs/pull-data-pipeline/${TAG}_nested50_500"
CONDA=/home/ps/miniconda3/bin/conda

MODELS=(
  "leroact_a2w_${TAG}_50_baseline_seed1000_50k_chunk100_exec50_bs16"
  "leroact_a2w_${TAG}_50_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16"
  "leroact_a2w_${TAG}_500_baseline_seed1000_50k_chunk100_exec50_bs16"
  "leroact_a2w_${TAG}_500_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16"
)

mkdir -p "$PIPE"
cd "$REPO"

for name in "${MODELS[@]}"; do
  checkpoint="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000/model_latest.pt"
  run_root="$REPO/high-level/logs/door-policy-success/${name}_wc_nomenkuang_seed${SEED}_open80_traverse0p8_64"
  log="$PIPE/eval_sequential_${name}.log"

  if [[ -s "$run_root/summary.json" ]]; then
    echo "SKIP_COMPLETE $name" | tee -a "$PIPE/eval_sequential_status.txt"
    continue
  fi
  if [[ -e "$run_root" ]]; then
    failed_root="${run_root}_failed_concurrent_$(date +%Y%m%d_%H%M%S)"
    mv "$run_root" "$failed_root"
    echo "ARCHIVED $failed_root" | tee -a "$PIPE/eval_sequential_status.txt"
  fi

  echo "$(date '+%F %T') START $name GPU=$GPU" | tee "$PIPE/eval_sequential_status.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$checkpoint" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
    --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
    --steps 2600 --pass_open_angle_deg 80 --pull_traversal_distance_m 0.8 \
    --base_seed "$SEED" --headless --graphics_device_id 0 \
    --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
    --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 --run_root "$run_root" -- \
    --door_name wc_nomenkuang --stop_distance 0.293 \
    --ikpush_door_wall_x_offset_rand 0 \
    --wc4_disable_door_open_resistance --door_auto_open_force 0 \
    --pull_vx_only_base --pull_panel_base_mode no_vy \
    --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
    --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
    --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    >"$log" 2>&1
  echo "$(date '+%F %T') COMPLETE $name" | tee "$PIPE/eval_sequential_status.txt"
done

/home/ps/miniconda3/bin/python - "$REPO" "$SEED" "${MODELS[@]}" <<'PY' | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

repo, seed = Path(sys.argv[1]), sys.argv[2]
for name in sys.argv[3:]:
    path = repo / "high-level/logs/door-policy-success" / f"{name}_wc_nomenkuang_seed{seed}_open80_traverse0p8_64" / "summary.json"
    data = json.loads(path.read_text())
    trials = int(data["trials"])
    opened = int(data["door_open_successes"])
    traversed = int(data["traversal_successes"])
    joint = int(data["successes"])
    print(
        f"{name}: open={opened}/{trials} ({100*opened/trials:.2f}%) "
        f"traversal={traversed}/{trials} ({100*traversed/trials:.2f}%) "
        f"joint={joint}/{trials} ({100*joint/trials:.2f}%)"
    )
PY

echo "$(date '+%F %T') COMPLETE_ALL" | tee "$PIPE/eval_sequential_status.txt"
