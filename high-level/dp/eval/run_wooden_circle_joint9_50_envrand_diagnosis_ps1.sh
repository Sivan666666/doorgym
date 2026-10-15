#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${ROOT:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
GPU=${GPU:-3}
SEED=${SEED:-615455575}
SUCCESS_ROOT="$ROOT/high-level/logs/door-policy-success"
LOG_ROOT="$ROOT/high-level/logs/wooden_circle_joint9_50_envrand_diagnosis_20260812"
mkdir -p "$SUCCESS_ROOT" "$LOG_ROOT"
cd "$ROOT"

eval_one() {
  local run=$1
  local ckpt="$ROOT/high-level/dp/logs/door-auto-wrapped/$run/model_latest.pt"
  local out="$SUCCESS_ROOT/${run}_wooden_circle_envrand_seed${SEED}_open80_64"
  [[ ! -s "$out/summary.json" ]] || return
  "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
    --mode ikpush --robot_body a2wz1 --num_envs 16 --total_trials 64 \
    --steps 1000 --pass_open_angle_deg 80 --door_dof_index 2 \
    --success_metric abs --base_seed "$SEED" --headless \
    --graphics_device_id "$GPU" --rl_device cuda:"$GPU" --sim_device cuda:"$GPU" \
    --dp_action_horizon 25 --dp_fps 25 --depth_only \
    --no_enable_depth_noise --no_enable_depth_gaussian_blur \
    --run_root "$out" --stream_output -- \
    --door_name wooden_circle --door_z_offset 0.01 --robot_y 0.2 \
    --arm_ik_solver gym_jacobian --ee_pose_frame robot_base_full \
    --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    >"$LOG_ROOT/$run.log" 2>&1
}

eval_one leroact_a2w_wooden_circle_joint9_50_baseline_seed1000_50k_chunk100_exec50_bs16
eval_one leroact_a2w_wooden_circle_joint9_50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16
echo COMPLETE
