#!/usr/bin/env bash
set -u

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
BASE_SEED=${BASE_SEED:-615455575}
TAG=${TAG:-0813_wc_nomenkuang_fixed15_observable_v1}
PIPE="$REPO/high-level/logs/pull-data-pipeline/${TAG}_500_phasebalanced_multiseed"
DATA="local/door_a2w_${TAG}_500_interaction_state"
# Evaluate intermediate checkpoints as soon as they appear. The main pipeline
# owns the final 500K evaluation so both processes cannot race on that run.
CHECKPOINTS=(100000 200000 300000)

cd "$REPO" || exit 1
mkdir -p "$PIPE"
exec 9>"$PIPE/checkpoint_eval_monitor.lock"
flock -n 9 || exit 0
exec >>"$PIPE/checkpoint_eval_monitor.log" 2>&1

stamp() {
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

job_name() {
  local seed=$1
  echo "leroact_a2w_${TAG}_500_plucker_interaction_phasebalanced_hold01_critical2_seed${seed}_500k_chunk100_exec50_bs16"
}

official_ready() {
  local seed=$1 step=$2 name
  name=$(job_name "$seed")
  [[ -s "$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/$step/pretrained_model/model.safetensors" ]]
}

export_one() {
  local gpu=$1 seed=$2 step=$3 name official wrapped
  name=$(job_name "$seed")
  official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/$step"
  wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/$step"
  [[ -s "$wrapped/model_latest.pt" ]] && return 0
  stamp "EXPORT seed=$seed step=$step gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
    --official_checkpoint "$official" --root high-level/data/lerobot \
    --repo_id "$DATA" --out_dir "$wrapped/model_latest" \
    --manifest_name model_latest.pt --device cuda:0 --depth_only \
    >"$PIPE/export_seed${seed}_${step}.log" 2>&1
}

eval_one() {
  local gpu=$1 seed=$2 step=$3 name ckpt run
  name=$(job_name "$seed")
  ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$name/$step/model_latest.pt"
  run="$REPO/high-level/logs/door-policy-success/${name}_step${step}_wc_nomenkuang_seed${BASE_SEED}_open80_traverse0p8_64"
  [[ -s "$run/summary.json" ]] && return 0
  stamp "EVAL64 seed=$seed step=$step gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
    --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
    --steps 2600 --pass_open_angle_deg 80 --pull_traversal_distance_m 0.8 \
    --base_seed "$BASE_SEED" --headless --graphics_device_id 0 \
    --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
    --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5 --run_root "$run" -- \
    --door_name wc_nomenkuang --stop_distance 0.293 \
    --ikpush_door_wall_x_offset_rand 0 --wc4_disable_door_open_resistance \
    --door_auto_open_force 0 --pull_vx_only_base --pull_panel_base_mode no_vy \
    --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
    --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
    --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    >"$PIPE/eval_seed${seed}_${step}_64.log" 2>&1
}

worker() {
  local gpu=$1 seed=$2 step
  for step in "${CHECKPOINTS[@]}"; do
    while ! official_ready "$seed" "$step"; do
      if grep -q "FAILED" "$PIPE/status.txt" 2>/dev/null; then
        stamp "STOP seed=$seed because main pipeline failed"
        return 1
      fi
      sleep 60
    done
    until export_one "$gpu" "$seed" "$step" && eval_one "$gpu" "$seed" "$step"; do
      stamp "RETRY seed=$seed step=$step after export/eval failure"
      sleep 120
    done
    stamp "DONE64 seed=$seed step=$step"
  done
}

stamp "START checkpoint-triggered evaluation monitor"
worker 2 1000 & p0=$!
worker 3 2000 & p1=$!
wait "$p0" || exit 1
wait "$p1" || exit 1
stamp "COMPLETE checkpoint-triggered evaluation monitor"
