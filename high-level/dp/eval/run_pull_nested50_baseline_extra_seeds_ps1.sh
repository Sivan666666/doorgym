#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
TRAIN="$REPO/high-level/lerobot/src/lerobot/scripts/lerobot_train.py"
EXPORT="$REPO/high-level/dp/export_official_lerobot_act_to_door_checkpoint.py"
EVAL="$REPO/high-level/dp/eval/eval_door_policy_success.py"
DATA="$REPO/high-level/data/lerobot/local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_prefix_50"
DATA_REPO=local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_prefix_50
PIPE="$REPO/high-level/logs/pull-data-pipeline/pull_scaling_diagnosis_20260731/extra_seeds"
mkdir -p "$PIPE"

run_seed() {
  local seed="$1" gpu="$2"
  local job="leroact_a2w_wc4_pull_joint9_nested50_baseline_seed${seed}_50k_chunk100_exec50_bs16_0731"
  local run="$REPO/high-level/dp/logs/lerobot-train/$job"
  local wrap_base="$REPO/high-level/dp/logs/door-auto-wrapped/$job/050000"
  local eval_root="$REPO/high-level/logs/door-policy-success/pull_nested50_base50k_trainseed${seed}_evalseed615455575_open60_traverse0p8_64"
  rm -rf "$run" "$wrap_base" "$eval_root"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python "$TRAIN" \
    --dataset.root="$DATA" --dataset.repo_id="$DATA_REPO" --dataset.video_backend=torchcodec \
    --dataset.use_imagenet_stats=false --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=100 --policy.n_action_steps=50 --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.use_action_loss_weight=false --policy.plucker_conditioning=false \
    --policy.interaction_state_conditioning=false --batch_size=16 --steps=50000 --seed="$seed" \
    --num_workers=4 --save_freq=50000 --log_freq=10 --keyframe_sampling_ratio=0.0 \
    --wandb.enable=false --job_name="$job" --output_dir="$run" >"$PIPE/train_seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python "$EXPORT" \
    --official_checkpoint "$run/checkpoints/050000" --root "$REPO/high-level/data/lerobot" \
    --repo_id "$DATA_REPO" --out_dir "$wrap_base/model_latest" --manifest_name model_latest.pt \
    --device cuda:0 --depth_only >"$PIPE/export_seed${seed}.log" 2>&1
  "$CONDA" run --no-capture-output -n txc_vbc python "$EVAL" \
    --checkpoint "$wrap_base/model_latest.pt" --yaml "$REPO/high-level/data/cfg/b1z1_opendoor.yaml" \
    --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 --steps 1750 \
    --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 --base_seed 615455575 \
    --headless --graphics_device_id "$gpu" --rl_device "cuda:$gpu" --sim_device "cuda:$gpu" \
    --dp_action_horizon 25 --dp_fps 25 --depth_only --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 --depth_camera_rot_rand_deg 5.0 --run_root "$eval_root" -- \
    --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
    --robot_pitch 0.0 --ikpush_robot_pitch_rand_min 0.0 --ikpush_robot_pitch_rand_max 0.0 \
    --ee_pose_frame robot_base_full --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    >"$PIPE/eval_seed${seed}.log" 2>&1
}

printf 'RUNNING\n' >"$PIPE/status.txt"
run_seed 2000 2 & p2=$!
run_seed 3000 3 & p3=$!
wait "$p2"
wait "$p3"
printf 'COMPLETE\n' >"$PIPE/status.txt"
