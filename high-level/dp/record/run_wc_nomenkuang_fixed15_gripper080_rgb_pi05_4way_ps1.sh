#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
TRAIN_ENV=${TRAIN_ENV:-b1z1_lerobot}
SIM_ENV=${SIM_ENV:-txc_vbc}
BASE_SEED=${BASE_SEED:-615455575}
TAG=${TAG:-0829_wc_nomenkuang_fixed15_gripper080_rgb_v1}
ARCHIVE=${ARCHIVE:-/mnt/sda/ps_archive/doorgym}
PIPE="$ARCHIVE/high-level/logs/rgb_pi05_pipeline/$TAG"
LR_ROOT="$ARCHIVE/high-level/data/lerobot"
TRAIN_ROOT="$ARCHIVE/high-level/dp/logs/lerobot-train-pi05"
EVAL_ROOT="$ARCHIVE/high-level/logs/door-policy-success"
PRETRAINED=${PRETRAINED:-$REPO/high-level/dp/models/pi05_base_cached_localtokenizer}

cd "$REPO"
mkdir -p "$PIPE" "$LR_ROOT/local" "$TRAIN_ROOT" "$EVAL_ROOT" high-level/data/lerobot/local
exec >>"$PIPE/pipeline.log" 2>&1

stage() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt"
}

fail() {
  stage "FAILED: $*"
  exit 1
}

episode_count() {
  local root=$1
  if [[ ! -d "$root" ]]; then
    echo 0
  else
    find -L "$root" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l
  fi
}

TRAJECTORY_ARGS=(
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml
  --door_name wc_nomenkuang --stop_distance 0.293
  --ikpush_door_wall_x_offset_rand 0
  --handle_rotate_steps 100
  --rotate_pull_overlap_steps 20 --rotate_pull_overlap_ee_distance 0.01
  --pull_then_panel_push
  --pull_panel_release_angle_deg 15 --pull_release_angle_tolerance_deg 0
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
  --pull_late_retreat_distance 0.12 --pull_late_retreat_steps 110
  --pull_base_follow_max_speed 0.36 --pull_base_follow_position_gain 2.5
  --pull_vx_monotonic_retreat --pull_vx_monotonic_max_accel_mps2 0.8
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
  --panel_reposition_path right_arc --panel_right_arc_continuous_spline
  --panel_reposition_steps 170 --panel_reposition_completion_gate
  --panel_reposition_completion_backside_margin_m 0.01
  --panel_reposition_completion_position_tolerance_m 0.12
  --panel_reposition_completion_max_extra_steps 300
  --panel_reposition_base_forward_distance 0
  --panel_right_arc_joint1_hold_rad -1.0
  --panel_right_arc_radius 0.27
  --panel_right_arc_outward_clearance 0.27
  --panel_right_arc_lift 0 --panel_edge_clearance 0.22
  --panel_contact_steps 80 --panel_push_goal_angle_deg 80
  --panel_push_motion_mode hinge_arc --panel_push_planned_arc_steps 320
  --panel_push_arc_lead_angle_deg 4
  --panel_contact_initial_penetration 0.03 --panel_contact_penetration 0.12
  --panel_target_max_step_m 0.03 --panel_base_follow_ratio 0
  --panel_base_follow_max_speed 0.15 --panel_postpush_home_steps 50
  --wc4_disable_door_open_resistance --door_auto_open_force 0
  --pass_open_angle_deg 80 --pull_record_traversal_distance_m 0.8
  --gripper_close_ratio 0.8
  --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 --ikpush_robot_pitch_rand_max 0
  --ee_pose_frame robot_base_full --log_interval 100
)

SHARD_RELS=()
for shard in 0 1 2 3; do
  seed=$((BASE_SEED + shard * 100000))
  SHARD_RELS+=("high-level/data/door_dp_raw/a2w_${TAG}_shard${shard}_50_seed${seed}")
done

record_shard() {
  local gpu=$1 shard=$2
  local raw=${SHARD_RELS[$shard]}
  local seed=$((BASE_SEED + shard * 100000))
  local existing
  existing=$(episode_count "$raw")
  if (( existing >= 50 )); then
    return 0
  fi
  if (( existing > 0 )); then
    seed=$((seed + 10000 + existing))
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n "$SIM_ENV" python \
    high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
    --mode ikpull --num_episodes 50 --num_envs 4 \
    --max_quota_rollouts 1000 --steps 2600 --seed "$seed" \
    --continue_after_cleanup_sigsegv --headless \
    --raw_root "$raw" --state_action_mode joint_state9 \
    --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
    --rgb_only --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --keyframe_loss_weight 3 --keyframe_loss_radius 3 \
    --record_camera_pose --record_handle_bbox --record_gripper_handle_contact \
    --filter_gripper_handle_contact --filter_gripper_handle_contact_min_frames 5 \
    --filter_gripper_handle_contact_phase_names close_gripper,rotate_handle \
    --filter_gripper_handle_contact_require_both -- \
    "${TRAJECTORY_ARGS[@]}" >"$PIPE/record_shard${shard}.log" 2>&1
}

stage RECORD_RGB_200_START
record_shard 0 0 & p0=$!
record_shard 2 1 & p1=$!
wait "$p0" || fail "RGB shard0 recording failed"
wait "$p1" || fail "RGB shard1 recording failed"
record_shard 0 2 & p2=$!
record_shard 2 3 & p3=$!
wait "$p2" || fail "RGB shard2 recording failed"
wait "$p3" || fail "RGB shard3 recording failed"

stage VALIDATE_RGB_RAW
"$CONDA" run --no-capture-output -n "$SIM_ENV" python - "${SHARD_RELS[@]}" <<'PY'
import sys
from pathlib import Path
import numpy as np

paths=[]
for root_value in sys.argv[1:]:
    root=Path(root_value)
    shard=sorted(root.glob("episode_*.npz"))
    expected=[f"episode_{i:06d}.npz" for i in range(50)]
    if [p.name for p in shard] != expected:
        raise RuntimeError(f"{root}: expected 50 contiguous episodes, got {len(shard)}")
    paths.extend(shard)
lengths=[]; opened=[]; traversed=[]; releases=[]
for path in paths:
    with np.load(path, allow_pickle=True) as data:
        state=np.asarray(data["state"]); action=np.asarray(data["action"])
        if state.ndim != 2 or state.shape[1] != 9 or action.shape != state.shape:
            raise RuntimeError(f"{path}: state/action={state.shape}/{action.shape}")
        for key in ("front_rgb","wrist_rgb","front_camera_pose_base","wrist_camera_pose_base",
                    "replay_door_dof_pos","gripper_handle_contact_both"):
            if key not in data:
                raise RuntimeError(f"{path}: missing {key}")
        if data["front_rgb"].shape[1:] != (480,640,3) or data["wrist_rgb"].shape[1:] != (480,640,3):
            raise RuntimeError(f"{path}: invalid RGB shape")
        lengths.append(len(state))
        releases.append(float(np.asarray(data["pull_release_angle_observed_deg"]).reshape(-1)[0]))
        opened.append(float(np.asarray(data["pull_max_open_deg"]).reshape(-1)[0]))
        traversed.append(float(np.asarray(data["pull_max_traversal_m"]).reshape(-1)[0]))
if len(paths) != 200 or min(releases) < 14.5 or max(releases) > 16.0:
    raise RuntimeError(f"episode/release gate failed: n={len(paths)} release={min(releases)}..{max(releases)}")
if min(opened) < 80.0 or min(traversed) < 0.8:
    raise RuntimeError(f"task-quality gate failed: open={min(opened)} traversal={min(traversed)}")
print(f"episodes={len(paths)} frames={sum(lengths)} length={min(lengths)}..{max(lengths)} "
      f"release={min(releases):.3f}..{max(releases):.3f} open_min={min(opened):.3f} "
      f"traversal_min={min(traversed):.3f}")
PY

RGB50="local/door_a2w_${TAG}_50_joint9"
RGB200="local/door_a2w_${TAG}_200_joint9"
RGB50_ROOT="$LR_ROOT/$RGB50"
RGB200_ROOT="$LR_ROOT/$RGB200"

convert_rgb() {
  local count=$1 repo_id=$2
  local args=(--raw_root "${SHARD_RELS[0]}")
  if (( count >= 100 )); then args+=(--additional_raw_root "${SHARD_RELS[1]}"); fi
  if (( count >= 150 )); then args+=(--additional_raw_root "${SHARD_RELS[2]}"); fi
  if (( count >= 200 )); then args+=(--additional_raw_root "${SHARD_RELS[3]}"); fi
  "$CONDA" run --no-capture-output -n "$TRAIN_ENV" python \
    high-level/dp/convert_door_raw_to_lerobot.py "${args[@]}" \
    --root "$LR_ROOT" --repo_id "$repo_id" --state_action_mode joint_state9 \
    --rgb_only --image_storage video --video_codec h264 --num_workers 4 --overwrite \
    >"$PIPE/convert_${count}.log" 2>&1
}

stage CONVERT_RGB_50
convert_rgb 50 "$RGB50"
stage CONVERT_RGB_200
convert_rgb 200 "$RGB200"

ln -sfn "$RGB50_ROOT" "$REPO/high-level/data/lerobot/$RGB50"
ln -sfn "$RGB200_ROOT" "$REPO/high-level/data/lerobot/$RGB200"

"$CONDA" run --no-capture-output -n "$TRAIN_ENV" python - "$RGB50_ROOT" "$RGB200_ROOT" <<'PY'
import json,sys
for root in sys.argv[1:]:
    side=json.load(open(root+"/door_dp_feature_names.json"))
    expected=["observation.images.wrist_rgb","observation.images.front_rgb"]
    if side.get("image_features") != expected:
        raise RuntimeError(f"{root}: image_features={side.get('image_features')}")
    info=json.load(open(root+"/meta/info.json"))
    images=[k for k,v in info["features"].items() if v.get("dtype") in ("video","image")]
    if images != expected:
        raise RuntimeError(f"{root}: visual dataset features={images}")
    print(root, "episodes=",info["total_episodes"],"frames=",info["total_frames"],"images=",images)
PY

DEPTH50="local/door_a2w_0813_wc_nomenkuang_fixed15_observable_v1_nested_50"
DEPTH200="local/door_a2w_0813_wc_nomenkuang_fixed15_observable_v1_nested_200"
DEPTH50_ROOT="$LR_ROOT/$DEPTH50"
DEPTH200_ROOT="$LR_ROOT/$DEPTH200"
for root in "$DEPTH50_ROOT" "$DEPTH200_ROOT"; do
  [[ -f "$root/meta/info.json" ]] || fail "missing existing ratio0.8 depth dataset: $root"
done
ln -sfn "$DEPTH50_ROOT" "$REPO/high-level/data/lerobot/$DEPTH50"
ln -sfn "$DEPTH200_ROOT" "$REPO/high-level/data/lerobot/$DEPTH200"

RUN_NAMES=(
  "pi05_pull_fixed15_g080_depth50_joint9_seed1000_30k"
  "pi05_pull_fixed15_g080_depth200_joint9_seed1000_30k"
  "pi05_pull_fixed15_g080_rgb50_joint9_seed1000_30k"
  "pi05_pull_fixed15_g080_rgb200_joint9_seed1000_30k"
)
DATASETS=("$DEPTH50" "$DEPTH200" "$RGB50" "$RGB200")
DATA_ROOTS=("$DEPTH50_ROOT" "$DEPTH200_ROOT" "$RGB50_ROOT" "$RGB200_ROOT")

train_one() {
  local gpu=$1 index=$2
  local run=${RUN_NAMES[$index]} dataset=${DATASETS[$index]} data_root=${DATA_ROOTS[$index]}
  local out="$TRAIN_ROOT/$run"
  if [[ -f "$out/checkpoints/030000/pretrained_model/model.safetensors" || -f "$out/checkpoints/last/pretrained_model/model.safetensors" ]]; then
    return 0
  fi
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$CONDA" run --no-capture-output -n "$TRAIN_ENV" \
    accelerate launch --num-processes=1 --mixed_precision=bf16 \
    /home/ps/miniconda3/envs/b1z1_lerobot/bin/lerobot-train \
    --dataset.root="$data_root" --dataset.repo_id="$dataset" \
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false \
    --policy.type=pi05 --policy.device=cuda --policy.pretrained_path="$PRETRAINED" \
    --policy.dtype=bfloat16 --policy.push_to_hub=false --policy.gradient_checkpointing=true \
    --policy.chunk_size=100 --policy.n_action_steps=50 --policy.num_inference_steps=10 \
    --batch_size=16 --steps=30000 --num_workers=4 --save_freq=10000 --log_freq=50 \
    --seed=1000 --job_name="$run" --output_dir="$out" \
    --wandb.enable=false --wandb.disable_artifact=true \
    >"$PIPE/train_${index}_${run}.log" 2>&1
}

stage TRAIN_PI05_4WAY_START
train_one 0 0 & t0=$!
train_one 1 1 & t1=$!
train_one 2 2 & t2=$!
train_one 3 3 & t3=$!
wait "$t0" || fail "depth50 pi0.5 training failed"
wait "$t1" || fail "depth200 pi0.5 training failed"
wait "$t2" || fail "rgb50 pi0.5 training failed"
wait "$t3" || fail "rgb200 pi0.5 training failed"

eval_one() {
  local gpu=$1 index=$2
  local run=${RUN_NAMES[$index]} data=${DATASETS[$index]}
  local checkpoint="$TRAIN_ROOT/$run/checkpoints/030000"
  [[ -f "$checkpoint/pretrained_model/model.safetensors" ]] || checkpoint="$TRAIN_ROOT/$run/checkpoints/last"
  local out="$EVAL_ROOT/${run}_pull_open60_traverse080_seed${BASE_SEED}"
  local vision=(--depth_only)
  if (( index >= 2 )); then vision=(--rgb); fi
  rm -rf "$out"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DOOR_AUTO_WRAP_ROOT="$ARCHIVE/high-level/dp/logs/door-auto-wrapped" \
    "$CONDA" run --no-capture-output -n "$SIM_ENV" python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$checkpoint" --mode ikpull --robot_body a2wz1 \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --num_envs 16 --total_trials 64 --steps 2600 --base_seed "$BASE_SEED" \
    --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 \
    --dp_action_horizon 25 --dp_inference_steps 10 --dp_fps 25 \
    --headless --parallel_batches 1 --run_root "$out" "${vision[@]}" -- \
    --door_name wc_nomenkuang \
    --wc4_disable_door_open_resistance --door_auto_open_force 0 \
    --gripper_close_ratio 0.8 --robot_pitch 0 \
    --ikpush_robot_pitch_rand_min 0 --ikpush_robot_pitch_rand_max 0 \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur --no_show_seg \
    $([[ "$index" -ge 2 ]] && printf '%s' '--dp_rgb_without_masks') \
    >"$PIPE/eval_${index}_${run}.log" 2>&1
}

stage EVAL_PI05_4WAY_START
eval_one 0 0 & e0=$!
eval_one 1 1 & e1=$!
eval_one 2 2 & e2=$!
eval_one 3 3 & e3=$!
wait "$e0" || fail "depth50 pi0.5 eval failed"
wait "$e1" || fail "depth200 pi0.5 eval failed"
wait "$e2" || fail "rgb50 pi0.5 eval failed"
wait "$e3" || fail "rgb200 pi0.5 eval failed"

stage SUMMARIZE
"$CONDA" run --no-capture-output -n "$SIM_ENV" python - "$EVAL_ROOT" "$BASE_SEED" "${RUN_NAMES[@]}" <<'PY' | tee "$PIPE/final_results.txt"
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); seed=sys.argv[2]
for run in sys.argv[3:]:
    d=root/f"{run}_pull_open60_traverse080_seed{seed}"
    summary=json.load(open(d/"summary.json"))
    print(run)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
PY
stage COMPLETE
