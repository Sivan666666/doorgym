#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
PY=${PY:-/home/ps/miniconda3/envs/txc_vbc/bin/python}
LR_PY=${LR_PY:-/home/ps/miniconda3/envs/b1z1_lerobot/bin/python}
BASE_SEED=${BASE_SEED:-615455575}
TAG=${TAG:-0813_wc_nomenkuang_fixed15_observable_v1}
PIPE="$REPO/high-level/logs/pull-data-pipeline/${TAG}_500_phasebalanced_multiseed"
DATA="local/door_a2w_${TAG}_500_interaction_state"
DATA_ROOT="$REPO/high-level/data/lerobot/$DATA"
PHASE_WEIGHTS=${PHASE_WEIGHTS:-8:2,11:0.1,12:2,13:2,14:2}
TRAIN_SEEDS=(1000 2000)
CHECKPOINTS=(100000 200000 300000 500000)

cd "$REPO"
mkdir -p "$PIPE"
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

SHARD_RELS=()
SHARD_ROOTS=()
SHARD_SEEDS=()
for shard in $(seq 0 9); do
  seed=$((BASE_SEED + shard * 100000))
  SHARD_SEEDS+=("$seed")
  SHARD_RELS+=("high-level/data/door_dp_raw/a2w_${TAG}_shard${shard}_50_seed${seed}")
  SHARD_ROOTS+=("$REPO/${SHARD_RELS[$shard]}")
done

TRAJECTORY_ARGS=(
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml
  --door_name wc_nomenkuang --stop_distance 0.293
  --ikpush_door_wall_x_offset_rand 0
  --handle_rotate_steps 100
  --rotate_pull_overlap_steps 20
  --rotate_pull_overlap_ee_distance 0.01
  --pull_then_panel_push
  --pull_panel_release_angle_deg 15
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
  --log_interval 100
)

record_shard() {
  local gpu=$1 shard=$2
  local raw_rel=${SHARD_RELS[$shard]}
  local raw=${SHARD_ROOTS[$shard]}
  local seed=${SHARD_SEEDS[$shard]}
  local existing
  existing=$(episode_count "$raw")
  (( existing < 50 )) || return 0
  if (( existing > 0 )); then
    seed=$((seed + 10000 + existing))
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
    --mode ikpull --num_episodes 50 --num_envs 16 \
    --max_quota_rollouts 1000 --steps 2600 --seed "$seed" \
    --continue_after_cleanup_sigsegv --headless \
    --raw_root "$raw_rel" --state_action_mode joint_state9 \
    --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
    --depth_only --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
    --enable_depth_noise --depth_noise_prob 0.5 \
    --depth_gaussian_std_m 0.005 --depth_gaussian_distance_factor 0.05 \
    --depth_edge_noise_prob 0.1 --depth_edge_gradient_threshold_m 0.05 \
    --depth_edge_dilation_kernel_size 3 --depth_hole_noise_prob 0.005 \
    --depth_hole_block_size_min 3 --depth_hole_block_size 15 \
    --depth_hole_white_prob 0.5 --depth_dropout_prob 0.002 \
    --depth_salt_pepper_prob 0.002 --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --keyframe_loss_weight 3 --keyframe_loss_radius 3 \
    --record_camera_pose --record_handle_bbox --record_gripper_handle_contact \
    --filter_gripper_handle_contact \
    --filter_gripper_handle_contact_min_frames 5 \
    --filter_gripper_handle_contact_phase_names close_gripper,rotate_handle \
    --filter_gripper_handle_contact_require_both -- \
    "${TRAJECTORY_ARGS[@]}" >"$PIPE/record_shard${shard}.log" 2>&1
}

record_worker() {
  local gpu=$1
  shift
  for shard in "$@"; do
    stage "RECORD gpu=$gpu shard=$shard existing=$(episode_count "${SHARD_ROOTS[$shard]}")/50"
    record_shard "$gpu" "$shard"
  done
}

stage RECORD_500_FIXED15
record_worker 0 0 4 8 & r0=$!
record_worker 1 1 5 9 & r1=$!
record_worker 2 2 6 & r2=$!
record_worker 3 3 7 & r3=$!
wait "$r0" || fail "record worker GPU0 failed"
wait "$r1" || fail "record worker GPU1 failed"
wait "$r2" || fail "record worker GPU2 failed"
wait "$r3" || fail "record worker GPU3 failed"

stage VALIDATE_RAW
"$PY" - "${SHARD_ROOTS[@]}" <<'PY' | tee "$PIPE/raw_validation.txt"
import sys
from pathlib import Path
import numpy as np

paths=[]
for root_value in sys.argv[1:]:
    root=Path(root_value)
    shard=sorted(root.glob("episode_*.npz"))
    expected=[f"episode_{i:06d}.npz" for i in range(50)]
    if [p.name for p in shard] != expected:
        raise RuntimeError(f"{root}: non-contiguous {len(shard)} episodes")
    paths.extend(shard)
lengths=[]; releases=[]; opened=[]; traversed=[]; max_vy=[]
for path in paths:
    with np.load(path, allow_pickle=True) as data:
        state=np.asarray(data["state"]); action=np.asarray(data["action"])
        if state.ndim != 2 or state.shape[1] != 9 or action.shape != state.shape:
            raise RuntimeError(f"{path}: state/action={state.shape}/{action.shape}")
        required=("front_masked_depth","wrist_masked_depth","front_camera_pose_base",
                  "wrist_camera_pose_base","replay_door_dof_pos",
                  "gripper_handle_contact_both","pull_release_angle_observed_deg",
                  "pull_max_open_deg","pull_max_traversal_m","pull_max_abs_body_vy_mps")
        missing=[key for key in required if key not in data]
        if missing: raise RuntimeError(f"{path}: missing {missing}")
        lengths.append(len(state))
        releases.append(float(np.asarray(data["pull_release_angle_observed_deg"]).reshape(-1)[0]))
        opened.append(float(np.asarray(data["pull_max_open_deg"]).reshape(-1)[0]))
        traversed.append(float(np.asarray(data["pull_max_traversal_m"]).reshape(-1)[0]))
        max_vy.append(float(np.asarray(data["pull_max_abs_body_vy_mps"]).reshape(-1)[0]))
if len(paths) != 500 or min(releases) < 14.5 or max(releases) > 16.0:
    raise RuntimeError(f"episode/release gate: n={len(paths)} range={min(releases)}/{max(releases)}")
if min(opened) < 80 or min(traversed) < .8 or max(max_vy) > 1e-4:
    raise RuntimeError("task-quality gate failed")
print(f"episodes={len(paths)} frames={sum(lengths)} length={min(lengths)}..{max(lengths)} "
      f"release={min(releases):.3f}..{max(releases):.3f} "
      f"open_min={min(opened):.3f} traversal_min={min(traversed):.3f} max_vy={max(max_vy):.8f}")
PY

stage CONVERT_500
if [[ ! -s "$PIPE/convert.complete" ]]; then
  additional=()
  for shard in $(seq 1 9); do
    additional+=(--additional_raw_root "${SHARD_RELS[$shard]}")
  done
  "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/convert_door_raw_to_lerobot.py \
    --raw_root "${SHARD_RELS[0]}" "${additional[@]}" \
    --root high-level/data/lerobot --repo_id "$DATA" \
    --state_action_mode joint_state9 --depth_only \
    --image_storage video --video_codec h264 \
    --add_interaction_state --num_workers 4 >"$PIPE/convert.log" 2>&1
  date >"$PIPE/convert.complete"
fi

stage PHASE_DISTRIBUTION
"$LR_PY" - "$DATA_ROOT" "$PHASE_WEIGHTS" <<'PY' | tee "$PIPE/phase_sampling_distribution.tsv"
import sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

root=Path(sys.argv[1]); configured={int(k):float(v) for k,v in (x.split(":") for x in sys.argv[2].split(","))}
values=[]
for path in sorted((root/"data").glob("chunk-*/file-*.parquet")):
    values.extend(pq.read_table(path, columns=["subtask_index"]).column(0).to_pylist())
values=[int(v[0] if isinstance(v,list) else v) for v in values]
counts=Counter(values); total=sum(counts.values())
weighted=sum(n*configured.get(k,1.0) for k,n in counts.items())
print("phase_id\tframes\tnatural_ratio\tweight\texpected_ratio")
for phase in sorted(counts):
    n=counts[phase]; w=configured.get(phase,1.0)
    print(f"{phase}\t{n}\t{n/total:.8f}\t{w:g}\t{n*w/weighted:.8f}")
PY

job_name() {
  local seed=$1
  echo "leroact_a2w_${TAG}_500_plucker_interaction_phasebalanced_hold01_critical2_seed${seed}_500k_chunk100_exec50_bs16"
}

train_seed() {
  local gpu=$1 seed=$2 name out
  name=$(job_name "$seed")
  out="$REPO/high-level/dp/logs/lerobot-train/$name"
  [[ ! -s "$out/checkpoints/500000/pretrained_model/model.safetensors" ]] || return 0
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.root="$DATA_ROOT" --dataset.repo_id="$DATA" \
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=100 --policy.n_action_steps=50 \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.use_action_loss_weight=false \
    --policy.plucker_conditioning=true --policy.plucker_horizontal_fov_deg=55 \
    --policy.interaction_state_conditioning=true \
    --policy.interaction_state_prediction_mode=decoder_chunk \
    --policy.interaction_state_probe_only=false \
    --policy.interaction_state_auxiliary_only=false \
    --policy.interaction_contact_loss_weight=0.05 \
    --policy.interaction_handle_loss_weight=0.05 \
    --policy.interaction_door_loss_weight=0.05 \
    --batch_size=16 --steps=500000 --seed="$seed" --num_workers=3 \
    --save_freq=100000 --log_freq=10 --keyframe_sampling_ratio=0 \
    --phase_sampling_feature=subtask_index \
    --phase_sampling_weights="$PHASE_WEIGHTS" \
    --phase_sampling_default_weight=1 \
    --wandb.enable=true --wandb.project=door-act --wandb.disable_artifact=true \
    --job_name="$name" --output_dir="$out" >"$PIPE/train_seed${seed}.log" 2>&1
}

stage TRAIN_TWO_SEEDS_TO_500K
train_seed 0 1000 & t0=$!
train_seed 1 2000 & t1=$!
wait "$t0" || fail "training seed1000 failed"
wait "$t1" || fail "training seed2000 failed"

export_one() {
  local gpu=$1 seed=$2 step=$3 name official wrapped
  name=$(job_name "$seed")
  official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/$step"
  wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/$step"
  [[ -s "$wrapped/model_latest.pt" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
    --official_checkpoint "$official" --root high-level/data/lerobot \
    --repo_id "$DATA" --out_dir "$wrapped/model_latest" \
    --manifest_name model_latest.pt --device cuda:0 --depth_only \
    >"$PIPE/export_seed${seed}_${step}.log" 2>&1
}

eval_one() {
  local gpu=$1 seed=$2 step=$3 trials=$4 suffix=$5 name ckpt run
  name=$(job_name "$seed")
  ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$name/$step/model_latest.pt"
  run="$REPO/high-level/logs/door-policy-success/${name}_step${step}_wc_nomenkuang_seed${BASE_SEED}_open80_traverse0p8_${suffix}"
  [[ -s "$run/summary.json" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
    --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials "$trials" \
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
    >"$PIPE/eval_seed${seed}_${step}_${suffix}.log" 2>&1
}

eval_curve_seed() {
  local gpu=$1 seed=$2 step
  for step in "${CHECKPOINTS[@]}"; do
    stage "EXPORT_EVAL64 seed=$seed step=$step gpu=$gpu"
    export_one "$gpu" "$seed" "$step"
    eval_one "$gpu" "$seed" "$step" 64 64
  done
}

stage EVAL_64_CURVES
eval_curve_seed 2 1000 & e0=$!
eval_curve_seed 3 2000 & e1=$!
wait "$e0" || fail "64-curve seed1000 failed"
wait "$e1" || fail "64-curve seed2000 failed"

stage SELECT_COMMON_BEST_STEP
"$PY" - "$REPO" "$BASE_SEED" "$TAG" "${CHECKPOINTS[@]}" <<'PY' | tee "$PIPE/curve64.tsv"
import json,sys
from pathlib import Path
repo=Path(sys.argv[1]); base=sys.argv[2]; tag=sys.argv[3]; steps=sys.argv[4:]
print("step\tseed1000_joint\tseed2000_joint\tcombined_joint_rate")
scores=[]
for step in steps:
    vals=[]
    for seed in (1000,2000):
        name=f"leroact_a2w_{tag}_500_plucker_interaction_phasebalanced_hold01_critical2_seed{seed}_500k_chunk100_exec50_bs16"
        run=repo/"high-level/logs/door-policy-success"/f"{name}_step{step}_wc_nomenkuang_seed{base}_open80_traverse0p8_64"/"summary.json"
        d=json.loads(run.read_text()); vals.append((int(d["successes"]),int(d["trials"])))
    rate=sum(v[0] for v in vals)/sum(v[1] for v in vals)
    scores.append((rate,int(step)))
    print(f"{step}\t{vals[0][0]}/{vals[0][1]}\t{vals[1][0]}/{vals[1][1]}\t{rate:.8f}")
best=max(scores)[1]
(repo/"high-level/logs/pull-data-pipeline"/f"{tag}_500_phasebalanced_multiseed"/"best_step.txt").write_text(str(best)+"\n")
print(f"BEST_STEP\t{best}")
PY
BEST_STEP=$(cat "$PIPE/best_step.txt")

stage EVAL_SELECTED_256 step=$BEST_STEP
eval_one 2 1000 "$BEST_STEP" 256 256 & f0=$!
eval_one 3 2000 "$BEST_STEP" 256 256 & f1=$!
wait "$f0" || fail "256 eval seed1000 failed"
wait "$f1" || fail "256 eval seed2000 failed"

stage SUMMARIZE
"$PY" - "$REPO" "$BASE_SEED" "$TAG" "$BEST_STEP" <<'PY' | tee "$PIPE/final_results.txt"
import json,sys
from pathlib import Path
repo=Path(sys.argv[1]); base=sys.argv[2]; tag=sys.argv[3]; step=sys.argv[4]
for seed in (1000,2000):
    name=f"leroact_a2w_{tag}_500_plucker_interaction_phasebalanced_hold01_critical2_seed{seed}_500k_chunk100_exec50_bs16"
    run=repo/"high-level/logs/door-policy-success"/f"{name}_step{step}_wc_nomenkuang_seed{base}_open80_traverse0p8_256"/"summary.json"
    d=json.loads(run.read_text()); n=int(d["trials"]); o=int(d["door_open_successes"]); t=int(d["traversal_successes"]); j=int(d["successes"])
    print(f"seed={seed} step={step} open={o}/{n} ({o/n:.4%}) traversal={t}/{n} ({t/n:.4%}) joint={j}/{n} ({j/n:.4%})")
PY

stage COMPLETE

