#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
TAG=${TAG:-0809_overlap20_slowhome_out16_v1}
SEED=${SEED:-615455575}
TRAIN_SEED=${TRAIN_SEED:-1000}
PIPE="$REPO/high-level/logs/pull-data-pipeline/wc4_pull_${TAG}_nested50_200"

SHARD_SEEDS=("$SEED" "$((SEED + 100000))" "$((SEED + 200000))" "$((SEED + 300000))")
SHARD_RELS=()
SHARD_ROOTS=()
for shard in 0 1 2 3; do
    SHARD_RELS+=("high-level/data/door_dp_raw/a2w_wc4_pull_joint9_${TAG}_shard${shard}_50_seed${SHARD_SEEDS[$shard]}")
    SHARD_ROOTS+=("$REPO/${SHARD_RELS[$shard]}")
done
RAW50_REL=${SHARD_RELS[0]}
RAW50=${SHARD_ROOTS[0]}

DATA50="local/door_a2w_wc4_pull_joint9_${TAG}_50_interaction_state"
DATA200="local/door_a2w_wc4_pull_joint9_${TAG}_200_interaction_state"
DATA50_ROOT="$REPO/high-level/data/lerobot/$DATA50"
DATA200_ROOT="$REPO/high-level/data/lerobot/$DATA200"

BASE50="leroact_a2w_wc4_pull_joint9_${TAG}_50_baseline_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"
INTER50="leroact_a2w_wc4_pull_joint9_${TAG}_50_plucker_interaction_decoderchunk_w005_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"
BASE200="leroact_a2w_wc4_pull_joint9_${TAG}_200_baseline_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"
INTER200="leroact_a2w_wc4_pull_joint9_${TAG}_200_plucker_interaction_decoderchunk_w005_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"

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
        printf '0\n'
        return
    fi
    find -L "$root" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l
}

record_shard() {
    local gpu=$1 shard=$2
    local raw_rel=${SHARD_RELS[$shard]}
    local raw=${SHARD_ROOTS[$shard]}
    local shard_seed=${SHARD_SEEDS[$shard]}
    local existing
    existing=$(episode_count "$raw")
    (( existing < 50 )) || return 0
    # A restarted shard must not replay the exact same rollout seeds that
    # produced its existing episodes.
    if (( existing > 0 )); then
        shard_seed=$((shard_seed + 10000 + existing))
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
        --mode ikpull --num_episodes 50 --num_envs 16 \
        --max_quota_rollouts 400 --steps 2000 --seed "$shard_seed" \
        --continue_after_cleanup_sigsegv --headless \
        --raw_root "$raw_rel" --state_action_mode joint_state9 \
        --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
        --depth_only --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        --enable_depth_noise --depth_noise_prob 0.5 \
        --depth_gaussian_std_m 0.005 --depth_gaussian_distance_factor 0.05 \
        --depth_edge_noise_prob 0.1 --depth_edge_gradient_threshold_m 0.05 \
        --depth_edge_dilation_kernel_size 3 \
        --depth_hole_noise_prob 0.005 --depth_hole_block_size_min 3 \
        --depth_hole_block_size 15 --depth_hole_white_prob 0.5 \
        --depth_dropout_prob 0.002 --depth_salt_pepper_prob 0.002 \
        --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 \
        --keyframe_loss_weight 3 --keyframe_loss_radius 3 \
        --record_camera_pose --record_handle_bbox --record_gripper_handle_contact \
        --filter_gripper_handle_contact \
        --filter_gripper_handle_contact_min_frames 5 \
        --filter_gripper_handle_contact_phase_names close_gripper,rotate_handle \
        --filter_gripper_handle_contact_require_both -- \
        --door_cfg high-level/data/cfg/b1z1_opendoor.yaml --door_name wc4 \
        --handle_rotate_steps 100 \
        --rotate_pull_overlap_steps 20 \
        --rotate_pull_overlap_ee_distance 0.01 \
        --pull_then_panel_push --pull_panel_release_angle_deg 35 \
        --pull_release_angle_tolerance_deg 0 \
        --pull_vx_only_base --pull_panel_base_mode no_vy \
        --pull_base_retreat_distance 0.48 \
        --pull_base_progress_reference_angle_deg 42 \
        --pull_late_retreat_start_angle_deg 22 \
        --pull_late_retreat_distance 0.12 \
        --pull_late_retreat_steps 110 \
        --pull_base_follow_max_speed 0.36 \
        --pull_base_follow_position_gain 2.5 \
        --pull_vx_monotonic_retreat \
        --pull_vx_monotonic_max_accel_mps2 0.8 \
        --release_handle_steps 60 --release_handle_motion_steps 60 \
        --release_gripper_open_start_fraction 0.0 \
        --release_gripper_open_fraction 0.05 \
        --release_outward_distance 0.14 \
        --release_outward_fraction 0.20 \
        --release_extra_outward_distance 0.02 \
        --release_extra_outward_end_fraction 0.24 \
        --release_joint_home_start_fraction 0.25 \
        --release_joint_home_motion_profile trapezoid \
        --release_joint_home_ramp_fraction 0.18 \
        --panel_reposition_path right_arc \
        --panel_right_arc_continuous_spline \
        --panel_reposition_steps 150 \
        --panel_reposition_base_forward_distance=0.0 \
        --panel_right_arc_joint1_hold_rad=-1.0 \
        --panel_right_arc_radius 0.24 \
        --panel_right_arc_outward_clearance 0.24 \
        --panel_right_arc_lift 0 --panel_edge_clearance 0.20 \
        --panel_contact_steps 80 --panel_push_goal_angle_deg 80 \
        --panel_push_motion_mode hinge_arc --panel_push_planned_arc_steps 320 \
        --panel_push_arc_lead_angle_deg 4 \
        --panel_contact_initial_penetration 0.03 \
        --panel_contact_penetration 0.12 --panel_target_max_step_m 0.03 \
        --panel_base_follow_ratio 1.0 --panel_base_follow_max_speed 0.3 \
        --panel_postpush_home_steps 50 \
        --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pass_open_angle_deg 80 --pull_record_traversal_distance_m 0.8 \
        --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
        --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
        --log_interval 100 >"$PIPE/record_shard${shard}.log" 2>&1
}

stage RECORD_FOUR_SHARDS
record_shard 0 0 & r0=$!
record_shard 1 1 & r1=$!
record_shard 2 2 & r2=$!
record_shard 3 3 & r3=$!
wait "$r0" || fail "record shard0 failed"
wait "$r1" || fail "record shard1 failed"
wait "$r2" || fail "record shard2 failed"
wait "$r3" || fail "record shard3 failed"

stage VALIDATE_FOUR_SHARDS
"$PY" - "${SHARD_ROOTS[@]}" <<'PY'
import sys
from pathlib import Path
import numpy as np

roots = [Path(arg) for arg in sys.argv[1:]]
if len(roots) != 4:
    raise RuntimeError(f"expected four raw shards, got {len(roots)}")
all_paths = []
for root in roots:
    paths = sorted(root.glob("episode_*.npz"))
    expected_names = [f"episode_{i:06d}.npz" for i in range(50)]
    if [path.name for path in paths] != expected_names:
        raise RuntimeError(f"{root}: expected 50 contiguous episodes, got {len(paths)}")
    all_paths.extend(paths)
lengths, releases, opens, traversals, vys = [], [], [], [], []
for path in all_paths:
    with np.load(path, allow_pickle=True) as d:
        state, action = np.asarray(d["state"]), np.asarray(d["action"])
        if state.ndim != 2 or state.shape[1] != 9 or action.shape != state.shape:
            raise RuntimeError(f"{path.name}: bad state/action {state.shape}/{action.shape}")
        required = (
            "front_masked_depth", "wrist_masked_depth",
            "front_camera_pose_base", "wrist_camera_pose_base",
            "replay_door_dof_pos", "gripper_handle_contact_both",
            "pull_release_angle_observed_deg", "pull_max_open_deg",
            "pull_max_traversal_m", "pull_max_abs_body_vy_mps",
        )
        missing = [key for key in required if key not in d]
        if missing:
            raise RuntimeError(f"{path.name}: missing {missing}")
        lengths.append(len(state))
        releases.append(float(np.asarray(d["pull_release_angle_observed_deg"]).reshape(-1)[0]))
        opens.append(float(np.asarray(d["pull_max_open_deg"]).reshape(-1)[0]))
        traversals.append(float(np.asarray(d["pull_max_traversal_m"]).reshape(-1)[0]))
        vys.append(float(np.asarray(d["pull_max_abs_body_vy_mps"]).reshape(-1)[0]))
if min(releases) < 34.5 or min(opens) < 80 or min(traversals) < 0.8:
    raise RuntimeError(
        f"quality gate release/open/traversal={min(releases):.3f}/{min(opens):.3f}/{min(traversals):.3f}"
    )
if max(vys) > 1e-4:
    raise RuntimeError(f"vy constraint failed: {max(vys)}")
print(
    f"RAW_VALID shards={len(roots)} episodes={len(all_paths)} frames={sum(lengths)} "
    f"length=[{min(lengths)},{max(lengths)}] release_min={min(releases):.3f} "
    f"open_min={min(opens):.3f} traversal_min={min(traversals):.3f} max_vy={max(vys):.8f}"
)
PY

convert_one() {
    local scope=$1 data=$2 data_root=$3 marker=$4
    [[ ! -s "$marker" ]] || return
    rm -rf -- "$data_root"
    local raw_args=(--raw_root "${SHARD_RELS[0]}")
    if [[ "$scope" == 200 ]]; then
        raw_args+=(
            --additional_raw_root "${SHARD_RELS[1]}"
            --additional_raw_root "${SHARD_RELS[2]}"
            --additional_raw_root "${SHARD_RELS[3]}"
        )
    fi
    "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/convert_door_raw_to_lerobot.py \
        "${raw_args[@]}" --root high-level/data/lerobot \
        --repo_id "$data" --state_action_mode joint_state9 \
        --depth_only --image_storage video --video_codec h264 \
        --add_interaction_state --num_workers 4 \
        >"${marker%.complete}.log" 2>&1
    printf 'completed_at=%s\n' "$(date '+%F %T')" >"$marker"
}

stage CONVERT_50
convert_one 50 "$DATA50" "$DATA50_ROOT" "$PIPE/convert_50.complete"
stage CONVERT_200
convert_one 200 "$DATA200" "$DATA200_ROOT" "$PIPE/convert_200.complete"

common_train=(
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false
    --policy.chunk_size=100 --policy.n_action_steps=50
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
    --policy.use_action_loss_weight=false
    --batch_size=16 --steps=50000 --seed="$TRAIN_SEED" --num_workers=3
    --save_freq=10000 --log_freq=10 --keyframe_sampling_ratio=0.0
    --wandb.enable=true --wandb.project=door-act --wandb.disable_artifact=true
)

train_one() {
    local gpu=$1 data=$2 data_root=$3 name=$4 kind=$5
    local out="$REPO/high-level/dp/logs/lerobot-train/$name"
    [[ ! -s "$out/checkpoints/050000/pretrained_model/model.safetensors" ]] || return
    local extra=(--policy.plucker_conditioning=false --policy.interaction_state_conditioning=false)
    if [[ "$kind" == interaction ]]; then
        extra=(
            --policy.plucker_conditioning=true --policy.plucker_horizontal_fov_deg=55.0
            --policy.interaction_state_conditioning=true
            --policy.interaction_state_prediction_mode=decoder_chunk
            --policy.interaction_state_probe_only=false
            --policy.interaction_state_auxiliary_only=false
            --policy.interaction_contact_loss_weight=0.05
            --policy.interaction_handle_loss_weight=0.05
            --policy.interaction_door_loss_weight=0.05
        )
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
        --dataset.root="$data_root" --dataset.repo_id="$data" \
        "${common_train[@]}" "${extra[@]}" \
        --job_name="$name" --output_dir="$out" >"$PIPE/train_${name}.log" 2>&1
}

stage TRAIN_FOUR
train_one 0 "$DATA50" "$DATA50_ROOT" "$BASE50" baseline & p0=$!
train_one 1 "$DATA50" "$DATA50_ROOT" "$INTER50" interaction & p1=$!
train_one 2 "$DATA200" "$DATA200_ROOT" "$BASE200" baseline & p2=$!
train_one 3 "$DATA200" "$DATA200_ROOT" "$INTER200" interaction & p3=$!
wait "$p0" || fail "50 baseline training failed"
wait "$p1" || fail "50 interaction training failed"
wait "$p2" || fail "200 baseline training failed"
wait "$p3" || fail "200 interaction training failed"

export_one() {
    local gpu=$1 data=$2 name=$3
    local official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/050000"
    local wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000"
    [[ ! -s "$wrapped/model_latest.pt" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$data" --out_dir "$wrapped/model_latest" \
        --manifest_name model_latest.pt --device cuda:0 --depth_only \
        >"$PIPE/export_${name}.log" 2>&1
}

stage EXPORT_FOUR
export_one 0 "$DATA50" "$BASE50"
export_one 1 "$DATA50" "$INTER50"
export_one 2 "$DATA200" "$BASE200"
export_one 3 "$DATA200" "$INTER200"

eval_one() {
    local gpu=$1 name=$2
    local ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000/model_latest.pt"
    local run="$REPO/high-level/logs/door-policy-success/${name}_wc4_seed${SEED}_open80_traverse0p8_64"
    [[ ! -s "$run/summary.json" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
        --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
        --steps 2600 --pass_open_angle_deg 80 --pull_traversal_distance_m 0.8 \
        --base_seed "$SEED" --headless --graphics_device_id 0 \
        --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
        --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 --run_root "$run" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pull_vx_only_base --pull_panel_base_mode no_vy \
        --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
        --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
        --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        >"$PIPE/eval_${name}.log" 2>&1
}

stage EVAL_FOUR
eval_one 0 "$BASE50" & e0=$!
eval_one 1 "$INTER50" & e1=$!
eval_one 2 "$BASE200" & e2=$!
eval_one 3 "$INTER200" & e3=$!
wait "$e0" || fail "50 baseline evaluation failed"
wait "$e1" || fail "50 interaction evaluation failed"
wait "$e2" || fail "200 baseline evaluation failed"
wait "$e3" || fail "200 interaction evaluation failed"

stage SUMMARIZE
"$PY" - "$REPO" "$SEED" "$BASE50" "$INTER50" "$BASE200" "$INTER200" <<'PY' \
    | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

repo, seed = Path(sys.argv[1]), sys.argv[2]
for name in sys.argv[3:]:
    path = repo / "high-level/logs/door-policy-success" / f"{name}_wc4_seed{seed}_open80_traverse0p8_64" / "summary.json"
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

stage COMPLETE
