#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
TAG=${TAG:-0809_contactloss_single15_v2_steps2000}
TARGET=${TARGET:-50}
SEED=${SEED:-615455575}
TRAIN_SEED=${TRAIN_SEED:-1000}

PIPE="$REPO/high-level/logs/pull-data-pipeline/wc4_pull_${TAG}_${TARGET}"
RAW_REL="high-level/data/door_dp_raw/a2w_wc4_pull_joint9_${TAG}_${TARGET}_seed${SEED}"
RAW="$REPO/$RAW_REL"
DATA="local/door_a2w_wc4_pull_joint9_${TAG}_${TARGET}_interaction_state"
DATA_ROOT="$REPO/high-level/data/lerobot/$DATA"
BASE_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_${TARGET}_baseline_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"
INTER_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_${TARGET}_plucker_interaction_decoderchunk_w005_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"

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
    if [[ ! -d "$RAW" ]]; then
        printf '0\n'
        return
    fi
    find "$RAW" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l
}

stage RECORD
existing=$(episode_count)
if (( existing < TARGET )); then
    record_seed=$SEED
    if (( existing > 0 )); then
        # Avoid replaying the same randomized environments after a resumed run.
        record_seed=$((SEED + 100000 + existing))
    fi
    CUDA_VISIBLE_DEVICES=0 "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
        --mode ikpull --num_episodes "$TARGET" --num_envs 16 \
        --max_quota_rollouts 800 --steps 2000 --seed "$record_seed" \
        --continue_after_cleanup_sigsegv --headless \
        --raw_root "$RAW_REL" --state_action_mode joint_state9 \
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
        --record_camera_pose --record_handle_bbox --record_gripper_handle_contact -- \
        --door_cfg high-level/data/cfg/b1z1_opendoor.yaml --door_name wc4 \
        --handle_rotate_steps 100 \
        --rotate_pull_overlap_steps 20 \
        --rotate_pull_overlap_ee_distance 0.01 \
        --pull_then_panel_push \
        --pull_panel_release_angle_randomization \
        --pull_panel_release_angle_min_deg 10 \
        --pull_panel_release_angle_max_deg 35 \
        --pull_panel_contact_loss_release \
        --pull_panel_contact_loss_min_angle_deg 10 \
        --pull_panel_contact_loss_min_frames 3 \
        --pull_contact_loss_defer_base_retreat \
        --pull_contact_loss_near_base_retreat_distance 0.36 \
        --pull_contact_loss_collision_retreat_angle_deg 35 \
        --pull_contact_loss_collision_retreat_distance 0.15 \
        --pull_contact_loss_collision_retreat_ramp_deg 5 \
        --pull_panel_adaptive_release_retreat \
        --pull_panel_low_angle_base_retreat_distance 0.36 \
        --pull_panel_low_angle_right_arc_radius 0.10 \
        --pull_panel_low_angle_edge_clearance 0.10 \
        --pull_panel_low_angle_outward_clearance 0.18 \
        --pull_panel_low_angle_reposition_steps 260 \
        --pull_panel_release_angle_deg 35 \
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
        --release_handle_steps 90 \
        --release_handle_motion_steps 90 \
        --release_to_clearance_pose \
        --release_clearance_retreat_distance 0.10 \
        --release_clearance_right_distance 0.14 \
        --release_clearance_lift_distance 0.03 \
        --release_clearance_retreat_fraction 0.30 \
        --release_clearance_low_angle_base_retreat_extra 0.15 \
        --release_clearance_base_retreat_extra 0.30 \
        --release_clearance_arc_radius_extra 0.08 \
        --release_clearance_edge_extra 0.06 \
        --release_clearance_outward_extra 0.06 \
        --panel_reposition_path right_arc \
        --panel_right_arc_continuous_spline \
        --panel_reposition_steps 170 \
        --panel_reposition_completion_gate \
        --panel_reposition_completion_backside_margin_m 0.01 \
        --panel_reposition_completion_position_tolerance_m 0.12 \
        --panel_reposition_completion_max_extra_steps 300 \
        --panel_reposition_base_forward_distance 0.0 \
        --panel_right_arc_joint1_hold_rad -1.0 \
        --panel_right_arc_radius 0.24 \
        --panel_right_arc_outward_clearance 0.24 \
        --panel_right_arc_lift 0 \
        --panel_edge_clearance 0.20 \
        --panel_contact_steps 80 \
        --panel_push_goal_angle_deg 80 \
        --panel_push_motion_mode hinge_arc \
        --panel_push_planned_arc_steps 320 \
        --panel_push_arc_lead_angle_deg 4 \
        --panel_contact_initial_penetration 0.03 \
        --panel_contact_penetration 0.12 \
        --panel_target_max_step_m 0.03 \
        --panel_base_follow_ratio 0.0 \
        --panel_base_follow_max_speed 0.15 \
        --panel_postpush_home_steps 50 \
        --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pass_open_angle_deg 80 --pull_record_traversal_distance_m 0.8 \
        --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
        --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
        --log_interval 100 >"$PIPE/record.log" 2>&1
fi

stage VALIDATE_RAW
"$PY" - "$RAW" "$TARGET" <<'PY'
import sys
from pathlib import Path
import numpy as np

root, target = Path(sys.argv[1]), int(sys.argv[2])
paths = sorted(root.glob("episode_*.npz"))
expected = [f"episode_{i:06d}.npz" for i in range(target)]
if [p.name for p in paths] != expected:
    raise RuntimeError(f"expected {target} contiguous episodes, got {len(paths)}")

frames = 0
lengths, release, opened, traversal, max_vy = [], [], [], [], []
for path in paths:
    with np.load(path, allow_pickle=True) as data:
        state, action = np.asarray(data["state"]), np.asarray(data["action"])
        if state.ndim != 2 or state.shape[1] != 9 or action.shape != state.shape:
            raise RuntimeError(f"{path.name}: bad state/action {state.shape}/{action.shape}")
        required = (
            "front_masked_depth", "wrist_masked_depth",
            "front_camera_pose_base", "wrist_camera_pose_base",
            "replay_door_dof_pos", "gripper_handle_contact_both",
            "pull_release_angle_observed_deg", "pull_max_open_deg",
            "pull_max_traversal_m", "pull_max_abs_body_vy_mps",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(f"{path.name}: missing {missing}")
        frames += len(state)
        lengths.append(len(state))
        release.append(float(np.asarray(data["pull_release_angle_observed_deg"]).reshape(-1)[0]))
        opened.append(float(np.asarray(data["pull_max_open_deg"]).reshape(-1)[0]))
        traversal.append(float(np.asarray(data["pull_max_traversal_m"]).reshape(-1)[0]))
        max_vy.append(float(np.asarray(data["pull_max_abs_body_vy_mps"]).reshape(-1)[0]))

if min(release) < 9.5 or min(opened) < 80.0 or min(traversal) < 0.8:
    raise RuntimeError(
        f"quality gate release/open/traversal="
        f"{min(release):.3f}/{min(opened):.3f}/{min(traversal):.3f}"
    )
if max(max_vy) > 1e-4:
    raise RuntimeError(f"vy constraint failed max={max(max_vy)}")
print(
    f"RAW_VALID episodes={target} frames={frames} "
    f"length=[{min(lengths)},{max(lengths)}] "
    f"release=[{min(release):.3f},{np.mean(release):.3f},{max(release):.3f}] "
    f"open_min={min(opened):.3f} traversal_min={min(traversal):.3f} "
    f"max_abs_vy={max(max_vy):.8f}"
)
PY

stage CONVERT
if [[ ! -s "$PIPE/convert.complete" ]]; then
    rm -rf -- "$DATA_ROOT"
    "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/convert_door_raw_to_lerobot.py \
        --raw_root "$RAW_REL" --root high-level/data/lerobot \
        --repo_id "$DATA" --state_action_mode joint_state9 \
        --depth_only --image_storage video --video_codec h264 \
        --add_interaction_state --num_workers 4 \
        >"$PIPE/convert.log" 2>&1
    printf 'episodes=%s\ncompleted_at=%s\n' "$TARGET" "$(date '+%F %T')" \
        >"$PIPE/convert.complete"
fi

stage WAIT_FOR_GPUS_2_3
while pgrep -f 'lerobot_train.py.*leroact_a2w_glass_door_joint9_skill200_' >/dev/null 2>&1; do
    stage "WAIT_FOR_GPUS_2_3 glass-door training still active"
    sleep 120
done

common_train=(
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false
    --policy.chunk_size=100 --policy.n_action_steps=50
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
    --policy.optimizer_lr=1e-5 --policy.use_action_loss_weight=false
    --batch_size=16 --steps=50000 --seed="$TRAIN_SEED" --num_workers=3
    --save_freq=10000 --log_freq=10 --keyframe_sampling_ratio=0.0
    --wandb.enable=true --wandb.project=door-act --wandb.disable_artifact=true
)

train_one() {
    local gpu=$1 name=$2 kind=$3
    local out="$REPO/high-level/dp/logs/lerobot-train/$name"
    [[ ! -s "$out/checkpoints/050000/pretrained_model/model.safetensors" ]] || return
    local extra=(
        --policy.plucker_conditioning=false
        --policy.interaction_state_conditioning=false
    )
    if [[ "$kind" == interaction ]]; then
        extra=(
            --policy.plucker_conditioning=true
            --policy.plucker_horizontal_fov_deg=55.0
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
        --dataset.root="$DATA_ROOT" --dataset.repo_id="$DATA" \
        "${common_train[@]}" "${extra[@]}" \
        --job_name="$name" --output_dir="$out" \
        >"$PIPE/train_${name}.log" 2>&1
}

stage TRAIN_TWO
train_one 2 "$BASE_JOB" baseline & p0=$!
train_one 3 "$INTER_JOB" interaction & p1=$!
wait "$p0" || fail "baseline training failed"
wait "$p1" || fail "interaction training failed"

export_one() {
    local gpu=$1 name=$2
    local official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/050000"
    local wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000"
    [[ ! -s "$wrapped/model_latest.pt" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$DATA" --out_dir "$wrapped/model_latest" \
        --manifest_name model_latest.pt --device cuda:0 --depth_only \
        >"$PIPE/export_${name}.log" 2>&1
}

stage EXPORT_TWO
export_one 2 "$BASE_JOB"
export_one 3 "$INTER_JOB"

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
        --rl_device cuda:0 --sim_device cuda:0 \
        --dp_action_horizon 25 --dp_fps 25 \
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

stage EVAL_TWO
eval_one 2 "$BASE_JOB" & e0=$!
eval_one 3 "$INTER_JOB" & e1=$!
wait "$e0" || fail "baseline evaluation failed"
wait "$e1" || fail "interaction evaluation failed"

stage SUMMARIZE
"$PY" - "$REPO" "$SEED" "$BASE_JOB" "$INTER_JOB" <<'PY' \
    | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

repo, seed = Path(sys.argv[1]), sys.argv[2]
for name in sys.argv[3:]:
    path = (
        repo / "high-level/logs/door-policy-success"
        / f"{name}_wc4_seed{seed}_open80_traverse0p8_64"
        / "summary.json"
    )
    data = json.loads(path.read_text())
    print(
        f"{name}: open={data['door_open_successes']}/{data['trials']} "
        f"traversal={data['traversal_successes']}/{data['trials']} "
        f"joint={data['successes']}/{data['trials']}"
    )
PY

stage COMPLETE
