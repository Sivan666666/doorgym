#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
cd "$REPO"

TAG=0804_pull30_panel80_minvy01
PIPE="$REPO/high-level/logs/pull-data-pipeline/wc4_pull_hybrid_panelpush_2x200_${TAG}"
mkdir -p "$PIPE"
exec >>"$PIPE/pipeline.log" 2>&1

RAW_NOVY_REL=high-level/data/door_dp_raw/a2w_wc4_pull30_panel80_novy_joint9_noisy_rand_200_${TAG}
RAW_WITHVY_REL=high-level/data/door_dp_raw/a2w_wc4_pull30_panel80_withvy_joint10_noisy_rand_200_${TAG}
RAW_NOVY="$REPO/$RAW_NOVY_REL"
RAW_WITHVY="$REPO/$RAW_WITHVY_REL"

DATA_NOVY=local/door_a2w_wc4_pull30_panel80_novy_joint9_200_${TAG}_interaction_state
DATA_WITHVY=local/door_a2w_wc4_pull30_panel80_withvy_joint10_200_${TAG}_interaction_state
DATA_NOVY_ROOT="$REPO/high-level/data/lerobot/$DATA_NOVY"
DATA_WITHVY_ROOT="$REPO/high-level/data/lerobot/$DATA_WITHVY"

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt"
}

fail() {
    stage "FAILED: $*"
    exit 1
}

common_record_args=(
    --mode ikpull
    --num_episodes 200
    --num_envs 16
    --max_quota_rollouts 500
    --steps 2200
    --seed "${RECORD_SEED:-615455604}"
    --continue_after_cleanup_sigsegv
    --headless
    --depth_only
    --camera_depth_clip_lower 0.2
    --camera_depth_clip_far 1.5
    --enable_depth_noise
    --depth_noise_prob 0.5
    --depth_gaussian_std_m 0.005
    --depth_gaussian_distance_factor 0.05
    --depth_edge_noise_prob 0.1
    --depth_edge_gradient_threshold_m 0.05
    --depth_edge_dilation_kernel_size 3
    --depth_hole_noise_prob 0.005
    --depth_hole_block_size_min 3
    --depth_hole_block_size 15
    --depth_hole_white_prob 0.5
    --depth_dropout_prob 0.002
    --depth_salt_pepper_prob 0.002
    --no_enable_depth_gaussian_blur
    --enable_depth_camera_randomization
    --depth_camera_pos_rand_m 0.02
    --depth_camera_rot_rand_deg 5.0
    --keyframe_loss_weight 3
    --keyframe_loss_radius 3
    --record_camera_pose
    --record_handle_bbox
    --record_gripper_handle_contact
    --filter_gripper_handle_contact
    --filter_gripper_handle_contact_min_frames 5
    --filter_gripper_handle_contact_phase_names close_gripper,rotate_handle
    --filter_gripper_handle_contact_require_both
)

common_pull_args=(
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml
    --door_name wc4
    --pull_then_panel_push
    --pull_panel_release_angle_deg 30
    --panel_push_goal_angle_deg 80
    --panel_contact_penetration 0.10
    --panel_base_follow_ratio 0.65
    --panel_base_follow_min_lateral_speed 0.10
    --panel_base_follow_max_lateral_speed 0.14
    --wc4_disable_door_open_resistance
    --door_auto_open_force 0
    --pass_open_angle_deg 80
    --pull_record_traversal_distance_m 0.8
    --pull_release_angle_tolerance_deg 0
    --release_handle_steps 50
    --release_handle_motion_steps 50
    --robot_pitch 0.0
    --ikpush_robot_pitch_rand_min 0.0
    --ikpush_robot_pitch_rand_max 0.0
    --ee_pose_frame robot_base_full
    --log_interval 100
)

stage RECORD_BOTH
for raw in "$RAW_NOVY" "$RAW_WITHVY"; do
    if [[ -e "$raw" ]]; then
        count=$(find "$raw" -maxdepth 1 -name 'episode_*.npz' | wc -l)
        (( count <= 200 )) || fail "raw path contains more than 200 episodes: $raw ($count)"
        stage "RESUME_RECORDING $raw count=$count/200 seed=${RECORD_SEED:-615455604}"
    fi
done

CUDA_VISIBLE_DEVICES=0 "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
    "${common_record_args[@]}" \
    --raw_root "$RAW_NOVY_REL" \
    --state_action_mode joint_state9 \
    --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
    -- "${common_pull_args[@]}" --pull_panel_base_mode no_vy \
    >>"$PIPE/record_novy.log" 2>&1 &
record_novy_pid=$!

CUDA_VISIBLE_DEVICES=1 "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
    "${common_record_args[@]}" \
    --raw_root "$RAW_WITHVY_REL" \
    --state_action_mode joint_state10 \
    --rl_device cuda:0 --sim_device cuda:0 --graphics_device_id 0 \
    -- "${common_pull_args[@]}" --pull_panel_base_mode with_vy \
    >>"$PIPE/record_withvy.log" 2>&1 &
record_withvy_pid=$!

set +e
wait "$record_novy_pid"; record_novy_rc=$?
wait "$record_withvy_pid"; record_withvy_rc=$?
set -e
[[ "$record_novy_rc" -eq 0 && "$record_withvy_rc" -eq 0 ]] || \
    fail "recording rc: no_vy=$record_novy_rc with_vy=$record_withvy_rc"

stage VALIDATE_RAW
"$PY" - "$RAW_NOVY" 9 no_vy "$RAW_WITHVY" 10 with_vy <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

for offset in (1, 4):
    root = Path(sys.argv[offset])
    dim = int(sys.argv[offset + 1])
    mode = sys.argv[offset + 2]
    paths = sorted(root.glob("episode_*.npz"))
    expected = [f"episode_{i:06d}.npz" for i in range(200)]
    if [p.name for p in paths] != expected:
        raise RuntimeError(f"{mode}: expected 200 contiguous episodes, got {len(paths)}")
    max_vys = []
    active_vys = []
    frames = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            state = np.asarray(data["state"])
            action = np.asarray(data["action"])
            if state.ndim != 2 or state.shape[1] != dim or action.shape != state.shape:
                raise RuntimeError(f"{path.name}: bad state/action {state.shape}/{action.shape}")
            for key in (
                "front_masked_depth", "wrist_masked_depth",
                "front_camera_pose_base", "wrist_camera_pose_base",
                "replay_door_dof_pos", "gripper_handle_contact_both",
                "pull_release_angle_observed_deg", "pull_max_open_deg",
                "pull_max_traversal_m", "pull_max_abs_body_vy_mps",
            ):
                if key not in data:
                    raise RuntimeError(f"{path.name}: missing {key}")
            release = float(np.asarray(data["pull_release_angle_observed_deg"]).reshape(-1)[0])
            opened = float(np.asarray(data["pull_max_open_deg"]).reshape(-1)[0])
            traversal = float(np.asarray(data["pull_max_traversal_m"]).reshape(-1)[0])
            max_vy = float(np.asarray(data["pull_max_abs_body_vy_mps"]).reshape(-1)[0])
            if release < 30.0 or opened < 80.0 or traversal < 0.8:
                raise RuntimeError(
                    f"{path.name}: release/open/traversal={release}/{opened}/{traversal}"
                )
            max_vys.append(max_vy)
            if mode == "with_vy":
                action_vy = np.abs(action[:, 1])
                state_vy = np.abs(state[:, 1])
                active_vys.extend(action_vy[action_vy > 0.0].tolist())
                active_vys.extend(state_vy[state_vy > 0.0].tolist())
            frames += state.shape[0]
    if mode == "no_vy" and max(max_vys) > 1e-4:
        raise RuntimeError(f"no_vy lateral motion is nonzero: {max(max_vys)}")
    if mode == "with_vy":
        if not active_vys:
            raise RuntimeError("with_vy dataset contains no lateral command")
        min_active_vy = min(active_vys)
        if min_active_vy < 0.1 - 1e-5:
            raise RuntimeError(
                f"with_vy contains a non-executable lateral command: min={min_active_vy}"
            )
    else:
        min_active_vy = 0.0
    print(
        f"RAW_VALID mode={mode} episodes=200 frames={frames} dim={dim} "
        f"min_active_vy={min_active_vy:.6f} max_vy={max(max_vys):.6f}"
    )
PY

stage CONVERT_BOTH
for root in "$DATA_NOVY_ROOT" "$DATA_WITHVY_ROOT"; do
    [[ ! -e "$root" ]] || fail "dataset already exists: $root"
done

"$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/convert_door_raw_to_lerobot.py \
    --raw_root "$RAW_NOVY_REL" --root high-level/data/lerobot \
    --repo_id "$DATA_NOVY" --state_action_mode joint_state9 \
    --depth_only --image_storage video --video_codec h264 \
    --add_interaction_state --num_workers 4 \
    >"$PIPE/convert_novy.log" 2>&1 &
convert_novy_pid=$!

"$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/convert_door_raw_to_lerobot.py \
    --raw_root "$RAW_WITHVY_REL" --root high-level/data/lerobot \
    --repo_id "$DATA_WITHVY" --state_action_mode joint_state10 \
    --depth_only --image_storage video --video_codec h264 \
    --add_interaction_state --num_workers 4 \
    >"$PIPE/convert_withvy.log" 2>&1 &
convert_withvy_pid=$!

set +e
wait "$convert_novy_pid"; convert_novy_rc=$?
wait "$convert_withvy_pid"; convert_withvy_rc=$?
set -e
[[ "$convert_novy_rc" -eq 0 && "$convert_withvy_rc" -eq 0 ]] || \
    fail "conversion rc: no_vy=$convert_novy_rc with_vy=$convert_withvy_rc"

train_names=(
    leroact_a2w_wc4_pull30_panel80_novy_joint9_200_baseline_seed1000_50k_${TAG}
    leroact_a2w_wc4_pull30_panel80_novy_joint9_200_plucker_interaction_seed1000_50k_${TAG}
    leroact_a2w_wc4_pull30_panel80_withvy_joint10_200_baseline_seed1000_50k_${TAG}
    leroact_a2w_wc4_pull30_panel80_withvy_joint10_200_plucker_interaction_seed1000_50k_${TAG}
)
train_data=("$DATA_NOVY" "$DATA_NOVY" "$DATA_WITHVY" "$DATA_WITHVY")
train_gpu=(0 1 2 3)

common_train=(
    --dataset.video_backend=torchcodec
    --dataset.use_imagenet_stats=false
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.chunk_size=100
    --policy.n_action_steps=50
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
    --policy.use_action_loss_weight=false
    --batch_size=16
    --steps=50000
    --seed=1000
    --num_workers=4
    --save_freq=10000
    --log_freq=10
    --keyframe_sampling_ratio=0.0
    --wandb.enable=true
    --wandb.project=door-act
    --wandb.disable_artifact=true
)

stage TRAIN_FOUR
pids=()
for i in 0 1 2 3; do
    name="${train_names[$i]}"
    data="${train_data[$i]}"
    gpu="${train_gpu[$i]}"
    out="$REPO/high-level/dp/logs/lerobot-train/$name"
    [[ ! -e "$out" ]] || fail "training output already exists: $out"
    extra=(--policy.plucker_conditioning=false --policy.interaction_state_conditioning=false)
    if (( i % 2 == 1 )); then
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
        --dataset.root="$REPO/high-level/data/lerobot/$data" \
        --dataset.repo_id="$data" \
        "${common_train[@]}" "${extra[@]}" \
        --job_name="$name" --output_dir="$out" \
        >"$PIPE/train_${i}_${name}.log" 2>&1 &
    pids+=("$!")
done

for i in 0 1 2 3; do
    wait "${pids[$i]}" || fail "training failed: ${train_names[$i]}"
done

stage EXPORT_FOUR
for i in 0 1 2 3; do
    name="${train_names[$i]}"
    data="${train_data[$i]}"
    gpu="${train_gpu[$i]}"
    official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/050000"
    wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000"
    [[ -s "$official/pretrained_model/model.safetensors" ]] || fail "missing 50K: $official"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$data" --out_dir "$wrapped/model_latest" \
        --manifest_name model_latest.pt --device cuda:0 --depth_only \
        >"$PIPE/export_${i}_${name}.log" 2>&1
done

stage EVAL_FOUR
eval_pids=()
for i in 0 1 2 3; do
    name="${train_names[$i]}"
    gpu="${train_gpu[$i]}"
    mode=no_vy
    if (( i >= 2 )); then mode=with_vy; fi
    ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000/model_latest.pt"
    run="$REPO/high-level/logs/door-policy-success/${name}_wc4_seed615455575_open80_traverse0p8_64"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
        --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
        --steps 2200 --pass_open_angle_deg 80 --pull_traversal_distance_m 0.8 \
        --base_seed 615455575 --headless --graphics_device_id 0 \
        --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
        --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 --run_root "$run" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pull_panel_base_mode "$mode" \
        --panel_base_follow_min_lateral_speed 0.10 \
        --panel_base_follow_max_lateral_speed 0.14 \
        --robot_pitch 0.0 \
        --ikpush_robot_pitch_rand_min 0.0 --ikpush_robot_pitch_rand_max 0.0 \
        --ee_pose_frame robot_base_full --camera_depth_clip_lower 0.2 \
        --camera_depth_clip_far 1.5 \
        >"$PIPE/eval_${i}_${name}.log" 2>&1 &
    eval_pids+=("$!")
done
for i in 0 1 2 3; do
    wait "${eval_pids[$i]}" || fail "evaluation failed: ${train_names[$i]}"
done

stage SUMMARIZE
"$PY" - "$REPO" "${train_names[@]}" <<'PY' | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
for name in sys.argv[2:]:
    path = repo / "high-level/logs/door-policy-success" / f"{name}_wc4_seed615455575_open80_traverse0p8_64" / "summary.json"
    data = json.loads(path.read_text())
    print(
        f"{name}: open={data['door_open_successes']}/{data['trials']} "
        f"traversal={data['traversal_successes']}/{data['trials']} "
        f"joint={data['successes']}/{data['trials']}"
    )
PY

stage COMPLETE
