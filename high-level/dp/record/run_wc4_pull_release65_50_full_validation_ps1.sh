#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
cd "$REPO"

RAW_REL=high-level/data/door_dp_raw/a2w_wc4_pull_joint9_release65_novy_noinertia_noisy_rand_50_v3_seed615455575
RAW="$REPO/$RAW_REL"
DATASET_REPO=local/door_a2w_wc4_pull_joint9_release65_novy_noinertia_noisy_rand_50_v3_interaction_state
DATASET_ROOT="$REPO/high-level/data/lerobot/$DATASET_REPO"

TAG=0802_release65_novy_v3
BASE_JOB=leroact_a2w_wc4_pull_joint9_release65_50_baseline_seed1000_50k_chunk100_exec50_bs16_${TAG}
INTER_JOB=leroact_a2w_wc4_pull_joint9_release65_50_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_${TAG}
TRAIN_ROOT="$REPO/high-level/dp/logs/lerobot-train"
WRAP_ROOT="$REPO/high-level/dp/logs/door-auto-wrapped"
BASE_RUN="$TRAIN_ROOT/$BASE_JOB"
INTER_RUN="$TRAIN_ROOT/$INTER_JOB"
BASE_WRAP="$WRAP_ROOT/$BASE_JOB/050000"
INTER_WRAP="$WRAP_ROOT/$INTER_JOB/050000"

EVAL_ROOT="$REPO/high-level/logs/door-policy-success"
BASE_EVAL="$EVAL_ROOT/wc4_pull_release65_novy_50_baseline_seed615455575_open60_traverse0p8_64"
INTER_EVAL="$EVAL_ROOT/wc4_pull_release65_novy_50_plucker_interaction_seed615455575_open60_traverse0p8_64"

PIPE="$REPO/high-level/logs/pull-data-pipeline/wc4_pull_release65_novy_50_full_validation_20260802"
STATUS="$PIPE/status.txt"
mkdir -p "$PIPE"
exec >>"$PIPE/pipeline.log" 2>&1

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$STATUS"
}

fail() {
    stage "FAILED: $*"
    exit 1
}

stage RECORD
"$CONDA" run --no-capture-output -n txc_vbc python \
    "$REPO/high-level/dp/record/record_door_dp_dataset_a2w_state10.py" \
    --mode ikpull \
    --num_episodes 50 \
    --num_envs 16 \
    --max_quota_rollouts 100 \
    --raw_root "$RAW_REL" \
    --state_action_mode joint_state9 \
    --steps 1750 \
    --seed 615455575 \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --graphics_device_id 0 \
    --headless \
    --depth_only \
    --camera_depth_clip_lower 0.2 \
    --camera_depth_clip_far 1.5 \
    --enable_depth_noise \
    --depth_noise_prob 0.5 \
    --depth_gaussian_std_m 0.005 \
    --depth_gaussian_distance_factor 0.05 \
    --depth_edge_noise_prob 0.1 \
    --depth_edge_gradient_threshold_m 0.05 \
    --depth_edge_dilation_kernel_size 3 \
    --depth_hole_noise_prob 0.005 \
    --depth_hole_block_size_min 3 \
    --depth_hole_block_size 15 \
    --depth_hole_white_prob 0.5 \
    --depth_dropout_prob 0.002 \
    --depth_salt_pepper_prob 0.002 \
    --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --keyframe_loss_weight 3 \
    --keyframe_loss_radius 3 \
    --record_camera_pose \
    --record_handle_bbox \
    --record_gripper_handle_contact \
    --filter_gripper_handle_contact \
    --filter_gripper_handle_contact_min_frames 5 \
    --filter_gripper_handle_contact_phase_names close_gripper,rotate_handle \
    --filter_gripper_handle_contact_require_both \
    -- \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --door_name wc4 \
    --wc4_disable_door_open_resistance \
    --door_auto_open_force 0 \
    --pass_open_angle_deg 60 \
    --pull_record_traversal_distance_m 0.8 \
    --pull_release_angle_deg 65 \
    --pull_release_angle_tolerance_deg 0 \
    --pull_late_base_lateral_follow_ratio 1.8 \
    --pull_base_follow_max_speed 0.32 \
    --pull_base_follow_max_yaw_rate 0.30 \
    --release_handle_steps 50 \
    --release_handle_motion_steps 50 \
    --robot_pitch 0.0 \
    --ikpush_robot_pitch_rand_min 0.0 \
    --ikpush_robot_pitch_rand_max 0.0 \
    --ee_pose_frame robot_base_full \
    --log_interval 100 \
    >"$PIPE/record.log" 2>&1 || fail "recording failed"

stage VALIDATE_RAW
"$PY" - "$RAW" <<'PY'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
paths = sorted(root.glob("episode_*.npz"))
expected = [f"episode_{i:06d}.npz" for i in range(50)]
if [p.name for p in paths] != expected:
    raise RuntimeError(f"Expected 50 contiguous episodes, got {len(paths)}")
rows = []
for path in paths:
    with np.load(path, allow_pickle=True) as data:
        state = np.asarray(data["state"])
        action = np.asarray(data["action"])
        required = {
            "front_masked_depth",
            "wrist_masked_depth",
            "front_camera_pose_base",
            "wrist_camera_pose_base",
            "replay_door_dof_pos",
            "gripper_handle_contact_both",
            "pull_max_abs_body_vy_mps",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"{path.name}: missing {missing}")
        release = float(data["pull_release_angle_observed_deg"])
        opened = float(data["pull_max_open_deg"])
        traversal = float(data["pull_max_traversal_m"])
        body_vy = float(data["pull_max_abs_body_vy_mps"])
        if state.ndim != 2 or state.shape[1] != 9 or action.shape != state.shape:
            raise RuntimeError(f"{path.name}: invalid Joint9 shapes {state.shape}/{action.shape}")
        if release < 65.0 or opened < 60.0 or traversal < 0.8 or body_vy > 1.0e-4:
            raise RuntimeError(
                f"{path.name}: invalid pull metrics "
                f"release={release} open={opened} traversal={traversal} body_vy={body_vy}"
            )
        rows.append((len(state), release, opened, traversal, body_vy))
a = np.asarray(rows)
print(
    f"RAW_VALID episodes={len(paths)} frames={int(a[:,0].sum())} "
    f"release={a[:,1].min():.4f}/{a[:,1].mean():.4f}/{a[:,1].max():.4f} "
    f"open={a[:,2].min():.4f}/{a[:,2].mean():.4f}/{a[:,2].max():.4f} "
    f"traversal={a[:,3].min():.4f}/{a[:,3].mean():.4f}/{a[:,3].max():.4f} "
    f"body_vy_max={a[:,4].max():.8f}"
)
PY

if [[ ! -d "$DATASET_ROOT/meta" ]]; then
    stage CONVERT
    "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        "$REPO/high-level/dp/convert_door_raw_to_lerobot.py" \
        --raw_root "$RAW_REL" \
        --root "$REPO/high-level/data/lerobot" \
        --repo_id "$DATASET_REPO" \
        --depth_only \
        --image_storage video \
        --video_codec h264 \
        --add_interaction_state \
        --num_workers 4 \
        >"$PIPE/convert.log" 2>&1 || fail "conversion failed"
else
    stage CONVERT_REUSE_EXISTING
fi

stage VALIDATE_DATASET
"$PY" - "$DATASET_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
if int(info["total_episodes"]) != 50:
    raise RuntimeError(info["total_episodes"])
if info["features"]["observation.state"]["shape"] != [9]:
    raise RuntimeError("state is not Joint9")
if info["features"]["action"]["shape"] != [9]:
    raise RuntimeError("action is not Joint9")
for key in (
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
):
    if key not in info["features"]:
        raise RuntimeError(f"missing {key}")
print(f"DATASET_VALID episodes={info['total_episodes']} frames={info['total_frames']}")
PY

for output in "$BASE_RUN" "$INTER_RUN"; do
    [[ ! -e "$output" ]] || fail "refusing to overwrite $output"
done

COMMON=(
    "--dataset.root=$DATASET_ROOT"
    "--dataset.repo_id=$DATASET_REPO"
    "--dataset.video_backend=torchcodec"
    "--dataset.use_imagenet_stats=false"
    "--policy.type=act"
    "--policy.device=cuda"
    "--policy.push_to_hub=false"
    "--policy.chunk_size=100"
    "--policy.n_action_steps=50"
    "--policy.vision_backbone=resnet18"
    "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1"
    "--policy.use_action_loss_weight=false"
    "--batch_size=16"
    "--steps=50000"
    "--seed=1000"
    "--num_workers=4"
    "--save_freq=10000"
    "--log_freq=10"
    "--keyframe_sampling_ratio=0.0"
    "--wandb.enable=false"
)

stage TRAIN_BOTH
CUDA_VISIBLE_DEVICES=1 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/lerobot/src/lerobot/scripts/lerobot_train.py" \
    "${COMMON[@]}" \
    --policy.plucker_conditioning=false \
    --policy.interaction_state_conditioning=false \
    "--job_name=$BASE_JOB" \
    "--output_dir=$BASE_RUN" \
    >"$PIPE/train_baseline.log" 2>&1 &
base_pid=$!

CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/lerobot/src/lerobot/scripts/lerobot_train.py" \
    "${COMMON[@]}" \
    --policy.plucker_conditioning=true \
    --policy.plucker_horizontal_fov_deg=55.0 \
    --policy.interaction_state_conditioning=true \
    --policy.interaction_state_prediction_mode=decoder_chunk \
    --policy.interaction_state_probe_only=false \
    --policy.interaction_state_auxiliary_only=false \
    --policy.interaction_contact_loss_weight=0.05 \
    --policy.interaction_handle_loss_weight=0.05 \
    --policy.interaction_door_loss_weight=0.05 \
    "--job_name=$INTER_JOB" \
    "--output_dir=$INTER_RUN" \
    >"$PIPE/train_interaction.log" 2>&1 &
inter_pid=$!
printf 'baseline_pid=%s interaction_pid=%s\n' "$base_pid" "$inter_pid"

set +e
wait "$base_pid"; base_rc=$?
wait "$inter_pid"; inter_rc=$?
set -e
[[ "$base_rc" -eq 0 && "$inter_rc" -eq 0 ]] || fail "training rc=$base_rc/$inter_rc"
[[ -s "$BASE_RUN/checkpoints/050000/pretrained_model/model.safetensors" ]] || fail "baseline checkpoint missing"
[[ -s "$INTER_RUN/checkpoints/050000/pretrained_model/model.safetensors" ]] || fail "interaction checkpoint missing"

stage EXPORT
CUDA_VISIBLE_DEVICES=1 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/dp/export_official_lerobot_act_to_door_checkpoint.py" \
    --official_checkpoint "$BASE_RUN/checkpoints/050000" \
    --root "$REPO/high-level/data/lerobot" --repo_id "$DATASET_REPO" \
    --out_dir "$BASE_WRAP/model_latest" --manifest_name model_latest.pt \
    --device cuda:0 --depth_only >"$PIPE/export_baseline.log" 2>&1 || fail "baseline export failed"

CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/dp/export_official_lerobot_act_to_door_checkpoint.py" \
    --official_checkpoint "$INTER_RUN/checkpoints/050000" \
    --root "$REPO/high-level/data/lerobot" --repo_id "$DATASET_REPO" \
    --out_dir "$INTER_WRAP/model_latest" --manifest_name model_latest.pt \
    --device cuda:0 --depth_only >"$PIPE/export_interaction.log" 2>&1 || fail "interaction export failed"

eval_one() {
    local ckpt="$1" out="$2" gpu="$3" log="$4"
    rm -rf "$out"
    "$CONDA" run --no-capture-output -n txc_vbc python \
        "$REPO/high-level/dp/eval/eval_door_policy_success.py" \
        --checkpoint "$ckpt" \
        --yaml "$REPO/high-level/data/cfg/b1z1_opendoor.yaml" \
        --mode ikpull --robot_body a2wz1 \
        --num_envs 16 --total_trials 64 --steps 1750 \
        --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 \
        --base_seed 615455575 --headless \
        --graphics_device_id "$gpu" --rl_device "cuda:$gpu" --sim_device "cuda:$gpu" \
        --dp_action_horizon 25 --dp_fps 25 --depth_only \
        --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization \
        --depth_camera_pos_rand_m 0.02 --depth_camera_rot_rand_deg 5.0 \
        --run_root "$out" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --robot_pitch 0.0 --ikpush_robot_pitch_rand_min 0.0 --ikpush_robot_pitch_rand_max 0.0 \
        --ee_pose_frame robot_base_full \
        --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        >"$PIPE/$log" 2>&1
}

stage EVAL_BOTH
eval_one "$BASE_WRAP/model_latest.pt" "$BASE_EVAL" 1 eval_baseline.log & ep0=$!
eval_one "$INTER_WRAP/model_latest.pt" "$INTER_EVAL" 2 eval_interaction.log & ep1=$!
set +e
wait "$ep0"; erc0=$?
wait "$ep1"; erc1=$?
set -e
[[ "$erc0" -eq 0 && "$erc1" -eq 0 ]] || fail "evaluation rc=$erc0/$erc1"

stage SUMMARIZE
"$PY" - "$BASE_EVAL/summary.json" "$INTER_EVAL/summary.json" >"$PIPE/final_results.txt" <<'PY'
import json, sys
for label, path in zip(("Baseline", "Plucker+Interaction"), sys.argv[1:]):
    d = json.load(open(path))
    batches = [(x["door_open_successes"], x["traversal_successes"], x["successes"]) for x in d["batch_logs"]]
    print(
        f"{label}: open={d['door_open_successes']}/{d['trials']} "
        f"traversal={d['traversal_successes']}/{d['trials']} "
        f"joint={d['successes']}/{d['trials']} batches={batches}"
    )
PY
cat "$PIPE/final_results.txt"
stage COMPLETE
