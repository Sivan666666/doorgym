#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
TXC_PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
cd "$REPO"

RAW_REL=high-level/data/door_dp_raw/a2w_wc4_pull_joint9_noresistance_noisy_rand_200_seed615455575
RAW="$REPO/$RAW_REL"
DATASET_REPO=local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_interaction_state
DATASET_ROOT="$REPO/high-level/data/lerobot/$DATASET_REPO"

RECORD_LOG="$REPO/high-level/logs/pull-data-pipeline/record_wc4_pull_joint9_200_seed615455575_20260731.log"
RECORD_PID_FILE="$RECORD_LOG.pid"
PIPELINE_ROOT="$REPO/high-level/logs/pull-data-pipeline/wc4_pull_joint9_200_20260731"
STATUS="$PIPELINE_ROOT/status.txt"
MAIN_LOG="$PIPELINE_ROOT/pipeline.log"

BASE_JOB=leroact_a2w_wc4_pull_joint9_noisy_rand200_baseline_seed1000_50k_chunk100_exec50_bs16_0731
INTER_JOB=leroact_a2w_wc4_pull_joint9_noisy_rand200_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0731
BASE_RUN="$REPO/high-level/dp/logs/lerobot-train/$BASE_JOB"
INTER_RUN="$REPO/high-level/dp/logs/lerobot-train/$INTER_JOB"
BASE_WRAP="$REPO/high-level/dp/logs/door-auto-wrapped/$BASE_JOB/050000"
INTER_WRAP="$REPO/high-level/dp/logs/door-auto-wrapped/$INTER_JOB/050000"

BASE_EVAL="$REPO/high-level/logs/door-policy-success/wc4_pull_joint9_rand200_baseline_seed615455575_open60_traverse0p8_64"
INTER_EVAL="$REPO/high-level/logs/door-policy-success/wc4_pull_joint9_rand200_plucker_interaction_seed615455575_open60_traverse0p8_64"

mkdir -p "$PIPELINE_ROOT"
exec >>"$MAIN_LOG" 2>&1

stage() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee "$STATUS"
}

fail() {
    stage "FAILED: $*"
    exit 1
}

episode_count() {
    find "$RAW" -maxdepth 1 -type f -name 'episode_*.npz' 2>/dev/null | wc -l
}

validate_raw() {
    "$TXC_PY" - "$RAW" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
episodes = sorted(root.glob("episode_*.npz"))
expected_names = [f"episode_{index:06d}.npz" for index in range(200)]
actual_names = [path.name for path in episodes]
if actual_names != expected_names:
    raise RuntimeError(
        f"Expected contiguous episode_000000..episode_000199, got "
        f"{len(actual_names)} files ending at {actual_names[-1] if actual_names else None}."
    )

required = {
    "state",
    "action",
    "wrist_masked_depth",
    "front_masked_depth",
    "replay_door_dof_pos",
    "gripper_handle_contact_both",
    "front_camera_pose_base",
    "wrist_camera_pose_base",
}
frame_counts = []
for path in episodes:
    with np.load(path, allow_pickle=True) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"{path.name} missing required fields: {missing}")
        state = np.asarray(data["state"])
        action = np.asarray(data["action"])
        if state.ndim != 2 or state.shape[1] != 9:
            raise RuntimeError(f"{path.name} invalid state shape {state.shape}")
        if action.shape != state.shape:
            raise RuntimeError(f"{path.name} action/state mismatch {action.shape}/{state.shape}")
        frame_counts.append(int(state.shape[0]))
        for key in (
            "wrist_masked_depth",
            "front_masked_depth",
            "replay_door_dof_pos",
            "gripper_handle_contact_both",
            "front_camera_pose_base",
            "wrist_camera_pose_base",
        ):
            if int(np.asarray(data[key]).shape[0]) != int(state.shape[0]):
                raise RuntimeError(f"{path.name} field {key} is not frame-aligned")

sidecar = json.loads((root / "door_dp_feature_names.json").read_text())
if sidecar.get("state_format") != "a2w_last_command_joint_state9":
    raise RuntimeError(f"Unexpected state format: {sidecar.get('state_format')}")
if sidecar.get("door_dp_mode") != "ikpull":
    raise RuntimeError(f"Unexpected controller mode: {sidecar.get('door_dp_mode')}")
print(
    f"RAW_VALID episodes={len(episodes)} frames={sum(frame_counts)} "
    f"min_frames={min(frame_counts)} max_frames={max(frame_counts)}"
)
PY
}

validate_dataset() {
    "$TXC_PY" - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
sidecar = json.loads((root / "door_dp_feature_names.json").read_text())
if int(info["total_episodes"]) != 200:
    raise RuntimeError(f"Expected 200 episodes, got {info['total_episodes']}")
if info["features"]["observation.state"]["shape"] != [9]:
    raise RuntimeError("Converted state is not Joint9")
if info["features"]["action"]["shape"] != [9]:
    raise RuntimeError("Converted action is not Joint9")
for key in (
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
):
    if key not in info["features"]:
        raise RuntimeError(f"Missing converted feature {key}")
if sidecar.get("door_dp_mode") != "ikpull":
    raise RuntimeError("Converted dataset is not marked as ikpull")
print(
    f"DATASET_VALID episodes={info['total_episodes']} "
    f"frames={info['total_frames']} state={info['features']['observation.state']['shape']} "
    f"action={info['features']['action']['shape']}"
)
PY
}

stage "WAIT_RECORDING"
while true; do
    count="$(episode_count)"
    if [[ "$count" -ge 200 ]]; then
        recorder_pid="$(cat "$RECORD_PID_FILE")"
        if kill -0 "$recorder_pid" 2>/dev/null; then
            printf '%s recording files reached 200; waiting for recorder shutdown pid=%s\n' \
                "$(date '+%Y-%m-%d %H:%M:%S')" "$recorder_pid"
            sleep 30
            continue
        fi
        break
    fi
    if [[ ! -f "$RECORD_PID_FILE" ]]; then
        fail "recording PID file is missing at count=$count"
    fi
    recorder_pid="$(cat "$RECORD_PID_FILE")"
    if ! kill -0 "$recorder_pid" 2>/dev/null; then
        fail "recording process exited before reaching 200 episodes (count=$count)"
    fi
    printf '%s recording=%s/200 pid=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$count" "$recorder_pid"
    sleep 120
done

stage "VALIDATE_RAW"
validate_raw || fail "raw validation failed"

if [[ -e "$DATASET_ROOT" ]]; then
    fail "refusing to overwrite existing dataset $DATASET_ROOT"
fi

stage "CONVERT"
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
    >"$PIPELINE_ROOT/convert.log" 2>&1 || fail "conversion failed"

stage "VALIDATE_DATASET"
validate_dataset || fail "dataset validation failed"

for output in "$BASE_RUN" "$INTER_RUN"; do
    if [[ -e "$output" ]]; then
        fail "refusing to overwrite existing training output $output"
    fi
done

COMMON_ARGS=(
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
    "--wandb.enable=true"
    "--wandb.project=door-act"
    "--wandb.disable_artifact=true"
)

stage "TRAIN_BOTH"
CUDA_VISIBLE_DEVICES=1 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/lerobot/src/lerobot/scripts/lerobot_train.py" \
    "${COMMON_ARGS[@]}" \
    --policy.plucker_conditioning=false \
    --policy.interaction_state_conditioning=false \
    "--job_name=$BASE_JOB" \
    "--output_dir=$BASE_RUN" \
    >"$PIPELINE_ROOT/train_baseline.log" 2>&1 &
base_pid=$!

CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/lerobot/src/lerobot/scripts/lerobot_train.py" \
    "${COMMON_ARGS[@]}" \
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
    >"$PIPELINE_ROOT/train_interaction.log" 2>&1 &
inter_pid=$!

printf '%s baseline_pid=%s interaction_pid=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$base_pid" "$inter_pid"
set +e
wait "$base_pid"
base_rc=$?
wait "$inter_pid"
inter_rc=$?
set -e
if [[ "$base_rc" -ne 0 || "$inter_rc" -ne 0 ]]; then
    fail "training failed baseline_rc=$base_rc interaction_rc=$inter_rc"
fi
[[ -s "$BASE_RUN/checkpoints/050000/pretrained_model/model.safetensors" ]] ||
    fail "baseline 50K checkpoint missing"
[[ -s "$INTER_RUN/checkpoints/050000/pretrained_model/model.safetensors" ]] ||
    fail "interaction 50K checkpoint missing"

stage "EXPORT"
CUDA_VISIBLE_DEVICES=1 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/dp/export_official_lerobot_act_to_door_checkpoint.py" \
    --official_checkpoint "$BASE_RUN/checkpoints/050000" \
    --root "$REPO/high-level/data/lerobot" \
    --repo_id "$DATASET_REPO" \
    --out_dir "$BASE_WRAP/model_latest" \
    --manifest_name model_latest.pt \
    --device cuda:0 \
    --depth_only \
    >"$PIPELINE_ROOT/export_baseline.log" 2>&1 || fail "baseline export failed"

CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    "$REPO/high-level/dp/export_official_lerobot_act_to_door_checkpoint.py" \
    --official_checkpoint "$INTER_RUN/checkpoints/050000" \
    --root "$REPO/high-level/data/lerobot" \
    --repo_id "$DATASET_REPO" \
    --out_dir "$INTER_WRAP/model_latest" \
    --manifest_name model_latest.pt \
    --device cuda:0 \
    --depth_only \
    >"$PIPELINE_ROOT/export_interaction.log" 2>&1 || fail "interaction export failed"

stage "EVAL_BOTH"
"$CONDA" run --no-capture-output -n txc_vbc python \
    "$REPO/high-level/dp/eval/eval_door_policy_success.py" \
    --checkpoint "$BASE_WRAP/model_latest.pt" \
    --yaml "$REPO/high-level/data/cfg/b1z1_opendoor.yaml" \
    --mode ikpull \
    --robot_body a2wz1 \
    --num_envs 16 \
    --total_trials 64 \
    --steps 1750 \
    --pass_open_angle_deg 60 \
    --pull_traversal_distance_m 0.8 \
    --base_seed 615455575 \
    --headless \
    --graphics_device_id 1 \
    --rl_device cuda:1 \
    --sim_device cuda:1 \
    --dp_action_horizon 25 \
    --dp_fps 25 \
    --depth_only \
    --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --run_root "$BASE_EVAL" \
    -- \
    --door_name wc4 \
    --wc4_disable_door_open_resistance \
    --door_auto_open_force 0 \
    --robot_pitch 0.0 \
    --ikpush_robot_pitch_rand_min 0.0 \
    --ikpush_robot_pitch_rand_max 0.0 \
    --ee_pose_frame robot_base_full \
    --camera_depth_clip_lower 0.2 \
    --camera_depth_clip_far 1.5 \
    >"$PIPELINE_ROOT/eval_baseline.log" 2>&1 &
base_eval_pid=$!

"$CONDA" run --no-capture-output -n txc_vbc python \
    "$REPO/high-level/dp/eval/eval_door_policy_success.py" \
    --checkpoint "$INTER_WRAP/model_latest.pt" \
    --yaml "$REPO/high-level/data/cfg/b1z1_opendoor.yaml" \
    --mode ikpull \
    --robot_body a2wz1 \
    --num_envs 16 \
    --total_trials 64 \
    --steps 1750 \
    --pass_open_angle_deg 60 \
    --pull_traversal_distance_m 0.8 \
    --base_seed 615455575 \
    --headless \
    --graphics_device_id 2 \
    --rl_device cuda:2 \
    --sim_device cuda:2 \
    --dp_action_horizon 25 \
    --dp_fps 25 \
    --depth_only \
    --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization \
    --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 \
    --run_root "$INTER_EVAL" \
    -- \
    --door_name wc4 \
    --wc4_disable_door_open_resistance \
    --door_auto_open_force 0 \
    --robot_pitch 0.0 \
    --ikpush_robot_pitch_rand_min 0.0 \
    --ikpush_robot_pitch_rand_max 0.0 \
    --ee_pose_frame robot_base_full \
    --camera_depth_clip_lower 0.2 \
    --camera_depth_clip_far 1.5 \
    >"$PIPELINE_ROOT/eval_interaction.log" 2>&1 &
inter_eval_pid=$!

set +e
wait "$base_eval_pid"
base_eval_rc=$?
wait "$inter_eval_pid"
inter_eval_rc=$?
set -e
if [[ "$base_eval_rc" -ne 0 || "$inter_eval_rc" -ne 0 ]]; then
    fail "evaluation failed baseline_rc=$base_eval_rc interaction_rc=$inter_eval_rc"
fi

stage "SUMMARIZE"
"$TXC_PY" - "$BASE_EVAL/summary.json" "$INTER_EVAL/summary.json" \
    >"$PIPELINE_ROOT/final_results.txt" <<'PY'
import json
import sys

for label, path in zip(("Baseline", "Plucker+Interaction"), sys.argv[1:]):
    data = json.load(open(path))
    batches = [
        (
            item["door_open_successes"],
            item["traversal_successes"],
            item["successes"],
        )
        for item in data["batch_logs"]
    ]
    print(
        f"{label}: trials={data['trials']} "
        f"open={data['door_open_successes']}/{data['trials']} "
        f"traversal={data['traversal_successes']}/{data['trials']} "
        f"joint={data['successes']}/{data['trials']} "
        f"rates={data['door_open_success_rate']:.6f},"
        f"{data['traversal_success_rate']:.6f},{data['success_rate']:.6f} "
        f"batches={batches}"
    )
PY
cat "$PIPELINE_ROOT/final_results.txt"
stage "COMPLETE"
