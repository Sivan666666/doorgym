#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/home/ps/workspace/txc/doorgym}
SIM_PYTHON=${SIM_PYTHON:-/home/ps/miniconda3/envs/txc_vbc/bin/python}
TRAIN_PYTHON=${TRAIN_PYTHON:-/home/ps/miniconda3/envs/b1z1_lerobot/bin/python}
SIM_GPU=${SIM_GPU:-0}
BASELINE_GPU=${BASELINE_GPU:-0}
INTERACTION_GPU=${INTERACTION_GPU:-1}

RAW_NAME=a2w_wc_articraft2_wc4physics_roboty_state10_plucker_depthnoise_camerarand_contact_200_step1400
DATASET_ID=local/door_a2w_wc_articraft2_wc4physics_roboty_state10_200_step1400
RAW_ROOT="$ROOT/high-level/data/door_dp_raw/$RAW_NAME"
DATASET_ROOT="$ROOT/high-level/data/lerobot/$DATASET_ID"
PIPELINE_LOG_ROOT="$ROOT/high-level/logs/wc_articraft2_wc4physics_pipeline_20260730"

BASELINE_RUN=leroact_a2w_wc_articraft2_wc4physics_roboty_200_baseline_50k_chunk100_exec50_bs16_seed1000_0730
INTERACTION_RUN=leroact_a2w_wc_articraft2_wc4physics_roboty_200_plucker_fov55_interaction_decoderchunk_w005_50k_chunk100_exec50_bs16_seed1000_0730
BASELINE_OUTPUT="$ROOT/high-level/dp/logs/lerobot-train/$BASELINE_RUN"
INTERACTION_OUTPUT="$ROOT/high-level/dp/logs/lerobot-train/$INTERACTION_RUN"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

fail() {
  echo "[$(timestamp)] PIPELINE_ERROR: $*" >&2
  exit 1
}

on_error() {
  local status=$?
  echo "[$(timestamp)] PIPELINE_ERROR line=$1 status=$status" >&2
  exit "$status"
}
trap 'on_error "$LINENO"' ERR

mkdir -p "$RAW_ROOT" "$PIPELINE_LOG_ROOT"
cd "$ROOT"

echo "[$(timestamp)] Pipeline start."
echo "[$(timestamp)] Raw root: $RAW_ROOT"
echo "[$(timestamp)] Dataset: $DATASET_ROOT"
echo "[$(timestamp)] GPUs: simulation=$SIM_GPU baseline=$BASELINE_GPU interaction=$INTERACTION_GPU"

if [[ -e "$BASELINE_OUTPUT" || -e "$INTERACTION_OUTPUT" ]]; then
  fail "Refusing to overwrite an existing training output directory."
fi

echo "[$(timestamp)] Stage 1/5: recording 200 successful Articraft2 episodes."
CUDA_VISIBLE_DEVICES="$SIM_GPU" "$SIM_PYTHON" \
  high-level/dp/record/record_door_dp_dataset_a2w_state10.py \
  --num_episodes 200 \
  --num_envs 16 \
  --max_quota_rollouts 100 \
  --raw_root "$RAW_ROOT" \
  --state_action_mode ee_state10 \
  --steps 1400 \
  --seed -1 \
  --headless \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --depth_only \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --camera_render_horizontal_fov_deg 60.0 \
  --enable_depth_noise \
  --no_enable_depth_gaussian_blur \
  --enable_depth_camera_randomization \
  --depth_camera_pos_rand_m 0.02 \
  --depth_camera_rot_rand_deg 5.0 \
  --record_camera_pose \
  --record_end_signal \
  --record_handle_bbox \
  --record_gripper_handle_contact \
  --filter_gripper_handle_contact \
  --filter_gripper_handle_contact_min_frames 5 \
  --filter_gripper_handle_contact_phase_names close_gripper,rotate_handle \
  --keyframe_loss_weight 8 \
  --keyframe_loss_radius 3 \
  -- \
  --door_cfg "$ROOT/high-level/data/cfg/b1z1_opendoor.yaml" \
  --door_name wc_articraft2 \
  --skill_program_json "$ROOT/high-level/float_ik/door_twin/examples/wc_articraft2_push_traverse_skill.json" \
  --ee_pose_frame robot_base_full

echo "[$(timestamp)] Stage 2/5: validating every raw episode."
"$TRAIN_PYTHON" - "$RAW_ROOT" <<'PY'
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
paths = sorted(root.glob("episode_*.npz"))
assert len(paths) == 200, f"expected 200 episodes, found {len(paths)}"
required = {
    "state",
    "action",
    "wrist_masked_depth",
    "front_masked_depth",
    "front_camera_pose_base",
    "wrist_camera_pose_base",
    "gripper_handle_contact_any",
    "replay_door_dof_pos",
    "replay_door_open_stage",
}
early_handle_max = 0.0
early_unlock_count = 0
for index, path in enumerate(paths):
    with np.load(path, allow_pickle=True) as data:
        missing = required.difference(data.files)
        assert not missing, f"{path.name}: missing {sorted(missing)}"
        assert data["state"].shape == (700, 10), (path.name, data["state"].shape)
        assert data["action"].shape == (700, 10), (path.name, data["action"].shape)
        door_pos = np.asarray(data["replay_door_dof_pos"], dtype=np.float32)
        open_stage = np.asarray(data["replay_door_open_stage"], dtype=np.float32).reshape(-1)
        assert door_pos.shape[0] == 700 and door_pos.shape[1] >= 2, (path.name, door_pos.shape)
        # The first 200 recorded frames precede rotate_handle. The handle must
        # remain locked there; this catches the former startup impulse.
        early_handle_max = max(early_handle_max, float(np.max(np.abs(door_pos[:200, 1]))))
        early_unlock_count += int(np.any(open_stage[:200] > 0.5))
assert early_unlock_count == 0, f"{early_unlock_count}/200 episodes unlocked before rotate_handle"
print(
    f"RAW_VALIDATION_OK episodes={len(paths)} frames={len(paths) * 700} "
    f"early_handle_max_rad={early_handle_max:.6f} early_unlock_count={early_unlock_count}"
)
PY

echo "[$(timestamp)] Stage 3/5: converting to LeRobot."
"$TRAIN_PYTHON" \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root "$RAW_ROOT" \
  --root "$ROOT/high-level/data/lerobot" \
  --repo_id "$DATASET_ID" \
  --overwrite \
  --fps 25 \
  --depth_only \
  --image_storage video \
  --video_codec h264 \
  --num_workers 4 \
  --state_preprocess none \
  --action_preprocess none \
  --near_zero_rate_eps 1e-5 \
  --add_end_signal \
  --add_interaction_state

echo "[$(timestamp)] Stage 4/5: validating converted dataset."
"$TRAIN_PYTHON" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
info = json.loads(path.read_text())
features = info.get("features", {})
assert info.get("total_episodes") == 200, info.get("total_episodes")
assert info.get("total_frames") == 140000, info.get("total_frames")
assert info.get("fps") == 25, info.get("fps")
assert features["observation.state"]["shape"] == [10]
assert features["action"]["shape"] == [10]
for key in (
    "observation.images.front_masked_depth",
    "observation.images.wrist_masked_depth",
    "observation.camera_pose.front",
    "observation.camera_pose.wrist",
    "aux.interaction_contact",
    "aux.interaction_handle_progress",
    "aux.interaction_door_progress",
):
    assert key in features, key
print("LEROBOT_VALIDATION_OK episodes=200 frames=140000 fps=25")
PY

train_variant() {
  local variant=$1
  local gpu=$2
  local run_name=$3
  local output_dir=$4
  shift 4
  echo "[$(timestamp)] Starting $variant on physical GPU $gpu."
  CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN_PYTHON" \
    high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="$DATASET_ID" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.use_imagenet_stats=false \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=100 \
    --policy.n_action_steps=50 \
    --policy.vision_backbone=resnet18 \
    --policy.optimizer_lr=1e-5 \
    --policy.use_action_loss_weight=false \
    --policy.end_signal_prediction=false \
    "$@" \
    --seed=1000 \
    --steps=50000 \
    --batch_size=16 \
    --num_workers=3 \
    --save_freq=10000 \
    --log_freq=100 \
    --keyframe_sampling_ratio=0 \
    --output_dir="$output_dir" \
    --job_name="$run_name" \
    --wandb.enable=true \
    --wandb.project=door-act \
    --wandb.disable_artifact=true
}

echo "[$(timestamp)] Stage 5/5: starting both 50K ACT trainings."
(
  set -o pipefail
  train_variant baseline "$BASELINE_GPU" "$BASELINE_RUN" "$BASELINE_OUTPUT" \
    --policy.plucker_conditioning=false \
    --policy.interaction_state_conditioning=false \
    2>&1 | tee "$PIPELINE_LOG_ROOT/${BASELINE_RUN}.log"
) &
baseline_pid=$!

(
  set -o pipefail
  train_variant plucker_interaction "$INTERACTION_GPU" "$INTERACTION_RUN" "$INTERACTION_OUTPUT" \
    --policy.plucker_conditioning=true \
    --policy.plucker_intrinsics_mode=legacy_shared_fov \
    --policy.plucker_horizontal_fov_deg=55.0 \
    --policy.interaction_state_conditioning=true \
    --policy.interaction_state_prediction_mode=decoder_chunk \
    --policy.interaction_state_probe_only=false \
    --policy.interaction_state_auxiliary_only=false \
    --policy.interaction_contact_loss_weight=0.05 \
    --policy.interaction_handle_loss_weight=0.05 \
    --policy.interaction_door_loss_weight=0.05 \
    2>&1 | tee "$PIPELINE_LOG_ROOT/${INTERACTION_RUN}.log"
) &
interaction_pid=$!

echo "[$(timestamp)] Training PIDs: baseline=$baseline_pid interaction=$interaction_pid"
baseline_status=0
interaction_status=0
wait "$baseline_pid" || baseline_status=$?
wait "$interaction_pid" || interaction_status=$?
if (( baseline_status != 0 || interaction_status != 0 )); then
  fail "Training failed: baseline_status=$baseline_status interaction_status=$interaction_status"
fi

for checkpoint in \
  "$BASELINE_OUTPUT/checkpoints/050000/pretrained_model/config.json" \
  "$INTERACTION_OUTPUT/checkpoints/050000/pretrained_model/config.json"; do
  [[ -f "$checkpoint" ]] || fail "Missing final checkpoint marker: $checkpoint"
done

echo "[$(timestamp)] PIPELINE_COMPLETE"
echo "[$(timestamp)] Baseline checkpoint: $BASELINE_OUTPUT/checkpoints/050000"
echo "[$(timestamp)] Plucker Interaction checkpoint: $INTERACTION_OUTPUT/checkpoints/050000"
