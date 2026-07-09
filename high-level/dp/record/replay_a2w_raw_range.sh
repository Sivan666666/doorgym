#!/usr/bin/env bash
set -euo pipefail

# Replay a contiguous range of A2W raw Door DP episodes in Isaac Gym.
#
# Default:
#   episodes 200..250 under
#   high-level/data/door_dp_raw/a2w_5door_w3_plucker_depthnoise_camerarand_250
#
# Usage examples:
#   bash high-level/dp/record/replay_a2w_raw_range.sh
#   bash high-level/dp/record/replay_a2w_raw_range.sh 200 249
#   RAW_ROOT=high-level/data/door_dp_raw/my_raw bash high-level/dp/record/replay_a2w_raw_range.sh 0 10

START_EP="${1:-200}"
END_EP="${2:-250}"

REPO_ROOT="${REPO_ROOT:-/home/sivan/whole_body/visual_whole_body}"
RAW_ROOT="${RAW_ROOT:-high-level/data/door_dp_raw/a2w_5door_w3_plucker_depthnoise_camerarand_250}"
CONDA_ENV="${CONDA_ENV:-b1z1}"
DOOR_CFG="${DOOR_CFG:-high-level/data/cfg/b1z1_opendoor.yaml}"
MODE="${MODE:-ikpush}"
REPLAY_MODE="${REPLAY_MODE:-state}"
STEPS="${STEPS:-500}"
STRIDE="${STRIDE:-1}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"
DEPTH_CLIP_LOWER="${DEPTH_CLIP_LOWER:-0.2}"
DEPTH_CLIP_FAR="${DEPTH_CLIP_FAR:-1.5}"
REAL_TIME_FLAG="${REAL_TIME_FLAG:---real_time}"
SHOW_SEG_FLAG="${SHOW_SEG_FLAG:---show_seg}"

cd "${REPO_ROOT}"

echo "[replay-range] raw_root=${RAW_ROOT}"
echo "[replay-range] episodes=${START_EP}..${END_EP}"
echo "[replay-range] env=${CONDA_ENV} device=${SIM_DEVICE} graphics=${GRAPHICS_DEVICE_ID}"

for ep in $(seq "${START_EP}" "${END_EP}"); do
  episode_path="${RAW_ROOT}/episode_$(printf '%06d' "${ep}").npz"
  if [[ ! -f "${episode_path}" ]]; then
    echo "[replay-range] skip missing ${episode_path}"
    continue
  fi

  echo
  echo "============================================================"
  echo "[replay-range] replay episode ${ep}: ${episode_path}"
  echo "============================================================"

  conda run --no-capture-output -n "${CONDA_ENV}" python \
    high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
    --raw_episode "${episode_path}" \
    --door_cfg "${DOOR_CFG}" \
    --mode "${MODE}" \
    --replay_mode "${REPLAY_MODE}" \
    --start_step 0 \
    --steps "${STEPS}" \
    --stride "${STRIDE}" \
    --rl_device "${RL_DEVICE}" \
    --sim_device "${SIM_DEVICE}" \
    --graphics_device_id "${GRAPHICS_DEVICE_ID}" \
    --camera_depth \
    ${SHOW_SEG_FLAG} \
    --camera_depth_clip_lower "${DEPTH_CLIP_LOWER}" \
    --camera_depth_clip_far "${DEPTH_CLIP_FAR}" \
    ${REAL_TIME_FLAG}
done

echo
echo "[replay-range] done."
