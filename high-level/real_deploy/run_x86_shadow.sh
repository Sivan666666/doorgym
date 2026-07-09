#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/home/robo/txc/door_act_deploy}"
REPO_ROOT="${REPO_ROOT:-${DEPLOY_ROOT}/visual_whole_body}"
VENV_ROOT="${DOOR_ACT_VENV:-/home/robo/txc/venvs/door_act}"
DDS_CONFIG="${CYCLONEDDS_CONFIG:-${DEPLOY_ROOT}/config/cyclonedds.xml}"

export DOOR_ACT_DISABLE_BACKBONE_PRETRAINED="${DOOR_ACT_DISABLE_BACKBONE_PRETRAINED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LEROBOT_MINIMAL_ACT_IMPORTS="${LEROBOT_MINIMAL_ACT_IMPORTS:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
if [[ -f "${DDS_CONFIG}" ]]; then
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:-${DDS_CONFIG}}"
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

export PYTHONPATH="${REPO_ROOT}/high-level:${REPO_ROOT}/high-level/dp:${REPO_ROOT}/high-level/lerobot/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${DEPLOY_ROOT}/z1_sdk/lib:${LD_LIBRARY_PATH:-}"

exec "${VENV_ROOT}/bin/python" \
  "${REPO_ROOT}/high-level/real_deploy/door_act_shadow.py" "$@"
