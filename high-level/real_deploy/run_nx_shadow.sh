#!/usr/bin/env bash
set -euo pipefail

export DOOR_ACT_DISABLE_BACKBONE_PRETRAINED="${DOOR_ACT_DISABLE_BACKBONE_PRETRAINED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LEROBOT_MINIMAL_ACT_IMPORTS="${LEROBOT_MINIMAL_ACT_IMPORTS:-1}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

# PC2 robot_control uses CycloneDDS on the 124.x robot network.  Keep the NX
# ACT process on the same RMW/interface so /cmd_vel_safe and /vel_state discover
# each other without requiring extra terminal exports before every run.
if [ -z "${RMW_IMPLEMENTATION:-}" ] && [ -f /home/anx/cyclonedds.xml ]; then
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi
if [ -z "${CYCLONEDDS_URI:-}" ] && [ -f /home/anx/cyclonedds.xml ]; then
  export CYCLONEDDS_URI=/home/anx/cyclonedds.xml
fi

if [ -f /opt/ros/humble/setup.bash ]; then
  # The real deployment path publishes/subscribes ROS2 topics by default.
  # Temporarily relax nounset because ROS setup files may reference unset vars.
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

DEPLOY_ROOT="${DEPLOY_ROOT:-/home/anx/door_act_deploy}"
REPO_ROOT="${REPO_ROOT:-${DEPLOY_ROOT}/visual_whole_body}"

export PYTHONPATH="${REPO_ROOT}/high-level:${REPO_ROOT}/high-level/dp:${REPO_ROOT}/high-level/lerobot/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${DEPLOY_ROOT}/z1_sdk/lib:${LD_LIBRARY_PATH:-}"

exec python3 "${REPO_ROOT}/high-level/real_deploy/door_act_shadow.py" "$@"
