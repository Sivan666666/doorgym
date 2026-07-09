#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/home/robo/txc/door_act_deploy}"
REPO_ROOT="${REPO_ROOT:-${DEPLOY_ROOT}/visual_whole_body}"
VENV_ROOT="${DOOR_ACT_VENV:-/home/robo/txc/venvs/door_act}"

export Z1_SDK_LIB="${Z1_SDK_LIB:-${DEPLOY_ROOT}/z1_sdk/lib}"
export LD_LIBRARY_PATH="${Z1_SDK_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/high-level:${REPO_ROOT}/high-level/real_deploy:${PYTHONPATH:-}"

exec "${VENV_ROOT}/bin/python" \
  "${REPO_ROOT}/high-level/real_deploy/z1_act_ee_bridge.py" "$@"
