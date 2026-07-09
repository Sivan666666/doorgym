#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/home/robo/txc/door_act_deploy}"
cd "${DEPLOY_ROOT}/z1_controller/build"
exec ./z1_ctrl "$@"
