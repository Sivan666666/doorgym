#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET_EPISODES=200 \
BASE_GPU=2 \
INTER_GPU=2 \
TRAIN_SEQUENTIAL=true exec \
    "$SCRIPT_DIR/run_wc4_pull_widesafe35_lateretreat_50_full_pipeline_ps1.sh"
