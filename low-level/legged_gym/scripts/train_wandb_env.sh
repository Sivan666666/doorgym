#!/usr/bin/env bash

# Source this file before training on a shared server:
#   source scripts/train_env.sh
#
# Then run the original training command as usual.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this file instead of executing it:"
  echo "  source scripts/train_env.sh"
  exit 1
fi

# Fill in your own W&B API key before using this on the server.
# You can also comment this out if the current shell already has WANDB_API_KEY.
export WANDB_API_KEY="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"

# Your personal W&B account or team name.
export WANDB_ENTITY="1309519635"

# Keep the default project behavior aligned with the current training scripts.
export WANDB_PROJECT="dev"

# Put local W&B metadata in your own home/cache directory instead of mixing with others.
export WANDB_DIR="${WANDB_DIR:-$HOME/.cache/wandb_roboduet}"

# Optional. Leave online for normal training; switch to offline if the server network is unstable.
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "$WANDB_DIR"

echo "W&B environment loaded:"
echo "  WANDB_ENTITY=$WANDB_ENTITY"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
echo "  WANDB_DIR=$WANDB_DIR"
echo "  WANDB_MODE=$WANDB_MODE"
