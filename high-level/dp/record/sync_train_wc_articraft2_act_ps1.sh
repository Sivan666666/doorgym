#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO=/home/sivan/whole_body/visual_whole_body
REMOTE_HOST=ps@192.168.1.104
REMOTE_REPO=/home/ps/workspace/txc/doorgym
CONVERT_SERVICE=${CONVERT_SERVICE:-wc-articraft2-record-convert-200.service}
DATASET_REL=high-level/data/lerobot/local/door_a2w_wc_articraft2_ours_state10_200_step1400
TRAIN_SCRIPT_REL=high-level/dp/record/train_wc_articraft2_act_variant_ps1.sh
LOCAL_PYTHON=/home/sivan/miniconda3/envs/b1z1_lerobot/bin/python

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

retry() {
  local max_attempts=$1
  local delay_seconds=$2
  shift 2
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      echo "[$(timestamp)] Command failed after $attempt attempts: $*" >&2
      return 1
    fi
    echo "[$(timestamp)] Command failed (attempt $attempt/$max_attempts); retrying in ${delay_seconds}s: $*" >&2
    attempt=$((attempt + 1))
    sleep "$delay_seconds"
  done
}

cd "$LOCAL_REPO"

echo "[$(timestamp)] Waiting for $CONVERT_SERVICE."
while systemctl --user is-active --quiet "$CONVERT_SERVICE"; do
  latest=$(journalctl --user -u "$CONVERT_SERVICE" -n 30 --no-pager \
    | grep -E 'Converted episode_[0-9]+.*\\([0-9]+/200\\)' | tail -n 1 || true)
  [[ -n "$latest" ]] && echo "$latest"
  sleep 60
done

if ! journalctl --user -u "$CONVERT_SERVICE" --no-pager | grep -q 'PIPELINE_COMPLETE'; then
  echo "Conversion service stopped without PIPELINE_COMPLETE." >&2
  journalctl --user -u "$CONVERT_SERVICE" -n 80 --no-pager >&2
  exit 1
fi

"$LOCAL_PYTHON" - "$LOCAL_REPO/$DATASET_REL/meta/info.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
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
print("Local dataset validation passed.")
PY

echo "[$(timestamp)] Syncing dataset and training launcher to PS1."
retry 20 30 ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
  "mkdir -p '$REMOTE_REPO/high-level/data/lerobot/local' '$REMOTE_REPO/high-level/dp/record' '$REMOTE_REPO/high-level/logs/wc_articraft2_training'"
retry 20 30 rsync -a --partial --info=progress2 --timeout=120 \
  "$LOCAL_REPO/$DATASET_REL/" \
  "$REMOTE_HOST:$REMOTE_REPO/$DATASET_REL/"
retry 20 30 rsync -a --partial --timeout=120 \
  "$LOCAL_REPO/$TRAIN_SCRIPT_REL" \
  "$REMOTE_HOST:$REMOTE_REPO/$TRAIN_SCRIPT_REL"
retry 20 30 ssh -o ConnectTimeout=15 "$REMOTE_HOST" "chmod +x '$REMOTE_REPO/$TRAIN_SCRIPT_REL'"

BASELINE_LAUNCH_LOG="$REMOTE_REPO/high-level/logs/wc_articraft2_training/baseline_launcher.log"
INTERACTION_LAUNCH_LOG="$REMOTE_REPO/high-level/logs/wc_articraft2_training/plucker_interaction_launcher.log"

echo "[$(timestamp)] Launching Baseline ACT on physical GPU 1."
baseline_pid=$(retry 20 30 ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
  "cd '$REMOTE_REPO' && nohup env GPU=1 VARIANT=baseline bash '$TRAIN_SCRIPT_REL' >'$BASELINE_LAUNCH_LOG' 2>&1 </dev/null & echo \$!")
echo "Baseline launcher PID: $baseline_pid"

echo "[$(timestamp)] Launching Plucker Interaction ACT on physical GPU 3."
interaction_pid=$(retry 20 30 ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
  "cd '$REMOTE_REPO' && nohup env GPU=3 VARIANT=plucker_interaction bash '$TRAIN_SCRIPT_REL' >'$INTERACTION_LAUNCH_LOG' 2>&1 </dev/null & echo \$!")
echo "Plucker Interaction launcher PID: $interaction_pid"

sleep 30
retry 20 30 ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
  "echo '=== Baseline launcher ==='; tail -n 30 '$BASELINE_LAUNCH_LOG'; \
   echo '=== Plucker Interaction launcher ==='; tail -n 30 '$INTERACTION_LAUNCH_LOG'; \
   echo '=== GPU status ==='; \
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"

echo "[$(timestamp)] SYNC_AND_LAUNCH_COMPLETE"
