#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
LR_PY=${LR_PY:-/home/ps/miniconda3/envs/b1z1_lerobot/bin/python}
TAG=${TAG:-0813_wc_nomenkuang_fixed15_observable_v1}
PIPE="$REPO/high-level/logs/pull-data-pipeline/${TAG}_500_phasebalanced_multiseed"
FINAL_REPO="local/door_a2w_${TAG}_500_interaction_state"
FINAL_ROOT="$REPO/high-level/data/lerobot/$FINAL_REPO"

cd "$REPO"
mkdir -p "$PIPE"

raw_rel() {
  local shard=$1
  local seed=$((615455575 + shard * 100000))
  printf 'high-level/data/door_dp_raw/a2w_%s_shard%s_50_seed%s' "$TAG" "$shard" "$seed"
}

convert_pair() {
  local pair=$1 first=$2 second=$3
  local repo_id="local/door_a2w_${TAG}_part${pair}_100_interaction_state"
  local root="$REPO/high-level/data/lerobot/$repo_id"
  rm -rf "$root"
  "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/convert_door_raw_to_lerobot.py \
    --raw_root "$(raw_rel "$first")" \
    --additional_raw_root "$(raw_rel "$second")" \
    --root high-level/data/lerobot --repo_id "$repo_id" \
    --state_action_mode joint_state9 --depth_only \
    --image_storage video --video_codec h264 \
    --add_interaction_state --num_workers 2 \
    >"$PIPE/convert_part${pair}.log" 2>&1
}

printf '%s PARALLEL_CONVERT_5X\n' "$(date '+%F %T')" | tee "$PIPE/status.txt"
convert_pair 0 0 1 & p0=$!
convert_pair 1 2 3 & p1=$!
convert_pair 2 4 5 & p2=$!
convert_pair 3 6 7 & p3=$!
convert_pair 4 8 9 & p4=$!
for pid in "$p0" "$p1" "$p2" "$p3" "$p4"; do
  wait "$pid"
done

printf '%s VALIDATE_PARTS\n' "$(date '+%F %T')" | tee "$PIPE/status.txt"
"$LR_PY" - "$REPO" "$TAG" <<'PY' | tee "$PIPE/parallel_parts_validation.txt"
import json, sys
from pathlib import Path

repo = Path(sys.argv[1])
tag = sys.argv[2]
for index in range(5):
    root = repo / "high-level/data/lerobot/local" / f"door_a2w_{tag}_part{index}_100_interaction_state"
    info = json.loads((root / "meta/info.json").read_text())
    episodes = int(info["total_episodes"])
    frames = int(info["total_frames"])
    if episodes != 100 or frames != 130000:
        raise RuntimeError(f"part{index}: episodes={episodes}, frames={frames}")
    print(f"part{index}\tepisodes={episodes}\tframes={frames}")
PY

printf '%s MERGE_PARTS\n' "$(date '+%F %T')" | tee "$PIPE/status.txt"
rm -rf "$FINAL_ROOT"
"$LR_PY" - "$REPO" "$TAG" "$FINAL_REPO" <<'PY' >"$PIPE/merge_parts.log" 2>&1
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

repo = Path(sys.argv[1])
tag = sys.argv[2]
final_repo = sys.argv[3]
datasets = []
for index in range(5):
    repo_id = f"local/door_a2w_{tag}_part{index}_100_interaction_state"
    root = repo / "high-level/data/lerobot" / repo_id
    datasets.append(LeRobotDataset(repo_id=repo_id, root=root))
final_root = repo / "high-level/data/lerobot" / final_repo
merged = merge_datasets(datasets, output_repo_id=final_repo, output_dir=final_root)
print(f"episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}")
PY

cp "$REPO/high-level/data/lerobot/local/door_a2w_${TAG}_part0_100_interaction_state/door_dp_feature_names.json" \
   "$FINAL_ROOT/door_dp_feature_names.json"

# merge_datasets merges the LeRobot metadata, while this project-specific
# sidecar has to be aggregated explicitly instead of inheriting part0 only.
"$LR_PY" - "$FINAL_ROOT" "$TAG" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
tag = sys.argv[2]
path = root / "door_dp_feature_names.json"
sidecar = json.loads(path.read_text())
roots = [
    f"high-level/data/door_dp_raw/a2w_{tag}_shard{shard}_50_seed{615455575 + shard * 100000}"
    for shard in range(10)
]
sidecar["source_raw_root"] = roots[0]
sidecar["source_raw_roots"] = roots
sidecar["merged_source_dataset_count"] = 5
sidecar["merged_source_episode_count"] = 500
path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
PY

"$LR_PY" - "$FINAL_ROOT" <<'PY' | tee "$PIPE/merged_validation.txt"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
assert int(info["total_episodes"]) == 500, info
assert int(info["total_frames"]) == 650000, info
features = info["features"]
for key in (
    "observation.state", "action", "observation.images.front_masked_depth",
    "observation.images.wrist_masked_depth", "aux.interaction_contact",
    "aux.interaction_handle_progress", "aux.interaction_door_progress",
):
    if key not in features:
        raise RuntimeError(f"missing feature {key}")
print(f"episodes={info['total_episodes']} frames={info['total_frames']} features=ok")
PY

date >"$PIPE/convert.complete"
printf '%s PARALLEL_CONVERT_COMPLETE\n' "$(date '+%F %T')" | tee "$PIPE/status.txt"
