#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed-15-degree WC-No-Frame experiment:
#   1. Nested data scaling: 50/100/200/500 episodes, seeds 1000 and 2000.
#   2. Training-seed variance at 500 episodes: add seeds 3000 and 4000.
# Every newly launched run trains from scratch for exactly 200K updates and is
# evaluated on the same 64 WC-No-Frame trials (base seed 615455575).

REPO=${REPO:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
ARCHIVE=${ARCHIVE:-/mnt/sda/ps_archive/doorgym}
DATA_ARCHIVE=${DATA_ARCHIVE:-$ARCHIVE/high-level/data/lerobot}
TRAIN_ARCHIVE=${TRAIN_ARCHIVE:-$ARCHIVE/high-level/dp/logs/lerobot-train}
WRAP_ARCHIVE=${WRAP_ARCHIVE:-$ARCHIVE/high-level/dp/logs/door-auto-wrapped}
EVAL_ARCHIVE=${EVAL_ARCHIVE:-$ARCHIVE/high-level/logs/door-policy-success}
PIPE_ARCHIVE=${PIPE_ARCHIVE:-$ARCHIVE/high-level/logs/pull-data-pipeline/0815_fixed15_nested_scaling_seed_variance_200k}
PIPE_LINK=${PIPE_LINK:-$REPO/high-level/logs/pull-data-pipeline/0815_fixed15_nested_scaling_seed_variance_200k}

BASE_DATA_NAME=door_a2w_0813_wc_nomenkuang_fixed15_observable_v1_500_interaction_state
NESTED_PREFIX=door_a2w_0813_wc_nomenkuang_fixed15_observable_v1_nested
PHASE_WEIGHTS=${PHASE_WEIGHTS:-8:2,11:0.1,12:2,13:2,14:2}
BASE_SEED=${BASE_SEED:-615455575}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
# GPU 1 has a low-utilization renderer resident at roughly 6 GiB. ACT uses
# about 11 GiB, so accepting up to 8 GiB resident memory remains safely below
# the 48 GiB device capacity while still rejecting occupied training GPUs.
GPU_IDLE_MEMORY_MIB=${GPU_IDLE_MEMORY_MIB:-8000}
GPU_IDLE_UTIL_PCT=${GPU_IDLE_UTIL_PCT:-35}
GPU_IDLE_CONFIRMATIONS=${GPU_IDLE_CONFIRMATIONS:-3}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}

mkdir -p "$TRAIN_ARCHIVE" "$WRAP_ARCHIVE" "$EVAL_ARCHIVE" "$PIPE_ARCHIVE"
mkdir -p "$(dirname "$PIPE_LINK")" "$REPO/high-level/dp/logs/lerobot-train" \
  "$REPO/high-level/dp/logs/door-auto-wrapped" "$REPO/high-level/logs/door-policy-success"
if [[ ! -e "$PIPE_LINK" ]]; then
  ln -s "$PIPE_ARCHIVE" "$PIPE_LINK"
fi

exec > >(tee -a "$PIPE_ARCHIVE/supervisor.log") 2>&1

timestamp() { date '+%F %T'; }
log() { echo "[$(timestamp)] $*"; }
fail() { log "ERROR: $*"; exit 1; }

job_name() {
  local episodes=$1 seed=$2
  echo "leroact_a2w_0813_wc_nomenkuang_fixed15_nested${episodes}_plucker_interaction_phasebalanced_hold01_critical2_seed${seed}_200k_chunk100_exec50_bs16"
}

data_name() {
  local episodes=$1
  if [[ "$episodes" == 500 ]]; then
    echo "$BASE_DATA_NAME"
  else
    echo "${NESTED_PREFIX}_${episodes}"
  fi
}

data_root() {
  local episodes=$1 name
  name=$(data_name "$episodes")
  if [[ "$episodes" == 500 ]]; then
    echo "$REPO/high-level/data/lerobot/local/$name"
  else
    echo "$DATA_ARCHIVE/local/$name"
  fi
}

ensure_link() {
  local target=$1 link=$2
  [[ -e "$target" ]] || fail "Cannot link a target that does not exist: $target"
  [[ -e "$link" || -L "$link" ]] || ln -s "$target" "$link"
}

gpu_snapshot() {
  local gpu=$1
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v gpu="$gpu" '{gsub(/ /, "", $0); if ($1 == gpu) print $2, $3}'
}

wait_for_gpu() {
  local gpu=$1 count=0 mem util
  while true; do
    read -r mem util < <(gpu_snapshot "$gpu") || true
    if [[ -n "${mem:-}" && -n "${util:-}" \
      && "$mem" -lt "$GPU_IDLE_MEMORY_MIB" && "$util" -lt "$GPU_IDLE_UTIL_PCT" ]]; then
      count=$((count + 1))
      if (( count >= GPU_IDLE_CONFIRMATIONS )); then
        log "GPU $gpu is available (memory=${mem}MiB utilization=${util}%)."
        return 0
      fi
    else
      count=0
      log "Waiting for GPU $gpu (memory=${mem:-unknown}MiB utilization=${util:-unknown}%)."
    fi
    sleep "$GPU_POLL_SECONDS"
  done
}

validate_data() {
  local episodes=$1 root repo_id
  root=$(data_root "$episodes")
  repo_id="local/$(data_name "$episodes")"
  "$CONDA" run --no-capture-output -n b1z1_lerobot python - "$root" "$repo_id" "$episodes" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root, repo_id, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = LeRobotDataset(repo_id=repo_id, root=root, video_backend="torchcodec")
assert ds.num_episodes == expected, (ds.num_episodes, expected)
assert ds.num_frames == expected * 1300, (ds.num_frames, expected * 1300)
assert tuple(ds.meta.features["observation.state"]["shape"]) == (9,)
assert tuple(ds.meta.features["action"]["shape"]) == (9,)
for key in ("aux.interaction_contact", "aux.interaction_handle_progress", "aux.interaction_door_progress"):
    assert key in ds.meta.features, key
print(f"validated {repo_id}: episodes={ds.num_episodes} frames={ds.num_frames}")
PY
}

train_one() {
  local gpu=$1 episodes=$2 seed=$3 name data repo_id data_path out log_file
  name=$(job_name "$episodes" "$seed")
  data=$(data_name "$episodes")
  repo_id="local/$data"
  data_path=$(data_root "$episodes")
  out="$TRAIN_ARCHIVE/$name"
  log_file="$PIPE_ARCHIVE/train_${episodes}_seed${seed}.log"

  if [[ -s "$out/checkpoints/$TRAIN_STEPS/pretrained_model/model.safetensors" ]]; then
    ensure_link "$out" "$REPO/high-level/dp/logs/lerobot-train/$name"
    log "Training already complete: episodes=$episodes seed=$seed"
    return 0
  fi
  # A very early launcher revision created empty output directories before
  # waiting for GPUs. Remove only truly empty remnants; never overwrite a
  # partial or completed training run.
  if [[ -d "$out" ]]; then
    if [[ -n "$(find "$out" -mindepth 1 -print -quit)" ]]; then
      fail "Non-empty incomplete output exists; refusing to overwrite: $out"
    fi
    rmdir "$out"
    rm -f "$REPO/high-level/dp/logs/lerobot-train/$name"
  fi

  wait_for_gpu "$gpu"
  log "TRAIN START gpu=$gpu episodes=$episodes seed=$seed data=$repo_id"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
    --dataset.root="$data_path" --dataset.repo_id="$repo_id" \
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=100 --policy.n_action_steps=50 \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.use_action_loss_weight=false \
    --policy.plucker_conditioning=true --policy.plucker_horizontal_fov_deg=55 \
    --policy.interaction_state_conditioning=true \
    --policy.interaction_state_prediction_mode=decoder_chunk \
    --policy.interaction_state_probe_only=false \
    --policy.interaction_state_auxiliary_only=false \
    --policy.interaction_contact_loss_weight=0.05 \
    --policy.interaction_handle_loss_weight=0.05 \
    --policy.interaction_door_loss_weight=0.05 \
    --batch_size=16 --steps="$TRAIN_STEPS" --seed="$seed" --num_workers=3 \
    --save_freq="$TRAIN_STEPS" --log_freq=10 --keyframe_sampling_ratio=0 \
    --phase_sampling_feature=subtask_index \
    --phase_sampling_weights="$PHASE_WEIGHTS" \
    --phase_sampling_default_weight=1 \
    --wandb.enable=true --wandb.project=door-act --wandb.disable_artifact=true \
    --job_name="$name" --output_dir="$out" >"$log_file" 2>&1
  [[ -s "$out/checkpoints/$TRAIN_STEPS/pretrained_model/model.safetensors" ]] \
    || fail "Missing final checkpoint after training: $out"
  ensure_link "$out" "$REPO/high-level/dp/logs/lerobot-train/$name"
  log "TRAIN DONE gpu=$gpu episodes=$episodes seed=$seed"
}

export_one() {
  local gpu=$1 episodes=$2 seed=$3 name data official wrapped log_file
  name=$(job_name "$episodes" "$seed")
  data=$(data_name "$episodes")
  official="$TRAIN_ARCHIVE/$name/checkpoints/$TRAIN_STEPS"
  wrapped="$WRAP_ARCHIVE/$name/$TRAIN_STEPS"
  log_file="$PIPE_ARCHIVE/export_${episodes}_seed${seed}.log"
  if [[ -s "$wrapped/model_latest.pt" ]]; then
    ensure_link "$WRAP_ARCHIVE/$name" "$REPO/high-level/dp/logs/door-auto-wrapped/$name"
    return 0
  fi
  wait_for_gpu "$gpu"
  log "EXPORT START gpu=$gpu episodes=$episodes seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
    high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
    --official_checkpoint "$official" --root "$(dirname "$(dirname "$(data_root "$episodes")")")" \
    --repo_id "local/$data" --out_dir "$wrapped/model_latest" \
    --manifest_name model_latest.pt --device cuda:0 --depth_only >"$log_file" 2>&1
  [[ -s "$wrapped/model_latest.pt" ]] || fail "Export failed: episodes=$episodes seed=$seed"
  ensure_link "$WRAP_ARCHIVE/$name" "$REPO/high-level/dp/logs/door-auto-wrapped/$name"
  log "EXPORT DONE gpu=$gpu episodes=$episodes seed=$seed"
}

eval_one() {
  local gpu=$1 episodes=$2 seed=$3 name wrapped run log_file
  name=$(job_name "$episodes" "$seed")
  wrapped="$WRAP_ARCHIVE/$name/$TRAIN_STEPS/model_latest.pt"
  run="$EVAL_ARCHIVE/${name}_step${TRAIN_STEPS}_wc_nomenkuang_seed${BASE_SEED}_open80_traverse0p8_64"
  log_file="$PIPE_ARCHIVE/eval_${episodes}_seed${seed}.log"
  if [[ -s "$run/summary.json" ]]; then
    ensure_link "$run" "$REPO/high-level/logs/door-policy-success/$(basename "$run")"
    return 0
  fi
  wait_for_gpu "$gpu"
  log "EVAL START gpu=$gpu episodes=$episodes seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "$wrapped" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
    --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
    --steps 2600 --pass_open_angle_deg 80 --pull_traversal_distance_m 0.8 \
    --base_seed "$BASE_SEED" --headless --graphics_device_id 0 \
    --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
    --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5 --run_root "$run" -- \
    --door_name wc_nomenkuang --stop_distance 0.293 \
    --ikpush_door_wall_x_offset_rand 0 --wc4_disable_door_open_resistance \
    --door_auto_open_force 0 --pull_vx_only_base --pull_panel_base_mode no_vy \
    --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
    --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
    --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 >"$log_file" 2>&1
  [[ -s "$run/summary.json" ]] || fail "Evaluation failed: episodes=$episodes seed=$seed"
  ensure_link "$run" "$REPO/high-level/logs/door-policy-success/$(basename "$run")"
  log "EVAL DONE gpu=$gpu episodes=$episodes seed=$seed"
}

run_one() {
  local gpu=$1 episodes=$2 seed=$3
  train_one "$gpu" "$episodes" "$seed"
  export_one "$gpu" "$episodes" "$seed"
  eval_one "$gpu" "$episodes" "$seed"
}

worker() {
  local gpu=$1; shift
  local spec episodes seed
  for spec in "$@"; do
    episodes=${spec%%:*}
    seed=${spec##*:}
    run_one "$gpu" "$episodes" "$seed"
  done
}

summarize() {
  "$CONDA" run --no-capture-output -n b1z1_lerobot python - \
    "$EVAL_ARCHIVE" "$BASE_SEED" "$TRAIN_STEPS" >"$PIPE_ARCHIVE/results64.tsv" <<'PY'
import json, sys
from pathlib import Path
root, base, step = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
print("episodes\ttrain_seed\tdoor_open\ttraversal\tjoint")
for episodes in (50, 100, 200, 500):
    seeds = (1000, 2000, 3000, 4000) if episodes == 500 else (1000, 2000)
    for seed in seeds:
        if episodes == 500 and seed in (1000, 2000):
            old = f"leroact_a2w_0813_wc_nomenkuang_fixed15_observable_v1_500_plucker_interaction_phasebalanced_hold01_critical2_seed{seed}_500k_chunk100_exec50_bs16"
            path = Path("/home/ps/workspace/txc/doorgym/high-level/logs/door-policy-success") / f"{old}_step200000_wc_nomenkuang_seed{base}_open80_traverse0p8_64" / "summary.json"
        else:
            name = f"leroact_a2w_0813_wc_nomenkuang_fixed15_nested{episodes}_plucker_interaction_phasebalanced_hold01_critical2_seed{seed}_200k_chunk100_exec50_bs16"
            path = root / f"{name}_step{step}_wc_nomenkuang_seed{base}_open80_traverse0p8_64" / "summary.json"
        if not path.is_file():
            print(f"{episodes}\t{seed}\tMISSING\tMISSING\tMISSING")
            continue
        d = json.loads(path.read_text())
        n = int(d["trials"])
        print(f"{episodes}\t{seed}\t{int(d['door_open_successes'])}/{n}\t{int(d['traversal_successes'])}/{n}\t{int(d['successes'])}/{n}")
PY
  cat "$PIPE_ARCHIVE/results64.tsv"
}

cd "$REPO"
for episodes in 50 100 200 500; do validate_data "$episodes"; done

log "Launching persistent queues. Other users' GPU processes are never terminated."
# GPU 2 is currently the least occupied; GPUs 0 and 1 wait until their external
# rendering jobs release enough memory/utilization.
worker 2 50:1000 100:2000 500:3000 & w2=$!
worker 0 50:2000 100:1000 500:4000 & w0=$!
worker 1 200:1000 200:2000 & w1=$!
wait "$w2" || fail "GPU2 queue failed"
wait "$w0" || fail "GPU0 queue failed"
wait "$w1" || fail "GPU1 queue failed"

summarize
log "COMPLETE"
