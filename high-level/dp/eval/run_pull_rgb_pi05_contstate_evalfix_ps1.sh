#!/usr/bin/env bash
# Re-evaluate existing RGB-50/200 continuous-state 30K checkpoints; never retrain.
set -euo pipefail
REPO=${REPO:-/home/ps/workspace/txc/doorgym}
ARCHIVE=${ARCHIVE:-/mnt/sda/ps_archive/doorgym}
TAG=${TAG:-0907_pull_rgb_pi05_contstate_evalfix_v1}
BASE_SEED=615455575
TRAIN_PY=/home/ps/miniconda3/envs/b1z1_lerobot/bin/python
CONDA=/home/ps/miniconda3/bin/conda
PIPE="$ARCHIVE/high-level/logs/rgb_pi05_continuous_state/$TAG"
WRAP_ROOT="$ARCHIVE/high-level/dp/logs/door-auto-wrapped/$TAG"
EVAL_ROOT="$ARCHIVE/high-level/logs/door-policy-success"
TRAIN_ROOT="$ARCHIVE/high-level/dp/logs/lerobot-train-pi05"
cd "$REPO"
mkdir -p "$PIPE"
[[ ! -e "$PIPE/pipeline.log" || "${RESUME:-0}" == 1 ]] || { echo "Refusing existing pipeline $PIPE"; exit 1; }
exec >>"$PIPE/pipeline.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONPATH="$REPO/high-level:$REPO/high-level/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export DOOR_DP_VIDEO_BACKEND=torchcodec
export TMPDIR="$PIPE/tmp"
mkdir -p "$TMPDIR"
ulimit -c 0
echo "$(date -Is) START" | tee "$PIPE/status.txt"
"$TRAIN_PY" high-level/dp/tests/test_pi05_door_state_loading.py >"$PIPE/unit_tests.log" 2>&1

run_one() {
  local n=$1 gpu=$2
  export CUDA_VISIBLE_DEVICES="$gpu"
  local run="pi05_pull_fixed15_g080_rgb${n}_joint9_contstate_seed1000_30k"
  local dataset="local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_${n}_joint9"
  local root="$ARCHIVE/high-level/data/lerobot/$dataset"
  local official="$TRAIN_ROOT/$run/checkpoints/030000/pretrained_model"
  local wrap="$WRAP_ROOT/$run/030000/model_latest"
  local out="$EVAL_ROOT/${run}_pull_open60_traverse080_seed${BASE_SEED}_${TAG}"
  [[ ! -e "$out" ]] || { echo "Refusing existing evaluation for $run"; return 1; }
  if [[ "${RESUME:-0}" == 1 && -f "${wrap}.pt" && -f "$wrap/door_policy_meta.json" ]]; then
    echo "Reusing exported wrapper $wrap; parity checks still mandatory"
  else
    [[ ! -e "$wrap" ]] || { echo "Refusing existing wrapper for $run"; return 1; }
    echo "$(date -Is) EXPORT RGB-$n GPU-$gpu" | tee "$PIPE/status_rgb${n}.txt"
    "$TRAIN_PY" high-level/dp/export_official_lerobot_pi05_to_door_checkpoint.py \
    --official_checkpoint "$official" --root "$root" --repo_id "$dataset" \
    --out_dir "$wrap" --device cuda:0 --rgb --action_horizon 25 --num_inference_steps 10 \
    >"$PIPE/export_rgb${n}.log" 2>&1
  fi
  echo "$(date -Is) PARITY RGB-$n" | tee "$PIPE/status_rgb${n}.txt"
  "$TRAIN_PY" high-level/dp/eval/verify_pi05_saved_processor_parity.py \
    --official_checkpoint "$official" --door_checkpoint "$wrap" \
    --dataset_root "$root" --repo_id "$dataset" --device cuda:0 \
    --output "$PIPE/parity_rgb${n}.json" >"$PIPE/parity_rgb${n}.log" 2>&1
  echo "$(date -Is) EVAL RGB-$n" | tee "$PIPE/status_rgb${n}.txt"
  "$CONDA" run --no-capture-output -n txc_vbc python \
    high-level/dp/eval/eval_door_policy_success.py \
    --checkpoint "${wrap}.pt" --mode ikpull --robot_body a2wz1 \
    --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
    --num_envs 16 --total_trials 64 --steps 2600 --base_seed "$BASE_SEED" \
    --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 \
    --dp_action_horizon 25 --dp_inference_steps 10 --dp_fps 25 \
    --headless --parallel_batches 1 --run_root "$out" --rgb -- \
    --door_name wc_nomenkuang \
    --wc4_disable_door_open_resistance --door_auto_open_force 0 \
    --gripper_close_ratio 0.8 --robot_pitch 0 \
    --ikpush_robot_pitch_rand_min 0 --ikpush_robot_pitch_rand_max 0 \
    --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
    --depth_camera_rot_rand_deg 5.0 --no_enable_depth_noise \
    --no_enable_depth_gaussian_blur --no_show_seg --dp_rgb_without_masks \
    >"$PIPE/eval_rgb${n}.log" 2>&1
  echo "$(date -Is) COMPLETE RGB-$n" | tee "$PIPE/status_rgb${n}.txt"
}

run_one 50 "${GPU_50:-0}" & p50=$!
run_one 200 "${GPU_200:-1}" & p200=$!
failed=0
wait "$p50" || failed=1
wait "$p200" || failed=1
if (( failed )); then
  echo "$(date -Is) FAILED; inspect per-model logs" | tee "$PIPE/status.txt"
  exit 1
fi
"$TRAIN_PY" - "$EVAL_ROOT" "$TAG" "$PIPE" <<'PY'
import json
import sys
from pathlib import Path
root, tag, pipe = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
results = []
for n in (50, 200):
    run = f"pi05_pull_fixed15_g080_rgb{n}_joint9_contstate_seed1000_30k"
    path = root / f"{run}_pull_open60_traverse080_seed615455575_{tag}" / "summary.json"
    s = json.loads(path.read_text())
    assert s['trials'] == 64 and len(s['batch_logs']) == 4
    entry = {k: s[k] for k in ('door_open_successes','traversal_successes','successes','trials','base_seed')}
    entry.update(episodes=n, summary=str(path))
    results.append(entry)
    print(json.dumps(entry), flush=True)
(pipe / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
PY
echo "$(date -Is) COMPLETE" | tee "$PIPE/status.txt"
