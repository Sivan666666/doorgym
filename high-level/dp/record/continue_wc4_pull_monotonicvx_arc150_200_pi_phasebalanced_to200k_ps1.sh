#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
LR_PY=/home/ps/miniconda3/envs/b1z1_lerobot/bin/python
TAG=0807_widesafe35_monotonicvx_arc150_v7
SEED=615455575
DATA="local/door_a2w_wc4_pull_joint9_${TAG}_200_interaction_state"
SOURCE_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_200_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16"
JOB="leroact_a2w_wc4_pull_joint9_${TAG}_200_plucker_interaction_phasebalanced_hold01_critical3_continue50to200_seed1000"
SOURCE_OUT="$REPO/high-level/dp/logs/lerobot-train/$SOURCE_JOB"
OUT="$REPO/high-level/dp/logs/lerobot-train/$JOB"
PIPE="$REPO/high-level/logs/pull-phasebalanced/${JOB}"
PHASE_WEIGHTS="8:3,11:0.1,12:3,13:3,14:3"
STEPS=(100000 150000 200000)

cd "$REPO"
mkdir -p "$PIPE"
exec >>"$PIPE/pipeline.log" 2>&1

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt"
}

fail() {
    stage "FAILED: $*"
    exit 1
}

stage PREPARE
SOURCE_CKPT="$SOURCE_OUT/checkpoints/050000"
test -s "$SOURCE_CKPT/pretrained_model/model.safetensors" || fail "missing source 50K model"
test -s "$SOURCE_CKPT/training_state/optimizer_state.safetensors" || fail "missing source optimizer state"
if [[ ! -s "$OUT/checkpoints/050000/pretrained_model/model.safetensors" ]]; then
    mkdir -p "$OUT/checkpoints"
    cp -a "$SOURCE_CKPT" "$OUT/checkpoints/050000"
fi
CFG="$OUT/checkpoints/050000/pretrained_model/train_config.json"
test -s "$CFG" || fail "missing copied train config"

# Record the exact natural and weighted phase distributions before training.
"$LR_PY" - "$REPO/high-level/data/lerobot/$DATA" "$PHASE_WEIGHTS" <<'PY' \
    | tee "$PIPE/phase_sampling_distribution.tsv"
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
configured = {int(k): float(v) for k, v in (item.split(":") for item in sys.argv[2].split(","))}
table = pq.read_table(root / "data/chunk-000/file-000.parquet", columns=["subtask_index"])
values = table.column("subtask_index").to_pylist()
values = [int(v[0] if isinstance(v, list) else v) for v in values]
counts = {phase: values.count(phase) for phase in sorted(set(values))}
total = sum(counts.values())
weighted_total = sum(count * configured.get(phase, 1.0) for phase, count in counts.items())
print("phase_id\tframes\tnatural_ratio\trelative_weight\texpected_sampling_ratio")
for phase, count in counts.items():
    weight = configured.get(phase, 1.0)
    print(f"{phase}\t{count}\t{count/total:.8f}\t{weight:g}\t{count*weight/weighted_total:.8f}")
PY

train() {
    stage TRAIN_50K_TO_200K
    CUDA_VISIBLE_DEVICES=3 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
        --config_path="$CFG" --resume=true --output_dir="$OUT" --job_name="$JOB" \
        --steps=200000 --save_freq=50000 --log_freq=10 \
        --phase_sampling_feature=subtask_index \
        --phase_sampling_weights="$PHASE_WEIGHTS" \
        --phase_sampling_default_weight=1.0 \
        >"$PIPE/train.log" 2>&1
}

export_one() {
    local step=$1
    local official="$OUT/checkpoints/$step"
    local wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$JOB/$step"
    [[ ! -s "$wrapped/model_latest.pt" ]] || return
    CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$DATA" --out_dir "$wrapped/model_latest" \
        --manifest_name model_latest.pt --device cuda:0 --depth_only \
        >"$PIPE/export_${step}.log" 2>&1
}

eval_one() {
    local step=$1
    local ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$JOB/$step/model_latest.pt"
    local run="$REPO/high-level/logs/door-policy-success/${JOB}_step${step}_wc4_seed${SEED}_open80_traverse0p8_64"
    [[ ! -s "$run/summary.json" ]] || return
    CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
        --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
        --parallel_batches 2 --steps 2600 --pass_open_angle_deg 80 \
        --pull_traversal_distance_m 0.8 --base_seed "$SEED" --headless \
        --graphics_device_id 0 --rl_device cuda:0 --sim_device cuda:0 \
        --dp_action_horizon 25 --dp_fps 25 --depth_only \
        --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 --run_root "$run" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pull_vx_only_base --pull_panel_base_mode no_vy --robot_pitch 0 \
        --ikpush_robot_pitch_rand_min 0 --ikpush_robot_pitch_rand_max 0 \
        --ee_pose_frame robot_base_full --camera_depth_clip_lower 0.2 \
        --camera_depth_clip_far 1.5 >"$PIPE/eval_${step}.log" 2>&1
}

train & train_pid=$!

for step in "${STEPS[@]}"; do
    stage "WAIT_CHECKPOINT_$step"
    while [[ ! -s "$OUT/checkpoints/$step/pretrained_model/model.safetensors" ]] \
        || [[ ! -s "$OUT/checkpoints/$step/training_state/optimizer_state.safetensors" ]]; do
        if ! kill -0 "$train_pid" 2>/dev/null; then
            wait "$train_pid" || fail "training exited before checkpoint $step"
            fail "training ended without checkpoint $step"
        fi
        sleep 60
    done
    stage "EXPORT_$step"
    export_one "$step"
    stage "EVAL_$step"
    eval_one "$step"
done

wait "$train_pid" || fail "training failed"

stage SUMMARIZE
"$PY" - "$REPO" "$SEED" "$JOB" <<'PY' | tee "$PIPE/final_results.tsv"
import json
import sys
from pathlib import Path

repo, seed, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
print("step\topen\ttraversal\tjoint\topen_rate\ttraversal_rate\tjoint_rate")
for step in ("100000", "150000", "200000"):
    run = repo / "high-level/logs/door-policy-success" / f"{name}_step{step}_wc4_seed{seed}_open80_traverse0p8_64"
    data = json.loads((run / "summary.json").read_text())
    trials = int(data["trials"])
    opened = int(data["door_open_successes"])
    traversed = int(data["traversal_successes"])
    joint = int(data["successes"])
    print(
        f"{step}\t{opened}/{trials}\t{traversed}/{trials}\t{joint}/{trials}\t"
        f"{opened/trials:.8f}\t{traversed/trials:.8f}\t{joint/trials:.8f}"
    )
PY

stage COMPLETE
