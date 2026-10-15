#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
TAG=0807_widesafe35_monotonicvx_arc150_v7
TARGET=200
SEED=615455575
PIPE="$REPO/high-level/logs/pull-data-pipeline/wc4_pull_${TAG}_${TARGET}"
DATA="local/door_a2w_wc4_pull_joint9_${TAG}_${TARGET}_interaction_state"
BASE_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_${TARGET}_baseline_seed1000_50k_chunk100_exec50_bs16"
INTER_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_${TARGET}_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16"
BASE_OUT="$REPO/high-level/dp/logs/lerobot-train/$BASE_JOB"
INTER_OUT="$REPO/high-level/dp/logs/lerobot-train/$INTER_JOB"

cd "$REPO"
mkdir -p "$PIPE"
exec >>"$PIPE/resume_pipeline.log" 2>&1

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt"
}

fail() {
    stage "FAILED_RESUME: $*"
    exit 1
}

stage RESUME_TRAIN_TWO

# The baseline 20K checkpoint was fully committed before the disk filled.
BASE_CFG="$BASE_OUT/checkpoints/020000/pretrained_model/train_config.json"
test -s "$BASE_CFG"

# Interaction's 20K training state was interrupted while being written. Resume
# from the last complete checkpoint (10K) and remove only that incomplete 20K artifact.
INTER_CFG="$INTER_OUT/checkpoints/010000/pretrained_model/train_config.json"
test -s "$INTER_CFG"
rm -rf -- "$INTER_OUT/checkpoints/020000"

resume_one() {
    local gpu=$1 cfg=$2 name=$3
    local out="$REPO/high-level/dp/logs/lerobot-train/$name"
    [[ ! -s "$out/checkpoints/050000/pretrained_model/model.safetensors" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
        --config_path="$cfg" --resume=true \
        >"$PIPE/resume_train_${name}.log" 2>&1
}

resume_one 1 "$BASE_CFG" "$BASE_JOB" & p0=$!
resume_one 3 "$INTER_CFG" "$INTER_JOB" & p1=$!
wait "$p0" || fail "baseline resume failed"
wait "$p1" || fail "interaction resume failed"

export_one() {
    local gpu=$1 name=$2
    local official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/050000"
    local wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000"
    [[ ! -s "$wrapped/model_latest.pt" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$DATA" --out_dir "$wrapped/model_latest" \
        --manifest_name model_latest.pt --device cuda:0 --depth_only \
        >"$PIPE/resume_export_${name}.log" 2>&1
}

stage EXPORT_TWO
export_one 1 "$BASE_JOB"
export_one 3 "$INTER_JOB"

eval_one() {
    local gpu=$1 name=$2
    local ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$name/050000/model_latest.pt"
    local run="$REPO/high-level/logs/door-policy-success/${name}_wc4_seed${SEED}_open80_traverse0p8_64"
    [[ ! -s "$run/summary.json" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
        --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
        --steps 2600 --pass_open_angle_deg 80 --pull_traversal_distance_m 0.8 \
        --base_seed "$SEED" --headless --graphics_device_id 0 \
        --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
        --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 --run_root "$run" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pull_vx_only_base --pull_panel_base_mode no_vy \
        --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
        --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
        --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        >"$PIPE/resume_eval_${name}.log" 2>&1
}

stage EVAL_TWO
eval_one 1 "$BASE_JOB" & e0=$!
eval_one 3 "$INTER_JOB" & e1=$!
wait "$e0" || fail "baseline evaluation failed"
wait "$e1" || fail "interaction evaluation failed"

stage SUMMARIZE
"$PY" - "$REPO" "$SEED" "$BASE_JOB" "$INTER_JOB" <<'PY' | tee "$PIPE/final_results.txt"
import json
import sys
from pathlib import Path

repo, seed = Path(sys.argv[1]), sys.argv[2]
for name in sys.argv[3:]:
    path = repo / "high-level/logs/door-policy-success" / f"{name}_wc4_seed{seed}_open80_traverse0p8_64" / "summary.json"
    data = json.loads(path.read_text())
    print(
        f"{name}: open={data['door_open_successes']}/{data['trials']} "
        f"traversal={data['traversal_successes']}/{data['trials']} "
        f"joint={data['successes']}/{data['trials']}"
    )
PY

stage COMPLETE
