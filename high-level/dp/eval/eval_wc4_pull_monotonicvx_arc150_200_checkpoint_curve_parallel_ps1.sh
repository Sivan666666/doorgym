#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
PY=/home/ps/miniconda3/envs/txc_vbc/bin/python
TAG=0807_widesafe35_monotonicvx_arc150_v7
SEED=615455575
DATA="local/door_a2w_wc4_pull_joint9_${TAG}_200_interaction_state"
BASE_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_200_baseline_seed1000_50k_chunk100_exec50_bs16"
INTER_JOB="leroact_a2w_wc4_pull_joint9_${TAG}_200_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16"
PIPE="$REPO/high-level/logs/pull-checkpoint-curve/${TAG}_200_parallel"

cd "$REPO"
mkdir -p "$PIPE"
exec >>"$PIPE/pipeline.log" 2>&1

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt"
}

export_one() {
    local gpu=$1 name=$2 step=$3
    local official="$REPO/high-level/dp/logs/lerobot-train/$name/checkpoints/$step"
    local wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$name/$step"
    [[ ! -s "$wrapped/model_latest.pt" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$DATA" --out_dir "$wrapped/model_latest" \
        --manifest_name model_latest.pt --device cuda:0 --depth_only \
        >"$PIPE/export_${name}_${step}.log" 2>&1
}

eval_one() {
    local gpu=$1 name=$2 step=$3
    local ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$name/$step/model_latest.pt"
    local run="$REPO/high-level/logs/door-policy-success/${name}_step${step}_wc4_seed${SEED}_open80_traverse0p8_64"
    [[ ! -s "$run/summary.json" ]] || return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
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
        --camera_depth_clip_far 1.5 >"$PIPE/eval_${name}_${step}.log" 2>&1
}

stage EXPORT
export_one 2 "$BASE_JOB" 030000 & p0=$!
export_one 2 "$BASE_JOB" 040000 & p1=$!
export_one 3 "$INTER_JOB" 030000 & p2=$!
export_one 3 "$INTER_JOB" 040000 & p3=$!
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"

stage EVAL_30K_40K
eval_one 2 "$BASE_JOB" 030000 & p0=$!
eval_one 2 "$BASE_JOB" 040000 & p1=$!
eval_one 3 "$INTER_JOB" 030000 & p2=$!
eval_one 3 "$INTER_JOB" 040000 & p3=$!
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"

stage SUMMARIZE
"$PY" - "$REPO" "$SEED" "$BASE_JOB" "$INTER_JOB" <<'PY' | tee "$PIPE/final_results.tsv"
import json
import sys
from pathlib import Path

repo, seed, *names = sys.argv[1:]
repo = Path(repo)
print("model\tstep\topen\ttraversal\tjoint")
for name in names:
    for step in ("010000", "020000", "030000", "040000"):
        run = repo / "high-level/logs/door-policy-success" / f"{name}_step{step}_wc4_seed{seed}_open80_traverse0p8_64"
        data = json.loads((run / "summary.json").read_text())
        print(
            f"{name}\t{step}\t{data['door_open_successes']}/{data['trials']}\t"
            f"{data['traversal_successes']}/{data['trials']}\t{data['successes']}/{data['trials']}"
        )
PY

stage COMPLETE
