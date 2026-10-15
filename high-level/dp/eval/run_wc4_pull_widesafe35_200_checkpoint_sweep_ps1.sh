#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
SEED=615455575
DATA=local/door_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_interaction_state
BASE=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_baseline_seed1000_50k_chunk100_exec50_bs16
INTER=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16
OUT="$REPO/high-level/logs/pull-checkpoint-sweep/wc4_pull_0806_widesafe35_lateretreat_v5_200"

cd "$REPO"
mkdir -p "$OUT"
exec >>"$OUT/sweep.log" 2>&1

export_checkpoint() {
    local gpu=$1 job=$2 step=$3
    local official="$REPO/high-level/dp/logs/lerobot-train/$job/checkpoints/$step"
    local wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$job/$step/model_latest.pt"
    [[ -s "$wrapped" ]] && return
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" --root high-level/data/lerobot \
        --repo_id "$DATA" \
        --out_dir "${wrapped%.pt}" --manifest_name model_latest.pt \
        --device cuda:0 --depth_only
}

eval_checkpoint() {
    local gpu=$1 job=$2 step=$3
    local ckpt="$REPO/high-level/dp/logs/door-auto-wrapped/$job/$step/model_latest.pt"
    local run="$REPO/high-level/logs/door-policy-success/${job}_checkpoint${step}_wc4_seed${SEED}_open80_traverse0p8_64"
    [[ -s "$run/summary.json" ]] && return
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
        --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5
}

for step in 030000 040000; do
    export_checkpoint 2 "$BASE" "$step" & p0=$!
    export_checkpoint 3 "$INTER" "$step" & p1=$!
    wait "$p0"
    wait "$p1"
done

for step in 030000 040000; do
    eval_checkpoint 2 "$BASE" "$step" >"$OUT/eval_${BASE}_${step}.log" 2>&1 & p0=$!
    eval_checkpoint 3 "$INTER" "$step" >"$OUT/eval_${INTER}_${step}.log" 2>&1 & p1=$!
    wait "$p0"
    wait "$p1"
done

python3 - "$REPO" "$BASE" "$INTER" "$SEED" <<'PY' | tee "$OUT/results.txt"
import json
import sys
from pathlib import Path

repo, base, inter, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
root = repo / "high-level/logs/door-policy-success"
for job in (base, inter):
    for step in ("030000", "040000"):
        p = root / f"{job}_checkpoint{step}_wc4_seed{seed}_open80_traverse0p8_64" / "summary.json"
        d = json.loads(p.read_text())
        print(f"{job} {step}: open={d['door_open_successes']}/{d['trials']} "
              f"traversal={d['traversal_successes']}/{d['trials']} "
              f"joint={d['successes']}/{d['trials']}")
    p = root / f"{job}_wc4_seed{seed}_open80_traverse0p8_64" / "summary.json"
    d = json.loads(p.read_text())
    print(f"{job} 050000: open={d['door_open_successes']}/{d['trials']} "
          f"traversal={d['traversal_successes']}/{d['trials']} "
          f"joint={d['successes']}/{d['trials']}")
PY
