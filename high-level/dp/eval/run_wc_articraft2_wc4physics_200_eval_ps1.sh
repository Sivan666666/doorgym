#!/usr/bin/env bash
set -u -o pipefail

ROOT=/home/ps/workspace/txc/doorgym
PYTHON=/home/ps/miniconda3/bin/conda
ENV_NAME=txc_vbc
EVAL_SCRIPT="$ROOT/high-level/dp/eval/eval_door_policy_success.py"
DOOR_CFG="$ROOT/high-level/data/cfg/b1z1_opendoor.yaml"
SKILL="$ROOT/high-level/float_ik/door_twin/examples/wc_articraft2_push_traverse_skill.json"
LOG_ROOT="$ROOT/high-level/logs/door-policy-success"

BASELINE_CKPT="$ROOT/high-level/dp/logs/lerobot-train/leroact_a2w_wc_articraft2_wc4physics_roboty_200_baseline_50k_chunk100_exec50_bs16_seed1000_0730/checkpoints/050000"
INTERACTION_CKPT="$ROOT/high-level/dp/logs/lerobot-train/leroact_a2w_wc_articraft2_wc4physics_roboty_200_plucker_fov55_interaction_decoderchunk_w005_50k_chunk100_exec50_bs16_seed1000_0730/checkpoints/050000"

BASELINE_RUN="$LOG_ROOT/wc_articraft2_wc4physics_roboty200_baseline_act50k_pitch0_h25_seed615455575_n64_0730"
INTERACTION_RUN="$LOG_ROOT/wc_articraft2_wc4physics_roboty200_plucker_interaction_act50k_pitch0_h25_seed615455575_n64_0730"
PIPELINE_LOG="$ROOT/high-level/logs/wc_articraft2_wc4physics_eval_20260730"

mkdir -p "$PIPELINE_LOG" "$BASELINE_RUN" "$INTERACTION_RUN"

run_eval() {
    local checkpoint=$1
    local gpu=$2
    local run_root=$3

    "$PYTHON" run --no-capture-output -n "$ENV_NAME" python "$EVAL_SCRIPT" \
        --checkpoint "$checkpoint" \
        --door_cfg "$DOOR_CFG" \
        --mode ikpush \
        --robot_body a2wz1 \
        --num_envs 16 \
        --total_trials 64 \
        --steps 1400 \
        --pass_open_angle_deg 80 \
        --success_metric abs \
        --headless \
        --graphics_device_id "$gpu" \
        --rl_device "cuda:$gpu" \
        --sim_device "cuda:$gpu" \
        --depth_only \
        --dp_action_horizon 25 \
        --dp_fps 25 \
        --base_seed 615455575 \
        --no_enable_depth_noise \
        --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization \
        --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 \
        --run_root "$run_root" \
        --stream_output \
        -- \
        --door_name wc_articraft2 \
        --skill_program_json "$SKILL" \
        --ee_pose_frame robot_base_full \
        --robot_pitch 0.0 \
        --ikpush_robot_pitch_rand_min 0.0 \
        --ikpush_robot_pitch_rand_max 0.0
}

echo "[$(date '+%F %T')] Starting baseline evaluation on physical GPU 1."
run_eval "$BASELINE_CKPT" 1 "$BASELINE_RUN" \
    >"$PIPELINE_LOG/baseline.out" 2>&1 &
baseline_pid=$!

echo "[$(date '+%F %T')] Starting Plucker Interaction evaluation on physical GPU 3."
run_eval "$INTERACTION_CKPT" 3 "$INTERACTION_RUN" \
    >"$PIPELINE_LOG/plucker_interaction.out" 2>&1 &
interaction_pid=$!

echo "[$(date '+%F %T')] Evaluation PIDs: baseline=$baseline_pid interaction=$interaction_pid"

set +e
wait "$baseline_pid"
baseline_status=$?
wait "$interaction_pid"
interaction_status=$?
set -e

if [[ $baseline_status -ne 0 || $interaction_status -ne 0 ]]; then
    echo "[$(date '+%F %T')] EVAL_PIPELINE_ERROR baseline=$baseline_status interaction=$interaction_status"
    exit 1
fi

test -s "$BASELINE_RUN/summary.json"
test -s "$INTERACTION_RUN/summary.json"

/home/ps/miniconda3/envs/b1z1_lerobot/bin/python - \
    "$BASELINE_RUN/summary.json" "$INTERACTION_RUN/summary.json" <<'PY'
import json
import sys

for name, path in zip(("baseline", "plucker_interaction"), sys.argv[1:]):
    summary = json.load(open(path, encoding="utf-8"))
    batch_successes = [entry["successes"] for entry in summary["batch_logs"]]
    print(
        f"EVAL_RESULT {name} "
        f"{summary['successes']}/{summary['total_trials']}="
        f"{summary['success_rate']:.6f} batches={batch_successes}"
    )
PY

echo "[$(date '+%F %T')] EVAL_PIPELINE_COMPLETE"
