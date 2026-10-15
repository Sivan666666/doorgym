#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
TRAIN_PY=/home/ps/miniconda3/envs/b1z1_lerobot/bin/python
SIM_ENV=txc_vbc
DATASET_ID=${DATASET_ID:-local/door_a2w_glass_door_jointstate9_gymjacobian_skill_heavier_50}
DOOR_CFG="$ROOT/high-level/data/cfg/b1z1_opendoor.yaml"
DOOR_NAME=${DOOR_NAME:-glass_door}
SKILL=${SKILL:-$ROOT/high-level/float_ik/door_twin/examples/glass_door_push_traverse_skill.json}
EVAL_STEPS=${EVAL_STEPS:-1100}
EVAL_TAG=${EVAL_TAG:-glass_door}
DOOR_DOF_INDEX=${DOOR_DOF_INDEX:-0}
LOG_ROOT=${LOG_ROOT:-$ROOT/high-level/logs/glass_door_joint9_skill50_eval_20260809}
SUCCESS_ROOT="$ROOT/high-level/logs/door-policy-success"

BASE_RUN=${BASE_RUN:-leroact_a2w_glass_door_joint9_skill50_baseline_seed1000_50k_chunk100_exec50_bs16}
PLK_RUN=${PLK_RUN:-leroact_a2w_glass_door_joint9_skill50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16}
BASE_GPU=${BASE_GPU:-2}
PLK_GPU=${PLK_GPU:-3}
BASE_OFF="$ROOT/high-level/dp/logs/lerobot-train/$BASE_RUN/checkpoints/050000"
PLK_OFF="$ROOT/high-level/dp/logs/lerobot-train/$PLK_RUN/checkpoints/050000"
BASE_WRAP_ROOT="$ROOT/high-level/dp/logs/door-auto-wrapped/$BASE_RUN"
PLK_WRAP_ROOT="$ROOT/high-level/dp/logs/door-auto-wrapped/$PLK_RUN"
BASE_WRAP="$BASE_WRAP_ROOT/050000"
PLK_WRAP="$PLK_WRAP_ROOT/050000"
BASE_MANIFEST="$BASE_WRAP_ROOT/model_latest.pt"
PLK_MANIFEST="$PLK_WRAP_ROOT/model_latest.pt"
BASE_EVAL="$SUCCESS_ROOT/${BASE_RUN}_${EVAL_TAG}_seed615455575_open80_64"
PLK_EVAL="$SUCCESS_ROOT/${PLK_RUN}_${EVAL_TAG}_seed615455575_open80_64"

mkdir -p "$LOG_ROOT" "$BASE_WRAP" "$PLK_WRAP"
cd "$ROOT"

ts() { date '+%F %T'; }

export_one() {
    local official=$1
    local out=$2
    local manifest=$3
    local gpu=$4
    local name=$5
    if [[ -s "$manifest" ]]; then
        echo "[$(ts)] $name wrapper exists: $manifest"
        return
    fi
    echo "[$(ts)] exporting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$TRAIN_PY" \
        high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
        --official_checkpoint "$official" \
        --root "$ROOT/high-level/data/lerobot" \
        --repo_id "$DATASET_ID" \
        --out_dir "$out" \
        --manifest_name model_latest.pt \
        --device cuda:0 \
        --depth_only \
        >"$LOG_ROOT/export_${name}.log" 2>&1
    test -s "$manifest"
}

eval_one() {
    local checkpoint=$1
    local gpu=$2
    local run_root=$3
    local name=$4
    if [[ -s "$run_root/summary.json" ]]; then
        echo "[$(ts)] $name summary exists: $run_root/summary.json"
        return
    fi
    echo "[$(ts)] evaluating $name on GPU $gpu"
    "$CONDA" run --no-capture-output -n "$SIM_ENV" python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$checkpoint" \
        --yaml "$DOOR_CFG" \
        --mode ikpush \
        --robot_body a2wz1 \
        --num_envs 16 \
        --total_trials 64 \
        --steps "$EVAL_STEPS" \
        --pass_open_angle_deg 80 \
        --door_dof_index "$DOOR_DOF_INDEX" \
        --success_metric abs \
        --base_seed 615455575 \
        --headless \
        --graphics_device_id "$gpu" \
        --rl_device cuda:"$gpu" \
        --sim_device cuda:"$gpu" \
        --dp_action_horizon 25 \
        --dp_fps 25 \
        --depth_only \
        --no_enable_depth_noise \
        --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization \
        --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 \
        --run_root "$run_root" \
        --stream_output \
        -- \
        --door_name "$DOOR_NAME" \
        --skill_program_json "$SKILL" \
        --arm_ik_solver gym_jacobian \
        --robot_pitch 0.0 \
        --ikpush_robot_pitch_rand_min 0.0 \
        --ikpush_robot_pitch_rand_max 0.0 \
        --ee_pose_frame robot_base_full \
        --camera_depth_clip_lower 0.2 \
        --camera_depth_clip_far 1.5 \
        >"$LOG_ROOT/eval_${name}.log" 2>&1
    test -s "$run_root/summary.json"
}

export_one "$BASE_OFF" "$BASE_WRAP" "$BASE_MANIFEST" "$BASE_GPU" baseline
export_one "$PLK_OFF" "$PLK_WRAP" "$PLK_MANIFEST" "$PLK_GPU" plucker

eval_one "$BASE_MANIFEST" "$BASE_GPU" "$BASE_EVAL" baseline &
base_pid=$!
eval_one "$PLK_MANIFEST" "$PLK_GPU" "$PLK_EVAL" plucker &
plk_pid=$!

wait "$base_pid"
wait "$plk_pid"

"$TRAIN_PY" - "$BASE_EVAL/summary.json" "$PLK_EVAL/summary.json" <<'PY_SUMMARY'
import json
import sys

for name, path in zip(["baseline", "plucker"], sys.argv[1:]):
    data = json.load(open(path, encoding="utf-8"))
    batches = [entry.get("successes") for entry in data.get("batch_logs", [])]
    print(
        "EVAL_RESULT",
        name,
        f"{data.get('successes')}/{data.get('total_trials')}",
        f"success_rate={data.get('success_rate'):.6f}",
        f"door_open={data.get('door_open_successes')}",
        f"traversal={data.get('traversal_successes')}",
        f"batches={batches}",
    )
PY_SUMMARY

echo "[$(ts)] COMPLETE"
