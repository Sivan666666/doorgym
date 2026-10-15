#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/home/ps/workspace/txc/doorgym}
CONDA=${CONDA:-/home/ps/miniconda3/bin/conda}
SIM_ENV=${SIM_ENV:-txc_vbc}
SEED=${SEED:-615455575}
LOG_ROOT=${LOG_ROOT:-$ROOT/high-level/logs/wooden_circle_joint9_first3_eval_20260812}
SUCCESS_ROOT="$ROOT/high-level/logs/door-policy-success"

RUN_50_BASE=leroact_a2w_wooden_circle_joint9_50_baseline_seed1000_50k_chunk100_exec50_bs16
RUN_50_PLK=leroact_a2w_wooden_circle_joint9_50_plucker_fov55_seed1000_50k_chunk100_exec50_bs16
RUN_200_BASE=leroact_a2w_wooden_circle_joint9_200_baseline_seed1000_50k_chunk100_exec50_bs16

mkdir -p "$LOG_ROOT" "$SUCCESS_ROOT"
cd "$ROOT"

eval_one() {
    local run=$1 gpu=$2
    local checkpoint="$ROOT/high-level/dp/logs/door-auto-wrapped/$run/model_latest.pt"
    local output="$SUCCESS_ROOT/${run}_wooden_circle_seed${SEED}_open80_64"
    local log="$LOG_ROOT/eval_${run}.log"
    if [[ -s "$output/summary.json" ]]; then
        echo "REUSE $output/summary.json"
        return
    fi
    test -s "$checkpoint"
    echo "START run=$run gpu=$gpu"
    "$CONDA" run --no-capture-output -n "$SIM_ENV" python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$checkpoint" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
        --mode ikpush --robot_body a2wz1 --num_envs 16 --total_trials 64 \
        --steps 1000 --pass_open_angle_deg 80 --door_dof_index 2 \
        --success_metric abs --base_seed "$SEED" --headless \
        --graphics_device_id "$gpu" --rl_device cuda:"$gpu" --sim_device cuda:"$gpu" \
        --dp_action_horizon 25 --dp_fps 25 --depth_only \
        --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --run_root "$output" --stream_output -- \
        --door_name wooden_circle --door_z_offset 0.01 --robot_y 0.2 \
        --arm_ik_solver gym_jacobian --no_ikpush_env_randomization \
        --robot_pitch 0.0 --ikpush_robot_pitch_rand_min 0.0 \
        --ikpush_robot_pitch_rand_max 0.0 --ee_pose_frame robot_base_full \
        --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        >"$log" 2>&1
    test -s "$output/summary.json"
    echo "COMPLETE run=$run summary=$output/summary.json"
}

(eval_one "$RUN_50_BASE" 2; eval_one "$RUN_200_BASE" 2) & p_base=$!
eval_one "$RUN_50_PLK" 3 & p_plk=$!
wait "$p_base"
wait "$p_plk"

/home/ps/miniconda3/envs/b1z1_lerobot/bin/python - "$SUCCESS_ROOT" "$SEED" \
    "$RUN_50_BASE" "$RUN_50_PLK" "$RUN_200_BASE" <<'PY'
import json, pathlib, sys
root, seed = pathlib.Path(sys.argv[1]), sys.argv[2]
for run in sys.argv[3:]:
    path = root / f"{run}_wooden_circle_seed{seed}_open80_64" / "summary.json"
    data = json.loads(path.read_text())
    batches = [x.get("successes") for x in data.get("batch_logs", [])]
    print("EVAL_RESULT", run, f"{data['successes']}/{data['total_trials']}",
          f"rate={data['success_rate']:.6f}", f"batches={batches}")
PY

echo PIPELINE_COMPLETE
