#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
DATA=local/door_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_interaction_state
DATA_ROOT="$REPO/high-level/data/lerobot/$DATA"
EVAL_SEED=615455575
TAG=0808_lrstep40k_2e6_to100k
ROOT="$REPO/high-level/logs/pull-lr-sweep/$TAG"

SOURCE_BASE=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_baseline_seed1000_50k_chunk100_exec50_bs16
SOURCE_INTER=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16

BASE_1000=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_baseline_seed1000_lrstep40k2e6_100k
INTER_1000=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_plucker_interaction_seed1000_lrstep40k2e6_100k
BASE_2000=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_baseline_seed2000_lrstep40k2e6_100k
INTER_2000=leroact_a2w_wc4_pull_joint9_0806_widesafe35_lateretreat_v5_200_plucker_interaction_seed2000_lrstep40k2e6_100k

cd "$REPO"
mkdir -p "$ROOT"
exec >>"$ROOT/pipeline.log" 2>&1

status() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" | tee "$ROOT/status.txt"
}

common_train=(
    --dataset.video_backend=torchcodec --dataset.use_imagenet_stats=false
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false
    --policy.chunk_size=100 --policy.n_action_steps=50
    --policy.vision_backbone=resnet18
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
    --policy.use_action_loss_weight=false --policy.optimizer_lr=1e-5
    --batch_size=16 --steps=100000 --num_workers=3
    --save_freq=5000 --log_freq=10 --keyframe_sampling_ratio=0.0
    --learning_rate_override_step=40000
    --learning_rate_after_override=2e-6
    --wandb.enable=false --wandb.disable_artifact=true
)

interaction_args=(
    --policy.plucker_conditioning=true --policy.plucker_horizontal_fov_deg=55.0
    --policy.interaction_state_conditioning=true
    --policy.interaction_state_prediction_mode=decoder_chunk
    --policy.interaction_state_probe_only=false
    --policy.interaction_state_auxiliary_only=false
    --policy.interaction_contact_loss_weight=0.05
    --policy.interaction_handle_loss_weight=0.05
    --policy.interaction_door_loss_weight=0.05
)

train_resume_40k() {
    local gpu=$1 source=$2 target=$3
    local source_cfg="$REPO/high-level/dp/logs/lerobot-train/$source/checkpoints/040000/pretrained_model/train_config.json"
    local out="$REPO/high-level/dp/logs/lerobot-train/$target"
    if [[ -s "$out/checkpoints/100000/pretrained_model/model.safetensors" ]]; then
        return
    fi
    status "TRAIN resume seed1000 gpu=$gpu target=$target"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
        --config_path="$source_cfg" --resume=true \
        --output_dir="$out" --job_name="$target" \
        --steps=100000 --save_freq=5000 \
        --learning_rate_override_step=40000 \
        --learning_rate_after_override=2e-6 \
        --wandb.enable=false --wandb.disable_artifact=true \
        >"$ROOT/train_${target}.log" 2>&1
}

train_seed2000() {
    local gpu=$1 target=$2 kind=$3
    local out="$REPO/high-level/dp/logs/lerobot-train/$target"
    if [[ -s "$out/checkpoints/100000/pretrained_model/model.safetensors" ]]; then
        return
    fi
    local extra=(--policy.plucker_conditioning=false --policy.interaction_state_conditioning=false)
    if [[ "$kind" == interaction ]]; then
        extra=("${interaction_args[@]}")
    fi
    status "TRAIN from-scratch seed2000 gpu=$gpu target=$target"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
        --dataset.root="$DATA_ROOT" --dataset.repo_id="$DATA" \
        "${common_train[@]}" "${extra[@]}" --seed=2000 \
        --job_name="$target" --output_dir="$out" \
        >"$ROOT/train_${target}.log" 2>&1
}

train_gpu2() {
    train_resume_40k 2 "$SOURCE_BASE" "$BASE_1000"
    train_seed2000 2 "$BASE_2000" baseline
}

train_gpu3() {
    train_resume_40k 3 "$SOURCE_INTER" "$INTER_1000"
    train_seed2000 3 "$INTER_2000" interaction
}

status TRAIN_ALL
train_gpu2 & train_p0=$!
train_gpu3 & train_p1=$!
wait "$train_p0"
wait "$train_p1"
status TRAIN_COMPLETE

official_checkpoint() {
    local seed=$1 kind=$2 step=$3
    if [[ "$seed" == 1000 && "$step" == 040000 ]]; then
        if [[ "$kind" == baseline ]]; then
            printf '%s/high-level/dp/logs/lerobot-train/%s/checkpoints/%s' "$REPO" "$SOURCE_BASE" "$step"
        else
            printf '%s/high-level/dp/logs/lerobot-train/%s/checkpoints/%s' "$REPO" "$SOURCE_INTER" "$step"
        fi
        return
    fi
    local job
    if [[ "$seed/$kind" == 1000/baseline ]]; then job=$BASE_1000
    elif [[ "$seed/$kind" == 1000/interaction ]]; then job=$INTER_1000
    elif [[ "$seed/$kind" == 2000/baseline ]]; then job=$BASE_2000
    else job=$INTER_2000
    fi
    printf '%s/high-level/dp/logs/lerobot-train/%s/checkpoints/%s' "$REPO" "$job" "$step"
}

experiment_job() {
    local seed=$1 kind=$2
    if [[ "$seed/$kind" == 1000/baseline ]]; then printf %s "$BASE_1000"
    elif [[ "$seed/$kind" == 1000/interaction ]]; then printf %s "$INTER_1000"
    elif [[ "$seed/$kind" == 2000/baseline ]]; then printf %s "$BASE_2000"
    else printf %s "$INTER_2000"
    fi
}

export_checkpoint() {
    local gpu=$1 seed=$2 kind=$3 step=$4
    local official job wrapped
    official=$(official_checkpoint "$seed" "$kind" "$step")
    job=$(experiment_job "$seed" "$kind")
    wrapped="$REPO/high-level/dp/logs/door-auto-wrapped/$job/$step/model_latest.pt"
    if [[ ! -s "$wrapped" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
            high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
            --official_checkpoint "$official" --root high-level/data/lerobot \
            --repo_id "$DATA" --out_dir "${wrapped%.pt}" \
            --manifest_name model_latest.pt --device cuda:0 --depth_only \
            >"$ROOT/export_${job}_${step}.log" 2>&1
    fi
    printf %s "$wrapped"
}

eval_checkpoint() {
    local gpu=$1 seed=$2 kind=$3 step=$4
    local job ckpt run
    job=$(experiment_job "$seed" "$kind")
    ckpt=$(export_checkpoint "$gpu" "$seed" "$kind" "$step")
    run="$REPO/high-level/logs/door-policy-success/${job}_step${step}_wc4_evalseed${EVAL_SEED}_open80_traverse0p8_64"
    if [[ -s "$run/summary.json" ]]; then
        return
    fi
    status "EVAL seed=$seed kind=$kind step=$step gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA" run --no-capture-output -n txc_vbc python \
        high-level/dp/eval/eval_door_policy_success.py \
        --checkpoint "$ckpt" --yaml high-level/data/cfg/b1z1_opendoor.yaml \
        --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 \
        --parallel_batches 4 --steps 2600 --pass_open_angle_deg 80 \
        --pull_traversal_distance_m 0.8 --pull_handle_unlock_angle_deg 40 \
        --base_seed "$EVAL_SEED" --headless --graphics_device_id 0 \
        --rl_device cuda:0 --sim_device cuda:0 --dp_action_horizon 25 --dp_fps 25 \
        --depth_only --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 \
        --depth_camera_rot_rand_deg 5.0 --run_root "$run" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --pull_vx_only_base --pull_panel_base_mode no_vy \
        --robot_pitch 0 --ikpush_robot_pitch_rand_min 0 \
        --ikpush_robot_pitch_rand_max 0 --ee_pose_frame robot_base_full \
        --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        >"$ROOT/eval_${job}_${step}.log" 2>&1
}

eval_curve() {
    local gpu=$1 seed=$2 kind=$3 first=$4
    local step formatted
    for ((step=first; step<=100000; step+=5000)); do
        printf -v formatted '%06d' "$step"
        eval_checkpoint "$gpu" "$seed" "$kind" "$formatted"
    done
}

status EVAL_SEED1000
eval_curve 2 1000 baseline 40000 & eval_p0=$!
eval_curve 3 1000 interaction 40000 & eval_p1=$!
wait "$eval_p0"
wait "$eval_p1"

status EVAL_SEED2000
eval_curve 2 2000 baseline 5000 & eval_p0=$!
eval_curve 3 2000 interaction 5000 & eval_p1=$!
wait "$eval_p0"
wait "$eval_p1"

status BUILD_REPORT
python3 - "$REPO" "$ROOT" "$EVAL_SEED" "$BASE_1000" "$INTER_1000" "$BASE_2000" "$INTER_2000" <<'PY'
import csv
import json
import sys
from pathlib import Path

repo, out, eval_seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
jobs = {
    (1000, "Baseline ACT"): sys.argv[4],
    (1000, "Plucker + Interaction"): sys.argv[5],
    (2000, "Baseline ACT"): sys.argv[6],
    (2000, "Plucker + Interaction"): sys.argv[7],
}
rows = []
for (seed, method), job in jobs.items():
    first = 40000 if seed == 1000 else 5000
    for step in range(first, 100001, 5000):
        run = repo / "high-level/logs/door-policy-success" / (
            f"{job}_step{step:06d}_wc4_evalseed{eval_seed}_open80_traverse0p8_64"
        )
        data = json.loads((run / "summary.json").read_text())
        fail = data.get("failure_categories") or {}
        rows.append({
            "train_seed": seed,
            "method": method,
            "step": step,
            "learning_rate": 1e-5 if step <= 40000 else 2e-6,
            "door_open": data["door_open_successes"],
            "traversal": data["traversal_successes"],
            "trials": data["trials"],
            "grasp_failure": fail.get("grasp_failure", 0),
            "insufficient_opening": fail.get("insufficient_opening", 0),
            "opened_not_traversed": fail.get("opened_not_traversed", 0),
            "unknown_failure": fail.get("unknown_failure", 0),
            "summary": str(run / "summary.json"),
        })

with (out / "checkpoint_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# WC4 pull checkpoint/LR/multi-seed experiment",
    "",
    "Evaluation: 64 trials, eval base seed 615455575, open >=80 deg, traversal >=0.8 m.",
    "",
    "| Train seed | Method | Best step | Door opening | Traversal | Grasp fail | Open insufficient | Opened/no traversal |",
    "|---:|---|---:|---:|---:|---:|---:|---:|",
]
for key in jobs:
    seed, method = key
    candidates = [r for r in rows if r["train_seed"] == seed and r["method"] == method]
    best = max(candidates, key=lambda r: (r["traversal"], r["door_open"], -r["step"]))
    lines.append(
        f"| {seed} | {method} | {best['step']//1000}K | "
        f"{best['door_open']}/64 = {100*best['door_open']/64:.2f}% | "
        f"{best['traversal']}/64 = {100*best['traversal']/64:.2f}% | "
        f"{best['grasp_failure']} | {best['insufficient_opening']} | "
        f"{best['opened_not_traversed']} |"
    )
lines += ["", "Best checkpoint is selected by traversal count, then door opening count, then earlier step."]
(out / "RESULTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

status COMPLETE
