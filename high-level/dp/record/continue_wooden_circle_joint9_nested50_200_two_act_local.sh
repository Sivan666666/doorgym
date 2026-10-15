#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/home/sivan/whole_body/visual_whole_body}
CONDA=${CONDA:-/home/sivan/miniconda3/bin/conda}
GPU=${GPU:-0}
TRAIN_SEED=${TRAIN_SEED:-1000}
LOG_ROOT=${LOG_ROOT:-$ROOT/high-level/logs/wooden_circle_joint9_nested50_200_two_act_20260812}

DATA_PREFIX=door_a2w_wooden_circle_joint9_push_norotate
RUN_PREFIX=leroact_a2w_wooden_circle_joint9

mkdir -p "$LOG_ROOT"
cd "$ROOT"

stamp() { date '+%F %T'; }
marker() { printf '%s/checkpoints/050000/pretrained_model/config.json' "$1"; }

wait_for_existing() {
    local output=$1
    local name=$2
    while [[ ! -f "$(marker "$output")" ]]; do
        if ! pgrep -f -- "lerobot_train.py.*--output_dir=$output" >/dev/null; then
            echo "[$(stamp)] ERROR: $name stopped without a 50K checkpoint" | tee -a "$LOG_ROOT/status.log"
            return 1
        fi
        echo "[$(stamp)] WAIT: $name is still training" | tee -a "$LOG_ROOT/status.log"
        sleep 300
    done
    echo "[$(stamp)] COMPLETE: $name" | tee -a "$LOG_ROOT/status.log"
}

train_one() {
    local size=$1
    local variant=$2
    local data_id="local/${DATA_PREFIX}_${size}"
    local data_root="$ROOT/high-level/data/lerobot/$data_id"
    local suffix
    local extra
    if [[ "$variant" == baseline ]]; then
        suffix=baseline
        extra=(--policy.plucker_conditioning=false)
    else
        suffix=plucker_fov55
        extra=(
            --policy.plucker_conditioning=true
            --policy.plucker_intrinsics_mode=legacy_shared_fov
            --policy.plucker_horizontal_fov_deg=55.0
        )
    fi

    local name="${RUN_PREFIX}_${size}_${suffix}_seed${TRAIN_SEED}_50k_chunk100_exec50_bs16"
    local output="$ROOT/high-level/dp/logs/lerobot-train/$name"
    if [[ -f "$(marker "$output")" ]]; then
        echo "[$(stamp)] REUSE: $name" | tee -a "$LOG_ROOT/status.log"
        return
    fi
    # LeRobot does not materialize output_dir until the first checkpoint, so
    # process detection must precede the filesystem check to avoid launching a
    # duplicate run during steps 0..9999.
    if pgrep -f -- "lerobot_train.py.*--output_dir=$output" >/dev/null; then
        wait_for_existing "$output" "$name"
        return
    fi
    if [[ -e "$output" ]]; then
        wait_for_existing "$output" "$name"
        return
    fi

    echo "[$(stamp)] START: $name" | tee -a "$LOG_ROOT/status.log"
    CUDA_VISIBLE_DEVICES="$GPU" "$CONDA" run --no-capture-output -n b1z1_lerobot python \
        high-level/lerobot/src/lerobot/scripts/lerobot_train.py \
        --dataset.repo_id="$data_id" \
        --dataset.root="$data_root" \
        --dataset.video_backend=torchcodec \
        --dataset.use_imagenet_stats=false \
        --policy.type=act \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.chunk_size=100 \
        --policy.n_action_steps=50 \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.optimizer_lr=1e-5 \
        --policy.use_action_loss_weight=false \
        --policy.interaction_state_conditioning=false \
        --policy.end_signal_prediction=false \
        "${extra[@]}" \
        --seed="$TRAIN_SEED" \
        --steps=50000 \
        --batch_size=16 \
        --num_workers=3 \
        --save_freq=10000 \
        --log_freq=100 \
        --keyframe_sampling_ratio=0 \
        --output_dir="$output" \
        --job_name="$name" \
        --wandb.enable=false \
        >"$LOG_ROOT/train_${name}.log" 2>&1
    [[ -f "$(marker "$output")" ]] || {
        echo "[$(stamp)] ERROR: missing 50K marker for $name" | tee -a "$LOG_ROOT/status.log"
        return 1
    }
    echo "[$(stamp)] COMPLETE: $name" | tee -a "$LOG_ROOT/status.log"
}

# The first run may already have been launched interactively.  This call waits
# for it rather than starting a duplicate process.
train_one 50 baseline
train_one 50 plucker
train_one 200 baseline
train_one 200 plucker

echo "[$(stamp)] PIPELINE_COMPLETE" | tee -a "$LOG_ROOT/status.log"
