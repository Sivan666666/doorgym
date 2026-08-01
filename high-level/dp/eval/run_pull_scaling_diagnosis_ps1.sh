#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/ps/workspace/txc/doorgym
CONDA=/home/ps/miniconda3/bin/conda
TRAIN="$REPO/high-level/lerobot/src/lerobot/scripts/lerobot_train.py"
EXPORT="$REPO/high-level/dp/export_official_lerobot_act_to_door_checkpoint.py"
EVAL="$REPO/high-level/dp/eval/eval_door_policy_success.py"
RESULT="$REPO/high-level/dp/result/pull_50_200_push_comparison_20260731"
PIPE="$REPO/high-level/logs/pull-data-pipeline/pull_scaling_diagnosis_20260731"

DATA200="$REPO/high-level/data/lerobot/local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_interaction_state"
REPO200=local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_interaction_state
DATA50="$REPO/high-level/data/lerobot/local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_prefix_50"
REPO50=local/door_a2w_wc4_pull_joint9_noresistance_noisy_rand_200_prefix_50

BASE200="$REPO/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_pull_joint9_noisy_rand200_baseline_seed1000_50k_chunk100_exec50_bs16_0731"
INT200="$REPO/high-level/dp/logs/lerobot-train/leroact_a2w_wc4_pull_joint9_noisy_rand200_plucker_fov55_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0731"
BASE50_JOB=leroact_a2w_wc4_pull_joint9_nested50_baseline_seed1000_50k_chunk100_exec50_bs16_0731
INT50_JOB=leroact_a2w_wc4_pull_joint9_nested50_plucker_interaction_decoderchunk_w005_seed1000_50k_chunk100_exec50_bs16_0731
BASE50="$REPO/high-level/dp/logs/lerobot-train/$BASE50_JOB"
INT50="$REPO/high-level/dp/logs/lerobot-train/$INT50_JOB"

mkdir -p "$PIPE" "$RESULT"
echo $$ >"$PIPE/supervisor.pid"

stage() {
    printf '%s %s\n' "$(date '+%F %T')" "$1" | tee "$PIPE/status.txt" >>"$PIPE/pipeline.log"
}
fail() {
    printf '%s FAILED %s\n' "$(date '+%F %T')" "$*" | tee "$PIPE/status.txt" >>"$PIPE/pipeline.log"
    exit 1
}
trap 'fail "line=$LINENO command=$BASH_COMMAND"' ERR

stage TRAIN_50_AND_RESUME_200

BASE200_CFG="$BASE200/checkpoints/050000/pretrained_model/train_config.json"
INT200_CFG="$INT200/checkpoints/050000/pretrained_model/train_config.json"
[[ -s "$BASE200_CFG" && -s "$INT200_CFG" ]] || fail "missing 200-episode 50K resume config"

CUDA_VISIBLE_DEVICES=0 "$CONDA" run --no-capture-output -n b1z1_lerobot python "$TRAIN" \
    --config_path="$BASE200_CFG" --resume=true --steps=200000 --save_freq=50000 \
    --log_freq=10 --wandb.enable=false \
    >"$PIPE/train_base200_resume.log" 2>&1 &
p_base200=$!

CUDA_VISIBLE_DEVICES=1 "$CONDA" run --no-capture-output -n b1z1_lerobot python "$TRAIN" \
    --config_path="$INT200_CFG" --resume=true --steps=200000 --save_freq=50000 \
    --log_freq=10 --wandb.enable=false \
    >"$PIPE/train_int200_resume.log" 2>&1 &
p_int200=$!

COMMON50=(
    "--dataset.root=$DATA50"
    "--dataset.repo_id=$REPO50"
    "--dataset.video_backend=torchcodec"
    "--dataset.use_imagenet_stats=false"
    "--policy.type=act"
    "--policy.device=cuda"
    "--policy.push_to_hub=false"
    "--policy.chunk_size=100"
    "--policy.n_action_steps=50"
    "--policy.vision_backbone=resnet18"
    "--policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1"
    "--policy.use_action_loss_weight=false"
    "--batch_size=16"
    "--steps=50000"
    "--seed=1000"
    "--num_workers=4"
    "--save_freq=50000"
    "--log_freq=10"
    "--keyframe_sampling_ratio=0.0"
    "--wandb.enable=false"
)

if [[ -e "$BASE50" || -e "$INT50" ]]; then
    fail "nested-50 output already exists"
fi

CUDA_VISIBLE_DEVICES=2 "$CONDA" run --no-capture-output -n b1z1_lerobot python "$TRAIN" \
    "${COMMON50[@]}" \
    --policy.plucker_conditioning=false \
    --policy.interaction_state_conditioning=false \
    "--job_name=$BASE50_JOB" "--output_dir=$BASE50" \
    >"$PIPE/train_base50_nested.log" 2>&1 &
p_base50=$!

CUDA_VISIBLE_DEVICES=3 "$CONDA" run --no-capture-output -n b1z1_lerobot python "$TRAIN" \
    "${COMMON50[@]}" \
    --policy.plucker_conditioning=true \
    --policy.plucker_horizontal_fov_deg=55.0 \
    --policy.interaction_state_conditioning=true \
    --policy.interaction_state_prediction_mode=decoder_chunk \
    --policy.interaction_state_probe_only=false \
    --policy.interaction_state_auxiliary_only=false \
    --policy.interaction_contact_loss_weight=0.05 \
    --policy.interaction_handle_loss_weight=0.05 \
    --policy.interaction_door_loss_weight=0.05 \
    "--job_name=$INT50_JOB" "--output_dir=$INT50" \
    >"$PIPE/train_int50_nested.log" 2>&1 &
p_int50=$!

printf 'base200=%s int200=%s base50=%s int50=%s\n' "$p_base200" "$p_int200" "$p_base50" "$p_int50" >>"$PIPE/pipeline.log"

rc=0
for pid in "$p_base200" "$p_int200" "$p_base50" "$p_int50"; do
    wait "$pid" || rc=1
done
[[ "$rc" -eq 0 ]] || fail "one or more training jobs failed"

for run in "$BASE50" "$INT50"; do
    [[ -s "$run/checkpoints/050000/pretrained_model/model.safetensors" ]] || fail "missing $run 50K"
done
for run in "$BASE200" "$INT200"; do
    for step in 100000 150000 200000; do
        [[ -s "$run/checkpoints/$step/pretrained_model/model.safetensors" ]] || fail "missing $run $step"
    done
done

stage EXPORT_ALL

export_one() {
    local official="$1" dataset="$2" repo_id="$3" name="$4" log="$5"
    local base="$REPO/high-level/dp/logs/door-auto-wrapped/$name"
    local out="$base/model_latest"
    rm -rf "$base"
    CUDA_VISIBLE_DEVICES=0 "$CONDA" run --no-capture-output -n b1z1_lerobot python "$EXPORT" \
        --official_checkpoint "$official" --root "$REPO/high-level/data/lerobot" \
        --repo_id "$repo_id" --out_dir "$out" --manifest_name model_latest.pt \
        --device cuda:0 --depth_only >"$PIPE/$log" 2>&1
    [[ -s "$base/model_latest.pt" ]] || return 1
}

export_one "$BASE50/checkpoints/050000" "$DATA50" "$REPO50" "${BASE50_JOB}/050000" export_base50.log
export_one "$INT50/checkpoints/050000" "$DATA50" "$REPO50" "${INT50_JOB}/050000" export_int50.log
for step in 100000 150000 200000; do
    export_one "$BASE200/checkpoints/$step" "$DATA200" "$REPO200" "$(basename "$BASE200")/$step" "export_base200_$step.log"
    export_one "$INT200/checkpoints/$step" "$DATA200" "$REPO200" "$(basename "$INT200")/$step" "export_int200_$step.log"
done

stage EVAL_WAVE_1

eval_one() {
    local checkpoint="$1" run_name="$2" gpu="$3" log="$4"
    local run_root="$REPO/high-level/logs/door-policy-success/$run_name"
    rm -rf "$run_root"
    "$CONDA" run --no-capture-output -n txc_vbc python "$EVAL" \
        --checkpoint "$checkpoint" \
        --yaml "$REPO/high-level/data/cfg/b1z1_opendoor.yaml" \
        --mode ikpull --robot_body a2wz1 --num_envs 16 --total_trials 64 --steps 1750 \
        --pass_open_angle_deg 60 --pull_traversal_distance_m 0.8 --base_seed 615455575 \
        --headless --graphics_device_id "$gpu" --rl_device "cuda:$gpu" --sim_device "cuda:$gpu" \
        --dp_action_horizon 25 --dp_fps 25 --depth_only \
        --no_enable_depth_noise --no_enable_depth_gaussian_blur \
        --enable_depth_camera_randomization --depth_camera_pos_rand_m 0.02 --depth_camera_rot_rand_deg 5.0 \
        --run_root "$run_root" -- \
        --door_name wc4 --wc4_disable_door_open_resistance --door_auto_open_force 0 \
        --robot_pitch 0.0 --ikpush_robot_pitch_rand_min 0.0 --ikpush_robot_pitch_rand_max 0.0 \
        --ee_pose_frame robot_base_full --camera_depth_clip_lower 0.2 --camera_depth_clip_far 1.5 \
        >"$PIPE/$log" 2>&1
}

WRAP="$REPO/high-level/dp/logs/door-auto-wrapped"
eval_one "$WRAP/$BASE50_JOB/050000/model_latest.pt" pull_nested50_base50k_seed615455575_open60_traverse0p8_64 0 eval_nested50_base.log & p0=$!
eval_one "$WRAP/$INT50_JOB/050000/model_latest.pt" pull_nested50_int50k_seed615455575_open60_traverse0p8_64 1 eval_nested50_int.log & p1=$!
eval_one "$WRAP/$(basename "$BASE200")/100000/model_latest.pt" pull_200_base100k_seed615455575_open60_traverse0p8_64 2 eval_200_base100.log & p2=$!
eval_one "$WRAP/$(basename "$INT200")/100000/model_latest.pt" pull_200_int100k_seed615455575_open60_traverse0p8_64 3 eval_200_int100.log & p3=$!
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid"; done

stage EVAL_WAVE_2
eval_one "$WRAP/$(basename "$BASE200")/150000/model_latest.pt" pull_200_base150k_seed615455575_open60_traverse0p8_64 0 eval_200_base150.log & p0=$!
eval_one "$WRAP/$(basename "$INT200")/150000/model_latest.pt" pull_200_int150k_seed615455575_open60_traverse0p8_64 1 eval_200_int150.log & p1=$!
eval_one "$WRAP/$(basename "$BASE200")/200000/model_latest.pt" pull_200_base200k_seed615455575_open60_traverse0p8_64 2 eval_200_base200.log & p2=$!
eval_one "$WRAP/$(basename "$INT200")/200000/model_latest.pt" pull_200_int200k_seed615455575_open60_traverse0p8_64 3 eval_200_int200.log & p3=$!
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid"; done

stage SUMMARIZE
"$CONDA" run --no-capture-output -n txc_vbc python - "$REPO" "$RESULT" <<'PY'
import json, sys
from pathlib import Path

repo, out = map(Path, sys.argv[1:])
specs = [
    ("Pull50Old", "Baseline", 50, 50000, "wc4_pull_baseline_joint9_50k_seed615455575_open60_traverse0p8_64_rerun2"),
    ("Pull50Old", "Plucker+Interaction", 50, 50000, "wc4_pull_plucker_interaction_joint9_50k_seed615455575_open60_traverse0p8_64_rerun2"),
    ("Pull50Nested", "Baseline", 50, 50000, "pull_nested50_base50k_seed615455575_open60_traverse0p8_64"),
    ("Pull50Nested", "Plucker+Interaction", 50, 50000, "pull_nested50_int50k_seed615455575_open60_traverse0p8_64"),
    ("Pull200", "Baseline", 200, 50000, "wc4_pull_joint9_rand200_baseline_seed615455575_open60_traverse0p8_64"),
    ("Pull200", "Plucker+Interaction", 200, 50000, "wc4_pull_joint9_rand200_plucker_interaction_seed615455575_open60_traverse0p8_64"),
]
for step in (100000, 150000, 200000):
    specs += [
        ("Pull200", "Baseline", 200, step, f"pull_200_base{step//1000}k_seed615455575_open60_traverse0p8_64"),
        ("Pull200", "Plucker+Interaction", 200, step, f"pull_200_int{step//1000}k_seed615455575_open60_traverse0p8_64"),
    ]
rows=[]
for dataset, model, episodes, step, run in specs:
    path=repo/'high-level/logs/door-policy-success'/run/'summary.json'
    data=json.loads(path.read_text())
    rows.append({
        'dataset':dataset,'model':model,'episodes':episodes,'train_steps':step,
        'sample_exposures':step*16,'effective_frame_epochs':step*16/(episodes*875),
        'trials':data['trials'],'door_open':data['door_open_successes'],
        'traversal':data['traversal_successes'],'joint_success':data['successes'],
        'door_open_rate':data['door_open_success_rate'],
        'traversal_rate':data['traversal_success_rate'],'joint_success_rate':data['success_rate'],
        'run_root':str(path.parent),
    })
(out/'training_eval_results.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Pull dataset scaling diagnosis','','| Dataset | Model | Episodes | Train steps | Effective epochs | Door open | Traversal | Joint success |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    n=r['trials']
    lines.append(f"| {r['dataset']} | {r['model']} | {r['episodes']} | {r['train_steps']//1000}K | {r['effective_frame_epochs']:.2f} | {r['door_open']}/{n} = {100*r['door_open_rate']:.2f}% | {r['traversal']}/{n} = {100*r['traversal_rate']:.2f}% | **{r['joint_success']}/{n} = {100*r['joint_success_rate']:.2f}%** |")
(out/'training_eval_results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY

stage COMPLETE
