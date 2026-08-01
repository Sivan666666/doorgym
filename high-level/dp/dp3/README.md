# A2W Front-Depth DP3

This integration uses only `front_masked_depth`. It converts each 480×640
front depth frame into 1024 XYZ points in `robot_base`, trains the official
full DP3 model, and packages its EMA checkpoint for the existing Door
play/eval interface.

## 1. Environment

```bash
cd /home/sivan/whole_body/3D-Diffusion-Policy
bash scripts/setup_a2w_dp3_env.sh
```

The script creates `dp3` (Python 3.8, PyTorch 2.1.2 CUDA 12.1), installs the
official package editable, and runs a CUDA forward/backward + EMA + reload
smoke test. `pytorch3d` is not required by the PointNet DP3 policy.

## 2. Convert the 250 raw episodes

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n dp3 python \
  high-level/dp/dp3/convert_door_raw_to_dp3.py \
  --raw_root high-level/data/door_dp_raw/a2w_5door_robotbasefull_contactcheck_250 \
  --output /home/sivan/whole_body/3D-Diffusion-Policy/3D-Diffusion-Policy/data/a2w_5door_front_xyz_250.zarr \
  --num_points 1024 \
  --point_frame robot_base \
  --batch_frames 64 \
  --empty_depth_policy previous \
  --debug_dir high-level/dp/result/dp3/a2w_5door_front_xyz_250_debug \
  --debug_per_door 3 \
  --overwrite
```

`--empty_depth_policy error` remains the strict default. The supplied raw
dataset contains a small number of all-zero front frames, so the completed
dataset uses the explicitly recorded `previous` fallback. Online inference
uses the same previous-cloud rule.

The completed conversion contains 14 recorded fallback frames. Its global XYZ
range is `[0.337,-0.924,0.217]` to `[1.893,0.985,1.740]` metres. The debug
directory contains three depth PNGs, PLY files and interactive HTML viewers for
each of the five doors.

Output schema:

```text
data/point_cloud  float32 [125000,1024,3]
data/state        float32 [125000,10]
data/action       float32 [125000,10]
meta/episode_ends int64   [250]
```

## 3. Train full DP3 for 50K optimizer steps

```bash
cd /home/sivan/whole_body/3D-Diffusion-Policy

bash scripts/train_a2w_door_dp3.sh \
  0 \
  a2w_5door_front_dp3_full_50k \
  data/a2w_5door_front_xyz_250.zarr
```

The A2W config is `a2w_dp3.yaml`: prediction horizon 32, observation steps 2,
action steps 16, batch 64, AdamW 1e-4, EMA, DDIM 10 inference steps. Exact
global-step checkpoints are saved at 10K, 20K, 30K, 40K, and 50K.

## 4. Package a Door checkpoint

```bash
cd /home/sivan/whole_body/visual_whole_body

DP3_RUN=/home/sivan/whole_body/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/a2w_5door_front_dp3_full_50k
DOOR_RUN=$PWD/high-level/dp/logs/door-auto-wrapped/a2w_5door_front_dp3_full_50k

conda run --no-capture-output -n dp3 python \
  high-level/dp/dp3/export_dp3_to_door_checkpoint.py \
  --checkpoint "$DP3_RUN/checkpoints/step=050000.ckpt" \
  --zarr /home/sivan/whole_body/3D-Diffusion-Policy/3D-Diffusion-Policy/data/a2w_5door_front_xyz_250.zarr \
  --output_dir "$DOOR_RUN/050000" \
  --manifest "$DOOR_RUN/050000/model_latest.pt" \
  --overwrite
```

The packaged checkpoint contains the official payload (EMA, normalizer and
Hydra config) plus Door geometry metadata.

## 5. Isaac Gym play

```bash
cd /home/sivan/whole_body/visual_whole_body
export DOOR_DP3_CONDA_ENV=dp3
export DOOR_CKPT=$PWD/high-level/dp/logs/door-auto-wrapped/a2w_5door_front_dp3_full_50k/050000/model_latest.pt

conda run --no-capture-output -n b1z1 python \
  high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py \
  --num_envs 4 \
  --steps 1000 \
  --seed -1 \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_name wc4 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --enable_front_camera \
  --no_enable_wrist_camera \
  --camera_depth \
  --depth_only \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --dp_policy_checkpoint "$DOOR_CKPT" \
  --dp_control_all_envs \
  --dp_action_horizon 12 \
  --dp_inference_steps 10 \
  --dp_fps 25 \
  --dp3_draw_point_cloud \
  --dp3_point_cloud_env_id 0 \
  --no_enable_depth_noise \
  --no_enable_depth_gaussian_blur \
  --enable_depth_camera_randomization \
  --depth_camera_pos_rand_m 0.02 \
  --depth_camera_rot_rand_deg 5.0 \
  --robot_pitch 0.0 \
  --ikpush_robot_pitch_rand_min 0.0 \
  --ikpush_robot_pitch_rand_max 0.0 \
  --no_preview_trajectory_at_spawn \
  --no_draw_ik_target \
  --no_draw_camera_axes \
  --no_show_seg
```

Set `DOOR_DP3_PYTHON=/home/sivan/miniconda3/envs/dp3/bin/python` instead of
`DOOR_DP3_CONDA_ENV` if an explicit interpreter is preferred.

## 6. Standard WC4 evaluation

Run the command below three times with `HORIZON=8`, `12`, and `16`:

```bash
cd /home/sivan/whole_body/visual_whole_body
export DOOR_DP3_PYTHON=/home/sivan/miniconda3/envs/dp3/bin/python
export DOOR_CKPT=$PWD/high-level/dp/logs/door-auto-wrapped/a2w_5door_front_dp3_full_50k/050000/model_latest.pt
HORIZON=12

conda run --no-capture-output -n b1z1 python \
  high-level/dp/eval/eval_door_policy_success.py \
  --checkpoint "$DOOR_CKPT" \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --mode ikpush \
  --robot_body a2wz1 \
  --num_envs 16 \
  --total_trials 64 \
  --steps 1000 \
  --pass_open_angle_deg 80 \
  --success_metric abs \
  --headless \
  --graphics_device_id 0 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --depth_only \
  --dp_action_horizon "$HORIZON" \
  --dp_inference_steps 10 \
  --dp_fps 25 \
  --base_seed 615455575 \
  --no_enable_depth_noise \
  --no_enable_depth_gaussian_blur \
  --enable_depth_camera_randomization \
  --depth_camera_pos_rand_m 0.02 \
  --depth_camera_rot_rand_deg 5.0 \
  --run_root "high-level/logs/door-policy-success/wc4_dp3_50k_h${HORIZON}_seed615455575" \
  --stream_output \
  -- \
  --door_name wc4 \
  --camera_depth_clip_lower 0.2 \
  --camera_depth_clip_far 1.5 \
  --robot_pitch 0.0 \
  --ikpush_robot_pitch_rand_min 0.0 \
  --ikpush_robot_pitch_rand_max 0.0 \
  --no_enable_wrist_camera
```

The same evaluator/backend path was smoke-tested with four parallel Isaac Gym
environments and a dedicated DP3 subprocess.

## 7. Optional dual-view DP3

The original front-only path remains the default and existing checkpoints are
loaded through the unchanged `DP3Encoder`. The dual-view variant keeps front
and wrist clouds separate and uses two independently parameterized PointNet
encoders:

```text
front depth -> robot-base XYZ -> PointNet_front --+
                                                   +-> concat -> fusion MLP -> geometry latent
wrist depth -> robot-base XYZ -> PointNet_wrist --+
robot state -------------------------------------------> state latent
geometry latent + state latent -> DP3 diffusion policy
```

Convert both views (1024 XYZ points per view):

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n dp3 python \
  high-level/dp/dp3/convert_door_raw_to_dp3_dualview.py \
  --raw_root high-level/data/door_dp_raw/a2w_wc4_robotbasefull_contactcheck_200 \
  --output /home/sivan/whole_body/3D-Diffusion-Policy/3D-Diffusion-Policy/data/a2w_wc4_dual_xyz_200.zarr \
  --num_points 1024 \
  --batch_frames 64 \
  --empty_depth_policy previous \
  --debug_dir high-level/dp/result/dp3/a2w_wc4_dual_xyz_200_debug \
  --debug_per_door 3 \
  --overwrite
```

Train for 50K optimizer steps:

```bash
cd /home/sivan/whole_body/3D-Diffusion-Policy

bash scripts/train_a2w_door_dualview_dp3.sh \
  0 \
  a2w_wc4_dualview_dp3_full_50k \
  data/a2w_wc4_dual_xyz_200.zarr
```

Export uses the same `export_dp3_to_door_checkpoint.py`; it auto-detects the
dual-view Zarr format and records both camera intrinsics in the Door manifest.
At play/evaluation time both `--enable_front_camera` and
`--enable_wrist_camera` are required. Single-view checkpoints continue to
accept `--no_enable_wrist_camera`.

## 8. ACT 单/双相机点云输入

ACT 点云数据仍使用标准 LeRobotDataset，但不保存深度视频；每帧直接保存
`observation.point_cloud: float32[1024,3]`。双相机数据固定按照 Front 512 点在前、
Wrist 512 点在后，并统一位于逐帧真实 `robot_base` 坐标系。

以双相机 5-door 250 条数据为例：

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root high-level/data/door_dp_raw/a2w_5door_robotbasefull_contactcheck_250 \
  --root high-level/data/lerobot \
  --repo_id local/door_a2w_5door250_pcd_dual \
  --depth_only \
  --point_cloud_views front,wrist \
  --point_cloud_num_points 1024 \
  --point_cloud_candidate_rows 64 \
  --point_cloud_candidate_cols 64 \
  --point_cloud_workspace_min 0.20,-1.00,0.00 \
  --point_cloud_workspace_max 2.00,1.00,1.80 \
  --point_cloud_empty_depth_policy previous \
  --point_cloud_storage point_cloud_only \
  --num_workers 4
```

单 Front 或单 Wrist 只需将 `--point_cloud_views` 改为 `front` 或 `wrist`。单相机
转换只要求对应相机的深度、内参和逐帧 pose；双相机缺少任一路会直接报错。

推荐的 OBSBench-local ACT 训练命令：

```bash
cd /home/sivan/whole_body/visual_whole_body
export PYTHONPATH="$PWD/high-level/lerobot/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 accelerate launch --num-processes=1 --mixed_precision=no \
  "$(which lerobot-train)" \
  --dataset.root="$PWD/high-level/data/lerobot/local/door_a2w_5door250_pcd_dual" \
  --dataset.repo_id=local/door_a2w_5door250_pcd_dual \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --policy.point_cloud_conditioning=true \
  --policy.point_cloud_views=front,wrist \
  --policy.point_cloud_encoder_mode=obsbench_local \
  --policy.point_cloud_num_points=1024 \
  --policy.point_cloud_num_tokens=256 \
  --policy.point_cloud_knn_k=16 \
  --batch_size=16 \
  --steps=50000 \
  --num_workers=4 \
  --save_freq=10000 \
  --log_freq=10 \
  --job_name=leroact_a2w_5door250_pcd_dual_local_50k \
  --output_dir="$PWD/high-level/dp/logs/lerobot-train/leroact_a2w_5door250_pcd_dual_local_50k" \
  --wandb.enable=true \
  --wandb.project=door-act \
  --wandb.disable_artifact=true
```

`obsbench_local` 现在严格采用 OBSBench 默认的 post-sampling 顺序：先用五层
`1×1 Conv + BatchNorm + ReLU` PointNet 为全部输入点产生 512D 逐点特征，再执行
FPS、KNN、`Linear + BatchNorm + ReLU` 和邻域 max pooling，最后使用 OBSBench 的
metric-XYZ 三维正弦位置编码。由于 LeRobot 数据中保存的是定长稠密点集，这里用
数学等价的 dense `Conv1d(kernel_size=1)` 代替 OBSBench 的
`spconv.SubMConv3d(kernel_size=1)`，不需要额外安装 spconv。

早期实验中的简化逐点 MLP 实现保留为
`--policy.point_cloud_encoder_mode=obsbench_local_legacy`，只用于加载旧点云实验，
不再推荐用于新训练。`obsbench_post_pointnet` 是 `obsbench_local` 的显式别名。

将 `--policy.point_cloud_encoder_mode` 改成 `dp3_global` 即可训练单全局 token 的
基础消融。这个模式现在严格复用原始 DP3 的 `PointNetEncoderXYZ`：逐点执行
`3→64→128→256` 的 `Linear + LayerNorm + ReLU`，对点维做全局 max pooling，
再通过 `Linear(256,64) + LayerNorm` 得到 DP3 的 64D 点云特征。由于 ACT Transformer
要求视觉 token 的通道为 `dim_model`，其后只保留一个必要的 `Linear(64,dim_model)`
适配层；点云编码器本体与原始 DP3 对齐，后端仍是 ACT 而不是 Diffusion Policy。

早期 PointCloud ACT 的无 LayerNorm 简化全局编码器保留为
`--policy.point_cloud_encoder_mode=dp3_global_legacy`。加载本地旧 checkpoint 时会根据
`state dict` 自动切换到 legacy 模式，因此旧实验仍可严格复现。

图像 ACT 与点云 ACT 在 v1 中互斥；点云模式不能同时打开 Plücker、
camera gating 或 image handle latent，但可以组合 End Signal 和 Interaction State。

### OBSBench Local + 未来 100 步 Interaction State

`obsbench_local` 可以直接复用 ACT 的 decoder-chunk 交互状态头。ACT 的每个 action
query 同时产生一帧 10D motion action 和一组交互状态：

```text
ACT decoder tokens [B,100,dim_model]
  ├─ action_head                  → [B,100,10]
  └─ interaction_state_chunk_head → [B,100,3]
                                     contact
                                     handle_progress
                                     door_progress
```

交互状态不会拼入机器人 action，也不会改变 Door 控制器消费的 10D action。默认
`interaction_state_conditioning=false`，因此旧数据、旧图像/点云配置和旧 checkpoint
保持原行为。打开新功能并从旧 OBSBench Local checkpoint 微调时，只允许缺少新的
`interaction_state_chunk_head.*` 参数；加载完成、开始训练之前，motion action 与旧模型
逐元素完全一致。`interaction_state_probe_only=false` 时，三个辅助 loss 会通过 decoder
token 回传到 ACT decoder 和点云 encoder。

现有不带标签的点云 LeRobot 数据集需要从同一批 raw 重新转换，但不需要重新录制：

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/convert_door_raw_to_lerobot.py \
  --raw_root high-level/data/door_dp_raw/a2w_wc4_robotbasefull_contactcheck_200 \
  --root high-level/data/lerobot \
  --repo_id local/door_a2w_wc4_200_pcd_front_interaction_state \
  --depth_only \
  --point_cloud_views front \
  --point_cloud_num_points 1024 \
  --point_cloud_candidate_rows 64 \
  --point_cloud_candidate_cols 64 \
  --point_cloud_workspace_min 0.20,-1.00,0.00 \
  --point_cloud_workspace_max 2.00,1.00,1.80 \
  --point_cloud_empty_depth_policy previous \
  --point_cloud_storage point_cloud_only \
  --add_interaction_state \
  --interaction_contact_min_consecutive_frames 3 \
  --interaction_handle_unlock_angle_deg 40 \
  --interaction_door_goal_angle_deg 90 \
  --num_workers 4
```

训练时在 OBSBench Local 参数后增加：

```bash
  --policy.interaction_state_conditioning=true \
  --policy.interaction_state_prediction_mode=decoder_chunk \
  --policy.interaction_state_probe_only=false \
  --policy.interaction_contact_loss_weight=0.05 \
  --policy.interaction_handle_loss_weight=0.05 \
  --policy.interaction_door_loss_weight=0.05
```

数据工厂会给三个 target 使用和 action 相同的未来 100 帧时间索引；episode 末尾超出
范围的 target 与 action 一样由 padding mask 排除，不参与 interaction loss。

导出 Door checkpoint：

```bash
conda run --no-capture-output -n b1z1_lerobot python \
  high-level/dp/export_official_lerobot_act_to_door_checkpoint.py \
  --official_checkpoint high-level/dp/logs/lerobot-train/leroact_a2w_5door250_pcd_dual_local_50k/checkpoints/050000 \
  --root high-level/data/lerobot \
  --repo_id local/door_a2w_5door250_pcd_dual \
  --out_dir high-level/dp/logs/door-auto-wrapped/leroact_a2w_5door250_pcd_dual_local_50k/050000 \
  --depth_only \
  --action_horizon 25
```

Door manifest 会记录相机模式、点数、encoder、workspace、内参和坐标系。在线推理
复用与离线转换相同的反投影、裁剪与固定点数采样函数。Play 时可继续使用
`--dp3_draw_point_cloud`：双相机融合模式中 Front 为绿色，Wrist 为紫红色。
