# Door Asset PPO / Teacher-Student RL

这套脚本**不使用 `B1Z1OpenDoor`**。环境底层来自当前可运行的：

- `high-level/play_b1z1_walk_with_door_asset_camera.py`
- `high-level/play_b1z1_push_with_door_asset_camera.py`

新环境会加载同一批门资产、同一套 wrist/front camera、同一套 low-level locomanip policy，然后用 PPO 训练 high-level 9D action。

## Action

High-level action 是 9 维：

```text
delta_ee_x, delta_ee_y, delta_ee_z,
delta_roll, delta_pitch, delta_yaw,
gripper, vx, yaw
```

其中 `vx` 和 `yaw` 发给 low-level controller，EE delta 和 gripper 通过现有 IK/PD 目标下发给机械臂。

## Teacher PPO Smoke

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage teacher \
  --num_envs 4 \
  --timesteps 100 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --headless \
  --run_name door_asset_teacher_smoke \
  --save_interval 50
```

## Teacher PPO Full

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage teacher \
  --num_envs 128 \
  --timesteps 1000000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --headless \
  --run_name door_asset_teacher_v1 \
  --save_interval 500 \
  --write_interval 24 \
  --wandb \
  --wandb_project door-rl
```

Teacher checkpoint 默认在：

```text
high-level/logs/door-rl/<run_name>/teacher/checkpoints/
```

skrl 默认 checkpoint 文件名通常是 `agent_<step>.pt`，例如 `agent_500.pt`、`agent_1000000.pt`。

## Student DAgger Smoke

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage student \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_teacher_v1/teacher/checkpoints/agent_<step>.pt \
  --num_envs 2 \
  --timesteps 100 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --viewer \
  --run_name door_asset_student_smoke \
  --save_interval 50
```

## Student DAgger Full

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage student \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_teacher_v1/teacher/checkpoints/agent_<step>.pt \
  --num_envs 32 \
  --timesteps 300000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --viewer \
  --run_name door_asset_student_v1 \
  --save_interval 500 \
  --write_interval 24 \
  --wandb \
  --wandb_project door-rl
```

## W&B Metrics

加 `--wandb --wandb_project door-rl` 后，teacher 和 student 都会同步到 Weights & Biases。

Teacher PPO 会上传：

- `Reward / ...`：skrl 默认 instant / episode reward。
- `Loss / Policy loss`：PPO actor loss。
- `Loss / Value loss`：PPO critic loss。
- `Loss / Entropy loss`：PPO entropy loss。
- `Learning / Learning rate`：当前学习率。
- `Reward terms / ...`：开门环境的 reward 分项，比如 reach、handle、open、pass_align。
- `Door metrics / ...`：signed door angle、open ratio、handle ratio、EE 到把手距离等调试指标。
- `Success / open_80deg`、`Success / passed_door`：开门和过门成功率。

Student DAgger 会上传：

- `Reward / ...`：student rollout 中的环境 reward。
- `Loss / DAgger loss`：student 模仿 teacher action 的训练 loss。
- `Loss / Online teacher-student MSE`：当前 step student action 和 teacher action 的在线 MSE。
- `Reward terms / ...`、`Door metrics / ...`、`Success / ...`：和 teacher 一样的环境指标。
- `Student action / ...`、`Teacher action / ...`：9D high-level action 的均值和绝对均值。

脚本也会像 `train_multistate.py` 一样把关键源码和门配置文件作为 wandb files 保存，方便回看当次实验代码。

## Eval Teacher With Viewer

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage eval_teacher \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_teacher_v1/teacher/checkpoints/agent_<step>.pt \
  --num_envs 4 \
  --timesteps 2500 \
  --viewer \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

## Eval Student With Viewer

```bash
cd /home/sivan/whole_body/visual_whole_body

conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage eval_student \
  --student_ckpt_path high-level/logs/door-rl/door_asset_student_v1/student/checkpoints/agent_<step>.pt \
  --num_envs 4 \
  --timesteps 2500 \
  --viewer \
  --rl_device cuda:0 \
  --sim_device cuda:0
```

## Notes

- Teacher obs 包含门把手真值、door/handle DOF、开门方向等 privileged 信息。
- Student obs 只包含 proprio + wrist/front handle mask + wrist/front masked depth。
- 当前 low-level `BaseTask` 在 `headless=True` 时会把 `graphics_device_id` 设成 `-1`，Isaac Gym camera tensor 可能创建失败；所以 student 相机训练/检查建议用 `--viewer`。
- Push/Pull 在同一个环境里通过 `door_motion_sign = +1/-1` 采样。
- 如果电脑容易卡，先把 `--num_envs` 降到 4 到 16。
