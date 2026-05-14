# Door RL Stage-wise Reward Debug Log

## 2026-05-12: Stage-wise reward implementation

目标：把 `high-level/door_rl` 从一次性开门 reward 改成可分阶段调试的 reward curriculum，先只验证 reach 是否能学出来。

已修改：

- `high-level/door_rl/train_door_rl.py`
  - 新增 `--reward_curriculum {reach,grasp,handle,open,pass,full}`，默认 `full` 保持旧行为。
  - 新增 `--debug_cycle_timesteps`，默认 `2000`；当 curriculum 不是 `full` 且没有显式改 `--timesteps` 时，自动用 2000。
  - 新增 `--stagewise_log_path`，默认写入 `high-level/logs/door-rl/<run_name>/stagewise_debug.jsonl`。
  - W&B/track data 现在会记录 `Phase / ...` 和 `Success / reach|grasp|phase`。

- `high-level/door_rl/door_asset_rl_env.py`
  - 新增 phase 状态：`phase_id`、`phase_hold_buf`、`reach_hold_buf`、`grasp_hold_buf`、`phase_success_buf`。
  - 新增 progress 状态：`prev_ee_to_handle`、`best_ee_to_handle`、`prev_handle_ratio`、`prev_open_ratio`、`prev_pass_distance`。
  - `reward_curriculum=reach` 时只启用 reach 相关奖励：
    - `10 * reach_dense`
    - `150 * reach_progress`
    - `20 * reach_close_bonus`
    - `50 * reach_success`
    - `2 * gripper_open`
    - action / tilt penalty
  - `reach_success` 定义为 `ee_to_handle < 0.08m` 连续 10 step。
  - `reward_curriculum=grasp/handle/open/pass` 已预留阶段 reward，便于 reach 稳定后继续调。
  - JSONL 调试记录会包含 `ee_to_handle_m`、`best_ee_to_handle_m`、`reach_progress`、`reach_success_rate`、`base_vx_cmd`、`base_yaw_cmd` 等。
  - 后续调试中追加了 `ee_goal_to_handle_m`、`ee_goal_clamped_rate`、`arm_base_to_handle_m`、`base_reach_*`、`reach_under_25cm/20cm/15cm/12cm/08cm_rate`，用于判断是不是手臂目标被工作空间 clamp 或底盘没有靠近。

后续补丁：

- `reward_curriculum != full` 时默认 `teacher_initial_log_std=-2.0`，避免初始随机 action 过大导致 EE target 饱和、越训越远。
- `reach` reward 追加 EE goal shaping：
  - `3.0 * reach_goal_dense`
  - `50.0 * reach_goal_progress`
  - `-5.0 * ee_goal_clamp_penalty`
  - `-0.02 * action_pos_l2`
- `reach` reward 追加轻量底盘可达性 shaping，因为第一阶段允许底盘参与：
  - `3.0 * base_reach_dense`
  - `40.0 * base_reach_progress`
  - 其中 `base_reach_dense` 只鼓励 arm base 进入 `0.8 * ee_max_radius` 的可达范围，不改变 reach success 定义。

建议第一轮命令：

```bash
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage teacher \
  --reward_curriculum reach \
  --mode pull \
  --num_envs 64 \
  --timesteps 2000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --headless \
  --run_name door_asset_reach_debug_v1 \
  --write_interval 24 \
  --save_interval 2000 \
  --wandb \
  --wandb_project door-rl
```

判断标准：

- `Door metrics / ee_to_handle_m` 是否明显下降。
- `Reward terms / reach_progress` 是否长期为正。
- `Success / reach` 是否出现非零。
- 如果 `base_vx_cmd` / `base_yaw_cmd` 很大但距离不降，说明底盘在干扰 reach，需要加大底盘动作惩罚或临时固定底盘。

## 2026-05-12: Reach debug cycles

所有训练轮次均为 `--reward_curriculum reach --mode pull --num_envs 64 --timesteps 2000 --write_interval 24`。由于本机 W&B 登录状态不确定，实际用 `stagewise_debug.jsonl` 和终端日志判断。

| Run | 关键改动 | 结果 |
| --- | --- | --- |
| `door_asset_reach_debug_v1` | 初始 stage-wise reach reward | `ee_to_handle_m` 初始约 0.95m，最后约 0.85m，但前/后 5 个 log 均值基本没变，`Success / reach = 0`。 |
| `door_asset_reach_debug_v2` | 加 EE target / clamp 诊断 | 发现 EE target 经常被 clamp，后期 `ee_to_handle_m` 反而到 1.35m 左右，`ee_goal_clamped_rate` 最高约 0.92。 |
| `door_asset_reach_debug_v3` | 加 `reach_goal_dense/progress` 和 clamp penalty | 仍然失败，action 前 3 维均值约 0.64，EE target 继续被推远；判断初始探索方差太大。 |
| `door_asset_reach_debug_v4` | `teacher_initial_log_std=-2.0` | 明显改善：前 20 个 log `ee_to_handle_m` 均值 0.950m，后 20 个 log 0.758m；`ee_goal_to_handle_m` 1.000m -> 0.659m；最小 logged `best_ee_to_handle_m` 0.394m；但 `Success / reach = 0`。 |
| `door_asset_reach_debug_v5` | 放宽 `--reach_success_dist 0.12` | 仍无成功；前 20 个 log `ee_to_handle_m` 0.944m，后 20 个 log 0.638m；最小 `best_ee_to_handle_m` 0.396m，说明整体更近但没有稳定进入 0.12m。 |
| `door_asset_reach_debug_v6` | 加 base reach shaping 和 base 距离日志 | 底盘确实参与：前 20 个 log `arm_base_to_handle_m` 1.142m，后 20 个 log 1.045m；`ee_goal_clamped_rate` 0.120 -> 0.001；`ee_to_handle_m` 0.919m -> 0.763m，最小 `best_ee_to_handle_m` 0.418m；`reach_close_bonus` 最高 0.0625，表示少量 env 曾进 0.15m，但没有连续 hold 成功。 |

当前判断：

- reach 奖励已经从“完全不学/越训越远”改善为“能让 EE goal 和底盘朝把手方向靠近”，但 reach 还没有稳定成功。
- 暂不进入 grasp。原因是 `reach_success_rate` 仍为 0，且 0.12m/0.08m 连续保持条件还没满足。
- 下一轮建议继续只调 reach，用新加的 `reach_under_*_rate` 判断是否已经大量进入 0.20m 或 0.15m；如果 `reach_under_20cm_rate` 有明显非零但 success 仍为 0，再考虑把 `reach_hold_steps` 从 10 临时降到 3 验证 pipeline。

最近验证：

```bash
python -m py_compile high-level/door_rl/train_door_rl.py high-level/door_rl/door_asset_rl_env.py
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py --stage teacher --reward_curriculum reach --reach_success_dist 0.12 --mode pull --num_envs 4 --timesteps 100 --rl_device cuda:0 --sim_device cuda:0 --headless --run_name door_asset_reach_smoke_v7 --write_interval 24 --save_interval 100
```

## 2026-05-13: Base approach heuristic reward

目标：解决 policy 输出 `vx` 不合理、狗没有先向门前进的问题。按照论文里的 reachability heuristic 思路，先让底盘进入手臂可达范围，再切换到 EE reach。

已修改：

- `high-level/door_rl/train_door_rl.py`
  - 新增 `--base_approach_dist`，默认 `0.50m`。
  - 新增 `--base_approach_min_vx`，默认 `0.30m/s`。
  - 新增 `--base_approach_max_vx`，默认 `0.55m/s`。
  - 新增 `--base_approach_vx_gain`，默认 `0.60`。
  - `teacher` 训练现在支持 `--teacher_ckpt_path` 继续训练，便于每轮 2k step 接着调。

- `high-level/door_rl/door_asset_rl_env.py`
  - 用 `arm_base_to_handle_m` 判断底盘是否还在 approach 阶段。
  - 当 `arm_base_to_handle_m > base_approach_dist` 时：
    - 根据距离生成正向 `target_vx`，越远目标速度越大，并限制在 `0.30~0.55m/s`。
    - reward 主要鼓励 `base_vx_cmd` 跟踪 `target_vx`、底盘到把手距离变小、夹爪保持打开。
  - 当 `arm_base_to_handle_m <= base_approach_dist` 时：
    - `target_vx = 0`，奖励底盘停住。
    - 切换回 EE reach reward，鼓励末端靠近把手。
  - stagewise JSONL 追加：
    - `base_approach_active_rate`
    - `base_vx_target`
    - `base_vx_tracking`
    - `base_vx_stop`
    - `base_yaw_stop`
    - `base_under_70cm_rate`
    - `base_under_60cm_rate`
    - `base_under_50cm_rate`

验证：

```bash
python -m py_compile high-level/door_rl/train_door_rl.py high-level/door_rl/door_asset_rl_env.py
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py --stage teacher --reward_curriculum reach --reach_success_dist 0.12 --mode pull --num_envs 4 --timesteps 100 --rl_device cuda:0 --sim_device cuda:0 --headless --run_name door_asset_reach_base_smoke_v1 --write_interval 24 --save_interval 100 --no_print_high_level_commands
```

2k 调试结果：

| Run | 命令差异 | 结果 |
| --- | --- | --- |
| `door_asset_reach_base_v1` | `base_approach_dist=0.50`，从头训 2k | `base_vx_cmd` 从约 `-0.019` 变到 `+0.474`，`base_vx_tracking` 从 `0.063` 到 `0.636`。底盘能明显往前走，`arm_base_to_handle_m` 最小到 `0.344m`，`best_arm_base_to_handle_m` 最小到 `0.259m`。EE 最小 `best_ee_to_handle_m=0.326m`，`reach_under_20cm_rate` 最高 `0.078`，但 `reach_success_rate=0`。 |
| `door_asset_reach_base_v2` | `base_approach_dist=0.70`，从头训 2k | 比 v1 差。最后 `base_vx_cmd=-0.230`，`base_vx_tracking=0.022`，EE 最小 `best_ee_to_handle_m=0.389m`，`reach_under_20cm_rate` 最高 `0.047`，`reach_success_rate=0`。不建议继续用 `0.70m` 阈值。 |
| `door_asset_reach_base_v1_cont1` | 从 `door_asset_reach_base_v1/agent_2000.pt` 继续训 2k | 当前最好。`base_vx_cmd` 最后约 `+0.546`，`base_vx_tracking` 最后 `0.985`。EE 接近率提升：`reach_under_25cm_rate` 最高 `0.297`，`reach_under_20cm_rate` 最高 `0.219`，`reach_under_15cm_rate` 最高 `0.109`，第一次出现非零 `reach_success_rate=0.0156`。 |

当前判断：

- 论文式的底盘启发式 reward 有效：policy 已经开始学到“先给正向 `vx` 靠近门”。
- `base_approach_dist=0.50m` 比 `0.70m` 更好。
- 当前还没有稳定抓住把手；reach 刚开始出现非零成功率，建议继续只调 reach，不急着进入 grasp。
- 当前最好 checkpoint：

```text
high-level/logs/door-rl/door_asset_reach_base_v1_cont1/teacher/checkpoints/agent_2000.pt
```

下一轮建议从当前最好 checkpoint 继续 2k：

```bash
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage teacher \
  --reward_curriculum reach \
  --reach_success_dist 0.12 \
  --mode pull \
  --num_envs 64 \
  --timesteps 2000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --headless \
  --run_name door_asset_reach_base_v1_cont2 \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_reach_base_v1_cont1/teacher/checkpoints/agent_2000.pt \
  --write_interval 24 \
  --save_interval 2000 \
  --no_print_high_level_commands
```

## 2026-05-13: Straight-to-door approach 调试

目标：解决 play 时底盘虽然有 `vx`，但不是正对门走的问题。当前 reach curriculum 做成两段：

- `arm_base_to_handle_m > 0.50m`：只走底盘，实际 command 强制 `vx=heuristic target_vx`、`yaw=0`、`vy=0`，机械臂目标冻结在当前 EE pose，夹爪保持打开。
- `arm_base_to_handle_m <= 0.50m`：底盘实际 command 强制 `vx=0,yaw=0`，切回 EE reach，让末端靠近把手。

已修改：

- `high-level/door_rl/door_asset_rl_env.py`
  - reach 阶段实际执行时 yaw 固定为 0，防止狗斜着绕门。
  - approach 阶段机械臂冻结，避免一边走一边乱动末端。
  - 进入 0.50m 内后底盘停止，只允许 EE reach。
  - 新增 raw policy action shaping：
    - `policy_base_vx_tracking`
    - `policy_base_vx_signed`
    - `policy_base_yaw_stop`
    - `ee_pose_action_stop`
  - 新增 EE reach action shaping：
    - `ee_pos_action_tracking`
    - `ee_pos_action_alignment`
  - stagewise log 新增对应字段。

- `high-level/door_rl/train_door_rl.py`
  - `--reward_curriculum reach` 默认 `init_yaw_noise=0.0`，让第一阶段从正对门的设置开始学。
  - 新增 `--base_heading_sigma`。

验证：

```bash
python -m py_compile high-level/door_rl/train_door_rl.py high-level/door_rl/door_asset_rl_env.py
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py --stage teacher --reward_curriculum reach --reach_success_dist 0.12 --mode pull --num_envs 4 --timesteps 100 --rl_device cuda:0 --sim_device cuda:0 --headless --run_name door_asset_reach_ee_action_smoke_v1 --write_interval 24 --save_interval 100 --no_print_high_level_commands
```

2k 调试结果：

| Run | 改动 | 关键结果 |
| --- | --- | --- |
| `door_asset_reach_straight_v1` | 只加 straight/heading reward，不改 action 执行 | 失败。`base_yaw_cmd` 后 20 个 log 约 `0.174`，没有学会直走；`reach_success_rate=0`。 |
| `door_asset_reach_straight_mask_v1` | 强制 yaw=0、approach 时机械臂冻结，但 `vx` 仍用 policy raw action | 失败。raw `vx` 是反号，实际 `base_vx_cmd` 约 `-0.547`，狗直线远离门，`arm_base_to_handle_m` 后段到 `10m+`。 |
| `door_asset_reach_scripted_vx_v1` | approach 阶段实际 `vx` 改为 heuristic 正向速度，yaw=0，机械臂冻结 | 有效靠近门。`base_vx_cmd` 约 `0.29~0.31`，`base_under_50cm_rate` 最高 `0.281`，但进入 0.50m 后把 base 控制还给 policy，会再次离开。 |
| `door_asset_reach_scripted_vx_stop_v1` | 进入 0.50m 后实际 `vx=0,yaw=0`，只操作 EE | 修正了“进门前乱跑/倒退”。`base_yaw_cmd=0`，`base_vx_tracking≈0.987`，`base_under_50cm_rate` 最高 `0.438`；`ee_to_handle_m` 后 20 个 log `0.482m`，但 `reach_success_rate=0`。 |
| `door_asset_reach_straight_policy_vx_v1` | 加强 raw policy 的 `vx/yaw/EE-still` action shaping | raw policy 有改善：`policy_base_vx_cmd` 后 20 个 log 从 `-0.480` 改到 `-0.370`，raw yaw 从约 `0.124` 到 `0.026`；`reach_under_20cm_rate` 最高 `0.047`，但成功率仍 0。 |
| `door_asset_reach_ee_action_v1` | 加 EE delta 朝把手方向的 action shaping | 当前本轮最好。raw `policy_base_vx_cmd` 最后一条变为 `+0.053`，`policy_base_vx_tracking` 最高 `0.314`；`reach_under_20cm_rate` 最高 `0.0625`，`reach_success_rate` 最高 `0.0156`，后 20 个 log 均值 `0.00078`。 |

当前判断：

- “正对门直走”这部分已经在执行层被修正：reach debug 下实际 `yaw=0`，`vy=0`，只有 `vx`。
- 单纯 reward-only 不够快；必须保留论文式 heuristic base command，否则 2k 内 raw policy 的 `vx` 仍可能反号。

## 2026-05-13: Grasp / Open stage 调试

目标：在已经学会靠近门并停下后，继续学习靠近把手、闭合夹爪、拉开门。底盘在抓取/开门阶段实际命令保持 `vx=0,yaw=0`。

代码修改：

- `high-level/door_rl/door_asset_rl_env.py`
  - `grasp_entry_dist` 默认从 `0.10m` 放宽到 `0.16m`，`grasp_success_dist` 默认从 `0.08m` 放宽到 `0.12m`。
  - 参考 IsaacLab drawer reward 增加/强化：
    - `approach_ee_handle`
    - `align_ee_handle`
    - `approach_gripper_handle`
    - `grasp_around_handle`
    - `grasp_handle`
    - `multi_stage_open`
  - `approach_gripper_handle` 改成连续距离奖励，不再被 `grasp_around_handle` 硬门控。
  - 抓取成功不再强制要求 `grasp_around_handle > 0.5`，先以 `ee_to_handle < grasp_success_dist` 且夹爪闭合为主。
  - 在 `grasp/handle/open` curriculum 中，进入抓取距离后实际 `external_gripper_target` assist 到闭爪，先打通“抓住/拉开”的链路；raw policy 的 `gripper_close_action` 仍记录。
  - `open` 成功改用 `abs(signed_angle)`，因为 pull 模式日志里开门方向为负角度，旧逻辑只认正角度会导致 pull 永远拿不到 success。
  - `open_delta` 改用绝对开门比例 `door_open_ratio` 的 progress。
  - open 阶段新增 scripted pull direction action shaping：
    - `open_pull_action_tracking`
    - `open_pull_action_alignment`
  - stagewise JSONL 新增：
    - `finger_mid_to_handle_m`
    - `finger_close_bonus_rate`
    - `gripper_close_action`
    - `open_pull_action_tracking`
    - `open_pull_action_alignment`

- `high-level/door_rl/train_door_rl.py`
  - 同步把 `--grasp_entry_dist` 默认值改为 `0.16`。
  - 同步把 `--grasp_success_dist` 默认值改为 `0.12`。

验证：

```bash
python -m py_compile high-level/door_rl/train_door_rl.py high-level/door_rl/door_asset_rl_env.py
```

2k 调试结果：

| Run | 命令差异 | 结果 |
| --- | --- | --- |
| `door_asset_grasp_open_debug_v2` | `reward_curriculum=open`，未加 gripper assist | 底盘能停住，但 `gripper_closed` 最大只有 `0.061`，`grasp_success_rate=0`，`open_bonus_rate=0`；说明主要卡在进入抓取后闭爪信号太弱。 |
| `door_asset_grasp_debug_v3` | 抓取奖励加强，`approach_gripper_handle` 改连续奖励，grasp success 不再要求 `grasp_around` | reach 有改善：`best_ee_to_handle_m=0.149m`，`reach_under_12cm_rate` 最高 `0.141`；但 `gripper_closed` 最大仅 `0.074`，抓取成功仍为 0。 |
| `door_asset_grasp_debug_v4` | 加 gripper close assist，进入抓取距离后实际闭爪 | raw policy 闭爪意图明显变强：`gripper_close_action` 后期均值约 `0.79`；但 reach 变差，`best_ee_to_handle_m=0.270m`，`grasp_success_rate=0`。说明单独 grasp 训练会被闭爪/接触扰动，不能只看本轮最终均值。 |
| `door_asset_grasp_open_debug_v3` | 从 `grasp_debug_v4` 继续，`reward_curriculum=open`，`grasp_entry_dist=0.22`，`grasp_success_dist=0.18`，`grasp_hold_steps=3` | 明显进展：`open_stage_rate` 最高 `0.453`，`gripper_closed` 最高 `0.578`，`reach_under_12cm_rate` 最高 `0.266`，门角最多约 `3.29deg`，但 `open_bonus_rate=0`。 |
| `door_asset_grasp_open_debug_v4` | 加 open 阶段 pull-direction action shaping，从 `open_debug_v3` 继续 | 首次出现开门成功：`open_bonus_rate` 最高 `0.015625`，表示 64 个 env 中至少 1 个达到 `20deg` 开门阈值；`open_stage_rate` 最高 `0.641`，`gripper_closed` 最高 `0.703`，`reach_under_20cm_rate` 最高 `0.688`，`handle_open_ratio` 最高 `0.052`。仍不稳定，最后一段均值没有稳定保持成功。 |

当前最好 checkpoint：

```text
high-level/logs/door-rl/door_asset_grasp_open_debug_v4/teacher/checkpoints/agent_2000.pt
```

当前判断：

- 阶段链路已经打通：approach -> stop -> reach/grasp -> open stage -> 至少一个 env 达到 20 度开门成功。
- 还不是稳定策略。`open_bonus_rate` 最高只有 `0.015625`，需要继续训练或进一步加强 open 阶段“保持贴近 + 沿 pull_dir 拉”的奖励。
- `grasp_success_rate` 仍为 0 是因为 open curriculum 目前没有更新 `grasp_success_buf`，不能用它判断 open 阶段抓取是否触发；应看 `open_stage_rate`、`gripper_closed`、`grasp_handle` 和 `open_bonus_rate`。

## 2026-05-13: Disable action assist by default

用户要求：不要在执行层写死底盘前进/停止/闭爪，应该全部由 high-level RL policy 输出。

已修改：

- `high-level/door_rl/door_asset_rl_env.py`
  - 默认 `stagewise_action_assist=False`。
  - 默认执行层不再强制：
    - approach 阶段 `vx=heuristic target_vx`
    - stop 阶段 `vx=0`
    - `yaw=0`
    - approach 阶段冻结机械臂
    - 抓取距离内强制闭爪
  - 默认实际执行命令现在直接来自 policy：
    - `commands[:, 0] = actions[:, 7] * max_vx`
    - `commands[:, 2] = actions[:, 8] * max_yaw`
    - `external_gripper_target` 来自 `actions[:, 6]`
  - 原来的 action assist 只保留为显式调试开关。

- `high-level/door_rl/train_door_rl.py`
  - 新增 `--stagewise_action_assist`，只有手动加这个参数时才启用旧的 scripted action override。

验证：

```bash
python -m py_compile high-level/door_rl/train_door_rl.py high-level/door_rl/door_asset_rl_env.py
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py --stage teacher --reward_curriculum reach --reach_success_dist 0.12 --mode pull --num_envs 4 --timesteps 100 --rl_device cuda:0 --sim_device cuda:0 --headless --run_name door_asset_no_action_assist_smoke_v1 --write_interval 24 --save_interval 100 --no_print_high_level_commands
```

smoke log 确认默认无 assist：

- `base_vx_cmd == policy_base_vx_cmd`
- `base_yaw_cmd == policy_base_yaw_cmd`
- `gripper_closed == gripper_close_action`

因此从这一版开始，不加 `--stagewise_action_assist` 的训练/评估都是 policy 直接控制 high-level command。
- raw policy 正在被拉正：`policy_base_vx_cmd` 从之前稳定负值，最新 run 最后一条已经到正值 `+0.053`。
- reach 还没有稳定成功：虽然最新 run 出现非零 `reach_success_rate`，但 EE action tracking 仍很低，说明接下来应核对 EE delta 坐标系/低层 EE target 跟踪，而不是继续盲目堆 reward。

当前用于 play 的本轮最好 checkpoint：

```text
high-level/logs/door-rl/door_asset_reach_ee_action_v1/teacher/checkpoints/agent_2000.pt
```

推荐 play：

```bash
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage eval_teacher \
  --reward_curriculum reach \
  --reach_success_dist 0.12 \
  --mode pull \
  --num_envs 1 \
  --timesteps 2000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_reach_ee_action_v1/teacher/checkpoints/agent_2000.pt \
  --print_high_level_commands \
  --print_high_level_command_interval 20
```

推荐继续训练：

```bash
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage teacher \
  --reward_curriculum reach \
  --reach_success_dist 0.12 \
  --mode pull \
  --num_envs 64 \
  --timesteps 2000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --headless \
  --run_name door_asset_reach_ee_action_v2 \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_reach_ee_action_v1/teacher/checkpoints/agent_2000.pt \
  --write_interval 24 \
  --save_interval 2000 \
  --no_print_high_level_commands
```

## 2026-05-13: Stop latch / stop hold 修正

问题：用户用 `eval_teacher --reward_curriculum reach` play 时看到狗会一直往前走，不会稳定停在门前。

原因：

- 原先 stop 条件是瞬时判断 `arm_base_to_handle_m <= 0.50m`，如果低层惯性或身体漂移让距离再次变大，approach 会重新激活。
- 另外 `--print_high_level_commands` 之前打印的是 policy raw action 缩放后的 command，不是 env 经过 heuristic/latch 后真实发给 low-level 的 command，所以终端里会看到 `base_vx_cmd=0.55`，但真实 env command 可能已经是 0 或负向刹车。

已修改：

- `high-level/door_rl/door_asset_rl_env.py`
  - 新增 `base_stop_latch`：一旦 `arm_base_to_handle_m <= --base_stop_dist`，就锁住 stop，直到 reset。
  - 新增参数：
    - `--base_stop_dist`，默认 `0.60m`
    - `--base_stop_hold_gain`，默认 `1.0`
    - `--base_stop_hold_max_vx`，默认 `0.20m/s`
  - latch 后不再简单给 `vx=0`，而是用小的 stop-hold controller：
    - 太近/冲过头时给负 `vx` 刹车。
    - 太远时才给很小正 `vx`。
  - stagewise log 新增：
    - `base_stop_latched_rate`
    - `base_stop_hold_vx`
    - `base_under_stop_dist_rate`

- `high-level/door_rl/train_door_rl.py`
  - `--print_high_level_commands` 现在在 `env.step()` 后打印。
  - 输出中区分：
    - `scaled_policy_command`: policy raw action 按 scale 换算的命令。
    - `actual_command`: 实际发给 low-level 的 `env.commands`。
    - `metrics`: `base_stop_latched`、`base_approach_active`、`arm_base_to_handle_m`、`ee_to_handle_m`。

验证：

```bash
python -m py_compile high-level/door_rl/train_door_rl.py high-level/door_rl/door_asset_rl_env.py
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py --stage teacher --reward_curriculum reach --reach_success_dist 0.12 --mode pull --num_envs 4 --timesteps 100 --rl_device cuda:0 --sim_device cuda:0 --headless --run_name door_asset_reach_stop_latch_smoke_v1 --write_interval 24 --save_interval 100 --no_print_high_level_commands
```

Smoke 结果：

- step 24: `base_stop_latched_rate=0.0`，`base_vx_cmd=0.55`
- step 48: `base_stop_latched_rate=0.0`，`base_vx_cmd=0.418`
- step 72: `base_stop_latched_rate=1.0`，`base_vx_cmd=0.0`
- step 96: `base_stop_latched_rate=1.0`，`base_vx_cmd=0.0`

加入 stop-hold 后 eval 结果：

| Run | Checkpoint | 关键结果 |
| --- | --- | --- |
| `door_asset_reach_stop_hold_eval_check_v1` | `door_asset_reach_ee_action_v1/agent_2000.pt` | step 72 后 `base_stop_latched=1`，`base_approach_active=0`；真实 `actual_command.base_vx_cmd` 约 `-0.20 -> -0.10`，狗不再继续向前冲。 |
| `door_asset_reach_stop_hold_eval_check_v2` | `door_asset_reach_stop_hold_v1/agent_2000.pt` | step 72 后稳定 latch；step 120/180 的真实 `actual_command.base_vx_cmd` 分别约 `-0.104/-0.111`，`arm_base_to_handle_m` 稳在约 `0.64m`，EE 最好到约 `0.21m`，但 reach 仍未稳定成功。 |

2k 训练结果：

| Run | 结果 |
| --- | --- |
| `door_asset_reach_stop_hold_v1` | stop 行为已稳定：后 20 个 log `base_stop_latched_rate=1.0`，`base_vx_cmd≈-0.087`，`base_vx_tracking≈0.997`。但 reach 变弱，后 20 个 log `ee_to_handle_m≈0.898m`，`reach_success_rate=0`。当前这轮主要解决“不会停”的问题，不代表 reach 已经学好。 |

当前推荐 play：

```bash
conda run --no-capture-output -n b1z1 python -u high-level/door_rl/train_door_rl.py \
  --stage eval_teacher \
  --reward_curriculum reach \
  --reach_success_dist 0.12 \
  --mode pull \
  --num_envs 1 \
  --timesteps 2000 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --teacher_ckpt_path high-level/logs/door-rl/door_asset_reach_stop_hold_v1/teacher/checkpoints/agent_2000.pt \
  --print_high_level_commands \
  --print_high_level_command_interval 20
```

判断 play 时是否真的停住，请看打印里的：

- `actual_command.base_vx_cmd`：真实底层 command。
- `metrics.base_stop_latched`：为 `1.0` 后代表已进入 stop-hold。
- `scaled_policy_command.base_vx_cmd` 只是 raw policy 缩放值，不代表真实执行命令。
