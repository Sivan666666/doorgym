#!/usr/bin/env bash
set -euo pipefail
cd /home/sivan/whole_body/visual_whole_body

echo '===== Candidate 0 | geometric_misalignment | branch 641 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000000.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 20 | insufficient_interaction | branch 450 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000019.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 40 | insufficient_interaction | branch 387 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000035.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 60 | insufficient_interaction | branch 475 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000051.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 80 | insufficient_interaction | branch 457 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000067.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 102 | insufficient_interaction | branch 422 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000084.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 1 | geometric_misalignment | branch 641 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000001.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 22 | insufficient_interaction | branch 450 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000020.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 42 | insufficient_interaction | branch 387 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000036.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 62 | insufficient_interaction | branch 475 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_locked_base_grasphold0_v5_seed615455575/episode_000052.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 280 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25
