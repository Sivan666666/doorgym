#!/usr/bin/env bash
set -euo pipefail
cd /home/sivan/whole_body/visual_whole_body

echo '===== Candidate 0 | geometric_misalignment | branch 641 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000000.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 253 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 4 | geometric_misalignment | branch 641 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000004.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 253 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 5 | geometric_misalignment | branch 646 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000005.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 253 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 10 | geometric_misalignment | branch 651 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000010.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 253 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 15 | geometric_misalignment | branch 671 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000015.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 253 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 24 | insufficient_interaction | branch 450 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000024.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 253 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 44 | insufficient_interaction | branch 387 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000044.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 210 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 64 | insufficient_interaction | branch 475 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000064.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 210 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 84 | insufficient_interaction | branch 457 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000084.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 210 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25

echo '===== Candidate 104 | insufficient_interaction | branch 422 ====='
conda run --no-capture-output -n b1z1 python \
  high-level/dp/record/replay_door_dp_raw_in_isaacgym.py \
  --raw_episode /home/sivan/whole_body/visual_whole_body/high-level/data/door_dp_raw/a2w_recovery_verified_wc4_baseline_rerun_v2_seed615455575/episode_000104.npz \
  --door_cfg high-level/data/cfg/b1z1_opendoor.yaml \
  --door_asset_name wc4 \
  --mode ikpush \
  --replay_mode state \
  --start_step 0 \
  --steps 210 \
  --stride 1 \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --graphics_device_id 0 \
  --real_time \
  --no_show_seg \
  --log_interval 25
