"""Preflight invariant tests; never launch simulation or train in this module."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

DP=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(DP/'train'),str(DP/'eval')]
import run_pull_rgb_aug500_pi05_0914_ps1 as t
import run_pull_rgb_aug500_eval64_0914_ps1 as e


class PipelineTests(unittest.TestCase):
    def test_training_changes_only_dataset_and_output_identity(self):
        old=json.loads((t.OLD_OUT/'planned_train.command.json').read_text())['argv']
        new=t.training_command()
        allowed={'--dataset.root','--dataset.repo_id','--job_name','--output_dir'}
        self.assertEqual(len(old),len(new))
        changed=set()
        for a,b in zip(old,new):
            if a!=b:
                key=a.split('=',1)[0];self.assertIn(key,allowed);changed.add(key)
        self.assertEqual(changed,allowed)
        self.assertNotIn(str(t.OLD_OUT),str(t.TRAIN))

    def test_new_recording_seeds_do_not_overlap_train_or_eval(self):
        old=json.loads((t.OLD_OUT/'raw_manifest.json').read_text())
        seeds={t.NEW_SEED+lane*10000+attempt for lane in range(4) for attempt in range(1000)}
        self.assertEqual(len(seeds),4000)
        self.assertTrue(seeds.isdisjoint(x['seed'] for x in old))
        self.assertTrue(seeds.isdisjoint(range(715455575,715455639)))
        self.assertTrue(seeds.isdisjoint(range(815455575,815455639)))

    def test_recording_preserves_trajectory(self):
        with patch.object(t.r,'REPO',DP.parents[1]):
            for lane in range(4):
                cmd,preview=t.recording_command(lane,0,0)
                self.assertEqual(cmd[cmd.index('--num_envs')+1],'1')
                self.assertEqual(cmd[cmd.index('--steps')+1],'2600')
                self.assertEqual(cmd[cmd.index('--gripper_close_ratio')+1],'0.8')
                self.assertEqual(cmd[cmd.index('--rgb_visual_config')+1],t.OUT/'rgb_config.yaml')
                self.assertNotIn('--handle_unlock_ratio',cmd)
                self.assertEqual(cmd[cmd.index('--raw_root')+1],t.SHARDS/f'lane_{lane}')

    def test_evaluation_reuses_exact_scene_and_threshold_arguments(self):
        from eval_door_policy_success import parse_args
        for i in range(64):
            cmd=e.command(i) # also internally restores and compares the entire old argv
            with patch.object(sys,'argv',cmd[1:]):args=parse_args()
            self.assertEqual(args.total_trials,1);self.assertEqual(args.num_envs,1)
            self.assertEqual(args.base_seed,715455575+i)
            self.assertEqual(args.pull_traversal_distance_m,.5)
            self.assertEqual(args.pull_handle_unlock_angle_deg,25)
            self.assertEqual(args.pass_open_angle_deg,60)
            self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(e.WRAP.with_suffix('.pt')))
            self.assertEqual(cmd[cmd.index('--rgb_visual_seed')+1],str(815455575+i))


if __name__=='__main__':unittest.main(verbosity=2)
