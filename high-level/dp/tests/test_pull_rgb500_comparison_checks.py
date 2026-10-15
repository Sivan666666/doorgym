"""Reject incomplete/misconfigured rollouts before assigning video or ACT labels."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from run_pull_rgb500_act_comparison_0911_ps1 import checked_rollout


class RolloutChecks(unittest.TestCase):
    def check(self, records=2600, unlock=25, finite=True, code=0):
        with tempfile.TemporaryDirectory() as tmp:
            stem=Path(tmp)/'trial'
            stem.with_suffix('.log').write_text('[PullUnlockAudit] '+json.dumps(
                [{'handle_unlock_angle_deg':unlock}])+'\nDone after 2600 steps\nSIGSEGV\n')
            with stem.with_suffix('.jsonl').open('w') as f:
                for i in range(records):
                    f.write(json.dumps(dict(step=i,controlled_env_id=0,
                        door={'dof':[1.2,.6]},base={'xy':[.6,0]},
                        pull_task_geometry={'door_plane_xy':[0,0],'traversal_direction_xy':[1,0]},
                        state=[0.]*9,dp_action_raw=[0.]*8+([0.] if finite else [float('nan')])) )+'\n')
            return checked_rollout(stem,1,code)

    def test_valid_and_cleanup(self):
        for code in (0,1,139,-11):
            rows,_,_=self.check(code=code)
            self.assertTrue(rows[0]['success'])

    def test_reject_incomplete(self):
        with self.assertRaises(AssertionError): self.check(records=2599,code=1)

    def test_reject_wrong_physics(self):
        with self.assertRaises(AssertionError): self.check(unlock=40)

    def test_reject_nan(self):
        with self.assertRaises(AssertionError): self.check(finite=False)


if __name__=='__main__': unittest.main()
