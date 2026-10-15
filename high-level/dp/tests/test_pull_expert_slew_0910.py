"""Test the numpy-only helper without importing Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

source = Path(__file__).resolve().parents[2] / 'float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
            and n.name == 'limit_pull_expert_arm_command')
scope = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
limit = scope['limit_pull_expert_arm_command']


class SlewTests(unittest.TestCase):
    def test_legacy_exact(self):
        x=np.arange(7,dtype=np.float32); old=x.copy(); t={}
        limit(x,t,SimpleNamespace(),'panel_reposition',.02)
        np.testing.assert_array_equal(x,old); self.assertEqual(t,{})

    def test_prefix_exact(self):
        x=np.arange(7,dtype=np.float32); old=x.copy(); t={}
        limit(x,t,SimpleNamespace(pull_expert_arm_command_max_speed=1.5),'grasp',.02)
        np.testing.assert_array_equal(x,old)

    def test_limit_and_gripper_unchanged(self):
        a=SimpleNamespace(pull_expert_arm_command_max_speed=1.5); t={}; x=np.zeros(7)
        limit(x,t,a,'rotate_handle',.02)
        x[:]=1.; limit(x,t,a,'pull_door',.02)
        np.testing.assert_allclose(x[:6],.03); self.assertEqual(x[-1],1.)

    def test_policy_bypass(self):
        x=np.ones(7);old=x.copy();t={'pull_expert_previous_arm_command':np.zeros(6)}
        limit(x,t,SimpleNamespace(pull_expert_arm_command_max_speed=1.5),'panel_push',.02,external=True)
        np.testing.assert_array_equal(x,old)


if __name__=='__main__': unittest.main()
