import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

source=Path(__file__).resolve().parents[2]/'float_ik/isaacgym_float_ik_a2w_basearn_pull_door_parallel.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='audit_pull_unlock_threshold')
scope={'math':math}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
audit=scope['audit_pull_unlock_threshold']


class UnlockAuditTests(unittest.TestCase):
    def state(self,angle,ratio):
        return NS(index=0,door=NS(handle_unlock_threshold=math.radians(angle),handle_rest_angle=0.),
                  args=NS(handle_unlock_ratio=ratio))

    def test_physical_25(self):
        self.assertAlmostEqual(audit([self.state(25,25/45)],25)[0]['handle_unlock_angle_deg'],25)

    def test_classifier_change_alone_is_rejected(self):
        with self.assertRaises(ValueError): audit([self.state(40,40/45)],25)

    def test_40_unchanged(self):
        self.assertAlmostEqual(audit([self.state(40,40/45)],40)[0]['handle_unlock_angle_deg'],40)


if __name__=='__main__': unittest.main()
