import unittest
import numpy as np
from wc_blender_physx_common import score,wilson
from wc_blender_physx_hooks import integrate


class ProtocolTest(unittest.TestCase):
    def test_cadence(self):
        self.assertEqual(len(range(0,2600,2)),1300)
        self.assertEqual(len(range(0,1300,50)),26)

    def test_joint10_integration(self):
        a=np.array([.3,.02,.1,1,2,3,4,5,6,-.5])
        xy,yaw,joints=integrate(a,[0,0],0,.02)
        np.testing.assert_allclose(xy,[.006,.0004]);self.assertAlmostEqual(yaw,.002)
        self.assertEqual(joints['jointGripper'],-.5)
        xy2,_,_=integrate(a,xy,yaw,.02)
        self.assertGreater(xy2[0],xy[0])

    def test_score_no_failure_filter(self):
        r=dict(open_deg=59.,traversal_m=.6,handle_deg=30.,unlocked=True,base_door_contact=False)
        self.assertFalse(score([r])['success'])
        r['open_deg']=60;self.assertTrue(score([r])['success'])
        r['base_door_contact']=True;self.assertFalse(score([r])['contact_free_success'])
        self.assertTrue(score([r])['success'])

    def test_wilson(self):
        self.assertIsNone(wilson(0,0))
        self.assertAlmostEqual(wilson(0,64)[0],0)
        self.assertLess(wilson(0,64)[1],.06)
        self.assertGreater(wilson(64,64)[0],.94)

    def test_success_requires_simultaneous_thresholds(self):
        first=dict(step=0,open_deg=65.,traversal_m=-.5,handle_deg=30.,unlocked=True,base_door_contact=False)
        last=dict(first,step=1,open_deg=55.,traversal_m=.6)
        result=score([first,last])
        self.assertTrue(result['door_open_success'] and result['traversal_success'])
        self.assertFalse(result['success'])


if __name__=='__main__':unittest.main()
