import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import indices, transition, save, read, digest, validate_rollout


def fixture():
    names = ['joint'+str(i) for i in range(1,7)] + ['jointGripper']
    frames = []
    for i in range(11):
        targets = [float(i)]*7
        frames.append(dict(step=i, phase='hold_home', replay_door_root_state=[2.61],
            replay_root_state=[1.3],replay_door_dof_pos=[.8,0],
            control_record=dict(applied_action10=[i,.0,.0]+targets, measured_joint7=[-.1]*7,
                                commanded_dof_positions=targets,dof_names=names)))
    return dict(fps=50, frames=frames)


class PipelineTests(unittest.TestCase):
    def test_pre_action_alignment(self):
        state, action = transition(fixture(), 4)
        self.assertEqual(state[0],4)
        self.assertEqual(action[0],5)
        self.assertTrue(np.all(state[3:] == np.float32(-.1)))
        self.assertTrue(np.all(action[3:] == 5))

    def test_drop_unlabelled_last_frame(self):
        self.assertEqual(indices(fixture(),10),[0,5])
        self.assertEqual(indices(fixture(),25),[0,2,4,6,8])
        with self.assertRaises(AssertionError):
            indices(fixture(),12)

    def test_gap_rejected(self):
        d=fixture();d['frames'][5]['step']=7
        with self.assertRaises(AssertionError):transition(d,4)

    def test_nan_rejected(self):
        d=fixture();d['frames'][1]['control_record']['applied_action10'][0]=float('nan')
        with self.assertRaises(AssertionError):transition(d,0)

    def test_commands_checked_against_sent_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trajectory.json'
            d=fixture();save(path,d)
            self.assertTrue(validate_rollout(path,11)['passed'])
            d['frames'][3]['control_record']['applied_action10'][4]=123
            save(path,d)
            with self.assertRaises(AssertionError):validate_rollout(path,11)

    def test_atomic_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.json';save(path,{'a':1});save(path,{'a':2})
            self.assertEqual(read(path),{'a':2})
            self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))
            self.assertNotEqual(digest({'a':1}),digest({'a':2}))
            self.assertEqual(len(list(Path(folder).iterdir())),1)

    def test_relighting_copies_raw_without_old_scene(self):
        from pipeline import reuse_recording
        from common import sha
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root/'old/episodes/000000/raw_001'
            save(raw/'trajectory.json', fixture())
            save(raw/'scene_001/scene_verified.json', {'old':True})
            save(raw.parent/'status.json', dict(stages={'record':{'status':'complete'}},
                 raw=str(raw),raw_sha256=sha(raw/'trajectory.json'),physics_seed=123))
            result = reuse_recording(root/'old',0,root/'newraw',{'seed':123})
            self.assertTrue(result['reused'])
            self.assertEqual(sha(root/'newraw/trajectory.json'),sha(raw/'trajectory.json'))
            self.assertFalse((root/'newraw/scene_001').exists())
            with self.assertRaises(AssertionError):
                reuse_recording(root/'old',0,root/'bad',{'seed':999})


if __name__=='__main__':unittest.main()
