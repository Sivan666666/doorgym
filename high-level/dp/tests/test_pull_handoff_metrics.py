"""Metric regression: 0.5m pull threshold, unchanged push, simultaneous success."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from eval_door_policy_success import summarize_log


class MetricsTest(unittest.TestCase):
    def evaluate(self, pairs, mode='ikpull', threshold=None):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.jsonl'
            path.write_text(''.join(json.dumps(dict(step=i, controlled_env_id=0,
                door={'dof':[math.radians(angle),0]},base={'xy':[distance,0]},
                pull_task_geometry={'door_plane_xy':[0,0],'traversal_direction_xy':[1,0]}))+'\n'
                for i,(angle,distance) in enumerate(pairs)))
            kwargs={} if threshold is None else {'pull_traversal_distance_m':threshold}
            return summarize_log(path,1,60.,'abs',-1.,mode=mode,**kwargs)[0]

    def test_disputed_video_distance(self):
        self.assertTrue(self.evaluate([(90,.7360302)])['success'])
        self.assertFalse(self.evaluate([(90,.7360302)],threshold=.8)['success'])

    def test_boundary(self):
        self.assertTrue(self.evaluate([(61,.5)])['success'])
        self.assertFalse(self.evaluate([(61,.4999)])['success'])

    def test_simultaneous(self):
        result=self.evaluate([(90,.1),(20,.6)])
        self.assertTrue(result['door_open_success'])
        self.assertTrue(result['traversal_success'])
        self.assertFalse(result['success'])

    def test_push_unchanged(self):
        self.assertTrue(self.evaluate([(90,-2)],mode='ikpush')['success'])


if __name__=='__main__': unittest.main()
