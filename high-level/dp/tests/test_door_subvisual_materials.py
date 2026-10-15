"""Regression checks: wood-only binding must not replace glass or physics."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

HIGH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HIGH / 'float_ik'))
import rgb_door_materials as m
import rgb_doorman as r


class MaterialsTests(unittest.TestCase):
    def test_disabled_materials_do_not_bake_textures(self):
        runtime = SimpleNamespace(config=dict(materials=dict(enabled=False, preserve_subvisual_materials=True)))
        self.assertEqual(m.prepare_visual_door(runtime, '/unused', 'original.urdf'), ('/unused', 'original.urdf'))

    def test_glass_seals_and_metal_not_wood(self):
        tree = ET.parse(HIGH / 'data/asset/door_set/wc_nomenkuang/model.urdf')
        for v in tree.findall('.//visual'):
            if 'glass' in v.get('name', '') or v.get('name') == 'latch_edge_plate':
                self.assertIsNone(m.wood_visual_group(v))

    def test_nonvisual_and_nonwood_unchanged(self):
        source = HIGH / 'data/asset/door_set/wc_nomenkuang/model.urdf'
        runtime = SimpleNamespace(num_envs=1,
            environment_materials=lambda i: dict(panel={}, frame={}),
            texture=lambda desc: np.full((8,8,3), 210, np.uint8))
        root, file = m.prepare_per_visual_door(runtime, str(source.parent), source.name)
        before, after = ET.parse(source), ET.parse(Path(root) / file)
        for node in before.findall('.//mesh') + before.findall('.//texture'):
            node.set('filename', str((source.parent / node.get('filename')).resolve()))
        self.assertEqual(m.nonvisual_xml(before.getroot()), m.nonvisual_xml(after.getroot()))
        for link in before.findall('link'):
            for v in link.findall('visual'):
                counterpart = after.find(f".//link[@name='{link.get('name')}']/visual[@name='{v.get('name')}']")
                if m.wood_visual_group(v) is None:
                    self.assertEqual(ET.tostring(v), ET.tostring(counterpart), v.get('name'))
        glass = after.find(".//visual[@name='glass_panel']/material/color")
        self.assertEqual(glass.get('rgba'), '0.2 0.3 0.28 0.45')
        oldmesh = before.find(".//visual[@name='door_slab']/geometry/mesh").get('filename')
        newmesh = after.find(".//visual[@name='door_slab']/geometry/mesh").get('filename')
        def positions(path):
            return [s for s in Path(path).read_text().splitlines() if s.startswith('v ')]
        self.assertEqual(positions(oldmesh), positions(newmesh))
        def faces(path):
            return [[ref.split('/')[0] for ref in s.split()[1:]]
                    for s in Path(path).read_text().splitlines() if s.startswith('f ')]
        self.assertEqual(faces(oldmesh), faces(newmesh))
        self.assertTrue(runtime.per_visual_door_materials['nonvisual_unchanged'])

    def test_multiple_environments_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'one environment'):
            m.prepare_per_visual_door(SimpleNamespace(num_envs=2), '', '')

    def test_attach_never_overrides_mixed_body(self):
        runtime = r.VisualRandomizer.__new__(r.VisualRandomizer)
        runtime.config = dict(materials=dict(enabled=True, preserve_subvisual_materials=True), scene=dict(enabled=True))
        runtime.per_visual_door_materials = dict(nonvisual_unchanged=True)
        runtime.environment_materials = lambda _: {}
        runtime.metadata = {}
        runtime.envs = {}
        runtime.seed = 0
        runtime.scene_geometry = {}
        runtime.preview = None
        def forbidden(*args):
            raise AssertionError('Whole-body material setter called')
        gym = SimpleNamespace(get_actor_count=lambda env: 1, get_actor_name=lambda env, actor: 'door',
                              set_rigid_body_color=forbidden, set_rigid_body_texture=forbidden,
                              create_texture_from_buffer=forbidden)
        old = r.physics_signature
        r.physics_signature = lambda *args: {'unchanged': True}
        try:
            runtime.attach(gym, None, None, None, 0, None, SimpleNamespace(parallel_env_id=0))
        finally:
            r.physics_signature = old


if __name__ == '__main__':
    unittest.main()
