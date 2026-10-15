import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'float_ik'))
import rgb_doorman as dr


class RGBTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'dp/config/rgb_doorman.yaml').read_text())
        self.rgb = np.random.default_rng(90).integers(0, 256, (24, 32, 3), dtype=np.uint8)

    def test_upstream_parameters(self):
        c = self.cfg['image_augmentation']
        self.assertEqual(c['brightness'], dict(probability=.25, range=[.7, 2.]))
        self.assertEqual(c['hue'], dict(probability=.5, range=[-.1, .1]))
        self.assertEqual(c['gaussian_blur'], dict(probability=.25, kernel_size=[3, 5], sigma=[.1, 1.5]))

    def test_disabled_identity_and_global_rng(self):
        np.random.seed(41)
        state = np.random.get_state()
        self.assertIs(dr.augment_rgb(self.rgb, {}, dr.seeded(3)), self.rgb)
        for i in range(10):
            p = dr.image_parameters(self.cfg['image_augmentation'], dr.seeded(9, i))
            dr.augment_rgb(self.rgb, p, dr.seeded(10, i))
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_deterministic_and_nonmutating(self):
        old = self.rgb.copy()
        p = dict(brightness=1.3, saturation=.7, hue=-.07, contrast=1.2, gaussian_noise=.04,
                 gaussian_blur=self.cfg['image_augmentation']['gaussian_blur'])
        a = dr.augment_rgb(self.rgb, p, dr.seeded(3))
        b = dr.augment_rgb(self.rgb, p, dr.seeded(3))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.rgb, old)
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(a.shape, self.rgb.shape)
        self.assertFalse(np.array_equal(a, old))

    def test_torchvision_fixed_parameter_parity(self):
        try:
            import torch
            from torchvision.transforms import functional as F
        except ImportError:
            self.skipTest('torchvision not available')
        x = torch.from_numpy(self.rgb.copy()).permute(2, 0, 1).float() / 255
        for key, value in [('brightness', 1.3), ('saturation', .7), ('hue', -.07), ('contrast', 1.2)]:
            ref = getattr(F, 'adjust_' + key)(x, value)
            expected = (ref.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            actual = dr.augment_rgb(self.rgb, {key: value}, dr.seeded(3))
            self.assertLessEqual(abs(expected.astype(int) - actual.astype(int)).max(), 1, key)
        ref = F.gaussian_blur(x, [3, 5], [.1, 1.5])
        expected = (ref.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        actual = dr.augment_rgb(self.rgb, {'gaussian_blur': dict(kernel_size=[3, 5], sigma=[.1, 1.5])}, dr.seeded(3))
        self.assertLessEqual(abs(expected.astype(int) - actual.astype(int)).max(), 1)

    def test_render_adapter_has_no_physics_setters(self):
        tree = ast.parse(Path(dr.__file__).read_text())
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'gym'}
        writes = {name for name in calls if name.startswith(('set_', 'create_', 'apply_', 'add_'))}
        self.assertEqual(writes, {'set_light_parameters', 'set_rigid_body_color',
                                 'set_rigid_body_texture', 'create_texture_from_buffer'})

    def test_physics_hash_ignores_struct_padding(self):
        dtype = np.dtype({'names': ['hasLimits', 'effort'], 'formats': ['?', 'f4'],
                          'offsets': [0, 4], 'itemsize': 8})
        a = np.zeros(1, dtype=dtype)
        b = a.copy()
        b.view(np.uint8)[1:4] = [9, 31, 75]
        self.assertNotEqual(a.tobytes(), b.tobytes())
        fake = SimpleNamespace(get_actor_dof_properties=lambda env, actor: a if env == 0 else b,
            get_actor_rigid_body_properties=lambda *a: [], get_actor_rigid_shape_properties=lambda *a: [])
        self.assertEqual(dr.physics_signature(fake, 0, [0]), dr.physics_signature(fake, 1, [0]))

    def test_mdl_tint_is_not_discarded(self):
        assets = Path(os.environ.get('DOORMAN_VISUAL_ASSETS', str(dr.DEFAULT_ASSETS)))
        manifest = assets / 'manifest.json'
        if not manifest.exists():
            self.skipTest('Offline assets are not installed in default path')
        import json
        runtime = object.__new__(dr.VisualRandomizer)
        runtime.assets = assets
        runtime.manifest = json.loads(manifest.read_text())
        runtime.config = self.cfg
        runtime.textures = {}
        results = []
        for name in ('Aluminum_Anodized_Blue', 'Aluminum_Anodized_Red'):
            i = next(i for i, m in enumerate(runtime.manifest['materials']) if m['name'] == name)
            entry = runtime.manifest['materials'][i]
            desc = dict(index=i, variant=0, tint=entry['linear_tint'], rotation_deg=0, translation=[0, 0])
            results.append(runtime.texture(desc))
        self.assertFalse(np.array_equal(*results))
        self.assertGreater(results[0][..., 2].mean(), results[0][..., 0].mean())
        self.assertGreater(results[1][..., 0].mean(), results[1][..., 2].mean())

    def test_floor_projection_occlusion_and_nonmutation(self):
        runtime = object.__new__(dr.VisualRandomizer)
        runtime.config = self.cfg
        runtime.rays = {}
        runtime.envs = {0: {'floor': {}}}
        runtime.texture = lambda _: np.full((4, 4, 3), [200, 50, 20], np.uint8)
        runtime.sky = np.ones((4, 8, 3), np.float32)
        runtime.intensity, runtime.rotation = 3000., 0.
        depth = np.full((4, 4), -1., np.float32)
        depth[0, 0] = -.5  # Object in front of ground, never recolor.
        depth[0, 1] = -np.inf  # Sky, not floor.
        view = np.eye(4)
        view[3, 2] = -1.  # Camera one metre above original ground.
        fake = SimpleNamespace(get_camera_proj_matrix=lambda *a: np.eye(4),
            get_camera_view_matrix=lambda *a: view, get_camera_image=lambda *a: depth,
            get_env_origin=lambda *a: SimpleNamespace(x=0, y=0))
        rgb = np.full((4, 4, 3), 150, np.uint8)
        result = runtime.composite(fake, None, None, None, rgb,
            SimpleNamespace(parallel_env_id=0), SimpleNamespace(IMAGE_DEPTH=0))
        np.testing.assert_array_equal(result[0, 0], rgb[0, 0])
        np.testing.assert_array_equal(result[1, 1], [200, 50, 20])
        np.testing.assert_array_equal(rgb, np.full_like(rgb, 150))
        self.assertEqual(depth[0, 0], -.5)
        self.assertTrue(np.isneginf(depth[0, 1]))

    def test_legacy_common_capture_exact(self):
        tree = ast.parse((ROOT / 'float_ik/door_common.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_camera_rgb_for_args')
        scope = {'remap_camera_image_for_args': lambda image, *a, **kw: image}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<helper>', 'exec'), scope)
        self.assertIs(scope[node.name](None, None, None, None, 'front', self.rgb, SimpleNamespace()), self.rgb)

    def test_stable_lighting_is_episode_fixed(self):
        assets = Path(os.environ.get('DOORMAN_VISUAL_ASSETS', str(dr.DEFAULT_ASSETS)))
        if not assets.exists():
            self.skipTest('Assets unavailable')
        runtime = dr.VisualRandomizer(SimpleNamespace(rgb=True, rgb_visual_config=str(dr.DEFAULT_CONFIG),
            rgb_visual_assets=str(assets), rgb_visual_seed=42, seed=9, rgb_visual_preview_dir=''))
        calls = []
        fake = SimpleNamespace(set_light_parameters=lambda *a: calls.append(a))
        api = SimpleNamespace(Vec3=lambda *a: a)
        for step in (0, 2, 50, 100, 1000):
            runtime.update(fake, None, api, step, .02)
        self.assertEqual(len(calls), 4)
        self.assertEqual(runtime.light_slot, 0)
        self.assertEqual(runtime.config['image_augmentation']['temporal_mode'], 'episode')

    def test_large_texture_translation_retains_grain(self):
        assets = Path(os.environ.get('DOORMAN_VISUAL_ASSETS', str(dr.DEFAULT_ASSETS)))
        if not assets.exists():
            self.skipTest('Assets unavailable')
        runtime = dr.VisualRandomizer(SimpleNamespace(rgb=True, rgb_visual_config=str(dr.DEFAULT_CONFIG),
            rgb_visual_assets=str(assets), rgb_visual_seed=20260911, seed=9, rgb_visual_preview_dir=''))
        desc = runtime.environment_materials(0)['floor']
        self.assertGreater(max(desc['translation']), 64)
        self.assertGreater(float(runtime.texture(desc).std((0, 1)).min()), 5.)

    def test_scene_has_lintel_and_no_collisions(self):
        import tempfile
        import xml.etree.ElementTree as ET
        from rgb_visual_scene import mesh_geometry, write_asset
        walls, floor = mesh_geometry([3, -1.1, 1.1], 'y', -1, 1., 1.2, 0., 3.2, 2.3, .08, 6., [3, 0])
        self.assertEqual(len(walls), 12)
        self.assertEqual(min(p[2] for q in walls[6:] for p in q), 2.3)
        self.assertEqual(max(p[1] for q in walls[6:] for p in q), 0.)
        with tempfile.TemporaryDirectory() as tmp:
            write_asset(tmp, [3,-1.1,1.1], walls, floor, self.rgb, self.rgb, 1.5, 1.5)
            tree = ET.parse(Path(tmp)/'scene.urdf')
            self.assertEqual(len(tree.findall('.//collision')), 0)
            self.assertEqual(len(tree.findall('.//link')), 1)
            self.assertEqual(len(tree.findall('.//joint')), 0)
            self.assertEqual(tree.find('.//mass').get('value'), '1.0')
            obj = (Path(tmp)/'scene.obj').read_text()
            self.assertIn('usemtl wall', obj)
            self.assertIn('usemtl floor', obj)
            uv = [list(map(float, row.split()[1:])) for row in obj.splitlines() if row.startswith('vt ')]
            self.assertGreater(len(uv), 52)
            self.assertTrue(np.all((np.asarray(uv) >= 0) & (np.asarray(uv) <= 1)))

    def test_indoor_room_is_geometry_not_photo(self):
        import tempfile
        import xml.etree.ElementTree as ET
        from rgb_visual_scene import indoor_parameters, indoor_geometry, write_asset, mesh_geometry
        config = yaml.safe_load(Path(dr.DEFAULT_CONFIG).read_text())
        self.assertFalse(config['hdr_background_rgb'])
        runtime = object.__new__(dr.VisualRandomizer)
        runtime.config = config
        # No image overlay and no camera/depth API access on the indoor path.
        self.assertIs(runtime.composite(None, None, None, None, self.rgb, None, None), self.rgb)
        cfg = config['scene']['indoor_room']
        state = np.random.get_state()
        rooms = [indoor_parameters(s, 0, cfg, 'y', np.pi, [3.,0.], 3.2) for s in (20260911,20260912)]
        self.assertNotEqual(rooms[0]['far_wall_distance_m'], rooms[1]['far_wall_distance_m'])
        for room in rooms:
            self.assertEqual(room['travel_sign'], -1)
            self.assertTrue(3 <= room['far_wall_distance_m'] <= 4)
            walls, ceiling = indoor_geometry(room, -1)
            self.assertAlmostEqual(np.max(np.asarray(walls[:6])[:,:,0]), 3-room['far_wall_distance_m'])
            self.assertAlmostEqual(np.min(np.asarray(ceiling)[:,:,2]), 3.2)
            self.assertEqual(np.max(np.asarray(walls)[:,:,1]), 0.)
            # Keep the far wall beyond the unchanged traversal endpoint.
            self.assertGreater(room['far_wall_distance_m'], 1.31)
            other_walls, _ = indoor_geometry(room, 1)
            self.assertEqual(np.min(np.asarray(other_walls)[:,:,1]), 0.)
        # Rotated door orientation selects +y as destination rather than hardcoded -x.
        rotated = indoor_parameters(42, 0, cfg, 'x', np.pi/2, [0.,3.], 3.2)
        self.assertEqual((rotated['normal_axis'], rotated['travel_sign']), (1, 1))
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(state[2:], np.random.get_state()[2:])
        walls, ceiling = indoor_geometry(rooms[0], -1)
        base_walls, floor = mesh_geometry([3,-1.1,1.1], 'y', -1, 1., 1.2, 0., 3.2, 2.3, .08, 6., [3,0])
        with tempfile.TemporaryDirectory() as tmp:
            write_asset(tmp, [3,-1.1,1.1], base_walls, floor, self.rgb, self.rgb, 1.5, 1.5,
                        [('far_wall', walls, self.rgb, 1.5), ('ceiling', ceiling, self.rgb, 1.5)])
            tree = ET.parse(Path(tmp)/'scene.urdf')
            self.assertEqual(len(tree.findall('.//collision')), 0)
            self.assertEqual(len(tree.findall('.//link')), 1)
            self.assertEqual(len(tree.findall('.//joint')), 0)
            obj = (Path(tmp)/'scene.obj').read_text()
            self.assertIn('usemtl far_wall', obj)
            self.assertIn('usemtl ceiling', obj)

    def test_wc_reference_materials(self):
        from rgb_door_materials import reference_sources, reference_descriptors, reference_texture
        cfg = yaml.safe_load(Path(dr.DEFAULT_CONFIG).read_text())['materials']['wc_reference']
        sources = reference_sources(ROOT.parent, cfg)
        a, b = [reference_descriptors(seed, 0, cfg, sources) for seed in (20260911,20260912)]
        self.assertEqual(a['handle'], b['handle'])
        panel, handle = reference_texture(a['panel'], 512), reference_texture(a['handle'], 512)
        self.assertGreater(float(panel.std((0,1)).min()), 5.)
        self.assertFalse(np.array_equal(panel, reference_texture(b['panel'], 512)))
        self.assertLess(int(handle.max()), 40)
        np.testing.assert_array_equal(handle[:,:,0], handle[:,:,1])
        np.testing.assert_array_equal(handle[:,:,1], handle[:,:,2])
        self.assertEqual(a, reference_descriptors(20260911, 0, cfg, sources))

    def test_wc_uv_copy_keeps_geometry_and_collisions(self):
        import xml.etree.ElementTree as ET
        from rgb_door_materials import prepare_visual_door
        runtime = SimpleNamespace(config=yaml.safe_load(Path(dr.DEFAULT_CONFIG).read_text()))
        source = ROOT/'data/asset/door_set/wc_nomenkuang/model.urdf'
        root, name = prepare_visual_door(runtime, str(source.parent), source.name)
        old, new = ET.parse(source), ET.parse(Path(root)/name)
        self.assertEqual([ET.tostring(x) for x in old.findall('.//joint')],
                         [ET.tostring(x) for x in new.findall('.//joint')])
        for a, b in zip(old.findall('.//collision'), new.findall('.//collision')):
            for m in a.findall('.//mesh'):
                m.set('filename', str((source.parent/m.get('filename')).resolve()))
            self.assertEqual(ET.tostring(a), ET.tostring(b))
        path = "./link[@name='door_panel']/visual[@name='door_slab']/geometry/mesh"
        orig = (source.parent/old.find(path).get('filename')).read_text().splitlines()
        mapped = Path(new.find(path).get('filename')).read_text().splitlines()
        self.assertEqual([x for x in orig if x.startswith('v ')], [x for x in mapped if x.startswith('v ')])
        self.assertEqual([x.split()[1:] for x in orig if x.startswith('f ')],
                         [[i.split('/')[0] for i in x.split()[1:]] for x in mapped if x.startswith('f ')])
        coords = np.asarray([list(map(float,x.split()[1:])) for x in mapped if x.startswith('vt ')])
        self.assertGreater(len(coords), 0)
        self.assertTrue(np.all((coords >= 0) & (coords <= 1)))


if __name__ == '__main__':
    unittest.main()
