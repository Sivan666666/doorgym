import copy
from pathlib import Path
import unittest
import numpy as np
from common import read
from augmentation import doorman, photo_plan, apply_photo


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read(Path(__file__).parent / 'configs/rgb_aug5_no_material.json')
        self.cfg['tools']['rgb_augmenter'] = str(Path(__file__).resolve().parents[2] / 'float_ik/rgb_doorman.py')
        self.module = doorman(self.cfg)
        self.rgb = np.arange(32*48*3, dtype=np.uint8).reshape(32,48,3)

    def test_deterministic_episode_plan(self):
        plans = [photo_plan(self.cfg,20260921+i) for i in range(5)]
        self.assertEqual(plans[0], photo_plan(self.cfg,20260921))
        self.assertEqual(len({p['parameters']['brightness'] for p in plans}),5)
        self.assertTrue(all('contrast' in p['parameters'] for p in plans))
        self.assertFalse(self.cfg['scene']['randomize'])

    def test_resume_and_no_mutation(self):
        plan = photo_plan(self.cfg, 20260921)
        plan['parameters']['gaussian_noise'] = 0.01
        original = self.rgb.copy()
        a = apply_photo(self.module, self.rgb, plan, 10, 'front')
        np.testing.assert_array_equal(a, apply_photo(self.module, self.rgb, plan, 10, 'front'))
        np.testing.assert_array_equal(original, self.rgb)
        self.assertEqual(a.shape, self.rgb.shape)
        self.assertEqual(a.dtype, self.rgb.dtype)
        self.assertFalse(np.array_equal(a, apply_photo(self.module,self.rgb,plan,11,'front')))
        self.assertFalse(np.array_equal(a, apply_photo(self.module,self.rgb,plan,10,'wrist')))

    def test_disabled_exact_passthrough(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['render']['image_augmentation']['enabled'] = False
        plan = photo_plan(cfg,1)
        self.assertIs(apply_photo(None,self.rgb,plan,0,'front'),self.rgb)

    def test_dim_variant_never_brightens_and_keeps_materials(self):
        cfg = read(Path(__file__).parent / 'configs/rgb_aug5_dim.json')
        cfg['tools']['rgb_augmenter'] = self.cfg['tools']['rgb_augmenter']
        for seed in range(20260921,20260926):
            plan = photo_plan(cfg,seed)
            self.assertLessEqual(plan['parameters']['brightness'],1.0)
        self.assertLessEqual(max(cfg['scene']['lighting']['energy_multiplier_range']),0.55)
        self.assertFalse(cfg['scene']['randomize'])

    def test_ceiling_only_disables_studio_and_world(self):
        from types import SimpleNamespace as S
        from augmentation import randomize_lighting
        cfg = read(Path(__file__).parent / 'configs/rgb_ceiling1.json')['scene']
        class Location(list):
            z = 3.24
        names = cfg['lighting']['ceiling_light_names'] + ['Soft front fill']
        objects = [S(name=name,type='LIGHT',location=Location([0,0,3.24]),
                     data=S(type='AREA',energy=500.,color=[1.,1.,1.],size=2.5)) for name in names]
        strength = S(default_value=.25,is_linked=False)
        world = S(use_nodes=True,node_tree=S(nodes=[S(name='Background',type='BACKGROUND',inputs={'Strength':strength})]))
        bpy = S(data=S(materials=[],objects=objects),context=S(scene=S(world=world)))
        result = randomize_lighting(bpy,cfg,20260921)
        self.assertEqual([o.data.energy for o in objects],[250.,250.,0.])
        self.assertEqual(strength.default_value,0.)
        self.assertTrue(result['materials_unchanged'])
        self.assertEqual(objects[0].rotation_euler,(0,0,0))

    def test_point_only_disables_all_existing_lights(self):
        from types import SimpleNamespace as S
        from augmentation import randomize_lighting
        cfg = read(Path(__file__).parent / 'configs/rgb_point_dark1.json')['scene']
        class Objects(list):
            def new(self,name,light):
                obj = S(name=name,type='LIGHT',data=light)
                self.append(obj)
                return obj
        objects = Objects([S(name='old ceiling',type='LIGHT',data=S(type='AREA',energy=500.,color=[1.,1.,1.]))])
        strength = S(default_value=.25,is_linked=False)
        world = S(use_nodes=True,node_tree=S(nodes=[S(name='Background',type='BACKGROUND',inputs={'Strength':strength})]))
        scene = S(world=world,collection=S(objects=S(link=lambda obj:None)))
        bpy = S(data=S(materials=[],objects=objects,lights=S(new=lambda name,kind:S(name=name,type=kind))),context=S(scene=scene))
        result = randomize_lighting(bpy,cfg,20260921)
        self.assertEqual([o.data.energy for o in objects],[0.,12.,12.])
        self.assertEqual([o.data.type for o in objects[1:]],['POINT','POINT'])
        self.assertEqual([o.location[0] for o in objects[1:]],[0.,0.])
        self.assertEqual([o.location[1] for o in objects[1:]],[-4.,3.5])
        self.assertEqual(strength.default_value,0.)
        self.assertTrue(result['materials_unchanged'])

    def test_final_point_light_positions_unchanged_small_power_increase(self):
        base = Path(__file__).parent / 'configs'
        old = read(base/'rgb_point_dark1.json')['scene']['lighting']['point_lights']
        new = read(base/'rgb_point_final1.json')['scene']['lighting']['point_lights']
        for before, after in zip(old,new):
            self.assertEqual(before['location'],after['location'])
            self.assertEqual(before['radius_m'],after['radius_m'])
            self.assertEqual(before['color'],after['color'])
            self.assertEqual(after['energy']/before['energy'],1.25)

    def test_point30_changes_only_lamp_energy_in_visual_settings(self):
        base = Path(__file__).parent / 'configs'
        before = read(base/'rgb_point_final1.json')
        after = read(base/'rgb_point30_1.json')
        scene = copy.deepcopy(before['scene'])
        for light in scene['lighting']['point_lights']:
            light['energy'] = 30.0
        self.assertEqual(scene,after['scene'])
        self.assertEqual(before['render'],after['render'])


if __name__ == '__main__':
    unittest.main()
