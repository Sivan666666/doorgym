"""Wood-only WC preview; retain historical experiments and nonwood materials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import export_rgb500_review_videos as capture
import preview_wc_original_white_wall as fixed

OUT = capture.A / 'dp/result/rgb500_wc_glass_fixed_20260918'
SOURCE = fixed.SOURCE
CHANGED = {'float_ik/rgb_doorman.py', 'float_ik/rgb_door_materials.py'}


def configure():
    capture.OUT = fixed.OUT = OUT
    fixed.SOURCE = SOURCE


def prepare():
    import yaml
    configure()
    config = fixed.prepare()
    cfg = yaml.safe_load(config.read_text())
    cfg['materials']['preserve_subvisual_materials'] = True
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config


def guard_sources():
    original = json.loads((capture.EXPERIMENT / 'experiment.json').read_text())['source_sha256']
    current = {}
    for name, expected in original.items():
        current[name] = hashlib.sha256((capture.REPO / 'high-level' / name).read_bytes()).hexdigest()
        assert name in CHANGED or current[name] == expected, name
    capture.save(OUT / 'source_audit.json', dict(original=original, current=current,
        intentional_changes=sorted(CHANGED), all_other_sources_unchanged=True))


def probe():
    # Import Isaac Gym before anything that may import torch.
    from isaacgym import gymapi
    import copy
    import cv2
    import numpy as np
    from types import SimpleNamespace
    import xml.etree.ElementTree as ET
    sys.path.insert(0, str(capture.REPO / 'high-level/float_ik'))
    import rgb_door_materials as material
    import rgb_doorman
    import yaml

    config = prepare()
    fixed.install_fixed_appearance(None)
    runtime = rgb_doorman.VisualRandomizer.__new__(rgb_doorman.VisualRandomizer)
    runtime.config = yaml.safe_load(config.read_text())
    runtime.envs, runtime.textures, runtime.num_envs = {}, {}, 1
    source = capture.REPO / 'high-level/data/asset/door_set/wc_nomenkuang/model.urdf'
    root, filename = material.prepare_visual_door(runtime, str(source.parent), source.name)
    root = Path(root)
    base = ET.parse(root / filename)
    output = OUT / 'glass_probe'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'material_manifest.json').write_text(json.dumps(runtime.per_visual_door_materials, indent=2))
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.physx.use_gpu = True
    params.physx.num_threads = 2
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    assert sim is not None
    try:
        for i in range(4):
            gym.set_light_parameters(sim, i, gymapi.Vec3(*((.5,)*3 if i == 0 else (0,)*3)),
                gymapi.Vec3(*((.7,)*3 if i == 0 else (0,)*3)), gymapi.Vec3(0, -1, 1))
        cases, physics = [], {}
        for mode in ('original_alpha', 'opaque_control', 'absent_visual_control'):
            tree = copy.deepcopy(base)
            link = tree.find(".//link[@name='door_panel']")
            glass = link.find("visual[@name='glass_panel']")
            assert glass is not None
            if mode == 'opaque_control':
                glass.find('material/color').set('rgba', '0.2 0.3 0.28 1')
            if mode == 'absent_visual_control':
                link.remove(glass)
            assert material.nonvisual_xml(tree.getroot()) == material.nonvisual_xml(base.getroot())
            tree.write(str(root / (mode + '.urdf')))
            opt = gymapi.AssetOptions()
            opt.fix_base_link = True
            opt.collapse_fixed_joints = True
            opt.use_mesh_materials = True
            opt.override_com = opt.override_inertia = True
            opt.disable_gravity = True
            opt.vhacd_enabled = True
            opt.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset = gym.load_asset(sim, str(root), mode + '.urdf', opt)
            assert asset is not None
            env = gym.create_env(sim, gymapi.Vec3(-3,-3,-1), gymapi.Vec3(3,3,3), 3)
            actor = gym.create_actor(env, asset, gymapi.Transform(), mode, len(cases), 0)
            physics[mode] = dict(bodies=gym.get_actor_rigid_body_count(env, actor),
                                shapes=gym.get_actor_rigid_shape_count(env, actor),
                                dofs=gym.get_actor_dof_count(env, actor))
            box_opt = gymapi.AssetOptions()
            box_opt.fix_base_link = True
            back_asset = gym.create_box(sim, .6, .02, 2.4, box_opt)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(.265, .35, 1.115)
            back = gym.create_actor(env, back_asset, pose, 'background', len(cases), 0)
            cp = gymapi.CameraProperties()
            cp.width, cp.height, cp.horizontal_fov = 640, 480, 50
            camera = gym.create_camera_sensor(env, cp)
            gym.set_camera_location(camera, env, gymapi.Vec3(.265,-.6,1.115), gymapi.Vec3(.265,0,1.115))
            cases.append((mode, env, back, camera))
        gym.prepare_sim(sim)
        frames = {}
        for color_name, color in [('red', (1.,0.,0.)), ('blue', (0.,0.,1.))]:
            for mode, env, back, camera in cases:
                gym.set_rigid_body_color(env, back, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            for mode, env, back, camera in cases:
                frame = np.asarray(gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR)).reshape(480,640,4)[...,:3].copy()
                frames[(mode, color_name)] = frame
                assert cv2.imwrite(str(output / (mode + '_' + color_name + '.png')), frame[...,::-1])
        roi = (slice(120,360), slice(300,340))
        response = {mode: float(np.abs(frames[(mode,'red')][roi].astype(float) -
                    frames[(mode,'blue')][roi].astype(float)).mean()) for mode, *_ in cases}
        alpha_delta = float(np.abs(frames[('original_alpha','red')][roi].astype(float) -
                                  frames[('opaque_control','red')][roi].astype(float)).mean())
        assert response['absent_visual_control'] > 30, response
        assert len({json.dumps(v, sort_keys=True) for v in physics.values()}) == 1
        result = dict(roi_xyxy=[300,120,340,360], background_swap_mean_abs_rgb_difference=response,
                      original_vs_opaque_mean_abs_rgb_difference=alpha_delta,
                      transparency_observed=response['original_alpha'] > 5,
                      physics_counts=physics, nonvisual_xml_unchanged=True,
                      tested_renderer='Isaac Gym camera IMAGE_COLOR', original_glass_rgba=[.2,.3,.28,.45])
        (output / 'verification.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        gym.destroy_sim(sim)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['probe', 'inference', 'compose', 'verify'])
    args = parser.parse_args()
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    guard_sources()
    if args.stage == 'probe':
        probe()
    elif args.stage == 'inference':
        capture.inference(visual_config=prepare(), module_setup=fixed.install_fixed_appearance)
    elif args.stage == 'compose':
        capture.compose()
    else:
        fixed.verify()
        previous = SOURCE.parent / 'inference_previews/physics_env00.json'
        current = OUT / 'inference_previews/physics_env00.json'
        assert json.loads(current.read_text()) == json.loads(previous.read_text()), 'Loaded physics changed'
        for path in (OUT / 'inference_previews').glob('env00_*_*.json'):
            manifest = json.loads(path.read_text())['episode']['per_visual_door_materials']
            assert manifest['nonvisual_unchanged'] and not manifest['rigid_body_texture_override']
            assert 'door_panel/glass_panel' in manifest['preserved_visuals']
        verification = json.loads((OUT / 'verification.json').read_text())
        verification.update(wood_only_bindings=True, preserved_glass_and_gaskets=True,
                            loaded_physics_equal_previous_preview=True,
                            transparency_probe=json.loads((OUT / 'glass_probe/verification.json').read_text()))
        capture.save(OUT / 'verification.json', verification)
        print('VERIFIED wood-only material bindings and preserved glass/gaskets', flush=True)


if __name__ == '__main__':
    main()
