"""DoorMan photometric augmentation and VISUAL-ONLY Isaac Gym adapter.

Photometric order/parameters follow NVlabs/GR00T-VisualSim2Real doorman
016c70c1e4e76f521963c36691ee69a6ab3ac9cd, Apache-2.0, NVIDIA 2026.
Implementation uses NumPy/OpenCV for Gym's older Torch runtime. No physics setters,
new actors, action edits, image geometry transforms, or global RNG consumption.
MDL/PBR and HDR dome rendering are not supported by Gym: explicitly approximated.
"""
import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
try:
    import cv2
    import yaml
except ImportError:
    cv2 = yaml = None  # Legacy operation must not acquire new mandatory dependencies.

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'dp/config/rgb_doorman_stable.yaml'
DEFAULT_ASSETS = ROOT / 'data/visual_randomization/doorman'


def cli_parameters():
    return [
        dict(name='--rgb_visual_mode', type=str, default='legacy', choices=['legacy', 'doorman']),
        dict(name='--rgb_visual_config', type=str, default=str(DEFAULT_CONFIG)),
        dict(name='--rgb_visual_assets', type=str, default=str(DEFAULT_ASSETS)),
        dict(name='--rgb_visual_seed', type=int, default=-1),
        dict(name='--rgb_visual_preview_dir', type=str, default=''),
    ]


def seeded(seed, *keys):
    values = [int(seed) % 2**32] + [int(k) % 2**32 for k in keys]
    return np.random.default_rng(np.random.SeedSequence(values))


def image_parameters(cfg, rng):
    params = {}
    if not cfg['enabled']:
        return params
    for key in ('brightness', 'saturation', 'hue', 'contrast', 'gaussian_noise', 'gaussian_blur'):
        spec = cfg[key]
        if rng.random() < spec['probability']:
            if key == 'gaussian_blur':
                params[key] = dict(kernel_size=spec['kernel_size'], sigma=spec['sigma'])
            else:
                params[key] = float(rng.uniform(*spec['std_range' if key == 'gaussian_noise' else 'range']))
    return params


def augment_rgb(rgb, params, rng):
    if not params:
        return rgb
    x = rgb.astype(np.float32) / 255.
    if 'brightness' in params:
        x = np.clip(x * params['brightness'], 0, 1)
    if 'saturation' in params:
        gray = (x * np.array([.2989, .587, .114], np.float32)).sum(-1, keepdims=True)
        f = params['saturation']
        x = np.clip(x * f + gray * (1 - f), 0, 1)
    if 'hue' in params:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + params['hue'] * 360.) % 360.
        x = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)
    if 'contrast' in params:
        mean = (x * np.array([.2989, .587, .114], np.float32)).sum(-1).mean()
        f = params['contrast']
        x = np.clip(x * f + mean * (1 - f), 0, 1)
    if 'gaussian_noise' in params:
        x = np.clip(x + rng.normal(0, params['gaussian_noise'], x.shape).astype(np.float32), 0, 1)
    if 'gaussian_blur' in params:
        b = params['gaussian_blur']
        x = cv2.GaussianBlur(x, tuple(b['kernel_size']), b['sigma'][0],
                             sigmaY=b['sigma'][1], borderType=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(np.rint(x * 255).clip(0, 255).astype(np.uint8))


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)


def srgb_to_linear(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def physics_properties(gym, env, actors):
    """Read-only comparison immediately before/after appearance setters."""
    def vector(v):
        if hasattr(v, 'x'):
            return [float(getattr(v, k)) for k in ('x', 'y', 'z')]
        return float(v)
    result = []
    for actor in actors:
        bodies = gym.get_actor_rigid_body_properties(env, actor)
        shapes = gym.get_actor_rigid_shape_properties(env, actor)
        dof = gym.get_actor_dof_properties(env, actor)
        result.append(dict(
            # Structured dtype padding can contain uninitialised bytes. Hash
            # named values, never raw .tobytes(), for cross-process comparison.
            dof={name: dof[name].tolist() for name in dof.dtype.names},
            bodies=[dict(mass=float(b.mass), invMass=float(b.invMass), flags=int(b.flags),
                         com=vector(b.com), inertia=[vector(b.inertia.x), vector(b.inertia.y), vector(b.inertia.z)])
                    for b in bodies],
            shapes=[{k: vector(getattr(s, k)) for k in ('friction', 'rolling_friction', 'torsion_friction',
                    'restitution', 'compliance', 'contact_offset', 'rest_offset', 'filter') if hasattr(s, k)} for s in shapes]))
    return result


def physics_signature(gym, env, actors):
    return hashlib.sha256(json.dumps(physics_properties(gym, env, actors), sort_keys=True).encode()).hexdigest()


class VisualRandomizer:
    def __init__(self, args):
        if cv2 is None or yaml is None:
            raise ImportError('Doorman RGB mode requires opencv-python and PyYAML')
        if not getattr(args, 'rgb', False):
            raise ValueError('doorman RGB mode requires --rgb / recorder --rgb_only; depth mode is untouched')
        self.config = yaml.safe_load(Path(args.rgb_visual_config).read_text())
        self.num_envs = int(getattr(args, 'num_envs', 1))
        if self.config['materials'].get('preserve_subvisual_materials', False) and getattr(args, 'door_use_urdf_rgba', False):
            raise ValueError('Per-visual texture binding requires mesh materials, not door_use_urdf_rgba')
        if self.config['version'] not in ('doorman_rgb_gym_v1', 'doorman_rgb_gym_v2'):
            raise ValueError('Unknown visual config version')
        if self.config['materials']['dynamic']:
            raise ValueError('Upstream DoorMan task disables dynamic materials; leave dynamic:false')
        self.assets = Path(args.rgb_visual_assets).resolve()
        manifest_bytes = (self.assets / 'manifest.json').read_bytes()
        self.manifest = json.loads(manifest_bytes)
        if self.manifest.get('version', 0) < 2:
            raise ValueError('Refresh asset diffuse tint metadata with prepare_doorman_visual_assets.py --offline_refresh')
        if not self.manifest['materials'] or not self.manifest['hdr']:
            raise ValueError('Empty material/HDR asset pool')
        for entry in self.manifest['materials'] + self.manifest['hdr']:
            if 'texture' in entry:
                p = self.assets / entry['texture']
                if hashlib.sha256(p.read_bytes()).hexdigest() != entry['prepared_sha256']:
                    raise ValueError('Visual asset changed or corrupt: ' + str(p))
        self.seed = int(args.rgb_visual_seed if args.rgb_visual_seed >= 0 else args.seed)
        self.frame = 0
        self.light_slot = None
        self.textures, self.skies, self.rays = {}, {}, {}
        self.envs = {}
        self.visual_assets = {}
        self.scene_geometry = {}
        reference_cfg = self.config['materials'].get('wc_reference', {})
        self.reference_sources = {}
        if reference_cfg.get('enabled'):
            from rgb_door_materials import reference_sources
            self.reference_sources = reference_sources(ROOT.parent, reference_cfg)
        self.preview = Path(args.rgb_visual_preview_dir) if args.rgb_visual_preview_dir else None
        self.preview_seen = set()
        self.metadata = dict(version=self.config['version'], mode='doorman', seed=self.seed,
            config=copy.deepcopy(self.config), source_commit=self.manifest['source_commit'],
            source_hashes=self.manifest['source_hashes'], asset_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            hdr_count=len(self.manifest['hdr']), material_count=len(self.manifest['materials']),
            hdr_per_folder=self.manifest['hdr_per_folder'], limitations=self.manifest['limitations'],
            rng='independent SeedSequence(seed, namespace, env, frame, camera); no physics RNG consumed',
            image_parameters_shared_across_views_and_envs=True, image_noise_independent_per_view=True,
            preview_is_policy_rgb=True, physics_changed=False,
            visual_geometry_changed=bool(self.config.get('scene', {}).get('enabled')),
            scene='Visual meshes on existing no-collision wall actors' if self.config.get('scene', {}).get('enabled')
                  else 'Original scene plus RGB-only ground projection',
            depth_note='New visual walls/floors are visible to rendering depth in RGB mode; no depth policy output')

    def material(self, group, rng):
        cfg = self.config['materials']
        groups = ('textured_material_list',) if group in ('wall', 'floor') else (
            ('textured_material_list', 'random_color_material_list') if group == 'frame' else (
            ('random_color_material_list',) if group == 'panel' else ('metallic_material_list',))
        )
        choices = []
        for i, m in enumerate(self.manifest['materials']):
            if m['group'] in groups:
                if group in ('wall', 'floor') and not m.get('texture'):
                    continue
                n = cfg['paint_variants'] if m['group'] == 'random_color_material_list' else cfg['texture_variants']
                choices.extend((i, variant) for variant in range(n))
        i, variant = choices[int(rng.integers(len(choices)))]
        entry = self.manifest['materials'][i]
        r = seeded(self.seed, 741, i, variant)
        desc = dict(index=i, name=entry['name'], variant=variant,
                    rotation_deg=float(r.uniform(*cfg['rotation_deg'])),
                    translation=r.uniform(*cfg['translation'], size=2).tolist(),
                    tint=r.exponential(cfg['paint_tint_exponential_scale'], 3).tolist()
                    if entry['group'] == 'random_color_material_list' else entry['linear_tint'])
        if entry['group'] == 'random_color_material_list' and 'paint_tint_range' in cfg:
            desc['tint'] = r.uniform(*cfg['paint_tint_range'], size=3).tolist()
        return desc

    def environment_materials(self, index):
        if index not in self.envs:
            r = seeded(self.seed, 742, index)
            stable = bool(self.config.get('scene', {}).get('enabled'))
            self.envs[index] = {g: self.material(g if g != 'floor' or stable else 'frame', r)
                               for g in ('frame', 'panel', 'handle', 'floor')}
            if stable:
                self.envs[index]['wall'] = self.material('wall', r)
                if self.config['scene'].get('indoor_room', {}).get('enabled'):
                    self.envs[index]['far_wall'] = self.material('wall', seeded(self.seed, 747, index))
            if self.reference_sources:
                from rgb_door_materials import reference_descriptors
                self.envs[index].update(reference_descriptors(self.seed, index,
                    self.config['materials']['wc_reference'], self.reference_sources))
        return self.envs[index]

    def texture(self, desc):
        if desc.get('kind') == 'wc_reference_v1':
            from rgb_door_materials import reference_texture
            key = json.dumps(desc, sort_keys=True)
            if key not in self.textures:
                self.textures[key] = reference_texture(desc, int(self.config['materials']['texture_resolution']))
            return self.textures[key]
        key = (desc['index'], desc['variant'])
        if key in self.textures:
            return self.textures[key]
        entry = self.manifest['materials'][desc['index']]
        size = int(self.config['materials']['texture_resolution'])
        if entry.get('texture'):
            bgr = cv2.imread(str(self.assets / entry['texture']))
            source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source = cv2.resize(source, (size, size))
            # Baked texture transform; not upstream's object-space MDL UV projection.
            y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
            a = math.radians(desc['rotation_deg'])
            # Wrap before scaling: OpenCV's internal short coordinates saturate
            # above 32767 (100 tiles * 512 px), otherwise flattening the texture.
            u = np.remainder(x * math.cos(a) - y * math.sin(a) + desc['translation'][0], 1.) * size
            v = np.remainder(x * math.sin(a) + y * math.cos(a) + desc['translation'][1], 1.) * size
            texture = cv2.remap(source, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            texture = np.rint(linear_to_srgb(srgb_to_linear(texture.astype(np.float32) / 255.) *
                                           np.asarray(desc['tint'])) * 255).astype(np.uint8)
        else:
            color = linear_to_srgb(np.array(entry['linear_color']) * desc['tint'])
            texture = np.empty((size, size, 3), np.uint8)
            texture[:] = np.rint(color * 255).astype(np.uint8)
        self.textures[key] = texture
        return texture

    def attach(self, gym, sim, gymapi, env, door_actor, door, args):
        index = int(args.parallel_env_id)
        actors = list(range(gym.get_actor_count(env)))
        before = physics_signature(gym, env, actors)
        descriptions = self.environment_materials(index)
        handles = {}
        if self.config['materials']['enabled']:
            per_visual = self.config['materials'].get('preserve_subvisual_materials', False)
            if per_visual and not hasattr(self, 'per_visual_door_materials'):
                raise RuntimeError('Per-visual door materials were not prepared before asset loading')
            for g in (() if per_visual else ('frame', 'panel', 'handle')):
                rgb = self.texture(descriptions[g])
                rgba = np.concatenate((rgb, np.full((*rgb.shape[:2], 1), 255, np.uint8)), axis=-1)
                handles[g] = gym.create_texture_from_buffer(sim, rgb.shape[1], rgb.shape[0],
                                                            np.ascontiguousarray(rgba.reshape(rgb.shape[0], -1)))
                if handles[g] == gymapi.INVALID_HANDLE:
                    raise RuntimeError('Gym texture allocation failed')
            for actor in actors:
                name = gym.get_actor_name(env, actor)
                if per_visual and actor == door_actor:
                    # Whole-body setters replace ALL submesh materials, including
                    # glass, seals and metal. The wood textures are already bound.
                    continue
                if actor != door_actor and not name.startswith('door_side_wall_'):
                    continue
                if actor != door_actor and self.config.get('scene', {}).get('enabled'):
                    # Each visual submesh has its own MTL texture; don't overwrite floor with wall texture.
                    assert gym.get_actor_rigid_shape_count(env, actor) == 0
                    continue
                if per_visual:
                    # No whole-body door handles were allocated in this mode.
                    continue
                for body in range(gym.get_actor_rigid_body_count(env, actor)):
                    group = ('handle' if body == door.handle_body_index else
                             'panel' if body == door.door_body_index else 'frame') if actor == door_actor else 'frame'
                    gym.set_rigid_body_color(env, actor, body, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
                    gym.set_rigid_body_texture(env, actor, body, gymapi.MESH_VISUAL, handles[group])
        after = physics_signature(gym, env, actors)
        if before != after:
            raise RuntimeError('Visual-only invariant violated: actor physics changed')
        self.envs[index] = descriptions
        args.rgb_visual_randomization = {k: v for k, v in self.metadata.items() if k != 'seed'}
        args.rgb_visual_episode = dict(seed=self.seed, env_id=index, materials=descriptions,
                                      physics_before=before, physics_after=after, physics_signature_version='named_fields_v2')
        if hasattr(self, 'per_visual_door_materials'):
            args.rgb_visual_episode['per_visual_door_materials'] = self.per_visual_door_materials
        if index in self.scene_geometry:
            args.rgb_visual_episode['indoor_room'] = self.scene_geometry[index]
        if self.preview:
            self.preview.mkdir(parents=True, exist_ok=True)
            (self.preview / f'physics_env{index:02d}.json').write_text(json.dumps(
                physics_properties(gym, env, actors), indent=2))

    def update(self, gym, sim, gymapi, step, dt):
        self.frame = int(step)
        slot = (0 if self.config['lighting'].get('temporal_mode') == 'episode' else
                int(math.floor((step * dt + 1e-8) / self.config['lighting']['interval_s'])))
        if slot == self.light_slot:
            return
        self.light_slot = slot
        r = seeded(self.seed, 743, slot)
        self.sky_index = int(r.integers(len(self.manifest['hdr'])))
        self.intensity = float(r.uniform(*self.config['lighting']['intensity']))
        self.rotation = math.radians(float(r.uniform(*self.config['lighting']['rotation_deg'])))
        if self.sky_index not in self.skies:
            self.skies[self.sky_index] = cv2.cvtColor(cv2.imread(str(self.assets /
                self.manifest['hdr'][self.sky_index]['texture']), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.sky = self.skies[self.sky_index]
        if not self.config['lighting']['enabled']:
            return
        # Mean radiance -> ambient plus three brightest separated HDR sectors.
        # Gym has globally shared directional lights, not per-environment PBR domes.
        small = cv2.resize(self.sky, (24, 12), interpolation=cv2.INTER_AREA)
        weights = np.sin((np.arange(12) + .5) * math.pi / 12)[:, None, None]
        mean = (small * weights).sum((0, 1)) / (weights.sum() * 24)
        mean = mean / max(float(mean.max()), 1e-6)
        neutral = float(self.config['lighting'].get('neutral_color_mix', 0))
        mean = mean * (1 - neutral) + neutral
        scale = self.intensity / self.config['lighting']['gym_intensity_reference']
        if self.config['lighting'].get('gym_hemisphere_key_fill'):
            # Bounded Gym-native key/fill lighting. HDR yaw/tint vary by episode;
            # unlike brightest-HDR-lobe extraction, lights cannot end up below the floor.
            for k, strength in enumerate((.65, .30, .20, .0)):
                angle = self.rotation + k * 2 * math.pi / 3
                direction = [.6*math.cos(angle), .6*math.sin(angle), .8]
                color = np.clip(mean * scale * strength, 0, 1)
                ambient = mean * .45
                gym.set_light_parameters(sim, k, gymapi.Vec3(*map(float, color)),
                    gymapi.Vec3(*map(float, ambient)), gymapi.Vec3(*map(float, direction)))
            return
        bright = small.mean(-1).copy()
        for k in range(3):
            y, x = np.unravel_index(bright.argmax(), bright.shape)
            azimuth = ((x + .5) / 24 * 2 * math.pi) + self.rotation
            elevation = (y + .5) / 12 * math.pi
            direction = [-math.sin(elevation) * math.cos(azimuth),
                         -math.sin(elevation) * math.sin(azimuth), -math.cos(elevation)]
            tint = small[y, x] / max(float(small[y, x].max()), 1e-6)
            color = ((tint * (1 - neutral) + neutral) * scale / 3 *
                     self.config['lighting']['gym_directional_gain'])
            ambient = mean * scale * self.config['lighting']['gym_ambient_gain'] if k == 0 else np.zeros(3)
            gym.set_light_parameters(sim, k, gymapi.Vec3(*map(float, color)),
                                     gymapi.Vec3(*map(float, ambient)), gymapi.Vec3(*map(float, direction)))
            bright[:, max(0, x - 3):min(24, x + 4)] = -1
        gym.set_light_parameters(sim, 3, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, -1))

    def composite(self, gym, sim, env, camera, rgb, args, gymapi):
        """Recolor existing ground and fill empty RGB sky; never alter depth or masks."""
        if not (self.config['ground_rgb_projection'] or self.config['hdr_background_rgb']):
            return rgb
        h, w = rgb.shape[:2]
        projection = np.asarray(gym.get_camera_proj_matrix(sim, env, camera))
        view_inv = np.linalg.inv(np.asarray(gym.get_camera_view_matrix(sim, env, camera)))
        key = (h, w, float(projection[0, 0]), float(projection[1, 1]))
        if key not in self.rays:
            v, u = np.mgrid[0:h, 0:w].astype(np.float32)
            self.rays[key] = np.stack(((2 * u / w - 1) / projection[0, 0],
                                     (1 - 2 * v / h) / projection[1, 1], -np.ones((h, w))), -1)
        rays = self.rays[key] @ view_inv[:3, :3]
        origin = view_inv[3, :3]
        depth = np.asarray(gym.get_camera_image(sim, env, camera, gymapi.IMAGE_DEPTH)).reshape(h, w)
        result = rgb.copy()
        if self.config['ground_rgb_projection'] and self.config['materials']['enabled']:
            t = np.divide(-origin[2], rays[..., 2], out=np.full((h, w), np.inf), where=abs(rays[..., 2]) > 1e-6)
            ground = np.isfinite(depth) & (t > 0) & (np.abs(-depth - t) < .004)
            if ground.any():
                xyz = origin + rays[ground] * t[ground, None]
                env_origin = gym.get_env_origin(env)
                tex = self.texture(self.envs[int(args.parallel_env_id)]['floor'])
                u = np.mod((xyz[:, 0] - env_origin.x), 1) * (tex.shape[1] - 1)
                v = np.mod((xyz[:, 1] - env_origin.y), 1) * (tex.shape[0] - 1)
                values = tex[v.astype(int), u.astype(int)].astype(np.float32)
                # Preserve rendered shading/shadows on the existing plane.
                shade = np.maximum(rgb[ground].mean(-1) / 150., .05)
                result[ground] = np.clip(values * shade[:, None], 0, 255).astype(np.uint8)
        if self.config['hdr_background_rgb']:
            background = ~np.isfinite(depth) | (depth < -1e6)
            if background.any():
                ray = rays[background]
                az = np.arctan2(ray[:, 1], ray[:, 0]) - self.rotation
                polar = np.arccos(np.clip(ray[:, 2] / np.linalg.norm(ray, axis=-1), -1, 1))
                u = np.mod(az / (2 * math.pi), 1) * (self.sky.shape[1] - 1)
                v = polar / math.pi * (self.sky.shape[0] - 1)
                linear = self.sky[v.astype(int), u.astype(int)] * self.intensity / 3000.
                result[background] = np.rint(linear_to_srgb(linear / (1 + linear)) * 255).astype(np.uint8)
        return result

    def process(self, gym, sim, env, camera, name, rgb, args, gymapi, remap):
        index = int(args.parallel_env_id)
        camera_key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        clean = remap(rgb)
        scene = remap(self.composite(gym, sim, env, camera, rgb, args, gymapi))
        parameter_frame = 0 if self.config['image_augmentation'].get('temporal_mode') == 'episode' else self.frame
        params = image_parameters(self.config['image_augmentation'], seeded(self.seed, 744, parameter_frame))
        image = augment_rgb(scene, params, seeded(self.seed, 745, self.frame, index, camera_key))
        key = (index, name, self.frame // 250)
        if self.preview and key not in self.preview_seen and index < 4:
            self.preview_seen.add(key)
            self.preview.mkdir(parents=True, exist_ok=True)
            stem = f'env{index:02d}_{name}_step{self.frame:05d}'
            cv2.imwrite(str(self.preview / (stem + '.jpg')), np.concatenate((clean, scene, image), 1)[..., ::-1])
            (self.preview / (stem + '.json')).write_text(json.dumps(dict(
                frame=self.frame, parameters=params, lighting_slot=self.light_slot,
                hdr=self.manifest['hdr'][self.sky_index], intensity=self.intensity,
                metadata=args.rgb_visual_randomization, episode=args.rgb_visual_episode), indent=2))
        return image


def initialize(args):
    mode = getattr(args, 'rgb_visual_mode', 'legacy')
    if mode == 'legacy':
        return
    if mode != 'doorman':
        raise ValueError('rgb_visual_mode must be legacy or doorman, got ' + repr(mode))
    args._rgb_visual_runtime = VisualRandomizer(args)
    print('[RGB DoorMan] independent visual RNG; physics/trajectory unchanged; Gym diffuse/HDR approximation', flush=True)


def attach(gym, sim, gymapi, env, actor, door, args):
    runtime = getattr(args, '_rgb_visual_runtime', None)
    if runtime is not None:
        runtime.attach(gym, sim, gymapi, env, actor, door, args)


def update(gym, sim, gymapi, args, step, dt):
    runtime = getattr(args, '_rgb_visual_runtime', None)
    if runtime is not None:
        runtime.update(gym, sim, gymapi, step, dt)
