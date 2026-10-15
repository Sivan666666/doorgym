"""Episode-stable lighting and DoorMan photometrics; never edit materials/poses."""
import hashlib
import importlib.util
import random
from common import digest, sha


def doorman(cfg):
    path = cfg['tools']['rgb_augmenter']
    spec = importlib.util.spec_from_file_location('wc_doorman_photometric', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def photo_plan(cfg, seed):
    settings = cfg['render'].get('image_augmentation', {'enabled': False})
    if not settings['enabled']:
        return dict(enabled=False, seed=seed, parameters={})
    assert settings['temporal_mode'] == 'episode', 'Avoid framewise parameter flicker'
    module = doorman(cfg)
    return dict(enabled=True, seed=seed, temporal_mode='episode',
                parameters=module.image_parameters(settings, module.seeded(seed, 744)),
                implementation_sha256=sha(cfg['tools']['rgb_augmenter']),
                reference='Existing DoorMan stable adapter; probabilities/ranges explicitly configured')


def apply_photo(module, rgb, plan, frame, view):
    if not plan['enabled']:
        return rgb
    key = int.from_bytes(hashlib.sha256(view.encode()).digest()[:4], 'little')
    return module.augment_rgb(rgb, plan['parameters'], module.seeded(plan['seed'], 745, frame, key))


def material_manifest(bpy):
    """Audit material values, shader sockets/links and referenced texture identity."""
    def value(v):
        if isinstance(v, (str, bool, int, float)):
            return v
        try:
            return list(v)
        except TypeError:
            return getattr(v, 'name', None)
    result = {}
    for m in sorted(bpy.data.materials, key=lambda m: m.name):
        item = dict(diffuse=list(m.diffuse_color), roughness=m.roughness,
                    metallic=m.metallic, use_nodes=m.use_nodes, nodes=[], links=[])
        if m.node_tree:
            for n in sorted(m.node_tree.nodes, key=lambda n: n.name):
                node = dict(name=n.name, type=n.bl_idname,
                            inputs={s.identifier: value(s.default_value) for s in n.inputs if hasattr(s, 'default_value')})
                if getattr(n, 'image', None):
                    im = n.image
                    node['image'] = dict(name=im.name, filepath=im.filepath, colorspace=im.colorspace_settings.name,
                        packed_sha256=hashlib.sha256(im.packed_file.data).hexdigest() if im.packed_file else None)
                item['nodes'].append(node)
            item['links'] = sorted((l.from_node.name, l.from_socket.identifier,
                                     l.to_node.name, l.to_socket.identifier) for l in m.node_tree.links)
        result[m.name] = item
    return result


def randomize_lighting(bpy, cfg, seed):
    settings = cfg.get('lighting', {'enabled': False})
    before = material_manifest(bpy)
    report = dict(enabled=settings['enabled'], seed=seed, temporal_mode='episode', lights=[], world=[])
    if settings['enabled']:
        assert not cfg.get('randomize', False), 'Combined room randomization changes materials'
        assert not (settings.get('point_only') and settings.get('ceiling_only'))
        rng = random.Random(seed)
        for obj in sorted(bpy.data.objects, key=lambda o: o.name):
            if obj.type != 'LIGHT':
                continue
            light = obj.data
            old_energy, old_color = light.energy, list(light.color)
            if settings.get('point_only'):
                light.energy = 0
                report['lights'].append(dict(name=obj.name, type=light.type, energy_before=old_energy,
                    energy_after=0, disabled_reason='point_only: all pre-existing lights disabled'))
                continue
            ceiling_only = settings.get('ceiling_only', False)
            if ceiling_only and obj.name not in settings['ceiling_light_names']:
                light.energy = 0
                report['lights'].append(dict(name=obj.name, type=light.type, energy_before=old_energy,
                    energy_after=0, disabled_reason='ceiling_only: studio/front/side lights disabled'))
                continue
            if ceiling_only:
                assert light.type == 'AREA' and obj.location.z > 3.0
                # Blender area lights emit along local -Z: physically downward from ceiling.
                obj.rotation_mode = 'XYZ'
                obj.rotation_euler = (0,0,0)
                light.shape = 'DISK'
                light.size = settings.get('ceiling_panel_size_m', 1.2)
            factor = rng.uniform(*settings['energy_multiplier_range'])
            # Neutral warm/cool shift in emitted light, NOT any surface material.
            warmth = rng.uniform(*settings['warmth_range'])
            tint = [1.0, 1.0 - abs(warmth)*0.2, 1.0-warmth] if warmth >= 0 else [1.0+warmth, 1.0-abs(warmth)*0.2, 1.0]
            light.energy = old_energy * factor
            light.color = [c*t for c,t in zip(old_color,tint)]
            report['lights'].append(dict(name=obj.name, type=light.type, energy_before=old_energy,
                energy_after=light.energy, multiplier=factor, color_before=old_color, color_after=list(light.color),
                location=list(obj.location), rotation_euler=list(obj.rotation_euler), size=getattr(light,'size',None)))
        if settings.get('point_only'):
            specs = settings['point_lights']
            assert len(specs) == 2 and len({v['name'] for v in specs}) == 2
            for spec in specs:
                assert spec['energy'] > 0 and spec['radius_m'] >= 0
                assert spec['name'] not in {o.name for o in bpy.data.objects}
                light = bpy.data.lights.new(spec['name'], 'POINT')
                light.energy = spec['energy']
                light.color = spec['color']
                light.shadow_soft_size = spec['radius_m']
                obj = bpy.data.objects.new(spec['name'], light)
                bpy.context.scene.collection.objects.link(obj)
                obj.location = spec['location']
                report['lights'].append(dict(name=obj.name,type='POINT',energy_before=0,
                    energy_after=light.energy,color_after=list(light.color),location=list(obj.location),
                    radius_m=light.shadow_soft_size,render_only=True))
            assert {l['name'] for l in report['lights'] if l['energy_after'] > 0} == {s['name'] for s in specs}
        world = bpy.context.scene.world
        if world and world.use_nodes:
            for node in sorted(world.node_tree.nodes, key=lambda n: n.name):
                if node.type == 'BACKGROUND' and not node.inputs['Strength'].is_linked:
                    old = node.inputs['Strength'].default_value
                    factor = rng.uniform(*settings['world_multiplier_range'])
                    node.inputs['Strength'].default_value = old * factor
                    report['world'].append(dict(name=node.name, strength_before=old,
                                                strength_after=old*factor, multiplier=factor))
        assert report['lights'], 'No scene lights to randomize'
        if settings.get('ceiling_only'):
            active = {l['name'] for l in report['lights'] if l['energy_after'] > 0}
            assert active == set(settings['ceiling_light_names']), active
    after = material_manifest(bpy)
    assert digest(before) == digest(after), 'Material changed during lighting augmentation'
    report.update(materials_unchanged=True, material_sha256=digest(after), material_count=len(after))
    return report
