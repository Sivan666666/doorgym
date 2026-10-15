"""Prepare offline Gym-compatible RGB assets from DoorMan's exact material lists.

No simulator, URDF, collision, trajectory or model files are modified. MDL files
and fetched asset URLs/hashes are retained for provenance; PBR is NOT emulated.
NVIDIA assets retain their original terms (not relicensed by this adapter).
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def fetch(url):
    # Bound total transfer time, not only socket inactivity, for large HDRs.
    result = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
        '--connect-timeout', '15', '--max-time', '180', '--retry', '2', url], capture_output=True)
    if result.returncode:
        raise RuntimeError('Asset download failed: ' + url + '\n' + result.stderr.decode(errors='replace'))
    return result.stdout


def material_lists(source):
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    'textured_material_list', 'random_color_material_list', 'metallic_material_list'):
                    found[t.id] = ast.literal_eval(node.value)
    assert len(found) == 3
    return found


def diffuse_parameters(text):
    result = {}
    for field, default in [('diffuse_color_constant', [.5, .5, .5]), ('diffuse_tint', [1., 1., 1.])]:
        match = re.search(r'\b' + field + r'\s*:\s*color\(([^)]+)\)', text)
        values = [float(x.strip().rstrip('f')) for x in match[1].split(',')] if match else default
        result['linear_color' if field == 'diffuse_color_constant' else 'linear_tint'] = values * 3 if len(values) == 1 else values
        result[field + '_explicit'] = bool(match)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--doorman_repo', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--hdr_per_folder', type=int, default=0, help='0: entire upstream HDR pool; >0: explicit preview subset')
    p.add_argument('--offline_refresh', action='store_true', help='Refresh diffuse metadata from retained MDLs; no downloads')
    args = p.parse_args()
    repo, out = Path(args.doorman_repo), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = out / 'reference'
    reference.mkdir(exist_ok=True)
    if args.offline_refresh:
        manifest = json.loads((out / 'manifest.json').read_text())
        for entry in manifest['materials']:
            mdl = (reference / (entry['name'] + '.mdl')).read_bytes()
            assert hashlib.sha256(mdl).hexdigest() == entry['mdl_sha256']
            entry.update(diffuse_parameters(mdl.decode()))
        manifest['version'] = 2
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print('Refreshed diffuse tint metadata without modifying assets')
        return
    files = [
        'gr00t/rl/isaac_utils/playground/env_rand/door.py',
        'gr00t/rl/config/domain_rand/domain_rand_visual_ImageRand.yaml',
        'gr00t/rl/envs/legged_base_task/legged_robot_base.py',
        'gr00t/rl/isaac_utils/playground/utils/usd_utils.py',
        'gr00t/rl/isaac_utils/playground/scripts/domelight_randomization.py',
        'gr00t/rl/simulator/isaacsim/isaacsim.py', 'LICENSE']
    hashes = {}
    for name in files:
        data = (repo / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        (reference / Path(name).name).write_bytes(data)
    groups = material_lists((repo / files[0]).read_text())
    def material(item):
        group, url = item
        name = Path(urllib.parse.urlparse(url).path).stem
        mdl_path = reference / (name + '.mdl')
        mdl = mdl_path.read_bytes() if mdl_path.exists() else fetch(url)
        mdl_path.write_bytes(mdl)
        text = mdl.decode()
        entry = dict(name=name, group=group, url=url, mdl_sha256=hashlib.sha256(mdl).hexdigest())
        match = re.search(r'diffuse_texture\s*:\s*texture_2d\("([^"]+)"', text)
        entry.update(diffuse_parameters(text))
        if match:
            tex_url = urllib.parse.urljoin(url, match[1])
            target = out / (name + '.png')
            if not target.exists():
                data = fetch(tex_url)
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(tex_url)
                cv2.imwrite(str(target), cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA))
            entry.update(texture=target.name, texture_url=tex_url,
                         prepared_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print('material', name, flush=True)
        return entry
    items = [(group, url) for group, urls in groups.items() for url in urls]
    with ThreadPoolExecutor(max_workers=6) as pool:
        materials = list(pool.map(material, items))
    simulator = (repo / files[5]).read_text()
    light_block = simulator[simulator.index('texture_file_folders=['):]
    light_block = light_block[:light_block.index('],')]
    folders = re.findall(r'"(https://[^\"]+/)"', light_block)
    assert folders
    hdr_urls = []
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    for folder in folders:
        parsed = urllib.parse.urlparse(folder)
        prefix = parsed.path.lstrip('/')
        params = {'list-type': '2', 'prefix': prefix}
        urls = []
        while True:
            xml = ET.fromstring(fetch(f'{parsed.scheme}://{parsed.netloc}/?' + urllib.parse.urlencode(params)))
            urls.extend(f'{parsed.scheme}://{parsed.netloc}/' + e.text
                        for e in xml.findall('s:Contents/s:Key', ns) if e.text.lower().endswith('.hdr'))
            if xml.findtext('s:IsTruncated', namespaces=ns) != 'true':
                break
            params['continuation-token'] = xml.findtext('s:NextContinuationToken', namespaces=ns)
        hdr_urls.extend(sorted(urls)[:args.hdr_per_folder] if args.hdr_per_folder else sorted(urls))
    def hdr(url):
        token = hashlib.sha256(url.encode()).hexdigest()[:16]
        target = out / (token + '.hdr')
        if not target.exists():
            print('fetch HDR', url, flush=True)
            data = fetch(url)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(url)
            cv2.imwrite(str(target), cv2.resize(image, (1024, 512), interpolation=cv2.INTER_AREA))
        print('hdr', url.split('/')[-1], flush=True)
        return dict(url=url, texture=target.name, prepared_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        skies = list(pool.map(hdr, sorted(set(hdr_urls))))
    assert skies
    manifest = dict(version=2, source_repo=str(repo), source_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), source_hashes=hashes,
        materials=materials, hdr=skies, hdr_per_folder=args.hdr_per_folder,
        limitations=['MDL diffuse textures/colors only; no PBR roughness/normal/metallic reflection',
                     'HDR downsampled to1024x512; Gym light approximation, not dome illumination'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('READY', len(materials), 'materials', len(skies), 'HDRs')


if __name__ == '__main__':
    main()
