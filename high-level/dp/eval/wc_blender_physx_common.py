"""Small dependency-free protocol and evaluation metrics."""
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def wait_file(path, root, timeout=900):
    start = time.monotonic()
    while not Path(path).exists():
        for error in Path(root).glob('*_error.json'):
            raise RuntimeError(str(error) + ': ' + error.read_text())
        if time.monotonic()-start > timeout:
            raise TimeoutError(str(path))
        time.sleep(.03)


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes/total
    den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0., center-half), min(1., center+half)]


def score(rows, open_deg=60., traversal_m=.5):
    assert rows and all(math.isfinite(r[k]) for r in rows
                        for k in ('open_deg','traversal_m','handle_deg'))
    opened = max(r['open_deg'] for r in rows) >= open_deg
    passed = max(r['traversal_m'] for r in rows) >= traversal_m
    successful = [r for r in rows if r['open_deg'] >= open_deg and r['traversal_m'] >= traversal_m]
    collision = any(r['base_door_contact'] for r in rows)
    return dict(success=bool(successful), door_open_success=bool(opened),
                traversal_success=bool(passed), base_door_contact=collision,
                contact_free_success=bool(successful and not collision),
                first_success_step=successful[0].get('step') if successful else None,
                max_open_deg=max(r['open_deg'] for r in rows),
                max_traversal_m=max(r['traversal_m'] for r in rows),
                final_traversal_m=rows[-1]['traversal_m'],
                max_handle_deg=max(r['handle_deg'] for r in rows),
                unlocked=any(r['unlocked'] for r in rows))
