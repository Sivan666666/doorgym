"""Audit same-seed recordings; extract real recorded RGB video, not a re-render."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def compare(a, b):
    if a.dtype.kind in 'bifuc' and b.dtype.kind in 'bifuc':
        same = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
        return dict(equal=bool(same), max_abs_diff=float(np.nanmax(abs(a.astype(float) - b.astype(float))))
                    if a.size and a.shape == b.shape else None)
    return dict(equal=json.dumps(plain(a), sort_keys=True) == json.dumps(plain(b), sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--legacy', required=True)
    p.add_argument('--visual', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.visual).glob('episode_*.npz'))
    assert len(files) == 2, files
    result = dict(legacy=args.legacy, visual=args.visual, episodes=[])
    for path in files:
        # Trusted local recorder outputs contain metadata object arrays.
        with np.load(path, allow_pickle=True) as new, np.load(Path(args.legacy) / path.name, allow_pickle=True) as old:
            assert new['state'].shape == new['action'].shape == (1300, 9)
            assert set(old.files) <= set(new.files)
            fields = {k: compare(old[k], new[k]) for k in old.files if k not in ('front_rgb', 'wrist_rgb')}
            meta = new['rgb_visual_episode'].item()
            assert meta['physics_before'] == meta['physics_after']
            summary = dict(episode=path.name, frames=len(new['state']), fields=fields,
                           extra_fields=sorted(set(new.files) - set(old.files)),
                           physics_properties_unchanged=True,
                           non_rgb_exact=all(v['equal'] for v in fields.values()), images={})
            streams = []
            for camera in ('front', 'wrist'):
                rgb, baseline = new[camera + '_rgb'], old[camera + '_rgb']
                assert rgb.shape == (1300, 480, 640, 3) and rgb.dtype == np.uint8
                means = rgb.mean((1, 2, 3))
                summary['images'][camera] = dict(shape=list(rgb.shape),
                    changed=not np.array_equal(rgb, baseline), mean=float(means.mean()),
                    frame_mean_min=float(means.min()), frame_mean_max=float(means.max()),
                    empty_frames=int(np.count_nonzero(np.max(rgb, axis=(1, 2, 3)) == 0)))
                for frame in (0, 150, 300, 500, 700):
                    cv2.imwrite(str(out / f'{path.stem}_{camera}_frame{frame:04d}_before_after.jpg'),
                                np.concatenate((baseline[frame], rgb[frame]), axis=1)[..., ::-1])
                if args.video:
                    # Keep video small; this is a preview, raw images remain 640x480.
                    streams.append(np.stack([cv2.resize(im, (480, 360)) for im in rgb[:900]]))
            if args.video:
                movie = out / (path.stem + '_front_wrist.mp4')
                writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), float(new['fps']), (960, 360))
                assert writer.isOpened()
                for left, right in zip(*streams):
                    writer.write(np.concatenate((left, right), axis=1)[..., ::-1])
                writer.release()
            result['episodes'].append(summary)
            print(path.name, 'non-RGB exact:', summary['non_rgb_exact'], flush=True)
    result['all_non_rgb_exact'] = all(e['non_rgb_exact'] for e in result['episodes'])
    (out / 'audit.json').write_text(json.dumps(result, indent=2))
    if not result['all_non_rgb_exact']:
        raise AssertionError('Numeric/metadata mismatch; inspect audit.json before claiming visual-only parity')


if __name__ == '__main__':
    main()
