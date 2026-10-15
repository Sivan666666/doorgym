"""Read-only comparison of old RGB50/200 and the new RGB500 dataset."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
PIPE = ARCHIVE / 'logs/0907_pull_rgb_joint9_pi05_evo'
ROOT = ARCHIVE / 'data/lerobot'


def name(n):
    return ('local/door_a2w_0907_wc_nomenkuang_fixed15_gripper080_rgb_500_joint9' if n == 500 else
            f'local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_{n}_joint9')


def table(n):
    keys = ['observation.state', 'action', 'episode_index', 'frame_index']
    return pa.concat_tables([pq.read_table(p, columns=keys) for p in sorted((ROOT / name(n) / 'data').rglob('*.parquet'))])


def main():
    torch.set_num_threads(4)
    full = table(500)
    assert len(full) == 650000
    report = dict(passed=False, exact_numeric_prefixes=[], exact_rgb_prefix_frames={})
    for n in (50, 200):
        old = table(n)
        assert len(old) == n * 1300
        for key in old.column_names:
            a = np.asarray(old[key].to_pylist())
            b = np.asarray(full[key].slice(0, len(old)).to_pylist())
            np.testing.assert_array_equal(a, b, err_msg=f'{n}: {key}')
        report['exact_numeric_prefixes'].append(n)
    new = LeRobotDataset(name(500), root=ROOT / name(500), video_backend='torchcodec')
    for n, indices in [(50, [0, 1299, 64999]), (200, [0, 65000, 129999, 259999])]:
        old = LeRobotDataset(name(n), root=ROOT / name(n), video_backend='torchcodec')
        for i in indices:
            a, b = old[i], new[i]
            for key in ('observation.images.front_rgb', 'observation.images.wrist_rgb'):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, msg=f'{n}/{i}/{key}')
            assert a['task'] == b['task'] == 'pull lever door open'
        report['exact_rgb_prefix_frames'][str(n)] = indices
    report.update(passed=True, total_episodes=500, total_frames=650000)
    (PIPE / 'dataset500_nesting_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
