"""Offline readback of downloaded portable LeRobot trial, including all labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a = p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim_data'))
    from common import read, save, transition, sha
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
    assert CODEBASE_VERSION == 'v3.0'
    dataset_path = a.root/'dataset'
    checksums = read(dataset_path/'file_sha256.json')
    for relative, expected in checksums.items():
        assert sha(dataset_path/relative) == expected, relative
    cfg = read(a.root/'config.json')
    ds = LeRobotDataset(cfg['dataset']['repo_id'],root=dataset_path,video_backend='pyav')
    n = cfg['batch']['episodes']
    assert ds.num_episodes == n and len(ds) == n*520
    checked = 0
    image_rows = []
    hashes = set()
    for eid in range(n):
        folder = a.root/'episodes'/f'{eid:06d}'
        status = read(folder/'status.json')
        raw = folder/Path(status['raw']).name/'trajectory.json'
        assert sha(raw) == status['raw_sha256']
        data = read(raw)
        rendered = read(folder/'render/render_complete.json')
        hashes.add(rendered['augmentation']['lighting']['material_sha256'])
        for record in rendered['frames']:
            state, action = transition(data,record['source_index'])
            row = ds.hf_dataset[checked]
            assert int(row['episode_index']) == eid
            np.testing.assert_allclose(np.asarray(row['observation.state']),state,atol=1e-7)
            np.testing.assert_allclose(np.asarray(row['action']),action,atol=1e-7)
            assert int(np.asarray(row['source_step']).item()) == record['source_index']
            checked += 1
        for index in (eid*520,eid*520+260,eid*520+519):
            sample = ds[index]
            for view in cfg['render']['views']:
                assert tuple(sample['observation.images.'+view].shape) == (3,480,640)
            image_rows.append(index)
    assert len(hashes)==1 and checked==n*520
    report = dict(passed=True,format=CODEBASE_VERSION,episodes=n,numeric_rows_checked=checked,
                  image_rows_read=image_rows,dataset_files_sha256_checked=len(checksums),
                  materials_identical=True, material_sha256=hashes.pop())
    save(a.root/'local_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
