"""Independent scalar-math rescore of all 64 complete closed-loop JSONL logs."""
import collections
import hashlib
import json
import math
from pathlib import Path
import time

OUT=Path('/mnt/sda/ps_archive/doorgym/high-level/dp/result/0913_pull_rgb_aug100_pi05_eval64')


def main(out=OUT,traversal_m=.8):
    summary=json.loads((out/'summary.json').read_text())
    assert summary['complete'] and summary['trials']==summary['total_trials']==64
    assert summary['opening_threshold_deg']==60 and summary['traversal_distance_m']==traversal_m
    rows=summary['trials_detail']
    assert {r['trial_id'] for r in rows}==set(range(64))
    assert len({r['physical_seed'] for r in rows})==len({r['visual_seed'] for r in rows})==64
    audits=[]
    for row in rows:
        opened=traversed=joint=False
        first_open=first_traverse=first_joint=None
        max_open=max_traverse=-math.inf
        digest=hashlib.sha256();count=0
        path=Path(row['log_path'])
        per_trial=json.loads((path.parent/'summary.json').read_text())
        assert per_trial['base_seed']==row['physical_seed']==715455575+row['trial_id']
        assert per_trial['pass_open_angle_deg']==60 and per_trial['pull_traversal_distance_m']==traversal_m
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line);r=json.loads(line)
                assert r['step']==count and r['controlled_env_id']==0
                for key in ('state','dp_action_raw'):
                    assert len(r[key])==9 and all(math.isfinite(float(v)) for v in r[key])
                angle=abs(math.degrees(float(r['door']['dof'][0])))
                geom=r['pull_task_geometry'];base=r['base']['xy'];door=geom['door_plane_xy']
                dx,dy=geom['traversal_direction_xy'][:2];norm=math.hypot(dx,dy)
                assert math.isfinite(norm) and norm>1e-8
                distance=((base[0]-door[0])*dx+(base[1]-door[1])*dy)/norm
                assert math.isfinite(angle) and math.isfinite(distance)
                max_open=max(max_open,angle);max_traverse=max(max_traverse,distance)
                if angle>=60:
                    opened=True
                    if first_open is None:first_open=count
                if distance>=traversal_m:
                    traversed=True
                    if first_traverse is None:first_traverse=count
                if angle>=60 and distance>=traversal_m:
                    joint=True
                    if first_joint is None:first_joint=count
                count+=1
        assert count==row['records']==2600
        assert (opened,traversed,joint)==(row['door_open_success'],row['traversal_success'],row['success'])
        assert (first_open,first_traverse,first_joint)==(row['first_door_open_step'],row['first_traversal_step'],row['first_success_step'])
        assert abs(max_open-row['max_open_deg'])<1e-6 and abs(max_traverse-row['max_pull_traversal_m'])<1e-6
        audits.append(dict(trial_id=row['trial_id'],log_sha256=digest.hexdigest(),records=count,
            opened=opened,traversed=traversed,joint=joint))
    totals={k:sum(r[k] for r in audits) for k in ('opened','traversed','joint')}
    assert (totals['opened'],totals['traversed'],totals['joint'])==tuple(summary[k] for k in
        ('door_open_success_count','traversal_success_count','success_count'))
    result=dict(passed=True,time=time.strftime('%F %T'),trials=64,records=sum(r['records'] for r in audits),
        totals=totals,failure_categories=dict(collections.Counter(r['failure_category'] for r in rows if not r['success'])),
        independent_scalar_rescore=True,per_trial=audits)
    temp=out/'final_audit.json.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(out/'final_audit.json')
    print(json.dumps({k:v for k,v in result.items() if k!='per_trial'},indent=2),flush=True)


if __name__=='__main__':main()
