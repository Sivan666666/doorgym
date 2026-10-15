"""Compare ACT wrapper weights, saved normalization and action outputs to training policy."""
import argparse
import copy
import json
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from dp.door_policy_backend import LeRobotActDoorPolicyBackend


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','wrapper','dataset','repo_id','out'): p.add_argument('--'+name,required=True)
    a=p.parse_args()
    official=ACTPolicy.from_pretrained(a.checkpoint,local_files_only=True,strict=True).cuda().eval()
    wrapped=LeRobotActDoorPolicyBackend.load(a.wrapper,device='cuda').eval()
    original_weights=official.state_dict(); wrapped_weights=wrapped.policy.state_dict()
    assert original_weights.keys()==wrapped_weights.keys()
    for key in original_weights:
        assert torch.equal(original_weights[key],wrapped_weights[key]), key
    pre,post=make_pre_post_processors(official.config,pretrained_path=a.checkpoint)
    dataset=LeRobotDataset(a.repo_id,root=a.dataset,video_backend='torchcodec')
    assert official.config.env_state_feature is None
    assert not official.config.plucker_conditioning and not official.config.interaction_state_conditioning
    # Dataset metadata includes camera poses; baseline ACT never consumes them.
    consumed=['observation.state',*official.config.image_features]
    rows=[]
    for index in (0,100,350,1299,259999,649999):
        sample=dataset[index]
        batch={k:sample[k].unsqueeze(0) for k in consumed}
        training=pre(copy.deepcopy(batch))
        deploy=wrapped.normalizer.normalize_batch(copy.deepcopy(batch),include_action=False)
        diffs={}
        for key in batch:
            torch.testing.assert_close(training[key],deploy[key],rtol=1e-6,atol=1e-6)
            diffs[key]=float((training[key]-deploy[key]).abs().max())
        with torch.inference_mode():
            prediction=post(official.predict_action_chunk(training))
            converted=wrapped.predict_action_chunks_from_batch(batch)
        assert prediction.shape==converted.shape==(1,100,9)
        torch.testing.assert_close(prediction.to(converted.device),converted,rtol=1e-5,atol=1e-5)
        rows.append(dict(frame=index,input_max_abs_error=diffs,
                         action_max_abs_error=float((prediction.to(converted.device)-converted).abs().max())))
    Path(a.out).write_text(json.dumps(dict(passed=True,identical_weight_tensors=len(original_weights),
        consumed_inputs=consumed,unused_metadata_inputs=sorted(set(official.config.input_features)-set(consumed)),
        rows=rows,criterion='strict weights; saved processor parity on all consumed inputs; real RGB/Joint9 samples; ACT deterministic inference'),indent=2))


if __name__=='__main__': main()
