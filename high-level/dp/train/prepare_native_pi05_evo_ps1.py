"""Prepare metadata only and audit the unmodified hwk PI05Evo implementation."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

REPO = Path('/home/ps/workspace/txc/doorgym')
ARCHIVE = Path('/mnt/sda/ps_archive/doorgym/high-level')
PIPE = ARCHIVE / 'logs/0907_pull_rgb_joint9_pi05_evo'
BASE = ARCHIVE / 'dp/models/pi05_evo_native_base_0907'
SOURCE = REPO / 'high-level/dp/models/pi05_base_cached_localtokenizer'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--save-audit-checkpoint', action='store_true')
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.pi05_evo.configuration_pi05_evo import PI05EvoConfig
    from lerobot.policies.pi05_evo.modeling_pi05_evo import PI05EvoPolicy
    from lerobot.policies.pi05_evo import processor_pi05_evo
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot

    assert str(REPO / 'high-level/hwk/lerobot/src') in lerobot.__file__, lerobot.__file__
    BASE.mkdir(parents=True, exist_ok=True); PIPE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'config.json').read_text())
    fields = {f.name for f in dataclasses.fields(PI05EvoConfig)} | {'type'}
    cfg = {k: v for k, v in cfg.items() if k in fields}
    cfg['type'] = 'pi05_evo'; cfg['device'] = 'cuda'; cfg['push_to_hub'] = False
    (BASE / 'config.json').write_text(json.dumps(cfg, indent=2))
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        data = json.loads((SOURCE / name).read_text())
        for step in data['steps']:
            if step['registry_name'] == 'pi05_prepare_state_tokenizer_processor_step':
                step['registry_name'] = 'pi05_evo_prepare_state_tokenizer_processor_step'
                assert step['config'] == {}
        (BASE / name).write_text(json.dumps(data, indent=2))
    weights = BASE / 'model.safetensors'
    if not weights.exists():
        weights.symlink_to(SOURCE / 'model.safetensors')
    assert weights.resolve() == (SOURCE / 'model.safetensors').resolve()
    if args.prepare_only:
        return
    torch.set_num_threads(4); torch.manual_seed(1000)
    repo_id = 'local/door_a2w_0829_wc_nomenkuang_fixed15_gripper080_rgb_v1_50_joint9'
    root = ARCHIVE / 'data/lerobot' / repo_id
    dataset = LeRobotDataset(repo_id, root=root, video_backend='torchcodec',
                            delta_timestamps={'action': [i/25 for i in range(100)]})
    config = PI05EvoConfig(device='cuda', dtype='bfloat16', chunk_size=100, n_action_steps=50,
                           gradient_checkpointing=True, push_to_hub=False,
                           input_features={
                               'observation.state': PolicyFeature(FeatureType.STATE, (9,)),
                               **{k: PolicyFeature(FeatureType.VISUAL, (3, 480, 640)) for k in
                                  ('observation.images.wrist_rgb', 'observation.images.front_rgb')}},
                           output_features={'action': PolicyFeature(FeatureType.ACTION, (9,))})
    policy = PI05EvoPolicy.from_pretrained(BASE, config=config, strict=True).cuda()
    assert not hasattr(policy.model, 'state_proj')
    # Native from_pretrained catches loading exceptions. Explicitly verify every
    # tensor so this experiment cannot accidentally start from random weights.
    original = load_file(str(weights), device='cpu')
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    fixed = {k if k.startswith('model.') else 'model.' + k: v for k, v in fixed.items()}
    actual = policy.state_dict()
    assert set(actual) == set(fixed), (set(actual)-set(fixed), set(fixed)-set(actual))
    for k, v in actual.items():
        torch.testing.assert_close(v.cpu(), fixed[k].to(v.dtype), atol=0, rtol=0, msg=k)
    loaded_count = len(actual)
    del original, fixed, actual
    pre, post = make_pre_post_processors(config, pretrained_path=str(BASE),
        preprocessor_overrides={
            'normalizer_processor': {'stats': dataset.meta.stats,
                'features': {**config.input_features, **config.output_features},
                'norm_map': config.normalization_mapping},
            'device_processor': {'device': 'cuda'}},
        postprocessor_overrides={'device_processor': {'device': 'cuda'},
            'unnormalizer_processor': {'stats': dataset.meta.stats,
                'features': config.output_features, 'norm_map': config.normalization_mapping}})
    assert any(isinstance(s, processor_pi05_evo.Pi05EvoPrepareStateTokenizerProcessorStep) for s in pre.steps)
    frame = dataset[0]
    batch = pre(torch.utils.data.default_collate([frame]))
    policy.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    policy.zero_grad(set_to_none=True); policy.eval()
    with torch.no_grad():
        action = post(policy.predict_action_chunk(batch))
    assert action.shape == (1, 100, 9) and torch.isfinite(action).all(), action.shape
    if args.save_audit_checkpoint:
        audit = ARCHIVE / 'dp/models/pi05_evo_native_audit_0907/pretrained_model'
        assert not audit.exists(), audit
        policy.save_pretrained(audit)
        pre.save_pretrained(audit)
        post.save_pretrained(audit)
    files = [REPO / 'high-level/hwk/lerobot/src/lerobot/policies/pi05_evo' / name for name in
             ('modeling_pi05_evo.py', 'processor_pi05_evo.py', 'configuration_pi05_evo.py')]
    report = dict(passed=True, loaded_tensor_count=loaded_count, loss=loss.item(),
                  action_shape=list(action.shape), continuous_state_projection=False,
                  native_quantized_state_prompt=True, task=frame['task'],
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (PIPE / 'evo_native_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
