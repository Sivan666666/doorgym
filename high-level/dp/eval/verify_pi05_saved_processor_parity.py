"""GPU integration audit: official checkpoint vs exported Door inference, no simulation."""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door_policy_backend import (
    DoorPolicyChunkDataset,
    LeRobotPI05DoorPolicyBackend,
    load_lerobot_policy_normalizer_stats,
    validate_pi05_state_projection,
    import_lerobot_pi05_modules,
    validate_native_pi05_evo_weights,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--official_checkpoint", type=Path, required=True)
    parser.add_argument("--door_checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = args.official_checkpoint
    policy_type = json.loads((source / "config.json").read_text())["type"]
    modules = import_lerobot_pi05_modules(policy_type)
    PI05Config, PI05Policy = modules["PI05Config"], modules["PI05Policy"]
    if policy_type == "pi05_evo":
        from lerobot.policies.pi05_evo import processor_pi05_evo  # noqa: F401
    else:
        from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    dataset = DoorPolicyChunkDataset(args.dataset_root, args.repo_id, 100, vision_mode="rgb")
    frames = [dataset._frame(i) for i in (0, 300, 1000)]
    task = str(frames[0]["task"])
    raw = {key: torch.as_tensor(frames[0][key]).unsqueeze(0) for key in ["observation.state", *dataset.image_keys]}
    raw["task"] = [task]
    config = PreTrainedConfig.from_pretrained(source)
    assert isinstance(config, PI05Config)
    config.device = args.device
    config.gradient_checkpointing = False
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(source),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )
    reference_batch = pre(raw)
    reference_inputs = {k: v.detach().cpu().clone() for k, v in reference_batch.items() if isinstance(v, torch.Tensor)}
    # Include real states from three phases, exact quantile boundaries, and
    # out-of-range states to catch accidental clipping or different bin edges.
    stats = load_lerobot_policy_normalizer_stats(source)["observation.state"]
    states = [torch.as_tensor(f["observation.state"]) for f in frames]
    lo, hi = torch.as_tensor(stats["q01"]), torch.as_tensor(stats["q99"])
    states += [lo, hi, (lo + hi) / 2, lo - 2, hi + 2]
    boundaries = {"observation.state": torch.stack(states), "task": [task] * len(states)}
    boundary_reference = {k: v.cpu().clone() for k, v in pre(boundaries).items() if isinstance(v, torch.Tensor)}
    model = PI05Policy.from_pretrained(source, config=config, strict=True, local_files_only=True).to(args.device).eval()
    validate_pi05_state_projection(model, source)
    if policy_type == "pi05_evo":
        validate_native_pi05_evo_weights(model, source)
    torch.manual_seed(12345)
    with torch.no_grad():
        reference_actions = post(model.predict_action_chunk(reference_batch)).cpu().clone()
    assert torch.isfinite(reference_actions).all()
    del model, reference_batch
    gc.collect()
    torch.cuda.empty_cache()

    wrapped = LeRobotPI05DoorPolicyBackend.load(args.door_checkpoint, device=args.device)
    assert wrapped.task_prompt == task, (wrapped.task_prompt, task)
    continuous_state = bool(getattr(config, "state_conditioning", False))
    assert bool(getattr(wrapped.config, "state_conditioning", False)) == continuous_state
    got = wrapped._prepare_batch(raw)
    for key, value in reference_inputs.items():
        torch.testing.assert_close(got[key].cpu(), value, rtol=0, atol=0, msg=f"input mismatch: {key}")
    boundary_got = wrapped._prepare_batch(boundaries)
    for key, value in boundary_reference.items():
        torch.testing.assert_close(boundary_got[key].cpu(), value, rtol=0, atol=0, msg=f"boundary mismatch: {key}")
    torch.manual_seed(12345)
    result_actions = wrapped.predict_action_chunks_from_batch(raw).cpu()
    torch.testing.assert_close(result_actions, reference_actions, rtol=0, atol=0)
    assert result_actions.shape == (1, 100, 9)
    # Verify the extra state projection participates in policy inference.
    captured = []
    if continuous_state:
        hook = wrapped.policy.model.state_proj.register_forward_hook(lambda m, i, o: captured.append(i[0].shape))
        torch.manual_seed(12345)
        wrapped.predict_action_chunks_from_batch(raw)
        hook.remove()
        assert captured and captured[0] == (1, config.max_state_dim), captured
    elif policy_type == "pi05_evo":
        assert not hasattr(wrapped.policy.model, "state_proj")
    summary = {
        "passed": True, "official_checkpoint": str(source), "door_checkpoint": str(args.door_checkpoint),
        "policy_type": policy_type, "state_conditioning": continuous_state, "task_prompt": task,
        "tested_real_frame_indices": [0, 300, 1000],
        "boundary_state_cases": len(states), "input_tensor_parity": "rtol=0, atol=0",
        "action_parity": "rtol=0, atol=0", "max_action_error": float((result_actions - reference_actions).abs().max()),
        "action_shape": list(result_actions.shape), "state_projection_hook_shapes": [list(s) for s in captured],
        "prompt_processor": [{"class": type(step).__name__, "config": step.get_config()}
                             for step in wrapped.preprocessor.steps if "PrepareStateTokenizer" in type(step).__name__],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
