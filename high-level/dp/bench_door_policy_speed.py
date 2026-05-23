import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from door_dp_common import IMAGE_HEIGHT, IMAGE_WIDTH  # noqa: E402
from door_policy_backend import DoorPolicyController  # noqa: E402


DEFAULT_DP_CKPT = REPO / "high-level/dp/logs/door-dp/lerodp_4door_ikpush/checkpoints/model_latest.pt"
DEFAULT_ACT_CKPT = REPO / "high-level/dp/logs/door-act/leroact_4door_ikpush/checkpoints/model_latest.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one Door DP/ACT policy inference call.")
    parser.add_argument("--dp_checkpoint", type=Path, default=DEFAULT_DP_CKPT)
    parser.add_argument("--act_checkpoint", type=Path, default=DEFAULT_ACT_CKPT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--act_warmup", type=int, default=5)
    parser.add_argument("--act_iters", type=int, default=20)
    return parser.parse_args()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_window(controller):
    state_dim = int(controller.config.get("state_dim", 73))
    state = np.zeros(state_dim, dtype=np.float32)
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    item = controller._make_item(state, image, image, image, image)
    return [item for _ in range(int(controller.obs_horizon))]


def measure(label, checkpoint, device, steps=None, scheduler=None, warmup=3, iters=10):
    kwargs = {"device": device}
    if steps is not None:
        kwargs["num_inference_steps"] = int(steps)
    if scheduler is not None:
        kwargs["noise_scheduler_type"] = scheduler

    controller = DoorPolicyController(checkpoint, **kwargs)
    window = make_window(controller)
    for _ in range(warmup):
        controller.predict_action_chunks_from_windows([window])
        cuda_sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        controller.predict_action_chunks_from_windows([window])
        cuda_sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(times) / len(times)
    print(
        f"{label}: mean={mean_ms:.2f} ms min={min(times):.2f} ms max={max(times):.2f} ms "
        f"obs_horizon={controller.obs_horizon} pred_horizon={controller.pred_horizon} "
        f"action_horizon={controller.action_horizon}",
        flush=True,
    )
    del controller, window
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    for steps in (100, 25, 10):
        for scheduler in ("DDPM", "DDIM"):
            measure(
                f"DP steps={steps} scheduler={scheduler}",
                args.dp_checkpoint,
                device=args.device,
                steps=steps,
                scheduler=scheduler,
                warmup=args.warmup,
                iters=args.iters,
            )
    measure(
        "ACT",
        args.act_checkpoint,
        device=args.device,
        warmup=args.act_warmup,
        iters=args.act_iters,
    )


if __name__ == "__main__":
    main()
