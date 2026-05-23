#!/usr/bin/env python3
"""Length-prefixed pickle worker for LeRobot Door policy inference.

The parent Isaac Gym process can be Python 3.8, while LeRobot 0.4.x runs in a
Python>=3.10 environment.  This worker keeps LeRobot imports and model weights
inside that compatible process.
"""

from __future__ import annotations

import pickle
import struct
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*numpy\\.core.*")

from door_policy_backend import DoorPolicyController  # noqa: E402


def write_message(payload: Mapping[str, Any]) -> None:
    data = pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL)
    PROTO_OUT.write(struct.pack(">I", len(data)))
    PROTO_OUT.write(data)
    PROTO_OUT.flush()


def read_exact(stream, n_bytes: int) -> bytes:
    chunks = []
    remaining = int(n_bytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Parent closed stdin.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message() -> Mapping[str, Any]:
    header = read_exact(sys.stdin.buffer, 4)
    size = struct.unpack(">I", header)[0]
    return pickle.loads(read_exact(sys.stdin.buffer, size))


def metadata(controller: DoorPolicyController) -> dict[str, Any]:
    return {
        "config": controller.config,
        "vision_mode": controller.vision_mode,
        "action_frame": controller.action_frame,
        "image_keys": controller.image_keys,
        "obs_horizon": controller.obs_horizon,
        "pred_horizon": controller.pred_horizon,
        "action_horizon": controller.action_horizon,
        "action_dim": controller.action_dim,
        "device": str(controller.device),
    }


def ok(**payload: Any) -> None:
    payload["ok"] = True
    write_message(payload)


def fail(exc: BaseException) -> None:
    write_message({"ok": False, "error": traceback.format_exc()})


def main() -> None:
    controller = None
    while True:
        request = {}
        try:
            request = read_message()
            cmd = request.get("cmd")
            if cmd == "init":
                controller = DoorPolicyController(
                    request["checkpoint"],
                    device=request.get("device"),
                    num_inference_steps=request.get("num_inference_steps"),
                    action_horizon=request.get("action_horizon"),
                    noise_scheduler_type=request.get("noise_scheduler_type"),
                )
                ok(metadata=metadata(controller))
            elif cmd == "reset":
                controller.reset()
                ok()
            elif cmd == "reset_envs":
                env_ids = request.get("env_ids")
                controller.reset_envs(None if env_ids is None else [int(x) for x in env_ids])
                ok()
            elif cmd == "clear_action_queue":
                controller.action_queue.clear()
                ok()
            elif cmd == "append_observation":
                controller.append_observation(
                    np.asarray(request["state"], dtype=np.float32),
                    request["mask_rgb"],
                    request["masked_depth_rgb"],
                    request.get("front_mask_rgb"),
                    request.get("front_masked_depth_rgb"),
                )
                ok()
            elif cmd == "append_observation_for_env":
                controller.append_observation_for_env(
                    int(request["env_id"]),
                    np.asarray(request["state"], dtype=np.float32),
                    request["mask_rgb"],
                    request["masked_depth_rgb"],
                    request.get("front_mask_rgb"),
                    request.get("front_masked_depth_rgb"),
                )
                ok()
            elif cmd == "sample_action_chunk":
                noise = request.get("noise")
                if noise is not None and not isinstance(noise, torch.Tensor):
                    noise = torch.as_tensor(noise, dtype=torch.float32)
                controller.sample_action_chunk(noise=noise)
                ok()
            elif cmd == "act":
                action = controller.act(
                    np.asarray(request["state"], dtype=np.float32),
                    request["mask_rgb"],
                    request["masked_depth_rgb"],
                    request.get("front_mask_rgb"),
                    request.get("front_masked_depth_rgb"),
                )
                ok(action=np.asarray(action, dtype=np.float32).tolist())
            elif cmd == "act_batch":
                actions = controller.act_batch(
                    [int(x) for x in request["env_ids"]],
                    np.asarray(request["states"], dtype=np.float32),
                    request["mask_rgbs"],
                    request["masked_depth_rgbs"],
                    request.get("front_mask_rgbs"),
                    request.get("front_masked_depth_rgbs"),
                )
                ok(actions=np.asarray(actions, dtype=np.float32).tolist())
            elif cmd == "close":
                ok()
                return
            else:
                raise ValueError(f"Unknown worker command: {cmd!r}")
        except BaseException as exc:
            fail(exc)
            if request.get("cmd") == "init":
                return


if __name__ == "__main__":
    main()
