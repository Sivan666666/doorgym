#!/usr/bin/env python3
"""Left-handle wrapper for the parallel float-IK push-door controller."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "isaacgym_float_ik_b1z1_basearn_push_door_parallel.py"


def main() -> None:
    defaults = [
        "--robot_y_alignment",
        "door_center",
        "--handle_rotate_right_distance",
        "-0.04",
        "--handle_rotate_down_distance",
        "0.06",
        "--handle_rotate_direction_sign",
        "1.0",
        "--draw_scripted_trajectory",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    spec = importlib.util.spec_from_file_location("ikpush_parallel_right_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
