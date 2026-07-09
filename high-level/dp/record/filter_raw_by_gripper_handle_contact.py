#!/usr/bin/env python3
"""Filter Door DP raw episodes by recorded gripper-vs-handle contact.

This script expects episodes recorded with:

    --record_gripper_handle_contact

Optionally during recording, `--filter_gripper_handle_contact` can discard bad
episodes immediately.  This offline tool is for auditing or copying already
recorded episodes that include the contact fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="", help="Directory containing episode_*.npz.")
    parser.add_argument(
        "--raw_episode",
        type=str,
        action="append",
        default=[],
        help="Specific raw episode path. Can be passed multiple times.",
    )
    parser.add_argument("--phase_names", type=str, default="close_gripper,rotate_handle")
    parser.add_argument("--min_frames", type=int, default=5)
    parser.add_argument("--require_both", dest="require_both", action="store_true", default=True)
    parser.add_argument("--no_require_both", dest="require_both", action="store_false")
    parser.add_argument("--copy_pass_to", type=str, default="", help="Optional output raw_root for passing episodes.")
    parser.add_argument("--copy_fail_to", type=str, default="", help="Optional output directory for failing episodes.")
    parser.add_argument("--jsonl", type=str, default="", help="Optional JSONL report path.")
    return parser.parse_args()


def episode_paths(args):
    paths = [Path(p).expanduser() for p in args.raw_episode]
    if args.raw_root:
        paths.extend(sorted(Path(args.raw_root).expanduser().glob("episode_*.npz")))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("Pass --raw_root or at least one --raw_episode.")
    return unique


def phase_mask(data, requested_phase_names):
    subtask = np.asarray(data["subtask_index"], dtype=np.int64).reshape(-1)
    names = [str(x) for x in data.get("phase_names", [])]
    phase_ids = {names.index(name) for name in requested_phase_names if name in names}
    if not phase_ids:
        return np.ones_like(subtask, dtype=bool)
    return np.asarray([int(x) in phase_ids for x in subtask], dtype=bool)


def evaluate_episode(path, phase_names, min_frames, require_both):
    with np.load(path, allow_pickle=True) as data:
        key = "gripper_handle_contact_both" if require_both else "gripper_handle_contact_any"
        if key not in data.files:
            return {
                "episode": str(path),
                "pass": False,
                "reason": f"missing_{key}",
                "has_contact_fields": False,
            }
        mask = phase_mask(data, phase_names)
        contact = np.asarray(data[key], dtype=np.float32).reshape(-1) > 0.5
        usable = contact & mask[: contact.shape[0]]
        contact_frames = int(np.count_nonzero(usable))
        score_total = 0.0
        score_max = 0.0
        if "gripper_handle_contact_score" in data.files:
            score = np.asarray(data["gripper_handle_contact_score"], dtype=np.float32).reshape(contact.shape[0], -1)
            score_window = score[mask[: score.shape[0]]]
            if score_window.size:
                score_total = float(np.sum(score_window))
                score_max = float(np.max(score_window))
        ok = contact_frames >= int(min_frames)
        return {
            "episode": str(path),
            "pass": bool(ok),
            "reason": "ok" if ok else "insufficient_gripper_handle_contact",
            "has_contact_fields": True,
            "phase_names": list(phase_names),
            "require_both": bool(require_both),
            "contact_frames": contact_frames,
            "min_frames": int(min_frames),
            "score_total": score_total,
            "score_max": score_max,
        }


def copy_episode(path, dst_root, next_index):
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / f"episode_{next_index:06d}.npz"
    shutil.copy2(path, dst)
    return dst


def main():
    args = parse_args()
    phases = [item.strip() for item in args.phase_names.split(",") if item.strip()]
    report_path = Path(args.jsonl).expanduser() if args.jsonl else None
    report_file = report_path.open("w", encoding="utf-8") if report_path else None
    pass_root = Path(args.copy_pass_to).expanduser() if args.copy_pass_to else None
    fail_root = Path(args.copy_fail_to).expanduser() if args.copy_fail_to else None
    pass_count = 0
    fail_count = 0
    missing_count = 0
    try:
        for path in episode_paths(args):
            result = evaluate_episode(path, phases, args.min_frames, args.require_both)
            if result["pass"]:
                if pass_root is not None:
                    result["copied_to"] = str(copy_episode(path, pass_root, pass_count))
                pass_count += 1
            else:
                if not result.get("has_contact_fields", False):
                    missing_count += 1
                if fail_root is not None:
                    result["copied_to"] = str(copy_episode(path, fail_root, fail_count))
                fail_count += 1
            line = json.dumps(result, ensure_ascii=False)
            print(line, flush=True)
            if report_file is not None:
                report_file.write(line + "\n")
        print(
            json.dumps(
                {
                    "summary": True,
                    "pass": pass_count,
                    "fail": fail_count,
                    "missing_contact_fields": missing_count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        if report_file is not None:
            report_file.close()


if __name__ == "__main__":
    main()
