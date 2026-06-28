#!/usr/bin/env python
"""Extract and visualize keyframes for every raw Door DP episode.

The script processes episodes sequentially to keep memory usage bounded.  It is
safe to rerun: completed episode directories are skipped unless --overwrite is
specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VISUALIZER = SCRIPT_DIR / "visualize_door_raw_keyframes.py"
EXPECTED_OUTPUT_FILES = (
    "keyframes.csv",
    "keyframe_timeline.png",
    "keyframe_contact_sheet.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", required=True, help="Directory containing episode_*.npz files.")
    parser.add_argument(
        "--output_root",
        default="",
        help="Output root. Defaults to <raw_root>/keyframe_viz.",
    )
    parser.add_argument("--start_episode", type=int, default=None, help="Lowest episode index to process.")
    parser.add_argument("--end_episode", type=int, default=None, help="Highest episode index to process, inclusive.")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Maximum number of selected episodes to process; 0 means all.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate already complete episode outputs.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately if one episode fails.")
    parser.add_argument("--cols", type=int, default=3, help="Columns in each camera contact sheet.")
    parser.add_argument("--thumb_width", type=int, default=240, help="Camera thumbnail width.")
    parser.add_argument("--max_label_chars", type=int, default=64)
    return parser.parse_args()


def episode_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def selected_episode_paths(args: argparse.Namespace, raw_root: Path) -> list[Path]:
    paths = sorted(raw_root.glob("episode_*.npz"), key=episode_index)
    if args.start_episode is not None:
        paths = [path for path in paths if episode_index(path) >= int(args.start_episode)]
    if args.end_episode is not None:
        paths = [path for path in paths if episode_index(path) <= int(args.end_episode)]
    if int(args.max_episodes) > 0:
        paths = paths[: int(args.max_episodes)]
    return paths


def output_complete(output_dir: Path) -> bool:
    return all((output_dir / filename).is_file() for filename in EXPECTED_OUTPUT_FILES)


def read_keyframe_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_dataset_outputs(
    output_root: Path,
    raw_root: Path,
    episode_summaries: list[dict],
    failures: list[dict],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "keyframe_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "episode_index",
                "keyframe_count",
                "keyframe_indices",
                "output_dir",
                "status",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for summary in episode_summaries:
            writer.writerow(summary)

    all_keyframes_path = output_root / "all_keyframes.csv"
    with all_keyframes_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "episode_index", "idx", "phase", "name", "rule"])
        writer.writeheader()
        for summary in episode_summaries:
            for row in summary["keyframe_rows"]:
                writer.writerow(
                    {
                        "episode": summary["episode"],
                        "episode_index": summary["episode_index"],
                        "idx": row.get("idx", ""),
                        "phase": row.get("phase", ""),
                        "name": row.get("name", ""),
                        "rule": row.get("rule", ""),
                    }
                )

    counts = [int(summary["keyframe_count"]) for summary in episode_summaries]
    summary_payload = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "episodes_completed": len(episode_summaries),
        "episodes_failed": len(failures),
        "total_keyframes": int(sum(counts)),
        "keyframes_per_episode": {
            "min": int(min(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
            "mean": float(statistics.mean(counts)) if counts else 0.0,
            "median": float(statistics.median(counts)) if counts else 0.0,
        },
        "failures": failures,
    }
    with (output_root / "keyframe_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2)

    # keyframe_rows is useful while building all_keyframes.csv but is too large
    # and nested for the compact per-episode manifest.
    for summary in episode_summaries:
        summary.pop("keyframe_rows", None)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else raw_root / "keyframe_viz"
    )
    paths = selected_episode_paths(args, raw_root)
    if not paths:
        raise FileNotFoundError(f"No selected episode_*.npz files under {raw_root}")

    episode_summaries: list[dict] = []
    failures: list[dict] = []
    total = len(paths)
    print(f"raw_root: {raw_root}")
    print(f"output_root: {output_root}")
    print(f"selected episodes: {total}")

    for position, episode_path in enumerate(paths, start=1):
        index = episode_index(episode_path)
        episode_name = episode_path.stem
        output_dir = output_root / episode_name
        complete_before = output_complete(output_dir)
        status = "skipped_existing" if complete_before and not args.overwrite else "generated"
        print(f"\n[{position}/{total}] {episode_name}: {status}", flush=True)

        if status == "generated":
            cmd = [
                sys.executable,
                str(DEFAULT_VISUALIZER),
                "--raw_root",
                str(raw_root),
                "--episode",
                str(index),
                "--output_dir",
                str(output_dir),
                "--cols",
                str(args.cols),
                "--thumb_width",
                str(args.thumb_width),
                "--max_label_chars",
                str(args.max_label_chars),
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                failure = {
                    "episode": episode_name,
                    "episode_index": index,
                    "returncode": int(exc.returncode),
                }
                failures.append(failure)
                print(f"FAILED: {failure}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
                continue

        csv_path = output_dir / "keyframes.csv"
        if not output_complete(output_dir):
            failure = {
                "episode": episode_name,
                "episode_index": index,
                "error": "expected output files are incomplete",
            }
            failures.append(failure)
            print(f"FAILED: {failure}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise RuntimeError(failure["error"])
            continue

        rows = read_keyframe_rows(csv_path)
        indices = [int(row["idx"]) for row in rows]
        episode_summaries.append(
            {
                "episode": episode_name,
                "episode_index": index,
                "keyframe_count": len(rows),
                "keyframe_indices": " ".join(str(value) for value in indices),
                "output_dir": str(output_dir),
                "status": status,
                "keyframe_rows": rows,
            }
        )
        print(f"keyframes: {len(rows)} indices={indices}", flush=True)

    write_dataset_outputs(output_root, raw_root, episode_summaries, failures)
    print("\nDone")
    print(f"episodes completed: {len(episode_summaries)}/{total}")
    print(f"episodes failed: {len(failures)}")
    print(f"total keyframes: {sum(item['keyframe_count'] for item in episode_summaries)}")
    print(f"manifest: {output_root / 'keyframe_manifest.csv'}")
    print(f"all keyframes: {output_root / 'all_keyframes.csv'}")
    print(f"summary: {output_root / 'keyframe_summary.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
