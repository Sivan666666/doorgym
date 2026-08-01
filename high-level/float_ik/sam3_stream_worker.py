#!/usr/bin/env python3
"""Persistent SAM3 text-prompt image segmentation worker.

The worker receives RGB-only ``npz`` files over stdin and writes a selected
binary mask plus JSON metadata.  It intentionally never receives simulator
segmentation or object pose data, so those signals cannot leak into the mask
used by FoundationPose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt", default="door handle")
    parser.add_argument("--confidence_threshold", type=float, default=0.005)
    parser.add_argument("--min_mask_pixels", type=int, default=20)
    parser.add_argument("--max_mask_pixels", type=int, default=10000)
    parser.add_argument(
        "--selection",
        choices=(
            "top_score",
            "wc4_front_roi",
            "wc4_model_free_ref_roi",
            "wc4_initial_handle_roi",
        ),
        default="wc4_front_roi",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def choose_candidate(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    selection: str,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int | None, str]:
    if len(scores) == 0:
        return None, "no_text_candidate"
    if selection == "top_score":
        return int(np.argmax(scores)), "top_score"

    height, width = masks.shape[-2:]
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    center_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    center_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    valid_area = (areas >= min_pixels) & (areas <= max_pixels)

    if selection == "wc4_initial_handle_roi":
        # During domain-randomized approach rollouts the handle may move from
        # the right edge toward the image center.  Select a compact,
        # predominantly horizontal text candidate in the lower interaction
        # band without consuming simulator pose or segmentation.
        aspect = box_width / np.maximum(box_height, 1.0)
        initial = np.flatnonzero(
            valid_area
            & (center_y >= 0.45 * height)
            & (center_y <= 0.98 * height)
            & (box_width >= 0.025 * width)
            & (box_width <= 0.40 * width)
            & (box_height <= 0.30 * height)
            & (aspect >= 0.75)
        )
        if len(initial):
            geometry = np.clip(aspect[initial], 0.75, 3.0)
            geometry *= np.exp(
                -0.5 * ((center_y[initial] - 0.72 * height) / (0.25 * height)) ** 2
            )
            return int(initial[np.argmax(scores[initial] * geometry)]), "wc4_initial_handle_roi"
        return None, "no_initial_handle_roi_candidate"

    if selection == "wc4_model_free_ref_roi":
        # The earlier dense-track reference capture used a different front
        # camera composition: the handle starts in the lower-right and moves
        # upward as the robot approaches the door.
        # This fixed image-space prior selects among SAM3 text candidates and
        # does not consume simulator segmentation or object pose.
        reference = np.flatnonzero(
            valid_area
            & (center_x >= 0.72 * width)
            & (boxes[:, 2] >= 0.90 * width)
            & (center_y >= 0.50 * height)
            & (center_y <= 0.96 * height)
            & (box_width >= 0.025 * width)
            & (box_width <= 0.30 * width)
            & (box_height <= 0.25 * height)
        )
        if len(reference):
            geometry = np.exp(-0.5 * ((center_x[reference] - 0.88 * width) / (0.12 * width)) ** 2)
            geometry *= np.exp(-0.5 * ((width - boxes[reference, 2]) / (0.08 * width)) ** 2)
            geometry *= np.clip(
                box_width[reference] / np.maximum(box_height[reference], 1.0),
                0.5,
                3.0,
            )
            return int(reference[np.argmax(scores[reference] * geometry)]), "wc4_model_free_ref_roi"
        return None, "no_model_free_ref_roi_candidate"

    # WC4's front camera sees the handle in the right-hand interaction band.
    # This is a fixed image-space prior, not a projected simulator pose or GT
    # mask.  The strict pass rejects the common false positive on the top door
    # frame while retaining thin, partially clipped handles.
    strict = np.flatnonzero(
        valid_area
        & (boxes[:, 2] >= 0.94 * width)
        & (center_y >= 0.35 * height)
        & (center_y <= 0.75 * height)
        & (box_height <= 0.20 * height)
    )
    if len(strict):
        return int(strict[np.argmax(scores[strict])]), "wc4_front_roi_strict"

    broad = np.flatnonzero(
        valid_area
        & (boxes[:, 2] >= 0.92 * width)
        & (center_y >= 0.32 * height)
        & (center_y <= 0.78 * height)
        & (box_height <= 0.35 * height)
    )
    if len(broad):
        geometry = np.exp(-0.5 * ((center_y[broad] - 0.52 * height) / (0.18 * height)) ** 2)
        geometry *= np.exp(-0.5 * ((width - boxes[broad, 2]) / (0.08 * width)) ** 2)
        geometry *= np.minimum(1.0, box_width[broad] / np.maximum(box_height[broad], 1.0))
        return int(broad[np.argmax(scores[broad] * geometry)]), "wc4_front_roi_fallback"
    return None, "no_roi_candidate"


def main() -> None:
    args = parse_args()
    root = args.sam3_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(root))
    os.chdir(root)

    import torch
    from PIL import Image, ImageDraw
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 worker requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(
        model,
        device="cuda",
        confidence_threshold=float(args.confidence_threshold),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    atomic_json(
        args.output_dir / "worker_ready.json",
        {
            "status": "ready",
            "checkpoint": str(checkpoint),
            "prompt": args.prompt,
            "selection": args.selection,
        },
    )

    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast.__enter__()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            if request.get("command") == "shutdown":
                break
            frame_id = int(request["frame_id"])
            result_path = Path(request["result_path"])
            try:
                with np.load(request["input_path"]) as frame:
                    rgb = np.asarray(frame["rgb"], dtype=np.uint8)
                state = processor.set_image(Image.fromarray(rgb, mode="RGB"))
                output = processor.set_text_prompt(prompt=args.prompt, state=state)
                masks = output["masks"].detach().cpu().numpy().reshape((-1, *rgb.shape[:2])).astype(bool)
                boxes = output["boxes"].detach().float().cpu().numpy().reshape((-1, 4))
                scores = output["scores"].detach().float().cpu().numpy().reshape((-1,))
                selected, selection_reason = choose_candidate(
                    masks,
                    boxes,
                    scores,
                    args.selection,
                    int(args.min_mask_pixels),
                    int(args.max_mask_pixels),
                )
                if selected is None:
                    selected_mask = np.zeros(rgb.shape[:2], dtype=bool)
                    selected_box = None
                    selected_score = 0.0
                else:
                    selected_mask = masks[selected]
                    selected_box = boxes[selected].tolist()
                    selected_score = float(scores[selected])

                mask_path = mask_dir / f"frame_{frame_id:06d}.npy"
                np.save(mask_path, selected_mask.astype(np.uint8))
                overlay = Image.fromarray(rgb, mode="RGB")
                pixels = np.asarray(overlay, dtype=np.float32).copy()
                pixels[selected_mask] = 0.45 * pixels[selected_mask] + 0.55 * np.asarray([255, 32, 32])
                overlay = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(overlay)
                if selected_box is not None:
                    draw.rectangle(tuple(selected_box), outline=(255, 255, 0), width=3)
                draw.text(
                    (8, 8),
                    f"SAM3 {args.prompt}: {selected_score:.4f} {selection_reason}",
                    fill=(255, 255, 0),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
                overlay_path = overlay_dir / f"frame_{frame_id:06d}.png"
                overlay.save(overlay_path)
                candidates = [
                    {
                        "index": int(index),
                        "score": float(score),
                        "mask_pixels": int(mask.sum()),
                        "box_xyxy": [float(value) for value in box],
                    }
                    for index, (mask, box, score) in enumerate(zip(masks, boxes, scores, strict=True))
                ]
                atomic_json(
                    result_path,
                    {
                        "status": "ok" if int(selected_mask.sum()) >= args.min_mask_pixels else "no_mask",
                        "frame_id": frame_id,
                        "prompt": args.prompt,
                        "selection": args.selection,
                        "selection_reason": selection_reason,
                        "selected_index": selected,
                        "selected_score": selected_score,
                        "selected_box_xyxy": selected_box,
                        "mask_pixels": int(selected_mask.sum()),
                        "mask_path": str(mask_path),
                        "overlay_path": str(overlay_path),
                        "candidate_count": int(len(scores)),
                        "candidates": candidates,
                    },
                )
                del state, output
            except Exception as exc:
                atomic_json(
                    result_path,
                    {
                        "status": "error",
                        "frame_id": frame_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        autocast.__exit__(None, None, None)


if __name__ == "__main__":
    main()
