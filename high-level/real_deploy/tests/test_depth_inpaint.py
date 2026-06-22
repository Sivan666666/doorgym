from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REAL_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(REAL_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOY_DIR))

from depth_inpaint import RGBGuidedDepthInpainter, _nearest_valid_initialize


def make_inpainter(**kwargs) -> RGBGuidedDepthInpainter:
    defaults = {
        "device": "cpu",
        "max_distance_px": 8,
        "iterations": 2,
        "rgb_sigma": 0.10,
        "output_size_hw": (32, 48),
        "depth_lower_m": 0.2,
        "depth_far_m": 1.5,
    }
    defaults.update(kwargs)
    return RGBGuidedDepthInpainter(**defaults)


def test_nearest_initialization_respects_max_distance() -> None:
    depth = np.zeros((25, 25), dtype=np.float32)
    depth[12, 12] = 0.75
    nearest, invalid, fillable, distance = _nearest_valid_initialize(depth, max_distance_px=5)

    assert nearest[12, 12] == depth[12, 12]
    assert fillable[12, 16]
    assert nearest[12, 16] == depth[12, 12]
    assert not fillable[0, 0]
    assert nearest[0, 0] == 0.0
    assert distance[0, 0] > 5
    assert invalid[0, 0]


def test_valid_measurements_are_bit_exact_and_far_holes_stay_zero() -> None:
    depth = np.zeros((1, 24, 32), dtype=np.float32)
    depth[0, 8:16, 10:22] = 0.6
    depth[0, 10, 12] = 0.1  # Real close measurement: black in policy, not a hole.
    color = np.full((1, 24, 32, 3), 127, dtype=np.uint8)
    original = depth.copy()

    result = make_inpainter(max_distance_px=4).process(depth, color)
    valid = original > 0

    assert np.array_equal(result.inpainted_depth_m[valid], original[valid])
    assert result.inpainted_depth_m[0, 0, 0] == 0.0
    assert result.policy_u8.shape == (1, 32, 48)
    # The 0.1m measured pixel remains a measurement and is not treated as a hole.
    assert not result.invalid_mask[0, 10, 12]
    assert result.inpainted_depth_m[0, 10, 12] == np.float32(0.1)


def test_close_measurements_are_not_used_as_hole_fill_seeds() -> None:
    height, width = 24, 32
    depth = np.zeros((1, height, width), dtype=np.float32)
    depth[0, 12, 10] = 0.1
    depth[0, 12, 18] = 0.8
    color = np.full((1, height, width, 3), 127, dtype=np.uint8)

    result = make_inpainter(
        max_distance_px=16,
        output_size_hw=(height, width),
        processing_downsample=1,
    ).process(depth, color)

    assert result.inpainted_depth_m[0, 12, 10] == np.float32(0.1)
    assert result.inpainted_depth_m[0, 12, 11] > 0.2


def test_thin_close_depth_crack_is_repaired_only_in_policy_image() -> None:
    height, width = 16, 20
    depth = np.full((1, height, width), 0.8, dtype=np.float32)
    depth[0, 3:13, 10] = 0.1
    color = np.full((1, height, width, 3), 127, dtype=np.uint8)
    base = np.full((1, height, width), 117, dtype=np.uint8)
    base[0, 3:13, 10] = 0
    output_invalid = np.zeros((1, height, width), dtype=bool)

    result = make_inpainter(
        output_size_hw=(height, width),
        processing_downsample=1,
    ).process(
        depth,
        color,
        base_policy_u8=base,
        base_invalid_mask=output_invalid,
    )

    assert np.all(result.inpainted_depth_m[0, 3:13, 10] == np.float32(0.1))
    assert np.all(result.policy_u8[0, 3:13, 10] > 0)


def test_rgb_edge_guides_opposite_depths_to_opposite_sides() -> None:
    height, width = 32, 48
    depth = np.zeros((1, height, width), dtype=np.float32)
    depth[:, :, :20] = 0.5
    depth[:, :, 28:] = 1.2
    color = np.zeros((1, height, width, 3), dtype=np.uint8)
    color[:, :, 24:] = 255

    result = make_inpainter(max_distance_px=8, iterations=4, output_size_hw=(height, width)).process(
        depth,
        color,
    )
    filled = result.inpainted_depth_m[0]

    assert filled[height // 2, 22] < 0.75
    assert filled[height // 2, 26] > 0.95
    assert np.array_equal(filled[:, :20], depth[0, :, :20])
    assert np.array_equal(filled[:, 28:], depth[0, :, 28:])


def test_batch_outputs_and_stats_cover_both_cameras() -> None:
    depth = np.full((2, 12, 16), 0.8, dtype=np.float32)
    depth[0, 4:7, 4:7] = 0.0
    depth[1, 2:5, 8:12] = np.nan
    color = np.zeros((2, 12, 16, 3), dtype=np.uint8)

    result = make_inpainter(output_size_hw=(12, 16)).process(depth, color)

    assert result.policy_u8.shape == (2, 12, 16)
    assert len(result.stats["per_camera"]) == 2
    assert result.stats["batch_size"] == 2
    assert result.stats["timing"]["samples"] == 1
    assert np.all(result.inpainted_depth_m[np.isfinite(depth) & (depth > 0)] == 0.8)


def test_supplied_raw_invalid_mask_overrides_filtered_nonzero_value() -> None:
    depth = np.full((1, 16, 20), 0.7, dtype=np.float32)
    depth[0, 8, 10] = 4.0  # Simulate a temporal-filter value at a raw invalid pixel.
    raw_invalid = np.zeros_like(depth, dtype=bool)
    raw_invalid[0, 8, 10] = True
    color = np.zeros((1, 16, 20, 3), dtype=np.uint8)

    result = make_inpainter(output_size_hw=(16, 20)).process(
        depth,
        color,
        invalid_mask=raw_invalid,
    )

    assert result.invalid_mask[0, 8, 10]
    assert np.isclose(result.inpainted_depth_m[0, 8, 10], 0.7, atol=1.0e-3)
    assert np.array_equal(
        result.inpainted_depth_m[~raw_invalid],
        depth[~raw_invalid],
    )


def test_torch_policy_path_preserves_base_valid_pixels() -> None:
    import torch

    depth = np.full((2, 16, 20), 0.7, dtype=np.float32)
    raw_invalid = np.zeros_like(depth, dtype=bool)
    raw_invalid[:, 6:10, 8:12] = True
    depth[raw_invalid] = 0.0
    color = np.zeros((2, 16, 20, 3), dtype=np.uint8)
    base = np.full((2, 16, 20), 73, dtype=np.uint8)
    output_invalid = raw_invalid.copy()

    result = make_inpainter(output_size_hw=(16, 20)).process(
        depth,
        color,
        invalid_mask=raw_invalid,
        base_policy_u8=base,
        base_invalid_mask=output_invalid,
        return_torch_policy=True,
        return_debug=False,
    )

    assert isinstance(result.policy_u8, torch.Tensor)
    assert tuple(result.policy_u8.shape) == (2, 16, 20)
    policy = result.policy_u8.cpu().numpy()
    assert np.all(policy[~output_invalid] == 73)
    assert np.all(policy[output_invalid] > 0)
