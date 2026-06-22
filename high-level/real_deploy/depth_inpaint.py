"""Realtime RGB-guided depth-hole filling for the Jetson deployment path.

The implementation is a bounded, realtime approximation of colorization-based
depth inpainting.  It never modifies a finite, positive sensor measurement:
only raw zero/NaN/Inf pixels within ``max_distance_px`` of a valid measurement
are initialized and refined.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class DepthInpaintBatchResult:
    inpainted_depth_m: np.ndarray | None
    resized_depth_m: np.ndarray | None
    policy_u8: Any
    nearest_depth_m: np.ndarray | None
    invalid_mask: np.ndarray
    fillable_mask: np.ndarray
    stats: dict[str, Any]


def _validate_inputs(depth_m: np.ndarray, color_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    color = np.asarray(color_rgb, dtype=np.uint8)
    if depth.ndim == 2:
        depth = depth[None, ...]
    if color.ndim == 3:
        color = color[None, ...]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape NHW or HW, got {depth.shape}.")
    if color.ndim != 4 or color.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape NHW3 or HW3, got {color.shape}.")
    if depth.shape[0] != color.shape[0] or depth.shape[1:3] != color.shape[1:3]:
        raise ValueError(f"Depth/RGB shapes do not match: {depth.shape}, {color.shape}.")
    return np.ascontiguousarray(depth), np.ascontiguousarray(color)


def _nearest_valid_initialize(
    depth_m: np.ndarray,
    max_distance_px: float,
    invalid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize bounded holes from their nearest valid depth measurement."""
    import cv2

    depth = np.asarray(depth_m, dtype=np.float32)
    sanitized = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    measured_valid = np.isfinite(depth) & (depth > 0.0)
    if invalid_mask is None:
        invalid = ~measured_valid
    else:
        supplied_invalid = np.asarray(invalid_mask, dtype=bool)
        if supplied_invalid.shape != depth.shape:
            raise ValueError(
                f"Invalid-mask shape {supplied_invalid.shape} does not match depth shape {depth.shape}."
            )
        invalid = supplied_invalid | ~measured_valid
    valid = ~invalid
    nearest = sanitized.copy()
    fillable = np.zeros_like(valid, dtype=bool)

    if not valid.any() or not invalid.any() or max_distance_px <= 0:
        nearest[invalid] = 0.0
        return nearest, invalid, fillable, np.full(depth.shape, np.inf, dtype=np.float32)

    # distanceTransform expects zero-valued seed pixels. DIST_LABEL_PIXEL gives
    # every valid seed a stable label, allowing its measured depth to be copied
    # to the nearest invalid pixel without an iterative CPU search.
    source = invalid.astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    labels = labels.astype(np.int32, copy=False)
    max_label = int(labels.max(initial=0))
    label_depth = np.zeros(max_label + 1, dtype=np.float32)
    valid_labels = labels[valid]
    label_depth[valid_labels] = sanitized[valid]
    nearest_candidate = label_depth[labels]
    fillable = invalid & (distance <= float(max_distance_px)) & (nearest_candidate > 0.0)
    nearest[invalid] = 0.0
    nearest[fillable] = nearest_candidate[fillable]
    return nearest, invalid, fillable, distance.astype(np.float32, copy=False)


def _bounded_fillable_from_invalid(
    invalid_mask: np.ndarray,
    max_distance_px: float,
    seed_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return invalid pixels that are close enough to any measured valid pixel."""
    import cv2

    invalid = np.asarray(invalid_mask, dtype=bool)
    if seed_valid_mask is None:
        seed_valid = ~invalid
    else:
        seed_valid = np.asarray(seed_valid_mask, dtype=bool)
        if seed_valid.shape != invalid.shape:
            raise ValueError(
                f"Seed-valid shape {seed_valid.shape} does not match invalid-mask shape {invalid.shape}."
            )
    fillable = np.zeros_like(invalid, dtype=bool)
    if max_distance_px <= 0 or not invalid.any() or not seed_valid.any():
        return fillable
    distance = cv2.distanceTransform((~seed_valid).astype(np.uint8), cv2.DIST_L2, 5)
    return invalid & (distance <= float(max_distance_px))


def _masked_resize_depth_np(
    depth_m: np.ndarray,
    active_mask: np.ndarray,
    size_wh: tuple[int, int],
) -> np.ndarray:
    """Resize depth without letting inactive zero pixels bleed into neighbors."""
    import cv2

    depth = np.asarray(depth_m, dtype=np.float32)
    active = np.asarray(active_mask, dtype=np.float32)
    numerator = cv2.resize(depth * active, size_wh, interpolation=cv2.INTER_LINEAR)
    denominator = cv2.resize(active, size_wh, interpolation=cv2.INTER_LINEAR)
    resized = numerator / np.maximum(denominator, 1.0e-6)
    resized[denominator <= 1.0e-6] = 0.0
    return resized.astype(np.float32, copy=False)


def _repair_black_cracks_u8_np(
    policy_u8: np.ndarray,
    editable_mask: np.ndarray,
    min_nonzero_neighbors: int = 5,
) -> np.ndarray:
    """Fill one-pixel black cracks while leaving large black regions untouched."""
    import cv2

    policy = np.asarray(policy_u8, dtype=np.uint8)
    editable = np.asarray(editable_mask, dtype=bool)
    repairable = (policy == 0) & editable
    if not repairable.any():
        return policy
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nonzero = (policy > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(
        nonzero,
        cv2.CV_16U,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    candidate = cv2.dilate(policy, np.ones((3, 3), dtype=np.uint8))
    repair = repairable & (neighbor_count >= int(min_nonzero_neighbors)) & (candidate > 0)
    if not repair.any():
        return policy
    output = policy.copy()
    output[repair] = candidate[repair]
    return output


def _fast_bounded_nearest_initialize(
    depth_m: np.ndarray,
    max_distance_px: float,
    invalid_mask: np.ndarray | None,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate nearest initialization cheaply while keeping an exact distance gate."""
    import cv2

    if downsample <= 1 or min(depth_m.shape) < downsample * 8:
        nearest, invalid, fillable, _ = _nearest_valid_initialize(
            depth_m,
            max_distance_px,
            invalid_mask=invalid_mask,
        )
        return nearest, invalid, fillable

    depth = np.asarray(depth_m, dtype=np.float32)
    sanitized = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    measured_valid = np.isfinite(depth) & (depth > 0.0)
    if invalid_mask is None:
        invalid = ~measured_valid
    else:
        supplied_invalid = np.asarray(invalid_mask, dtype=bool)
        if supplied_invalid.shape != depth.shape:
            raise ValueError(
                f"Invalid-mask shape {supplied_invalid.shape} does not match depth shape {depth.shape}."
            )
        invalid = supplied_invalid | ~measured_valid
    valid = ~invalid
    nearest = sanitized.copy()
    nearest[invalid] = 0.0
    fillable = np.zeros_like(valid, dtype=bool)
    if not valid.any() or not invalid.any() or max_distance_px <= 0:
        return nearest, invalid, fillable

    height, width = depth.shape
    low_width = max(1, (width + downsample - 1) // downsample)
    low_height = max(1, (height + downsample - 1) // downsample)
    valid_float = valid.astype(np.float32)
    low_valid_fraction = cv2.resize(
        valid_float,
        (low_width, low_height),
        interpolation=cv2.INTER_AREA,
    )
    low_depth_sum = cv2.resize(
        sanitized * valid_float,
        (low_width, low_height),
        interpolation=cv2.INTER_AREA,
    )
    low_depth = low_depth_sum / np.maximum(low_valid_fraction, 1.0e-6)
    low_depth[low_valid_fraction <= 0.0] = 0.0
    low_nearest, _, _, _ = _nearest_valid_initialize(
        low_depth,
        max_distance_px=max(low_height, low_width),
    )
    initialized = cv2.resize(
        low_nearest,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    distance = cv2.distanceTransform(invalid.astype(np.uint8), cv2.DIST_L2, 5)
    fillable = invalid & (distance <= float(max_distance_px)) & (initialized > 0.0)
    nearest[fillable] = initialized[fillable]
    return nearest, invalid, fillable


class RGBGuidedDepthInpainter:
    """Batch RGB-guided hole filler with bounded propagation and timing stats."""

    def __init__(
        self,
        device: str = "cuda:0",
        max_distance_px: float = 64.0,
        iterations: int = 2,
        rgb_sigma: float = 0.10,
        output_size_hw: tuple[int, int] = (480, 640),
        depth_lower_m: float = 0.2,
        depth_far_m: float = 1.5,
        processing_downsample: int = 4,
        timing_window: int = 512,
    ) -> None:
        import torch

        if max_distance_px < 0:
            raise ValueError("max_distance_px must be non-negative.")
        if iterations < 0:
            raise ValueError("iterations must be non-negative.")
        if rgb_sigma <= 0:
            raise ValueError("rgb_sigma must be positive.")
        if depth_far_m <= depth_lower_m:
            raise ValueError("depth_far_m must be greater than depth_lower_m.")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA inpainting requested on {device}, but CUDA is unavailable.")
        self.max_distance_px = float(max_distance_px)
        self.iterations = int(iterations)
        self.rgb_sigma = float(rgb_sigma)
        self.output_size_hw = (int(output_size_hw[0]), int(output_size_hw[1]))
        self.depth_lower_m = float(depth_lower_m)
        self.depth_far_m = float(depth_far_m)
        self.processing_downsample = max(1, int(processing_downsample))
        self._timings_ms: deque[float] = deque(maxlen=max(1, int(timing_window)))

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    @staticmethod
    def _shift(tensor, dy: int, dx: int):
        out = tensor.new_zeros(tensor.shape)
        height, width = tensor.shape[-2:]
        source_y = slice(max(0, dy), min(height, height + dy))
        target_y = slice(max(0, -dy), min(height, height - dy))
        source_x = slice(max(0, dx), min(width, width + dx))
        target_x = slice(max(0, -dx), min(width, width - dx))
        out[..., target_y, target_x] = tensor[..., source_y, source_x]
        return out

    def _refine(self, nearest, original, color, valid, fillable):
        torch = self.torch
        directions: Sequence[tuple[int, int]] = ((-1, 0), (1, 0), (0, -1), (0, 1))
        active = valid | fillable
        output = nearest
        weights = []
        for dy, dx in directions:
            neighbor_color = self._shift(color, dy, dx)
            color_delta = torch.mean(torch.abs(color - neighbor_color), dim=1, keepdim=True)
            weights.append(torch.exp(-color_delta / self.rgb_sigma))

        for _ in range(self.iterations):
            numerator = torch.zeros_like(output)
            denominator = torch.zeros_like(output)
            for weight, (dy, dx) in zip(weights, directions):
                neighbor_active = self._shift(active, dy, dx)
                weighted_active = weight * neighbor_active.to(weight.dtype)
                numerator.add_(weighted_active * self._shift(output, dy, dx))
                denominator.add_(weighted_active)
            candidate = numerator / denominator.clamp_min(1.0e-6)
            update = fillable & (denominator > 1.0e-6)
            output = torch.where(update, candidate, output)
            # This invariant is intentionally repeated after every iteration.
            output = torch.where(valid, original, output)
            output = torch.where(active, output, torch.zeros_like(output))
        return output

    def process(
        self,
        depth_m: np.ndarray,
        color_rgb: np.ndarray,
        invalid_mask: np.ndarray | None = None,
        base_policy_u8: np.ndarray | None = None,
        base_invalid_mask: np.ndarray | None = None,
        return_torch_policy: bool = False,
        return_debug: bool = True,
    ) -> DepthInpaintBatchResult:
        import cv2
        import torch.nn.functional as F

        if return_torch_policy and return_debug:
            raise ValueError("return_torch_policy and return_debug cannot both be enabled.")

        depth, color = _validate_inputs(depth_m, color_rgb)
        supplied_invalid = None
        if invalid_mask is not None:
            supplied_invalid = np.asarray(invalid_mask, dtype=bool)
            if supplied_invalid.ndim == 2:
                supplied_invalid = supplied_invalid[None, ...]
            if supplied_invalid.shape != depth.shape:
                raise ValueError(
                    f"Invalid-mask shape {supplied_invalid.shape} does not match depth shape {depth.shape}."
                )
        base_policy = None
        if base_policy_u8 is not None:
            base_policy = np.asarray(base_policy_u8, dtype=np.uint8)
            if base_policy.ndim == 4 and base_policy.shape[-1] in (1, 3):
                base_policy = base_policy[..., 0]
            if base_policy.ndim == 2:
                base_policy = base_policy[None, ...]
            expected = (depth.shape[0], *self.output_size_hw)
            if base_policy.shape != expected:
                raise ValueError(f"Base-policy shape {base_policy.shape} does not match expected {expected}.")
        output_invalid = None
        if base_invalid_mask is not None:
            output_invalid = np.asarray(base_invalid_mask, dtype=bool)
            if output_invalid.ndim == 2:
                output_invalid = output_invalid[None, ...]
            expected = (depth.shape[0], *self.output_size_hw)
            if output_invalid.shape != expected:
                raise ValueError(
                    f"Base invalid-mask shape {output_invalid.shape} does not match expected {expected}."
                )

        start = time.perf_counter()
        if supplied_invalid is None:
            sanitized = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            invalid_np = ~np.isfinite(depth) | (depth <= 0.0)
        else:
            # RealSense input is produced from uint16 depth multiplied by a
            # finite scale. The camera worker already computed the raw
            # zero/invalid mask, so another full nan_to_num/isfinite pass costs
            # ~10ms on NX without adding information.
            sanitized = depth
            invalid_np = supplied_invalid | (depth <= 0.0)
        valid_np = ~invalid_np
        seed_valid_np = valid_np & (sanitized >= self.depth_lower_m)
        mask_prep_ms = (time.perf_counter() - start) * 1000.0

        prep_start = time.perf_counter()
        work_depth = []
        work_color = []
        work_nearest = []
        work_valid = []
        work_fillable = []
        work_active = []
        full_fillable = []
        output_fillable = []
        factor = self.processing_downsample
        propagation_steps = max(0, int(np.ceil(self.max_distance_px / factor)))
        output_height, output_width = self.output_size_hw
        for index in range(depth.shape[0]):
            height, width = depth.shape[1:3]
            low_width = max(1, (width + factor - 1) // factor)
            low_height = max(1, (height + factor - 1) // factor)
            low_size_wh = (low_width, low_height)
            fillable_full = _bounded_fillable_from_invalid(
                invalid_np[index],
                self.max_distance_px,
                seed_valid_mask=seed_valid_np[index],
            )
            output_invalid_single = (
                output_invalid[index]
                if output_invalid is not None
                else cv2.resize(
                    invalid_np[index].astype(np.uint8),
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
            output_fillable_single = cv2.resize(
                fillable_full.astype(np.uint8),
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool) & output_invalid_single
            valid_float = seed_valid_np[index].astype(np.float32)
            low_valid_fraction = cv2.resize(
                valid_float,
                low_size_wh,
                interpolation=cv2.INTER_AREA,
            )
            low_depth_sum = cv2.resize(
                sanitized[index] * valid_float,
                low_size_wh,
                interpolation=cv2.INTER_AREA,
            )
            low_depth = low_depth_sum / np.maximum(low_valid_fraction, 1.0e-6)
            low_invalid = low_valid_fraction <= 0.0
            low_depth[low_invalid] = 0.0
            nearest, _, fillable, _ = _nearest_valid_initialize(
                low_depth,
                max_distance_px=self.max_distance_px / factor,
                invalid_mask=low_invalid,
            )
            low_active = (~low_invalid) | fillable
            work_depth.append(low_depth.astype(np.float32, copy=False))
            work_color.append(
                cv2.resize(color[index], low_size_wh, interpolation=cv2.INTER_AREA).astype(
                    np.uint8,
                    copy=False,
                )
            )
            work_nearest.append(nearest.astype(np.float32, copy=False))
            work_valid.append(~low_invalid)
            work_fillable.append(fillable)
            work_active.append(low_active)
            output_fillable.append(output_fillable_single)
            if return_debug:
                full_fillable.append(fillable_full)
        work_depth_np = np.stack(work_depth)
        work_color_np = np.stack(work_color)
        work_nearest_np = np.stack(work_nearest)
        work_valid_np = np.stack(work_valid)
        work_fillable_np = np.stack(work_fillable)
        work_active_np = np.stack(work_active)
        output_fillable_np = np.stack(output_fillable)
        fillable_np = np.stack(full_fillable) if return_debug else None
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        gpu_start = time.perf_counter()
        torch = self.torch
        compute_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        original_t = (
            torch.from_numpy(work_depth_np[:, None])
            .to(self.device, non_blocking=True)
            .to(compute_dtype)
        )
        nearest_t = (
            torch.from_numpy(work_nearest_np[:, None])
            .to(self.device, non_blocking=True)
            .to(compute_dtype)
        )
        valid_t = torch.from_numpy(work_valid_np[:, None]).to(self.device, non_blocking=True)
        fillable_t = torch.from_numpy(work_fillable_np[:, None]).to(self.device, non_blocking=True)
        color_t = (
            torch.from_numpy(np.ascontiguousarray(np.transpose(work_color_np, (0, 3, 1, 2))))
            .to(self.device, non_blocking=True)
            .to(compute_dtype)
            .div_(255.0)
        )
        refined_t = self._refine(
            nearest_t,
            original_t,
            color_t,
            valid_t,
            fillable_t,
        )
        policy_tensor = None
        fillable_counts = None
        refined_np = None
        active_low_t = torch.from_numpy(work_active_np[:, None]).to(self.device, non_blocking=True)
        if return_torch_policy:
            invalid_output_t = (
                torch.from_numpy(output_invalid[:, None]).to(self.device, non_blocking=True)
                if output_invalid is not None
                else F.interpolate(
                    torch.from_numpy(invalid_np[:, None])
                    .to(self.device, non_blocking=True)
                    .to(torch.float32),
                    size=self.output_size_hw,
                    mode="nearest",
                )
                > 0.5
            )
            fillable_output_t = torch.from_numpy(output_fillable_np[:, None]).to(
                self.device,
                non_blocking=True,
            )
            active_low_float_t = active_low_t.to(torch.float32)
            policy_denominator_t = F.interpolate(
                active_low_float_t,
                size=self.output_size_hw,
                mode="bilinear",
                align_corners=False,
            )
            policy_depth_t = F.interpolate(
                refined_t.to(torch.float32) * active_low_float_t,
                size=self.output_size_hw,
                mode="bilinear",
                align_corners=False,
            ) / policy_denominator_t.clamp_min(1.0e-6)
            policy_depth_t = torch.where(
                policy_denominator_t > 1.0e-6,
                policy_depth_t,
                torch.zeros_like(policy_depth_t),
            )
            guided_t = (
                (policy_depth_t.clamp(self.depth_lower_m, self.depth_far_m) - self.depth_lower_m)
                / (self.depth_far_m - self.depth_lower_m)
            ).clamp_(0.0, 1.0)
            guided_t = torch.where(
                policy_depth_t >= self.depth_lower_m,
                guided_t,
                torch.zeros_like(guided_t),
            )
            guided_u8_t = (guided_t * 255.0).to(torch.uint8)
            if base_policy is None:
                policy_tensor = torch.where(
                    invalid_output_t & ~fillable_output_t,
                    torch.zeros_like(guided_u8_t),
                    guided_u8_t,
                )
            else:
                base_policy_t = torch.from_numpy(base_policy[:, None]).to(self.device, non_blocking=True)
                policy_tensor = torch.where(fillable_output_t, guided_u8_t, base_policy_t)
                policy_tensor = torch.where(
                    invalid_output_t & ~fillable_output_t,
                    torch.zeros_like(policy_tensor),
                    policy_tensor,
                )
            editable_black_t = fillable_output_t | ~invalid_output_t
            nonzero_t = (policy_tensor > 0).to(torch.float32)
            neighbor_kernel_t = torch.ones(
                (1, 1, 3, 3),
                dtype=torch.float32,
                device=self.device,
            )
            neighbor_kernel_t[:, :, 1, 1] = 0.0
            nonzero_neighbor_count_t = F.conv2d(
                nonzero_t,
                neighbor_kernel_t,
                padding=1,
            )
            crack_candidate_t = F.max_pool2d(
                policy_tensor.to(torch.float32),
                kernel_size=3,
                stride=1,
                padding=1,
            ).to(torch.uint8)
            crack_repair_t = (
                (policy_tensor == 0)
                & editable_black_t
                & (nonzero_neighbor_count_t >= 5.0)
                & (crack_candidate_t > 0)
            )
            policy_tensor = torch.where(crack_repair_t, crack_candidate_t, policy_tensor)
            fillable_counts = fillable_output_t.flatten(1).sum(dim=1).cpu().numpy()
            policy_tensor = policy_tensor[:, 0]
        else:
            refined_np = refined_t[:, 0].to(torch.float32).cpu().numpy()
        self._synchronize()
        gpu_ms = (time.perf_counter() - gpu_start) * 1000.0

        post_start = time.perf_counter()
        policy_list = []
        nearest_list = []
        inpainted_list = []
        resized_list = []
        fillable_output_counts = []
        for index in range(depth.shape[0]) if not return_torch_policy else ():
            policy_depth = _masked_resize_depth_np(
                refined_np[index],
                work_active_np[index],
                (output_width, output_height),
            )
            normalized = (np.clip(policy_depth, self.depth_lower_m, self.depth_far_m) - self.depth_lower_m) / (
                self.depth_far_m - self.depth_lower_m
            )
            normalized[policy_depth < self.depth_lower_m] = 0.0
            guided_gray = (255.0 * np.clip(normalized, 0.0, 1.0)).astype(np.uint8)
            invalid_output = (
                output_invalid[index]
                if output_invalid is not None
                else cv2.resize(
                    invalid_np[index].astype(np.uint8),
                    (output_width, output_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
            fillable_output = output_fillable_np[index] & invalid_output
            fillable_output_counts.append(int(fillable_output.sum()))
            if base_policy is None:
                policy_gray = guided_gray
                policy_gray[invalid_output & ~fillable_output] = 0
            else:
                policy_gray = base_policy[index].copy()
                policy_gray[invalid_output & ~fillable_output] = 0
                policy_gray[fillable_output] = guided_gray[fillable_output]
            policy_gray = _repair_black_cracks_u8_np(
                policy_gray,
                editable_mask=fillable_output | ~invalid_output,
            )
            policy_list.append(policy_gray)

            if return_debug:
                nearest_full = _masked_resize_depth_np(
                    work_nearest_np[index],
                    work_active_np[index],
                    (depth.shape[2], depth.shape[1]),
                )
                inpainted_full = _masked_resize_depth_np(
                    refined_np[index],
                    work_active_np[index],
                    (depth.shape[2], depth.shape[1]),
                )
                nearest_full[valid_np[index]] = sanitized[index][valid_np[index]]
                inpainted_full[valid_np[index]] = sanitized[index][valid_np[index]]
                nearest_full[~(valid_np[index] | fillable_np[index])] = 0.0
                inpainted_full[~(valid_np[index] | fillable_np[index])] = 0.0
                resized = _masked_resize_depth_np(
                    inpainted_full,
                    valid_np[index] | fillable_np[index],
                    (output_width, output_height),
                )
                nearest_list.append(nearest_full)
                inpainted_list.append(inpainted_full)
                resized_list.append(resized)
        policy_np = np.stack(policy_list) if policy_list else None
        nearest_np = np.stack(nearest_list) if return_debug else None
        inpainted_np = np.stack(inpainted_list) if return_debug else None
        resized_np = np.stack(resized_list) if return_debug else None
        postprocess_ms = (time.perf_counter() - post_start) * 1000.0

        total_ms = (time.perf_counter() - start) * 1000.0
        self._timings_ms.append(total_ms)
        per_camera = []
        for index in range(depth.shape[0]):
            pixels = float(depth[index].size)
            invalid_count = int(invalid_np[index].sum())
            if fillable_np is not None:
                fillable_count = int(fillable_np[index].sum())
                fillable_pct = 100.0 * fillable_count / pixels
            elif fillable_counts is not None:
                output_pixels = float(output_height * output_width)
                fillable_count = int(fillable_counts[index])
                fillable_pct = 100.0 * fillable_count / output_pixels
            else:
                output_pixels = float(output_height * output_width)
                fillable_count = int(fillable_output_counts[index])
                fillable_pct = 100.0 * fillable_count / output_pixels
            per_camera.append(
                {
                    "input_invalid_pct": 100.0 * invalid_count / pixels,
                    "fillable_pct": fillable_pct,
                    "filled_pct": fillable_pct,
                    "remaining_invalid_pct": max(
                        0.0,
                        100.0 * invalid_count / pixels - fillable_pct,
                    ),
                }
            )
        stats = {
            "mode": "rgb_guided",
            "batch_size": int(depth.shape[0]),
            "input_shape_hw": [int(depth.shape[1]), int(depth.shape[2])],
            "output_shape_hw": list(self.output_size_hw),
            "max_distance_px": self.max_distance_px,
            "iterations": self.iterations,
            "rgb_sigma": self.rgb_sigma,
            "processing_downsample": self.processing_downsample,
            "propagation_steps": propagation_steps,
            "mask_prep_ms": mask_prep_ms,
            "prep_ms": prep_ms,
            "gpu_ms": gpu_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
            "timing": self.timing_summary(),
            "per_camera": per_camera,
        }
        return DepthInpaintBatchResult(
            inpainted_depth_m=None if inpainted_np is None else np.ascontiguousarray(inpainted_np),
            resized_depth_m=None if resized_np is None else np.ascontiguousarray(resized_np),
            policy_u8=policy_tensor if return_torch_policy else np.ascontiguousarray(policy_np),
            nearest_depth_m=None if nearest_np is None else np.ascontiguousarray(nearest_np),
            invalid_mask=np.ascontiguousarray(invalid_np),
            fillable_mask=(
                np.ascontiguousarray(fillable_np)
                if fillable_np is not None
                else np.zeros((0,), dtype=bool)
            ),
            stats=stats,
        )

    def timing_summary(self) -> dict[str, float | int]:
        if not self._timings_ms:
            return {"samples": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        values = np.asarray(self._timings_ms, dtype=np.float64)
        return {
            "samples": int(values.size),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max()),
        }


__all__ = [
    "DepthInpaintBatchResult",
    "RGBGuidedDepthInpainter",
    "_nearest_valid_initialize",
]
