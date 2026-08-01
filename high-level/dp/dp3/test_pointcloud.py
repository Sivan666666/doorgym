#!/usr/bin/env python3
"""Geometry tests for the shared A2W front-depth point cloud conversion."""

from __future__ import annotations

import unittest

import numpy as np

from pointcloud import (
    FrontDepthPointCloudConfig,
    fuse_depth_views_to_point_cloud,
    front_depth_to_point_cloud,
    front_depth_to_point_cloud_batch,
    quat_xyzw_to_matrix,
)


class FrontDepthPointCloudTest(unittest.TestCase):
    def setUp(self):
        self.k = {"fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0, "width": 2, "height": 2}
        self.cfg = FrontDepthPointCloudConfig(
            num_points=4,
            candidate_rows=2,
            candidate_cols=2,
            workspace_min=(-10.0, -10.0, -10.0),
            workspace_max=(10.0, 10.0, 10.0),
            near_clip_m=0.0,
            far_clip_m=2.55,
        )
        self.pose = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    def test_shape_finite_and_repeatable(self):
        depth = np.full((2, 2), 100, dtype=np.uint8)
        first = front_depth_to_point_cloud(depth, self.pose, self.k, self.cfg)
        second = front_depth_to_point_cloud(depth, self.pose, self.k, self.cfg)
        self.assertEqual(first.shape, (4, 3))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_isaac_camera_axis_and_pixel_center(self):
        depth = np.full((2, 2), 100, dtype=np.uint8)  # exactly 1 metre
        points = front_depth_to_point_cloud(depth, self.pose, self.k, self.cfg)
        np.testing.assert_allclose(points[:, 0], 1.0, atol=1e-6)
        # Top-left lies left/up: Isaac local +Y/+Z.
        self.assertGreater(points[0, 1], 0.0)
        self.assertGreater(points[0, 2], 0.0)
        # Bottom-right lies right/down: Isaac local -Y/-Z.
        self.assertLess(points[-1, 1], 0.0)
        self.assertLess(points[-1, 2], 0.0)

    def test_pose_transform_and_hwc_runtime_input(self):
        depth = np.full((2, 2, 3), 100, dtype=np.uint8)
        pose = np.asarray([1, 2, 3, 0, 0, 0, 1], dtype=np.float32)
        points = front_depth_to_point_cloud(depth, pose, self.k, self.cfg)
        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose(points[:, 0], 2.0, atol=1e-6)

    def test_batch_matches_single_exactly(self):
        depth = np.stack(
            [np.full((2, 2), 100, dtype=np.uint8), np.full((2, 2), 120, dtype=np.uint8)])
        poses = np.stack([self.pose, self.pose])
        batch = front_depth_to_point_cloud_batch(depth, poses, self.k, self.cfg)
        for index in range(2):
            single = front_depth_to_point_cloud(depth[index], poses[index], self.k, self.cfg)
            np.testing.assert_array_equal(batch[index], single)

    def test_quaternion_rotation_is_orthonormal(self):
        matrix = quat_xyzw_to_matrix(np.asarray([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]))
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)

    def test_empty_depth_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "No valid depth points"):
            front_depth_to_point_cloud(np.zeros((2, 2), dtype=np.uint8), self.pose, self.k, self.cfg)

    def test_front_wrist_fusion_is_equal_and_in_robot_base(self):
        config = FrontDepthPointCloudConfig(
            num_points=8,
            candidate_rows=2,
            candidate_cols=4,
            workspace_min=(-10.0, -10.0, -10.0),
            workspace_max=(10.0, 10.0, 10.0),
            near_clip_m=0.0,
            far_clip_m=2.55,
        )
        intrinsics = {"fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0, "width": 2, "height": 2}
        depth = np.full((2, 2), 100, dtype=np.uint8)
        front_pose = np.asarray([1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        wrist_pose = np.asarray([0, 1, 0, 0, 0, 0, 1], dtype=np.float32)
        fused, source, views = fuse_depth_views_to_point_cloud(
            front_depth_u8=depth,
            front_camera_pose_base=front_pose,
            front_intrinsics=intrinsics,
            wrist_depth_u8=depth,
            wrist_camera_pose_base=wrist_pose,
            wrist_intrinsics=intrinsics,
            config=config,
        )
        self.assertEqual(fused.shape, (8, 3))
        self.assertEqual(views.shape, (2, 4, 3))
        np.testing.assert_array_equal(source, np.asarray([0] * 4 + [1] * 4, dtype=np.uint8))
        np.testing.assert_allclose(fused[:4], views[0], atol=0, rtol=0)
        np.testing.assert_allclose(fused[4:], views[1], atol=0, rtol=0)
        np.testing.assert_allclose(
            views[0] - views[1],
            np.tile(np.asarray([1.0, -1.0, 0.0]), (4, 1)),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
