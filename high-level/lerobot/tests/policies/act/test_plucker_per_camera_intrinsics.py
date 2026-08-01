import torch

from lerobot.policies.act.modeling_act import (
    make_camera_local_unit_rays,
    make_camera_local_unit_rays_from_intrinsics,
)


def test_calibrated_principal_point_ray_is_optical_axis():
    rays = make_camera_local_unit_rays_from_intrinsics(
        height=480,
        width=640,
        fx=604.7375,
        fy=603.8005,
        cx=329.0,
        cy=241.0,
    )
    torch.testing.assert_close(rays[:, 241, 329], torch.tensor([0.0, 0.0, 1.0]))


def test_legacy_shared_fov_ray_builder_is_unchanged():
    expected = make_camera_local_unit_rays(height=8, width=10, horizontal_fov_deg=55.0)
    actual = make_camera_local_unit_rays(height=8, width=10, horizontal_fov_deg=55.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_front_and_wrist_calibrations_produce_different_rays():
    front = make_camera_local_unit_rays_from_intrinsics(
        height=480, width=640, fx=604.7375, fy=603.8005, cx=329.0904, cy=240.8226
    )
    wrist = make_camera_local_unit_rays_from_intrinsics(
        height=480, width=640, fx=606.1074, fy=605.9551, cx=325.8074, cy=259.1791
    )
    assert not torch.equal(front, wrist)
