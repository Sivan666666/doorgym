from pathlib import Path
import sys

import numpy as np


FLOAT_IK_ROOT = Path(__file__).resolve().parents[1]
if str(FLOAT_IK_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOAT_IK_ROOT))

from camera_intrinsics import (  # noqa: E402
    load_real_camera_intrinsics_config,
    remap_coverage,
    remap_image,
    reverse_remap_coordinates,
)


CFG = FLOAT_IK_ROOT.parent / "data" / "cfg" / "a2w_real_camera_intrinsics_640x480.yaml"


def test_real_camera_profile_has_full_render_coverage():
    profile = load_real_camera_intrinsics_config(CFG)
    for camera_name in ("front", "wrist"):
        map_x, map_y = reverse_remap_coordinates(
            profile["render_intrinsics"], profile["camera_intrinsics"][camera_name]
        )
        assert remap_coverage(map_x, map_y, profile["render_intrinsics"]) == 1.0


def test_identity_remap_is_pixel_exact_for_nearest_neighbor():
    intrinsics = {"fx": 4.0, "fy": 4.0, "cx": 2.0, "cy": 1.5, "width": 5, "height": 4}
    map_x, map_y = reverse_remap_coordinates(intrinsics, intrinsics)
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    np.testing.assert_array_equal(remap_image(image, map_x, map_y, interpolation="nearest"), image)


def test_depth_edge_remap_uses_nearest_without_mixing_values():
    source = {"fx": 6.0, "fy": 6.0, "cx": 3.0, "cy": 2.0, "width": 6, "height": 4}
    target = {"fx": 5.5, "fy": 5.5, "cx": 3.0, "cy": 2.0, "width": 6, "height": 4}
    image = np.zeros((4, 6), dtype=np.float32)
    image[:, :3] = 0.4
    image[:, 3:] = 1.4
    map_x, map_y = reverse_remap_coordinates(source, target)
    remapped = remap_image(image, map_x, map_y, interpolation="nearest")
    for value in np.unique(remapped):
        assert np.any(np.isclose(value, np.asarray([0.0, 0.4, 1.4], dtype=np.float32)))
