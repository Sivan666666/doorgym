from __future__ import annotations

import sys
from pathlib import Path

import pytest


REAL_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(REAL_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOY_DIR))

import door_act_shadow


class _Filter:
    def __init__(self) -> None:
        self.options = {}

    def set_option(self, option, value) -> None:
        self.options[option] = value


class _FakeRS:
    class option:
        filter_magnitude = "magnitude"
        filter_smooth_alpha = "alpha"
        filter_smooth_delta = "delta"
        holes_fill = "holes"

    def __init__(self) -> None:
        self.spatial = _Filter()
        self.temporal = _Filter()
        self.hole = _Filter()

    def spatial_filter(self):
        return self.spatial

    def temporal_filter(self):
        return self.temporal

    def hole_filling_filter(self):
        return self.hole


def test_default_mode_preserves_realsense_behavior() -> None:
    args = door_act_shadow.parse_args([])
    assert args.depth_inpaint_mode == "realsense"
    assert args.depth_inpaint_max_distance_px == 64.0
    assert args.depth_inpaint_iterations == 2
    assert args.depth_inpaint_rgb_sigma == 0.10
    assert args.enable_ros_base_bridge
    assert args.ros_cmd_vel_topic == "/cmd_vel_safe"
    assert args.ros_vel_state_topic == "/vel_state"
    assert args.vel_state_timeout_s == 0.5
    assert args.enable_z1_action_bridge
    assert args.z1_action_udp_host == "127.0.0.1"
    assert args.z1_action_udp_port == 15011
    assert args.enable_z1_state_receiver
    assert args.z1_state_udp_bind_host == "0.0.0.0"
    assert args.z1_state_udp_port == 15013
    assert args.z1_state_timeout_s == 0.5
    assert args.wait_for_z1_startup_zero
    assert args.z1_startup_zero_wait_timeout_s == 20.0


def test_real_bridges_can_be_disabled_for_shadow_only_runs() -> None:
    args = door_act_shadow.parse_args(
        [
            "--no_enable_ros_base_bridge",
            "--no_enable_z1_action_bridge",
            "--no_enable_z1_state_receiver",
            "--no_wait_for_z1_startup_zero",
        ]
    )
    door_act_shadow.validate_runtime_args(args)
    assert not args.enable_ros_base_bridge
    assert not args.enable_z1_action_bridge
    assert not args.enable_z1_state_receiver
    assert not args.wait_for_z1_startup_zero


def test_realsense_filter_profiles_enable_or_disable_hole_propagation() -> None:
    rs = _FakeRS()
    filters = door_act_shadow.make_realsense_filters(rs, True, fill_holes=True)
    assert filters == [rs.spatial, rs.hole, rs.temporal]
    assert rs.spatial.options["holes"] == door_act_shadow.DEFAULT_RS_SPATIAL_HOLES_FILL

    rs = _FakeRS()
    filters = door_act_shadow.make_realsense_filters(rs, True, fill_holes=False)
    assert filters == [rs.spatial, rs.temporal]
    assert rs.spatial.options["holes"] == 0

    assert door_act_shadow.make_realsense_filters(rs, False, fill_holes=True) == []


def test_rgb_guided_requires_color_alignment_but_allows_no_rs_filters() -> None:
    args = door_act_shadow.parse_args(
        [
            "--camera_mode",
            "realsense",
            "--depth_inpaint_mode",
            "rgb_guided",
            "--no_rs_filters",
        ]
    )
    door_act_shadow.validate_runtime_args(args)
    assert not args.rs_filters

    args.no_align_depth_to_color = True
    with pytest.raises(ValueError, match="requires depth aligned to color"):
        door_act_shadow.validate_runtime_args(args)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--depth_inpaint_max_distance_px", "-1"),
        ("--depth_inpaint_iterations", "-1"),
        ("--depth_inpaint_rgb_sigma", "0"),
        ("--vel_state_timeout_s", "-1"),
        ("--z1_state_timeout_s", "-1"),
        ("--z1_startup_zero_wait_timeout_s", "-1"),
    ],
)
def test_invalid_inpaint_parameters_are_rejected(flag: str, value: str) -> None:
    args = door_act_shadow.parse_args([flag, value])
    with pytest.raises(ValueError):
        door_act_shadow.validate_runtime_args(args)


def test_wait_for_z1_startup_zero_requires_state_receiver() -> None:
    args = door_act_shadow.parse_args(["--no_enable_z1_state_receiver"])
    with pytest.raises(ValueError, match="requires --enable_z1_state_receiver"):
        door_act_shadow.validate_runtime_args(args)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--z1_action_udp_port", "0"),
        ("--z1_state_udp_port", "70000"),
    ],
)
def test_invalid_z1_ports_are_rejected(flag: str, value: str) -> None:
    args = door_act_shadow.parse_args([flag, value])
    with pytest.raises(ValueError):
        door_act_shadow.validate_runtime_args(args)
