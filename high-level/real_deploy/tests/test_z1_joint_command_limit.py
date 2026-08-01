from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "z1_act_ee_bridge.py"
SPEC = importlib.util.spec_from_file_location("z1_act_ee_bridge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


def test_joint9_packet_and_state_bypass_ee_layout() -> None:
    action = np.asarray([0.2, -0.1, 0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -1.2], dtype=np.float32)
    packet = json.dumps({"action": action.tolist()}).encode("utf-8")
    parsed, mode, q_target = bridge.parse_action_command(packet, "joint9")

    np.testing.assert_allclose(parsed, action)
    assert mode == "joint"
    np.testing.assert_allclose(q_target, action[2:8])

    state = bridge.make_act_state(
        np.asarray([0.3, 0.05]),
        np.zeros(3),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        -0.7,
        state_action_mode="joint9",
        q=action[2:8],
    )
    assert state.shape == (9,)
    np.testing.assert_allclose(state, [0.3, 0.05, *action[2:8], -0.7], atol=1.0e-6)


def test_joint9_bridge_cli_is_explicit_and_legacy_default_remains_ee10(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["z1_act_ee_bridge.py"])
    assert bridge.parse_args().act_state_action_mode == "ee10"
    monkeypatch.setattr(sys, "argv", ["z1_act_ee_bridge.py", "--act_state_action_mode", "joint9"])
    assert bridge.parse_args().act_state_action_mode == "joint9"


def test_simulated_camera_poses_match_front_and_ee_mounts() -> None:
    front, wrist = bridge.simulated_camera_pose_transforms(
        np.eye(4, dtype=np.float64),
        np.eye(4, dtype=np.float64),
        bridge.make_transform([0.086, 0.0, 0.0], [0.0, 0.0, 0.0]),
    )
    front_pos, front_quat = bridge.transform_to_pose(front)
    wrist_pos, wrist_quat = bridge.transform_to_pose(wrist)

    np.testing.assert_allclose(front_pos, [0.29, 0.031, 0.165], atol=1.0e-7)
    np.testing.assert_allclose(
        front_quat,
        [0.0, -np.sin(np.deg2rad(22.5)), 0.0, np.cos(np.deg2rad(22.5))],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(wrist_pos, [-0.007, 0.031, 0.22], atol=1.0e-7)
    np.testing.assert_allclose(
        wrist_quat,
        [0.0, np.sin(np.deg2rad(30.0)), 0.0, np.cos(np.deg2rad(30.0))],
        atol=1.0e-7,
    )


def test_wrist_camera_translation_rotates_with_fk_and_ee() -> None:
    sdk_ee = bridge.make_transform([0.4, -0.2, 0.3], [0.0, 0.0, np.pi / 2.0])
    act_ee_from_sdk_ee = bridge.make_transform([0.086, 0.0, 0.0], [0.0, 0.0, 0.0])
    _, wrist = bridge.simulated_camera_pose_transforms(
        sdk_ee,
        np.eye(4),
        act_ee_from_sdk_ee,
    )
    np.testing.assert_allclose(
        wrist[:3, 3],
        [0.4 - 0.031, -0.2 - 0.007, 0.3 + 0.22],
        atol=1.0e-7,
    )


def test_joint_command_is_limited_once_before_send() -> None:
    q_current = np.zeros(6, dtype=np.float64)
    q_target = np.asarray([2.0, -2.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    max_speed = np.full(6, 0.6, dtype=np.float64)

    q_next, qd = bridge.clamp_joint_target(q_current, q_target, 0.002, max_speed)

    np.testing.assert_allclose(
        np.abs(q_next - q_current),
        np.full(6, 0.0012),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.abs(qd), max_speed, atol=1.0e-12)


def test_small_joint_target_is_not_overshot() -> None:
    q_current = np.asarray([0.1, 0.2, -0.3, 0.4, -0.5, 0.6], dtype=np.float64)
    q_target = q_current + np.asarray([0.0002, -0.0003, 0.0004, 0.0, 0.0001, -0.0005])

    q_next, _ = bridge.clamp_joint_target(
        q_current,
        q_target,
        0.002,
        np.full(6, 0.6),
    )

    np.testing.assert_allclose(q_next, q_target, atol=1.0e-12)


def test_cli_disallows_direct_hold_and_defaults_to_back_to_start(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["z1_act_ee_bridge.py"])
    args = bridge.parse_args()

    assert args.zero_joints_command_mode == "controller_home"
    assert args.startup_home_max_joint_speed == 0.2
    assert args.joint_command_mode == "online_quintic"
    assert args.back_to_start_on_exit
    assert args.max_joint_speed == 3.0
    assert args.max_gripper_speed == 3.14
    assert args.max_gripper_acceleration == 120.0
    assert args.max_joint_acceleration == 15.0
    assert args.joint_trajectory_duration_s == 0.02
    assert args.ik_max_joint_delta == 1.5
    assert args.soft_ik_fallback
    assert args.soft_ik_max_iterations == 10
    assert args.soft_ik_orientation_weight == 0.02
    assert args.joint_target_max_delta == 2.5


def test_startup_home_stability_is_relative_to_calibrated_home_not_zero() -> None:
    home_q = np.asarray([0.0, 0.0001, -0.00695, -0.07416, -0.00001, 0.0])
    q_measured = home_q + np.asarray([0.001, -0.002, 0.001, 0.003, 0.0, -0.001])
    qd_measured = np.asarray([0.01, -0.02, 0.01, 0.03, 0.0, -0.01])

    stable, max_drift, max_speed = bridge.startup_home_sample_is_stable(
        q_measured,
        qd_measured,
        home_q,
        position_tolerance=0.03,
        max_joint_speed=0.08,
    )

    assert stable
    assert max_drift == pytest.approx(0.003)
    assert max_speed == pytest.approx(0.03)


def test_startup_home_stability_rejects_motion_and_non_finite_feedback() -> None:
    home_q = np.asarray([0.0, 0.0, -0.01, -0.07, 0.0, 0.0])

    stable, _, max_speed = bridge.startup_home_sample_is_stable(
        home_q,
        np.asarray([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
        home_q,
        position_tolerance=0.03,
        max_joint_speed=0.08,
    )
    assert not stable
    assert max_speed == pytest.approx(0.2)

    bad_q = home_q.copy()
    bad_q[2] = np.nan
    stable, max_drift, max_speed = bridge.startup_home_sample_is_stable(
        bad_q,
        np.zeros(6),
        home_q,
        position_tolerance=0.03,
        max_joint_speed=0.08,
    )
    assert not stable
    assert np.isinf(max_drift)
    assert np.isinf(max_speed)


def test_startup_home_is_recorded_in_bridge_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["z1_act_ee_bridge.py"])
    args = bridge.parse_args()
    shared = bridge.SharedBridgeState()
    home_q = np.asarray([0.0, 0.0001, -0.00695, -0.07416, -0.00001, 0.0])

    bridge.update_shared_startup_zero_status(
        shared,
        requested=True,
        active=False,
        done=True,
        max_err=0.002,
        home_q=home_q,
        max_speed=0.03,
    )
    snapshot = shared.snapshot(args)

    np.testing.assert_allclose(snapshot["startup_home_q"], home_q, atol=1.0e-6)
    assert snapshot["startup_home_max_speed"] == pytest.approx(0.03)
    assert snapshot["startup_zero_done"] is True


def test_shutdown_back_to_start_status_is_recorded_in_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["z1_act_ee_bridge.py"])
    args = bridge.parse_args()
    shared = bridge.SharedBridgeState()

    bridge.update_shared_shutdown_status(
        shared,
        requested=True,
        active=False,
        done=True,
    )
    snapshot = shared.snapshot(args)

    assert snapshot["shutdown_requested"] is True
    assert snapshot["shutdown_active"] is False
    assert snapshot["shutdown_done"] is True
    assert snapshot["shutdown_error"] == ""


def test_online_quintic_uses_the_full_nominal_waypoint_interval() -> None:
    trajectory = bridge.OnlineQuinticJointTrajectory.hold(np.zeros(6), now=10.0)
    target = np.full(6, 0.02, dtype=np.float64)

    duration = trajectory.retarget(
        target,
        np.full(6, 0.5),
        now=10.0,
        nominal_duration=0.04,
        max_speed=np.full(6, 2.5),
        max_acceleration=np.full(6, 100.0),
    )

    assert duration == 0.04
    q0, qd0, qdd0 = trajectory.sample(10.0)
    q_half, qd_half, _ = trajectory.sample(10.02)
    q1, qd1, qdd1 = trajectory.sample(10.04)
    np.testing.assert_allclose(q0, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(qd0, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(qdd0, 0.0, atol=1.0e-12)
    assert np.all(q_half > 0.0)
    assert np.all(q_half < target)
    assert np.all(qd_half > 0.0)
    np.testing.assert_allclose(q1, target, atol=1.0e-10)
    np.testing.assert_allclose(qd1, 0.5, atol=1.0e-10)
    np.testing.assert_allclose(qdd1, 0.0, atol=1.0e-9)


def test_online_quintic_replanning_preserves_q_qd_qdd_continuity() -> None:
    trajectory = bridge.OnlineQuinticJointTrajectory.hold(np.zeros(6), now=20.0)
    trajectory.retarget(
        np.full(6, 0.02),
        np.full(6, 0.5),
        now=20.0,
        nominal_duration=0.04,
        max_speed=np.full(6, 2.5),
        max_acceleration=np.full(6, 100.0),
    )
    before = trajectory.sample(20.02)

    trajectory.retarget(
        np.full(6, 0.04),
        np.full(6, 0.5),
        now=20.02,
        nominal_duration=0.04,
        max_speed=np.full(6, 2.5),
        max_acceleration=np.full(6, 100.0),
    )
    after = trajectory.sample(20.02)

    for before_value, after_value in zip(before, after):
        np.testing.assert_allclose(after_value, before_value, atol=1.0e-9)


def test_far_online_quintic_target_is_stretched_to_speed_limit() -> None:
    trajectory = bridge.OnlineQuinticJointTrajectory.hold(np.zeros(6), now=30.0)
    max_speed = np.full(6, 2.5)
    duration = trajectory.retarget(
        np.full(6, 1.0),
        np.full(6, 2.5),
        now=30.0,
        nominal_duration=0.04,
        max_speed=max_speed,
        max_acceleration=np.full(6, 15.0),
    )

    assert duration > 0.04
    sampled_qd = np.asarray(
        [trajectory.sample(30.0 + duration * i / 200.0)[1] for i in range(201)]
    )
    assert float(np.max(np.abs(sampled_qd))) <= 2.5 + 1.0e-5


def test_online_quintic_respects_acceleration_limit() -> None:
    trajectory = bridge.OnlineQuinticJointTrajectory.hold(np.zeros(6), now=40.0)
    duration = trajectory.retarget(
        np.full(6, 0.02),
        np.full(6, 0.5),
        now=40.0,
        nominal_duration=0.04,
        max_speed=np.full(6, 2.5),
        max_acceleration=np.full(6, 15.0),
    )

    assert duration > 0.04
    sampled_qdd = np.asarray(
        [trajectory.sample(40.0 + duration * i / 200.0)[2] for i in range(201)]
    )
    assert float(np.max(np.abs(sampled_qdd))) <= 15.0 + 1.0e-4


def test_online_quintic_brake_reaches_zero_velocity_smoothly() -> None:
    trajectory = bridge.OnlineQuinticJointTrajectory.hold(np.zeros(6), now=50.0)
    trajectory.retarget(
        np.full(6, 0.02),
        np.full(6, 0.5),
        now=50.0,
        nominal_duration=0.04,
        max_speed=np.full(6, 2.5),
        max_acceleration=np.full(6, 15.0),
    )
    before = trajectory.sample(50.02)
    duration = trajectory.brake(
        now=50.02,
        nominal_duration=0.08,
        max_speed=np.full(6, 2.5),
        max_acceleration=np.full(6, 15.0),
    )
    after = trajectory.sample(50.02)
    for before_value, after_value in zip(before, after):
        np.testing.assert_allclose(after_value, before_value, atol=1.0e-9)
    _, qd_end, qdd_end = trajectory.sample(50.02 + duration)
    np.testing.assert_allclose(qd_end, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(qdd_end, 0.0, atol=1.0e-8)


def test_parse_explicit_joint_target_command() -> None:
    payload = (
        '{"action":[0,0,0.35,0,0.32,0,0,0,1,-1.0],'
        '"control_mode":"home","joint_target":[0,0.1,-0.2,0.3,-0.4,0.5]}'
    ).encode()
    action, mode, joint_target = bridge.parse_action_command(payload)
    assert action.shape == (10,)
    assert mode == "home"
    np.testing.assert_allclose(joint_target, [0.0, 0.1, -0.2, 0.3, -0.4, 0.5])


def test_lowcmd_inverse_dynamics_receives_online_qdd() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.received_qdd = None

        def jointProtect(self, q, qd):
            return q, qd

        def inverseDynamics(self, q, qd, qdd, ftip):
            self.received_qdd = np.asarray(qdd, dtype=np.float64).copy()
            return np.zeros(6, dtype=np.float64)

    class FakeArm:
        def setArmCmd(self, q, qd, tau):
            pass

        def setGripperCmd(self, q, qd, tau):
            pass

        def sendRecv(self):
            pass

    qdd = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    model = FakeModel()
    bridge.send_lowcmd_joint_target(
        FakeArm(),
        model,
        threading.Lock(),
        np.zeros(6),
        np.zeros(6),
        qdd,
        -1.0,
        0.0,
        np.full(6, -3.0),
        np.full(6, 3.0),
    )
    np.testing.assert_allclose(model.received_qdd, qdd)


def test_gripper_command_accelerates_and_settles_at_zero_velocity() -> None:
    dt = 0.002
    position = -np.pi / 2.0
    velocity = 0.0
    positions = []
    velocities = []

    for _ in range(1000):
        position, velocity = bridge.advance_gripper_command(
            position,
            velocity,
            -0.3141592654,
            dt,
            max_speed=2.5,
            max_acceleration=20.0,
            position_min=-np.pi / 2.0,
            position_max=0.0,
        )
        positions.append(position)
        velocities.append(velocity)

    assert np.all(np.diff(positions) >= -1.0e-12)
    assert max(velocities) <= 2.5 + 1.0e-12
    assert max(np.abs(np.diff(velocities))) <= 20.0 * dt + 1.0e-12
    np.testing.assert_allclose(position, -0.3141592654, atol=1.0e-12)
    assert velocity == 0.0

    for _ in range(100):
        position, velocity = bridge.advance_gripper_command(
            position,
            velocity,
            -0.3141592654,
            dt,
            max_speed=2.5,
            max_acceleration=20.0,
            position_min=-np.pi / 2.0,
            position_max=0.0,
        )
        assert velocity == 0.0


def test_soft_ik_reaches_position_when_orientation_is_unachievable() -> None:
    class PositionOnlyModel:
        def forwardKinematics(self, q, index):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = np.asarray(q, dtype=np.float64)[:3]
            return transform

        def CalcJacobian(self, q):
            jacobian = np.zeros((6, 6), dtype=np.float64)
            jacobian[3:, :3] = np.eye(3)
            return jacobian

    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [0.25, -0.15, 0.1]
    target[:3, :3] = bridge.rpy_to_rot(0.0, 0.0, np.pi / 2.0)
    result = bridge.solve_soft_pose_ik(
        PositionOnlyModel(),
        threading.Lock(),
        target,
        np.zeros(6),
        np.full(6, -1.0),
        np.full(6, 1.0),
        np.eye(4),
        max_iterations=24,
        damping=0.01,
        orientation_weight=0.02,
        seed_weight=1.0e-6,
        max_joint_step=0.08,
        position_tolerance=0.003,
        min_position_improvement=0.001,
    )

    assert result.success
    np.testing.assert_allclose(result.q[:3], target[:3, 3], atol=3.0e-3)
    assert result.position_error <= 3.0e-3
    assert result.orientation_error > 1.0
