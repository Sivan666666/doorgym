#!/usr/bin/env python3
"""Parallel float-base A2W+Z1 base+arm IK door-pull recorder.

This is the high-throughput float-base variant for the A2W+Z1 robot.
The complete walk-to-handle, grasp, close-gripper, and rotate-handle prefix is
kept identical to the push controller.  Only the post-rotation state machine is
replaced by pull, release, collision-aware arm retraction, and traversal phases.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except Exception:
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]
KINEMATICS_DIR = SCRIPT_DIR / "kinematics"
if str(KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(KINEMATICS_DIR))

from joint_trajectory_smoothing import evaluate_polynomial, polynomial_coefficients

import door_common as dc
import isaacgym_a2w_ik_push_door_parallel as a2w_ik
from door_twin import (
    DoorTwinSpec,
    MIN_DOOR_TWIN_FORWARD_DISTANCE_M,
    RolloutTracker,
    apply_program_to_args,
    compute_skill_waypoints,
    default_skill_program_from_args,
    load_skill_program,
    profile_from_program,
    write_rollout_reports,
)

base_ik = dc.base_ik
gymapi = dc.gymapi
gymutil = dc.gymutil
DEFAULT_DOOR_CFG = dc.DEFAULT_DOOR_CFG
DP_NUM_DOFS = dc.DP_NUM_DOFS
DP_NUM_ACTIONS = dc.DP_NUM_ACTIONS
FLOAT_ARM_TO_DP_DOF = dc.FLOAT_ARM_TO_DP_DOF
ThickAxesGeometry = dc.ThickAxesGeometry
DoorRuntime = dc.DoorRuntime
A2WZ1_DEFAULT_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "a2wz1"
A2WZ1_DEFAULT_ASSET_FILE = "urdf/a2wz1.urdf"
DEFAULT_Z1_TRACIK_URDF = HIGH_LEVEL_ROOT / "data" / "asset" / "z1" / "urdf" / "z1_arm.urdf"
DEFAULT_Z1_TRACIK_EE_LINK = "ee_gripper_link"
DEFAULT_Z1_TRACIK_MOUNT_CORRECTION = (0.026, 0.0, -0.042)
Z1_ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))

DP_ROOT = HIGH_LEVEL_ROOT / "dp"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

try:
    from door_dp_common import (
        ACTION_NAMES,
        DoorDPJsonlLogger,
        DoorDPPolicyController,
        RawDoorDPRecorder,
        image_to_three_channel_uint8,
        make_state_feature_names,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )
except ImportError:
    ACTION_NAMES = None
    DoorDPJsonlLogger = None
    DoorDPPolicyController = None
    RawDoorDPRecorder = None
    image_to_three_channel_uint8 = None
    make_state_feature_names = None
    normalize_vision_mode = None
    raw_image_keys_for_vision_mode = None


DP_PHASE_NAMES = [
    "walk",
    "initial_hold",
    "grasp",
    "grasp_hold",
    "close_gripper",
    "rotate_handle",
    "pull_door",
    "open_hold",
    "release_handle",
    "pass_through",
    "return_home",
    "hold_home",
]
DP_PHASE_ID = {name: idx for idx, name in enumerate(DP_PHASE_NAMES)}
IKPULL_STATE_VERSION = "zero_leg_dof_pos_prev_action_v1"
sorted_asset_entries = dc.sorted_asset_entries


VIEWER_PAUSE_ACTION = "pause_simulation"
A2W_FLOAT_IK_CONFIG_ENV_VAR = "A2W_FLOAT_IK_CONFIG"
A2W_FLOAT_IK_DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "a2w_float_ik_push_door_parallel.yaml"
A2W_FLOAT_IK_CONFIG_DERIVED_ATTRS = {
    "dp_print",
}
DOOR_TWIN_DEFAULT_LOG_ROOT = SCRIPT_DIR / "door_twin" / "experiments" / "runs"
DOOR_TWIN_KEYFRAME_PHASES = (
    "initial_hold",
    "grasp",
    "close_gripper",
    "rotate_handle",
    "pull_door",
    "open_hold",
    "release_handle",
    "pass_through",
    "return_home",
    "hold_home",
)
DOOR_TWIN_CAMERA_VIEWS = (
    "wrist",
    "front",
    "front_left",
    "front_right",
    "observer_left",
    "observer_right",
    "overhead",
    "handle_closeup",
)
DOOR_TWIN_DEFAULT_CAMERA_VIEWS = ("wrist", "front", "observer_left", "observer_right", "handle_closeup")
DEFAULT_STAGE_SCREENSHOT_PHASES = (
    "walk",
    "initial_hold",
    "grasp",
    "close_gripper",
    "rotate_handle",
    "pull_door",
    "open_hold",
    "release_handle",
    "pass_through",
    "return_home",
    "hold_home",
)


class SimOnlineQuinticTrajectory:
    """Causal C2 joint trajectory matching the real Z1 episode player."""

    def __init__(self, initial, max_speed, max_acceleration, max_duration):
        self.dof = int(np.asarray(initial).size)
        self.max_speed = np.broadcast_to(np.asarray(max_speed, dtype=np.float64), (self.dof,)).copy()
        self.max_acceleration = np.broadcast_to(
            np.asarray(max_acceleration, dtype=np.float64),
            (self.dof,),
        ).copy()
        self.max_duration = float(max_duration)
        self.coefficients = None
        self.start_time = 0.0
        self.duration = 0.0
        self.hold = np.asarray(initial, dtype=np.float64).reshape(self.dof).copy()

    def sample(self, now):
        if self.coefficients is None:
            zeros = np.zeros(self.dof, dtype=np.float64)
            return self.hold.copy(), zeros.copy(), zeros.copy(), zeros.copy()
        tau = float(np.clip(float(now) - self.start_time, 0.0, self.duration))
        values = evaluate_polynomial(self.coefficients, np.asarray([tau]))
        return tuple(value[0] for value in values)

    def retarget(self, goal, now, nominal_duration, goal_velocity=None):
        q0, qd0, qdd0, _ = self.sample(now)
        goal = np.asarray(goal, dtype=np.float64).reshape(self.dof)
        zeros = np.zeros(self.dof, dtype=np.float64)
        if goal_velocity is None:
            goal_velocity = zeros
        else:
            goal_velocity = np.asarray(goal_velocity, dtype=np.float64).reshape(self.dof)
            goal_velocity = np.clip(goal_velocity, -self.max_speed, self.max_speed)
        duration = max(float(nominal_duration), 1.0e-4)
        for _ in range(12):
            coefficients = polynomial_coefficients(
                q0,
                goal,
                qd0,
                goal_velocity,
                qdd0,
                zeros,
                zeros,
                zeros,
                duration,
                "quintic_hermite",
            )
            samples = evaluate_polynomial(coefficients, np.linspace(0.0, duration, 101))
            speed_ratio = float(np.max(np.abs(samples[1]) / np.maximum(self.max_speed, 1.0e-9)))
            acceleration_ratio = float(
                np.max(np.abs(samples[2]) / np.maximum(self.max_acceleration, 1.0e-9))
            )
            if speed_ratio <= 1.0001 and acceleration_ratio <= 1.0001:
                break
            scale = max(speed_ratio, np.sqrt(acceleration_ratio), 1.01) * 1.02
            duration *= scale
            if duration > self.max_duration:
                raise RuntimeError(
                    f"Quintic segment needs {duration:.3f}s, above "
                    f"max_duration={self.max_duration:.3f}s"
                )
        self.coefficients = coefficients
        self.start_time = float(now)
        self.duration = float(duration)
        self.hold = goal.copy()
        return self.duration


class TracIKTrajectoryController:
    """25 Hz TRAC-IK waypoint solver with 50 Hz online joint interpolation."""

    def __init__(self, solver, initial_q, lower, upper, sim_dt, args, seed):
        self.args = args
        self.solver = solver
        self.lower = np.asarray(lower, dtype=np.float64).reshape(6)
        self.upper = np.asarray(upper, dtype=np.float64).reshape(6)
        self.sim_dt = float(sim_dt)
        requested_hz = max(float(args.tracik_command_hz), 1.0e-6)
        self.command_stride = max(1, int(round(1.0 / (requested_hz * self.sim_dt))))
        self.command_period = self.command_stride * self.sim_dt
        self.max_restarts = max(1, int(args.tracik_max_restarts))
        self.initial_segment_duration = float(args.tracik_initial_segment_duration)
        self.projection_iterations = max(0, int(args.tracik_projection_iterations))
        self.projection_min_alpha = float(args.tracik_projection_min_alpha)
        self.project_unreachable = not bool(args.no_tracik_unreachable_projection)
        self.max_waypoint_step = float(args.tracik_max_waypoint_step)
        self.velocity_alpha = float(args.tracik_waypoint_velocity_alpha)
        self.rng = np.random.default_rng(int(seed))
        initial_q = np.clip(np.asarray(initial_q, dtype=np.float64).reshape(6), self.lower, self.upper)
        configured_initial_seed = getattr(args, "tracik_initial_seed_q", None)
        self.initial_solve_seed = (
            None
            if configured_initial_seed is None
            else np.clip(
                np.asarray(configured_initial_seed, dtype=np.float64).reshape(6),
                self.lower,
                self.upper,
            )
        )
        configured_fixed_seed = getattr(args, "tracik_fixed_seed_q", None)
        self.fixed_solve_seed = (
            None
            if configured_fixed_seed is None
            else np.clip(
                np.asarray(configured_fixed_seed, dtype=np.float64).reshape(6),
                self.lower,
                self.upper,
            )
        )
        self.trajectory = SimOnlineQuinticTrajectory(
            initial_q,
            args.tracik_max_joint_speed,
            args.tracik_max_joint_acceleration,
            args.tracik_max_segment_duration,
        )
        self.last_solution = initial_q.copy()
        self.branch_reference_q = initial_q.copy()
        self.last_reachable_pose = np.asarray(self.solver.fk(initial_q), dtype=np.float64).reshape(7)
        self.last_waypoint_velocity = np.zeros(6, dtype=np.float64)
        self.last_solve_step = None
        self.solve_count = 0
        self.failure_count = 0
        self.last_solve_ms = 0.0
        self.last_segment_duration = 0.0
        self.last_error = ""
        self.last_target_pose = None
        self.last_candidate = None
        self.last_solve_seed = None
        self.last_measured_q = initial_q.copy()
        self.last_waypoint_step = 0.0
        self.projection_count = 0
        self.projection_failure_count = 0
        self.nearest_position_count = 0
        self.branch_guard_count = 0
        self.branch_reject_count = 0
        self.branch_recovery_count = 0
        self.branch_position_fallback_count = 0
        self.local_servo_count = 0
        self.local_servo_reject_count = 0
        self.local_servo_projection_count = 0
        self.last_local_servo_alpha = 1.0
        self.last_local_servo_reason = ""
        self.last_branch_guard = None
        self.last_branch_candidates = 0
        self.last_branch_score = None
        self.last_branch_recovered = False
        self.last_branch_position_fallback = False
        self.last_projection_alpha = 1.0
        self.last_projection_ms = 0.0
        self.last_projection_target_pose = None
        self.initial_plan_active = False

    def reset(self, q, sim_time):
        q = np.clip(np.asarray(q, dtype=np.float64).reshape(6), self.lower, self.upper)
        self.trajectory = SimOnlineQuinticTrajectory(
            q,
            self.trajectory.max_speed,
            self.trajectory.max_acceleration,
            self.trajectory.max_duration,
        )
        self.trajectory.start_time = float(sim_time)
        self.last_solution = q.copy()
        self.branch_reference_q = q.copy()
        self.last_reachable_pose = np.asarray(self.solver.fk(q), dtype=np.float64).reshape(7)
        self.last_waypoint_velocity[:] = 0.0
        self.last_solve_step = None
        self.last_error = ""
        self.last_measured_q = q.copy()
        self.last_local_servo_alpha = 1.0
        self.last_local_servo_reason = ""
        self.initial_plan_active = False

    @staticmethod
    def _interpolate_target(start_pose, target_pose, alpha, position_only):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        position = (1.0 - alpha) * start_pose[:3] + alpha * target_pose[:3]
        if position_only:
            return position
        start_quat = np.asarray(start_pose[3:7], dtype=np.float64)
        target_quat = np.asarray(target_pose[3:7], dtype=np.float64)
        if float(np.dot(start_quat, target_quat)) < 0.0:
            target_quat = -target_quat
        quat = (1.0 - alpha) * start_quat + alpha * target_quat
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1.0e-12:
            quat = start_quat.copy()
        else:
            quat /= quat_norm
        return np.concatenate([position, quat])

    def _target_branch_guard_rule(self, target_pose, position_only=False):
        """Return the lightweight geometric joint1 branch constraint.

        For the A2W door trajectories, a target to the robot's negative lateral
        side (the handle-approach side) must not be solved by folding joint1 to
        the positive ~pi/2 branch.  The rule uses only the robot-base target, so
        the exact same selection can be used in simulation and on hardware.
        """
        if bool(position_only):
            return None
        pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if pose.size < 2:
            return None
        if (
            float(pose[0]) >= float(self.args.tracik_branch_forward_min)
            and float(pose[1]) <= -float(self.args.tracik_branch_lateral_deadband)
        ):
            return {
                "side": "negative_lateral",
                "joint1_upper": float(self.args.tracik_branch_joint1_upper),
                "target_azimuth": float(math.atan2(float(pose[1]), float(pose[0]))),
            }
        return None

    def _branch_guard_for_target(self, target_pose, position_only=False):
        if not bool(getattr(self.args, "tracik_branch_continuity", True)):
            return None
        return self._target_branch_guard_rule(target_pose, position_only=position_only)

    def _candidate_within_controller_limits(self, candidate):
        candidate = np.asarray(candidate, dtype=np.float64).reshape(6)
        tolerance = 1.0e-5
        return bool(
            np.all(np.isfinite(candidate))
            and np.all(candidate >= self.lower - tolerance)
            and np.all(candidate <= self.upper + tolerance)
        )

    @staticmethod
    def _append_unique_seed(seeds, seed):
        seed = np.asarray(seed, dtype=np.float64).reshape(6)
        if not any(float(np.max(np.abs(seed - existing))) < 1.0e-6 for existing in seeds):
            seeds.append(seed)

    def _branch_seed_options(self, solve_seed, measured_q, branch_guard):
        max_seeds = max(1, int(self.args.tracik_branch_candidate_seeds))
        seeds = []
        solve_seed = np.clip(np.asarray(solve_seed, dtype=np.float64).reshape(6), self.lower, self.upper)
        measured_q = np.clip(np.asarray(measured_q, dtype=np.float64).reshape(6), self.lower, self.upper)
        self._append_unique_seed(seeds, solve_seed)
        self._append_unique_seed(seeds, measured_q)
        self._append_unique_seed(seeds, self.last_solution)

        azimuth = float(branch_guard["target_azimuth"])
        upper = float(
            branch_guard.get("effective_joint1_upper", branch_guard["joint1_upper"])
        )
        for offset in (0.25, 0.55, 0.85, 1.15):
            if len(seeds) >= max_seeds:
                break
            candidate_seed = measured_q.copy()
            desired_q1 = min(upper - 0.05, azimuth - offset)
            desired_q1 = float(np.clip(desired_q1, self.lower[0], self.upper[0]))
            q1_delta = desired_q1 - float(candidate_seed[0])
            candidate_seed[0] = desired_q1
            # joint1 and joint5 largely compensate one another for the forward
            # gripper orientation.  Counter-rotating joint5 makes the alternate
            # seed useful without imposing a fixed full-arm posture.
            candidate_seed[4] = float(
                np.clip(candidate_seed[4] - q1_delta, self.lower[4], self.upper[4])
            )
            self._append_unique_seed(seeds, candidate_seed)

        if len(seeds) < max_seeds:
            demonstrated_approach_seed = np.asarray(
                [-1.0, 1.4, -0.7, -0.9, 1.0, 0.0],
                dtype=np.float64,
            )
            self._append_unique_seed(
                seeds,
                np.clip(demonstrated_approach_seed, self.lower, self.upper),
            )

        while len(seeds) < max_seeds:
            random_seed = self.rng.uniform(self.lower, self.upper)
            random_seed[0] = self.rng.uniform(
                self.lower[0],
                min(self.upper[0], upper),
            )
            self._append_unique_seed(seeds, random_seed)
        return seeds[:max_seeds]

    def _branch_candidate_score(self, candidate, measured_q, branch_guard):
        candidate = np.asarray(candidate, dtype=np.float64).reshape(6)
        measured_q = np.asarray(measured_q, dtype=np.float64).reshape(6)
        weights = np.ones(6, dtype=np.float64)
        weights[0] = float(self.args.tracik_branch_joint1_weight)
        measured_cost = float(np.sum(weights * np.square(candidate - measured_q)))
        command_cost = float(np.sum(weights * np.square(candidate - self.last_solution)))
        azimuth_cost = float(
            np.square(candidate[0] - float(branch_guard["target_azimuth"]))
        )
        return measured_cost + 0.5 * command_cost + 0.25 * azimuth_cost

    @staticmethod
    def _quat_orientation_error_xyzw(desired, current):
        desired = np.asarray(desired, dtype=np.float64).reshape(4)
        current = np.asarray(current, dtype=np.float64).reshape(4)
        desired /= max(float(np.linalg.norm(desired)), 1.0e-12)
        current /= max(float(np.linalg.norm(current)), 1.0e-12)
        x1, y1, z1, w1 = desired
        x2, y2, z2, w2 = -current[0], -current[1], -current[2], current[3]
        delta = np.asarray(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )
        return delta[:3] * (1.0 if delta[3] >= 0.0 else -1.0)

    def _advance_branch_reference(self, target_pose, measured_q, branch_guard):
        """Track a continuous posture branch with a portable FK-based local DLS.

        This reference is not sent to the robot directly.  It supplies a
        topology-stable full-pose target that TRAC-IK solves below.  Numerical
        FK keeps this implementation independent of Isaac Gym's Jacobian and
        therefore usable with the same URDF on hardware.
        """
        q = np.asarray(self.branch_reference_q, dtype=np.float64).reshape(6).copy()
        measured_q = np.asarray(measured_q, dtype=np.float64).reshape(6)
        # If the reference became stale (reset, external intervention, or a
        # long unguarded phase), restart it from the actual arm configuration.
        if float(np.max(np.abs(q - measured_q))) > float(
            self.args.tracik_branch_reference_reset_distance
        ):
            q = measured_q.copy()

        target_arr = np.asarray(target_pose, dtype=np.float64).reshape(7)
        finite_difference = float(self.args.tracik_branch_reference_fd_epsilon)
        damping = float(self.args.tracik_branch_reference_damping)
        max_joint_step = float(self.args.tracik_branch_reference_joint_step)
        iterations = max(1, int(self.args.tracik_branch_reference_iterations))
        for _ in range(iterations):
            current_pose = np.asarray(self.solver.fk(q), dtype=np.float64).reshape(7)
            error = np.concatenate(
                [
                    target_arr[:3] - current_pose[:3],
                    self._quat_orientation_error_xyzw(target_arr[3:7], current_pose[3:7]),
                ]
            )
            jacobian = np.zeros((6, 6), dtype=np.float64)
            for joint_index in range(6):
                q_minus = q.copy()
                q_plus = q.copy()
                q_minus[joint_index] = max(
                    self.lower[joint_index],
                    q_minus[joint_index] - finite_difference,
                )
                q_plus[joint_index] = min(
                    self.upper[joint_index],
                    q_plus[joint_index] + finite_difference,
                )
                denominator = float(q_plus[joint_index] - q_minus[joint_index])
                if denominator <= 1.0e-12:
                    continue
                pose_minus = np.asarray(self.solver.fk(q_minus), dtype=np.float64).reshape(7)
                pose_plus = np.asarray(self.solver.fk(q_plus), dtype=np.float64).reshape(7)
                jacobian[:3, joint_index] = (
                    pose_plus[:3] - pose_minus[:3]
                ) / denominator
                jacobian[3:6, joint_index] = (
                    2.0
                    * self._quat_orientation_error_xyzw(
                        pose_plus[3:7],
                        pose_minus[3:7],
                    )
                    / denominator
                )
            rotation_weight = float(self.args.tracik_branch_reference_rotation_weight)
            weights = np.asarray(
                [1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight],
                dtype=np.float64,
            )
            weighted_jacobian = jacobian * weights[:, None]
            weighted_error = error * weights
            jacobian_t = weighted_jacobian.T
            lhs = (
                weighted_jacobian @ jacobian_t
                + np.eye(6, dtype=np.float64) * (damping * damping)
            )
            try:
                delta = jacobian_t @ np.linalg.solve(lhs, weighted_error)
            except np.linalg.LinAlgError:
                break
            delta = np.clip(delta, -max_joint_step, max_joint_step)
            q = np.clip(q + delta, self.lower, self.upper)
            q[0] = min(q[0], float(branch_guard["joint1_upper"]))
        self.branch_reference_q = q.copy()
        return q

    def _solve_with_branch_continuity(
        self,
        target_pose,
        position_only,
        solve_seed,
        measured_q,
        branch_guard=None,
    ):
        """Solve once normally, then recover only if it selects a guarded branch."""
        if branch_guard is None:
            branch_guard = self._branch_guard_for_target(target_pose, position_only=position_only)
        self.last_branch_guard = branch_guard
        self.last_branch_candidates = 0
        self.last_branch_score = None
        self.last_branch_recovered = False
        self.last_branch_position_fallback = False

        first = self.solver.ik(
            target_pose,
            seed_joints=solve_seed,
            position_only=bool(position_only),
            max_restarts=1,
            strict_seed=True,
        )
        if branch_guard is None:
            return first

        self.branch_guard_count += 1
        # Keep the demonstrated branch bound strict.  Letting the bound follow
        # ``last_solution`` by even a small amount permits a sequence of locally
        # continuous IK results to creep into the opposite branch over time.
        # A recovered target may be far in joint space, but it is always handed
        # to the velocity/acceleration-limited quintic trajectory below.
        branch_guard = dict(branch_guard)
        branch_guard["effective_joint1_upper"] = float(branch_guard["joint1_upper"])
        self.last_branch_guard = branch_guard
        upper = float(branch_guard["effective_joint1_upper"])
        candidates = []
        if first is not None:
            first = np.asarray(first, dtype=np.float64).reshape(6)
            if self._candidate_within_controller_limits(first) and float(first[0]) <= upper:
                candidates.append(first)
            else:
                self.branch_reject_count += 1

        seed_options = self._branch_seed_options(solve_seed, measured_q, branch_guard)
        for seed in seed_options:
            candidate = self.solver.ik(
                target_pose,
                seed_joints=seed,
                position_only=bool(position_only),
                max_restarts=1,
                strict_seed=True,
            )
            if candidate is None:
                continue
            candidate = np.asarray(candidate, dtype=np.float64).reshape(6)
            if not self._candidate_within_controller_limits(candidate):
                continue
            if float(candidate[0]) > upper:
                self.branch_reject_count += 1
                continue
            if any(float(np.max(np.abs(candidate - saved))) < 1.0e-4 for saved in candidates):
                continue
            candidates.append(candidate)

        full_pose_candidates = candidates
        selected_full = None
        selected_full_score = None
        if full_pose_candidates:
            scored_full = [
                (self._branch_candidate_score(candidate, measured_q, branch_guard), candidate)
                for candidate in full_pose_candidates
            ]
            selected_full_score, selected_full = min(scored_full, key=lambda item: item[0])
            selected_full_step = float(
                np.max(np.abs(selected_full - self.last_solution))
            )
            if (
                not bool(getattr(self.args, "tracik_branch_position_fallback", True))
                or self.max_waypoint_step <= 0.0
                or selected_full_step <= self.max_waypoint_step
            ):
                self.last_branch_candidates = len(full_pose_candidates)
                self.last_branch_score = float(selected_full_score)
                self.last_branch_recovered = True
                self.branch_recovery_count += 1
                return selected_full.copy()

        candidates = []
        if bool(getattr(self.args, "tracik_branch_position_fallback", True)):
            # Old Gym-Jacobian demonstrations used a soft orientation
            # objective.  Around a Z1 wrist singularity, a strict 6D solve of
            # their instantaneous EE target can therefore jump to a folded q1
            # branch.  Advance a portable FK-based local reference on the
            # continuous branch, then ask TRAC-IK to solve the reference's full
            # pose.  The reference itself is never sent as a joint command:
            # every accepted target still comes from full-pose TRAC-IK.
            reference_q = np.asarray(self.branch_reference_q, dtype=np.float64).reshape(6)
            reference_pose = np.asarray(self.solver.fk(reference_q), dtype=np.float64).reshape(7)
            local_seed_options = [reference_q, measured_q, self.last_solution]
            for seed in local_seed_options:
                candidate = self.solver.ik(
                    reference_pose,
                    seed_joints=seed,
                    position_only=False,
                    max_restarts=1,
                    strict_seed=True,
                )
                if candidate is None:
                    continue
                candidate = np.asarray(candidate, dtype=np.float64).reshape(6)
                if not self._candidate_within_controller_limits(candidate):
                    continue
                if float(candidate[0]) > upper:
                    self.branch_reject_count += 1
                    continue
                if any(float(np.max(np.abs(candidate - saved))) < 1.0e-4 for saved in candidates):
                    continue
                candidates.append(candidate)
            if candidates:
                self.last_branch_position_fallback = True
                self.branch_position_fallback_count += 1

        self.last_branch_candidates = len(candidates)
        if not candidates:
            if selected_full is None:
                return None
            self.last_branch_score = float(selected_full_score)
            self.last_branch_recovered = False
            return selected_full.copy()
        scored = [
            (self._branch_candidate_score(candidate, measured_q, branch_guard), candidate)
            for candidate in candidates
        ]
        score, selected = min(scored, key=lambda item: item[0])
        self.last_branch_score = float(score)
        selected_step = float(np.max(np.abs(selected - self.last_solution)))
        self.last_branch_recovered = bool(
            self.max_waypoint_step <= 0.0
            or selected_step <= self.max_waypoint_step
        )
        self.branch_recovery_count += 1
        return selected.copy()

    def _local_servo_accepts(
        self,
        candidate,
        measured_q,
        branch_guard=None,
        allow_large_step=False,
    ):
        if candidate is None:
            self.last_local_servo_reason = "ik_failed"
            return False
        candidate = np.asarray(candidate, dtype=np.float64).reshape(6)
        measured_q = np.asarray(measured_q, dtype=np.float64).reshape(6)
        if not self._candidate_within_controller_limits(candidate):
            self.last_local_servo_reason = "joint_limits"
            return False
        if branch_guard is not None:
            upper = float(
                branch_guard.get("effective_joint1_upper", branch_guard["joint1_upper"])
            )
            if float(candidate[0]) > upper:
                self.last_local_servo_reason = (
                    f"branch_q1:{float(candidate[0]):.4f}>{upper:.4f}"
                )
                return False
        if bool(allow_large_step):
            self.last_local_servo_reason = ""
            return True
        delta = np.abs(candidate - measured_q)
        max_joint_step = float(getattr(self.args, "tracik_local_servo_max_joint_step", 0.25))
        max_q1_step = float(getattr(self.args, "tracik_local_servo_max_q1_step", 0.18))
        if max_q1_step > 0.0 and float(delta[0]) > max_q1_step:
            self.last_local_servo_reason = f"q1_step:{float(delta[0]):.4f}>{max_q1_step:.4f}"
            return False
        if max_joint_step > 0.0 and float(np.max(delta)) > max_joint_step:
            self.last_local_servo_reason = (
                f"joint_step:{float(np.max(delta)):.4f}>{max_joint_step:.4f}"
            )
            return False
        self.last_local_servo_reason = ""
        return True

    def _solve_local_servo(
        self,
        target_pose,
        position_only,
        measured_q,
        seed_q=None,
        allow_large_step=False,
    ):
        """Single-seed local TRAC-IK servo around the measured arm posture.

        This mode is intentionally small and portable: solve from q_current,
        reject large joint jumps, and if needed shrink the EE target toward the
        current FK pose.  It avoids task-specific branch search while keeping
        TRAC-IK as the only IK backend.
        """
        measured_q = np.clip(
            np.asarray(measured_q, dtype=np.float64).reshape(6),
            self.lower,
            self.upper,
        )
        seed_q = (
            measured_q
            if seed_q is None
            else np.clip(
                np.asarray(seed_q, dtype=np.float64).reshape(6),
                self.lower,
                self.upper,
            )
        )
        self.local_servo_count += 1
        self.last_local_servo_alpha = 1.0
        self.last_local_servo_reason = ""
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        branch_guard = (
            self._target_branch_guard_rule(
                target_pose,
                position_only=bool(position_only),
            )
            if bool(getattr(self.args, "tracik_local_servo_branch_anchor", False))
            else None
        )
        if branch_guard is not None:
            upper = float(
                branch_guard.get("effective_joint1_upper", branch_guard["joint1_upper"])
            )
            if float(seed_q[0]) > upper:
                anchored_seed = seed_q.copy()
                desired_q1 = float(np.clip(upper, self.lower[0], self.upper[0]))
                q1_delta = desired_q1 - float(anchored_seed[0])
                anchored_seed[0] = desired_q1
                # Joint1 and joint5 compensate for gripper-forward poses.  Keep
                # the seed on the same end-effector orientation family without
                # doing a multi-seed branch search.
                anchored_seed[4] = float(
                    np.clip(anchored_seed[4] - q1_delta, self.lower[4], self.upper[4])
                )
                seed_q = anchored_seed
        solution = self.solver.ik(
            target_pose,
            seed_joints=seed_q,
            position_only=bool(position_only),
            max_restarts=1,
            strict_seed=True,
        )
        if self._local_servo_accepts(
            solution,
            measured_q,
            branch_guard=branch_guard,
            allow_large_step=allow_large_step,
        ):
            return np.clip(np.asarray(solution, dtype=np.float64).reshape(6), self.lower, self.upper)

        self.local_servo_reject_count += 1
        if not self.project_unreachable or self.projection_iterations <= 0:
            return None

        reject_reason = self.last_local_servo_reason
        start_pose = np.asarray(self.solver.fk(measured_q), dtype=np.float64).reshape(7)
        low = 0.0
        high = 1.0
        best_solution = None
        for _ in range(self.projection_iterations):
            alpha = 0.5 * (low + high)
            candidate_target = self._interpolate_target(
                start_pose,
                target_pose,
                alpha,
                bool(position_only),
            )
            candidate = self.solver.ik(
                candidate_target,
                seed_joints=seed_q,
                position_only=bool(position_only),
                max_restarts=1,
                strict_seed=True,
            )
            if self._local_servo_accepts(candidate, measured_q, branch_guard=branch_guard):
                low = alpha
                best_solution = np.clip(
                    np.asarray(candidate, dtype=np.float64).reshape(6),
                    self.lower,
                    self.upper,
                )
            else:
                high = alpha

        self.last_local_servo_alpha = float(low)
        if best_solution is not None and low >= self.projection_min_alpha:
            self.local_servo_projection_count += 1
            self.last_local_servo_reason = ""
            return best_solution
        self.last_local_servo_reason = reject_reason
        return None

    def _project_to_reachable_target(
        self,
        target_pose,
        position_only,
        solve_seed=None,
        measured_q=None,
        branch_guard=None,
    ):
        if not self.project_unreachable or self.projection_iterations <= 0:
            return None, 0.0, None
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        start_pose = self.last_reachable_pose.copy()
        low = 0.0
        high = 1.0
        best_solution = self.last_solution.copy()
        best_target = start_pose[:3].copy() if position_only else start_pose.copy()
        for _ in range(self.projection_iterations):
            alpha = 0.5 * (low + high)
            candidate_target = self._interpolate_target(
                start_pose,
                target_pose,
                alpha,
                bool(position_only),
            )
            candidate_solution = self._solve_with_branch_continuity(
                candidate_target,
                bool(position_only),
                solve_seed=(
                    np.asarray(solve_seed, dtype=np.float64).reshape(6)
                    if solve_seed is not None
                    else (
                        self.fixed_solve_seed
                        if self.fixed_solve_seed is not None
                        else self.last_solution
                    )
                ),
                measured_q=(
                    self.last_solution
                    if measured_q is None
                    else np.asarray(measured_q, dtype=np.float64).reshape(6)
                ),
                branch_guard=branch_guard,
            )
            if candidate_solution is not None:
                candidate_solution = np.clip(
                    np.asarray(candidate_solution, dtype=np.float64).reshape(6),
                    self.lower,
                    self.upper,
                )
                waypoint_step = float(np.max(np.abs(candidate_solution - self.last_solution)))
                if self.max_waypoint_step > 0.0 and waypoint_step > self.max_waypoint_step:
                    candidate_solution = None
            if candidate_solution is None:
                high = alpha
            else:
                low = alpha
                best_solution = candidate_solution.copy()
                best_target = np.asarray(candidate_target, dtype=np.float64).copy()
        if low < self.projection_min_alpha:
            return None, low, best_target
        return best_solution, low, best_target

    def update(
        self,
        step,
        target_pose,
        position_only=False,
        nearest_position_solution=None,
        nearest_position_solve_ms=0.0,
        solve_seed_override=None,
        measured_q=None,
        initial_move=False,
    ):
        step = int(step)
        now = step * self.sim_dt
        leaving_initial_move = self.initial_plan_active and not bool(initial_move)
        if leaving_initial_move:
            self.initial_plan_active = False
        solve_due = (
            leaving_initial_move
            or self.last_solve_step is None
            or step - self.last_solve_step >= self.command_stride
        )
        if bool(initial_move) and self.initial_plan_active:
            solve_due = False
        if solve_due:
            first_solve_after_reset = self.last_solve_step is None
            # Keep the online trajectory anchored at the current commanded joints, but
            # optionally seed the first nonlinear IK solve from a known manipulation
            # posture.  This selects the demonstrated IK branch without teleporting the
            # arm to that posture before the first trajectory segment is generated.
            solve_seed = (
                np.asarray(solve_seed_override, dtype=np.float64).reshape(6)
                if solve_seed_override is not None
                else (
                    self.fixed_solve_seed
                    if self.fixed_solve_seed is not None
                    else (
                        self.initial_solve_seed
                        if first_solve_after_reset and self.initial_solve_seed is not None
                        else self.last_solution
                    )
                )
            )
            solve_seed = np.clip(solve_seed, self.lower, self.upper)
            measured_q = (
                self.last_solution.copy()
                if measured_q is None
                else np.clip(
                    np.asarray(measured_q, dtype=np.float64).reshape(6),
                    self.lower,
                    self.upper,
                )
            )
            self.last_measured_q = measured_q.copy()
            self.last_solve_seed = solve_seed.copy()
            self.last_target_pose = np.asarray(target_pose, dtype=np.float64).copy()
            self.last_projection_alpha = 1.0
            self.last_projection_ms = 0.0
            self.last_projection_target_pose = None
            used_projection = False
            used_nearest_position = False
            used_local_servo_projection = False
            projection_attempted = False
            branch_guard = None
            self.last_branch_recovered = False
            local_servo_enabled = bool(getattr(self.args, "tracik_local_servo", False))
            start_ns = time.perf_counter_ns()
            if bool(position_only) and nearest_position_solution is not None:
                solution = np.clip(
                    np.asarray(nearest_position_solution, dtype=np.float64).reshape(6),
                    self.lower,
                    self.upper,
                )
                if self.max_waypoint_step > 0.0:
                    solution = np.clip(
                        solution,
                        self.last_solution - self.max_waypoint_step,
                        self.last_solution + self.max_waypoint_step,
                    )
                strict_solution = solution
                needs_projection = False
                used_nearest_position = True
                self.nearest_position_count += 1
            elif local_servo_enabled:
                before_local_projections = self.local_servo_projection_count
                solution = self._solve_local_servo(
                    target_pose,
                    bool(position_only),
                    measured_q,
                    seed_q=solve_seed,
                    allow_large_step=bool(first_solve_after_reset or initial_move),
                )
                strict_solution = solution
                used_local_servo_projection = (
                    self.local_servo_projection_count > before_local_projections
                )
                used_projection = used_local_servo_projection
                needs_projection = False
            else:
                branch_guard = self._branch_guard_for_target(
                    target_pose,
                    position_only=bool(position_only),
                )
                if branch_guard is None:
                    self.branch_reference_q = measured_q.copy()
                else:
                    self._advance_branch_reference(
                        target_pose,
                        measured_q,
                        branch_guard,
                    )
                solution = self._solve_with_branch_continuity(
                    target_pose,
                    bool(position_only),
                    solve_seed,
                    measured_q,
                    branch_guard=branch_guard,
                )
                strict_branch_recovered = bool(self.last_branch_recovered)
                strict_solution = solution
                strict_waypoint_jump = 0.0
                if strict_solution is not None:
                    strict_solution = np.clip(
                        np.asarray(strict_solution, dtype=np.float64).reshape(6),
                        self.lower,
                        self.upper,
                    )
                    strict_waypoint_jump = float(
                        np.max(np.abs(strict_solution - self.last_solution))
                    )
                needs_projection = (
                    strict_solution is None
                    or (
                        self.max_waypoint_step > 0.0
                        and strict_waypoint_jump > self.max_waypoint_step
                        and not strict_branch_recovered
                    )
                )
            if needs_projection and not first_solve_after_reset:
                projection_attempted = self.project_unreachable and self.projection_iterations > 0
                projection_start_ns = time.perf_counter_ns()
                projected_solution, projection_alpha, projection_target = self._project_to_reachable_target(
                    target_pose,
                    bool(position_only),
                    solve_seed=solve_seed,
                    measured_q=measured_q,
                    branch_guard=branch_guard,
                )
                self.last_projection_ms = (time.perf_counter_ns() - projection_start_ns) * 1.0e-6
                self.last_projection_alpha = float(projection_alpha)
                self.last_projection_target_pose = projection_target
                if projected_solution is not None:
                    solution = projected_solution
                    used_projection = True
                    self.projection_count += 1
                elif self.project_unreachable:
                    solution = strict_solution
                    self.projection_failure_count += 1
            if solution is None and self.max_restarts > 1 and not projection_attempted:
                if branch_guard is None:
                    solution = self.solver.ik(
                        target_pose,
                        seed_joints=solve_seed,
                        position_only=bool(position_only),
                        max_restarts=self.max_restarts,
                        strict_seed=False,
                        rng=self.rng,
                    )
            self.last_solve_ms = (
                (time.perf_counter_ns() - start_ns) * 1.0e-6
                + float(nearest_position_solve_ms)
            )
            self.solve_count += 1
            self.last_solve_step = step
            if solution is None:
                self.failure_count += 1
                self.last_error = "ik_failed"
            else:
                solution = np.clip(np.asarray(solution, dtype=np.float64).reshape(6), self.lower, self.upper)
                self.last_candidate = solution.copy()
                waypoint_step = float(np.max(np.abs(solution - self.last_solution)))
                self.last_waypoint_step = waypoint_step
                if (
                    not first_solve_after_reset
                    and not local_servo_enabled
                    and self.max_waypoint_step > 0.0
                    and waypoint_step > self.max_waypoint_step
                    and not bool(self.last_branch_recovered)
                ):
                    self.failure_count += 1
                    self.last_error = (
                        f"ik_waypoint_jump:{waypoint_step:.4f}>"
                        f"{self.max_waypoint_step:.4f}"
                    )
                else:
                    if bool(initial_move):
                        waypoint_velocity = np.zeros(6, dtype=np.float64)
                        nominal_duration = self.initial_segment_duration
                    else:
                        raw_velocity = (solution - self.last_solution) / self.command_period
                        raw_velocity = np.clip(
                            raw_velocity,
                            -self.trajectory.max_speed,
                            self.trajectory.max_speed,
                        )
                        waypoint_velocity = (
                            self.velocity_alpha * raw_velocity
                            + (1.0 - self.velocity_alpha) * self.last_waypoint_velocity
                        )
                        nominal_duration = self.command_period
                    self.last_segment_duration = self.trajectory.retarget(
                        solution,
                        now,
                        nominal_duration,
                        goal_velocity=waypoint_velocity,
                    )
                    self.last_solution = solution.copy()
                    self.last_reachable_pose = np.asarray(
                        self.solver.fk(solution),
                        dtype=np.float64,
                    ).reshape(7)
                    self.last_waypoint_velocity = waypoint_velocity.copy()
                    if used_nearest_position:
                        self.last_error = "nearest_position_target"
                    elif local_servo_enabled and used_local_servo_projection:
                        self.last_error = "local_servo_projected_target"
                    elif local_servo_enabled:
                        self.last_error = "local_servo_target"
                    elif self.last_branch_recovered:
                        self.last_error = "branch_recovered_target"
                    elif used_projection:
                        self.last_error = "projected_target"
                    else:
                        self.last_error = ""
                    if bool(initial_move):
                        self.initial_plan_active = True
        q_command, qd_command, qdd_command, _ = self.trajectory.sample(now + self.sim_dt)
        return q_command, qd_command, qdd_command


def default_a2w_float_ik_config_path():
    override = os.environ.get(A2W_FLOAT_IK_CONFIG_ENV_VAR, "").strip()
    if override:
        return str(Path(override).expanduser())
    return str(A2W_FLOAT_IK_DEFAULT_CONFIG_PATH)


def load_yaml_mapping(path):
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError(f"PyYAML is required to read A2W float IK config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"A2W float IK config must be a mapping: {path}")
    return data


def flatten_a2w_config(data, prefix=()):
    flat = {}
    for key, value in dict(data or {}).items():
        if key in ("randomization_absolute_ranges",):
            continue
        if isinstance(value, dict):
            flat.update(flatten_a2w_config(value, prefix + (str(key),)))
        else:
            flat[str(key)] = value
    return flat


def cli_flag_present(argv, flag):
    prefix = flag + "="
    return any(str(token) == flag or str(token).startswith(prefix) for token in argv)


def cli_attr_was_set(argv, attr):
    attr = str(attr)
    return cli_flag_present(argv, f"--{attr}") or cli_flag_present(argv, f"--no_{attr}")


def apply_a2w_float_ik_config_defaults(args, argv):
    config_path = Path(getattr(args, "a2w_float_ik_config", default_a2w_float_ik_config_path())).expanduser()
    config = load_yaml_mapping(config_path)
    flat_defaults = flatten_a2w_config(config)
    setattr(args, "a2w_float_ik_config", str(config_path))
    setattr(args, "_a2w_float_ik_config_defaults", dict(flat_defaults))

    ranges = config.get("randomization_absolute_ranges", {}) if isinstance(config, dict) else {}
    if ranges:
        if not isinstance(ranges, dict):
            raise ValueError("randomization_absolute_ranges must be a mapping of name: [min, max].")
        for name, pair in ranges.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"randomization_absolute_ranges.{name} must contain exactly [min, max].")
            IKPUSH_DEFAULT_ENV_RANGES[str(name)] = (float(pair[0]), float(pair[1]))

    for attr, value in flat_defaults.items():
        if attr == "a2w_float_ik_config":
            continue
        if not hasattr(args, attr) and attr not in A2W_FLOAT_IK_CONFIG_DERIVED_ATTRS:
            raise ValueError(f"Unknown key in A2W float IK config {config_path}: {attr}")
        if cli_attr_was_set(argv, attr):
            continue
        current = getattr(args, attr, None)
        if isinstance(current, bool) and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                value = True
            elif lowered in ("0", "false", "no", "off"):
                value = False
        setattr(args, attr, value)
    return args


def _resolve_a2w_runtime_path(path_value):
    if path_value is None:
        return path_value
    text = str(path_value).strip()
    if not text:
        return path_value
    expanded = Path(text).expanduser()
    if expanded.exists():
        return str(expanded)

    # Config files are often copied between machines and may still contain an
    # absolute path like /home/sivan/.../high-level/data/asset/a2wz1.  If that
    # path does not exist locally, keep the part after "high-level/" and map it
    # onto this checkout's HIGH_LEVEL_ROOT.
    parts = expanded.parts
    if "high-level" in parts:
        high_level_idx = parts.index("high-level")
        suffix = Path(*parts[high_level_idx + 1 :])
        candidate = HIGH_LEVEL_ROOT / suffix
        if candidate.exists():
            return str(candidate)

    # Also support repo-root relative paths such as high-level/data/... and
    # high-level-root relative paths such as data/...
    candidates = []
    if not expanded.is_absolute():
        candidates.extend([REPO_ROOT / expanded, HIGH_LEVEL_ROOT / expanded])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(expanded)


def normalize_a2w_runtime_paths(args):
    for attr in ("asset_root", "door_cfg", "depth_aug_config", "dp_raw_root"):
        if hasattr(args, attr):
            setattr(args, attr, _resolve_a2w_runtime_path(getattr(args, attr)))
    return args


def setup_viewer_pause_shortcut(gym, viewer):
    if viewer is None:
        return
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, VIEWER_PAUSE_ACTION)
    print("Viewer control: press SPACE to pause/resume simulation.", flush=True)


def handle_viewer_pause(gym, sim, viewer):
    """Return False when the viewer closes; otherwise handle SPACE pause/resume."""
    if viewer is None:
        return True

    pause_requested = any(
        event.action == VIEWER_PAUSE_ACTION and event.value > 0
        for event in gym.query_viewer_action_events(viewer)
    )
    if not pause_requested:
        return not gym.query_viewer_has_closed(viewer)

    print("Simulation paused. Press SPACE to resume.", flush=True)
    while not gym.query_viewer_has_closed(viewer):
        # Keep the viewer responsive without advancing physics or the trajectory.
        gym.draw_viewer(viewer, sim, True)
        if cv2 is not None:
            cv2.waitKey(1)
        for event in gym.query_viewer_action_events(viewer):
            if event.action == VIEWER_PAUSE_ACTION and event.value > 0:
                print("Simulation resumed.", flush=True)
                return True
        time.sleep(0.1)
    return False


def configure_door_twin_args(args):
    skill_ref = str(getattr(args, "skill_program_json", "") or "").strip()
    program = None
    if skill_ref:
        if skill_ref.lower() in ("auto", "default"):
            program = default_skill_program_from_args(args)
        else:
            program = load_skill_program(skill_ref)
        apply_program_to_args(args, program)

    if (bool(getattr(args, "save_failed_rollouts", False)) or bool(getattr(args, "dump_keyframe_images", False))) and not str(
        getattr(args, "door_twin_log_dir", "") or ""
    ).strip():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.door_twin_log_dir = str(DOOR_TWIN_DEFAULT_LOG_ROOT / f"a2w_float_ik_{timestamp}")

    args.door_twin_skill_program = program
    args.door_twin_enabled = bool(
        program is not None
        or str(getattr(args, "door_twin_log_dir", "") or "").strip()
        or bool(getattr(args, "save_failed_rollouts", False))
        or bool(getattr(args, "dump_keyframe_images", False))
    )
    if (
        args.door_twin_enabled
        and str(getattr(args, "door_twin_log_dir", "") or "").strip()
        and not bool(getattr(args, "disable_collision_geom_check", False))
    ):
        # Digital-twin success accounting needs at least the cheap geometric collision check.
        args.enable_collision_geom_check = True
    if program is not None:
        print("DoorTwin skill_program:", json.dumps(program.to_dict(), ensure_ascii=False), flush=True)
    if str(getattr(args, "door_twin_log_dir", "") or "").strip():
        print(f"DoorTwin log dir: {args.door_twin_log_dir}", flush=True)
    return args


def parse_args():
    args = gymutil.parse_arguments(
        description="A2W+Z1 base+arm float IK door-pull demo.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": str(A2WZ1_DEFAULT_ASSET_ROOT)},
            {"name": "--asset_file", "type": str, "default": A2WZ1_DEFAULT_ASSET_FILE},
            {"name": "--rl_device", "type": str, "default": "cuda:0"},
            {"name": "--num_envs", "type": int, "default": 1},
            {"name": "--steps", "type": int, "default": 2405},
            {"name": "--seed", "type": int, "default": -1},
            {
                "name": "--a2w_float_ik_config",
                "type": str,
                "default": default_a2w_float_ik_config_path(),
                "help": (
                    "YAML file providing default A2W float IK parameters. "
                    f"Set {A2W_FLOAT_IK_CONFIG_ENV_VAR} to change the global default."
                ),
            },
            {"name": "--door_cfg", "type": str, "default": str(DEFAULT_DOOR_CFG)},
            {"name": "--door_name", "type": str, "default": ""},
            {"name": "--door_index", "type": int, "default": -1},
            {
                "name": "--door_asset_path_override",
                "type": str,
                "default": "",
                "help": "Override the selected door URDF path relative to assetFileDoor, useful for repaired digital-twin assets.",
            },
            {
                "name": "--door_selection",
                "type": str,
                "default": "diverse",
                "help": "Door cycling mode when --door_name/--door_index are unset: default, diverse, all, push_left, or push_right.",
            },
            {"name": "--door_prefer_name", "type": str, "default": "wc4"},
            {
                "name": "--door_max_unique_assets",
                "type": int,
                "default": 0,
                "help": "Max unique door assets for diverse selection; <=0 uses min(num_envs, available doors).",
            },
            {"name": "--door_actor_scale", "type": float, "default": 1.2},
            {
                "name": "--flip_door_motion_sign",
                "action": "store_true",
                "help": "Multiply the selected door hinge/open direction by -1 at runtime without editing the door cfg.",
            },
            {
                "name": "--pull_flip_door_actor",
                "action": "store_true",
                "default": False,
                "help": (
                    "Experimental: rotate the complete door asset by 180 degrees. The default "
                    "pull setup instead keeps the asset unchanged and spawns the robot on the "
                    "opposite side, which preserves the pre-pull grasp/rotate commands."
                ),
            },
            {
                "name": "--ignore_door_controller_overrides",
                "action": "store_true",
                "help": "Do not apply per-door controller_multipliers/controller_overrides from the door cfg; command-line values win.",
            },
            {
                "name": "--door_use_urdf_rgba",
                "action": "store_true",
                "help": "Use URDF <material><color rgba=...> for door visuals instead of mesh/MTL materials.",
            },
            {"name": "--door_x", "type": float, "default": 2.5},
            {"name": "--door_y", "type": float, "default": 0.0},
            {"name": "--door_z_offset", "type": float, "default": 0.01},
            {"name": "--no_door_side_walls", "action": "store_true", "help": "Disable the static side-wall actors beside each door."},
            {"name": "--door_wall_height", "type": float, "default": 2.2},
            {"name": "--door_wall_opening_width", "type": float, "default": 0.0, "help": "Wall opening width; <=0 uses the scaled door bounding-box width."},
            {"name": "--door_wall_opening_clearance", "type": float, "default": 0.0, "help": "Extra clearance added to each side of the automatic wall opening."},
            {"name": "--door_wall_side_width", "type": float, "default": 1.0},
            {"name": "--door_wall_thickness", "type": float, "default": 0.08},
            {"name": "--door_wall_gap", "type": float, "default": 0.0},
            {"name": "--door_wall_x_offset", "type": float, "default": 0.0},
            {"name": "--door_wall_y_offset", "type": float, "default": 0.0},
            {"name": "--robot_x", "type": float, "default": 4.1},
            {"name": "--robot_y", "type": float, "default": 0.0},
            {
                "name": "--robot_y_alignment",
                "type": str,
                "default": "handle",
                "help": "How to place the robot in Y relative to each door: handle or door_center.",
            },
            {"name": "--robot_z", "type": float, "default": 0.50},
            {"name": "--robot_pitch", "type": float, "default": 0.0},
            {"name": "--robot_yaw", "type": float, "default": math.pi},
            {"name": "--robot_front_offset", "type": float, "default": 0.55},
            {"name": "--robot_rear_offset", "type": float, "default": 0.65},
            {"name": "--stop_distance", "type": float, "default": 0.15},
            {"name": "--push_base_distance", "type": float, "default": 0.35},
            {"name": "--base_push_time_scale", "type": float, "default": 1.35},
            {"name": "--door_pass_clearance", "type": float, "default": 0.55},
            {
                "name": "--door_twin_min_forward_distance",
                "type": float,
                "default": MIN_DOOR_TWIN_FORWARD_DISTANCE_M,
                "help": "Minimum total base progress from approach stop through push+traverse; values below 1.94 m are clamped.",
            },
            {
                "name": "--no_pass_through_door",
                "action": "store_true",
                "default": False,
                "help": "Disable the default behavior that moves the base through the door during the push phase.",
            },
            {"name": "--push_base_yaw_delta", "type": float, "default": 0.0},
            {"name": "--walk_steps", "type": int, "default": 260},
            {"name": "--walk_min_speed", "type": float, "default": 0.20},
            {"name": "--no_dynamic_walk_steps", "action": "store_true"},
            {"name": "--initial_hold_steps", "type": int, "default": 150},
            {"name": "--initial_hold_move_steps", "type": int, "default": 100},
            {"name": "--grasp_steps", "type": int, "default": 50},
            {"name": "--grasp_hold_steps", "type": int, "default": 0},
            {"name": "--gripper_close_steps", "type": int, "default": 50},
            {"name": "--handle_rotate_steps", "type": int, "default": 100},
            {"name": "--door_push_steps", "type": int, "default": 300},
            {
                "name": "--door_pull_steps",
                "type": int,
                "default": 420,
                "help": (
                    "Nominal pull duration used to scale base retreat and EE pull progress. "
                    "The controller does not release until the measured door angle reaches "
                    "--pull_release_angle_deg."
                ),
            },
            {
                "name": "--pull_base_retreat_distance",
                "type": float,
                "default": 0.50,
                "help": "Primary distance the A2W base retreats while physically pulling the door.",
            },
            {
                "name": "--pull_base_retreat_steps",
                "type": int,
                "default": 200,
                "help": "Smooth base-retreat duration at the beginning of pull_door.",
            },
            {
                "name": "--pull_base_progress_lead_deg",
                "type": float,
                "default": 0.0,
                "help": (
                    "Small measured-angle lead used only for the fixed-axis base "
                    "retreat. It ramps in over pull_base_retreat_steps and makes "
                    "the physical pull faster without adding force to the door."
                ),
            },
            {
                "name": "--pull_base_progress_reference_angle_deg",
                "type": float,
                "default": 48.0,
                "help": (
                    "Door angle used to normalize the primary straight-back base retreat. "
                    "It is intentionally independent of pull_release_angle_deg so raising "
                    "the physical release threshold does not slow the established early pull."
                ),
            },
            {
                "name": "--pull_late_retreat_start_angle_deg",
                "type": float,
                "default": 38.0,
                "help": (
                    "Measured door angle that starts the additional slow base retreat "
                    "used to continue the physical pull from about 40 degrees to the "
                    "release angle."
                ),
            },
            {
                "name": "--pull_late_retreat_distance",
                "type": float,
                "default": 0.23,
                "help": (
                    "Additional straight-back physical base retreat applied only late "
                    "in pull_door. No force or torque is applied directly to the door."
                ),
            },
            {
                "name": "--pull_late_retreat_steps",
                "type": int,
                "default": 140,
                "help": (
                    "Frames used to smoothly apply the late pull retreat. The "
                    "default keeps physical traction active through the last few "
                    "degrees instead of finishing the retreat early and stalling."
                ),
            },
            {
                "name": "--pull_high_angle_base_lateral_follow_ratio",
                "type": float,
                "default": 1.0,
                "help": (
                    "Fraction of the high-angle handle-arc lateral sweep followed smoothly by "
                    "the base. This preserves tangential pulling geometry after about 40 deg."
                ),
            },
            {
                "name": "--pull_late_base_lateral_follow_ratio",
                "type": float,
                "default": 1.80,
                "help": (
                    "Late-pull lateral follow ratio reached smoothly at large door angles. "
                    "Keeping the early ratio near 1 avoids tearing the fingers off the handle, "
                    "while a small late over-travel maintains tangential pulling force."
                ),
            },
            {
                "name": "--pull_late_base_lateral_boost_start_deg",
                "type": float,
                "default": 55.0,
                "help": "Measured door angle at which late lateral over-travel starts.",
            },
            {
                "name": "--pull_late_base_lateral_boost_full_deg",
                "type": float,
                "default": 65.0,
                "help": "Measured door angle at which the late lateral ratio is fully applied.",
            },
            {
                "name": "--pull_high_angle_base_lateral_max_distance",
                "type": float,
                "default": 0.40,
                "help": (
                    "Maximum lateral base displacement used for smooth high-angle arc following."
                ),
            },
            {
                "name": "--release_base_retreat_distance",
                "type": float,
                "default": 0.76,
                "help": (
                    "Total retreat from base_stop reached while releasing/retracting the arm, "
                    "placing the A2W body outside the opening door's sweep."
                ),
            },
            {
                "name": "--pull_target_max_distance",
                "type": float,
                "default": 0.04,
                "help": (
                    "Maximum per-frame Cartesian lead from the measured EE to the circular-arc "
                    "pull target. Bounding this lead keeps the Jacobian IK command reachable "
                    "while maintaining physical tensile force. Set <=0 to disable the bound."
                ),
            },
            {
                "name": "--pull_gripper_tighten_steps",
                "type": int,
                "default": 1,
                "help": (
                    "Frames after rotate_handle used to tighten from the unchanged scripted "
                    "close ratio to the fully closed command required for tensile pulling."
                ),
            },
            {
                "name": "--pull_gripper_contact_friction",
                "type": float,
                "default": 4.0,
                "help": "Rigid-shape friction used by the Z1 arm/gripper in the pull script.",
            },
            {
                "name": "--pull_handle_contact_friction",
                "type": float,
                "default": 4.0,
                "help": "Rigid-shape friction used by the door/handle in the pull script.",
            },
            {
                "name": "--pull_gripper_stiffness",
                "type": float,
                "default": 300.0,
                "help": "Position-drive stiffness of jointGripper in the pull script.",
            },
            {
                "name": "--pull_gripper_damping",
                "type": float,
                "default": 20.0,
                "help": "Position-drive damping of jointGripper in the pull script.",
            },
            {
                "name": "--pull_gripper_effort_limit",
                "type": float,
                "default": 60.0,
                "help": (
                    "Effort limit of jointGripper while pulling. This only strengthens "
                    "the physical grasp; it does not apply force to the door."
                ),
            },
            {
                "name": "--pull_arc_lead_angle_deg",
                "type": float,
                "default": 0.6,
                "help": (
                    "Maximum angular lead of the commanded grasp pose ahead of the "
                    "measured door angle while following the hinge-centered pull arc."
                ),
            },
            {
                "name": "--pull_arc_lead_ramp_steps",
                "type": int,
                "default": 30,
                "help": "Frames used to ramp in the circular-arc angular lead.",
            },
            {
                "name": "--pull_mid_arc_lead_start_angle_deg",
                "type": float,
                "default": 10.0,
                "help": (
                    "Measured door angle after which the established grasp may "
                    "use a faster circular-arc lead."
                ),
            },
            {
                "name": "--pull_mid_arc_lead_angle_deg",
                "type": float,
                "default": 0.85,
                "help": "Circular-arc lead used through the middle of pull_door.",
            },
            {
                "name": "--pull_mid_arc_lead_ramp_steps",
                "type": int,
                "default": 50,
                "help": "Frames used to ramp from the initial to middle arc lead.",
            },
            {
                "name": "--pull_late_arc_lead_angle_deg",
                "type": float,
                "default": 5.0,
                "help": (
                    "Circular-arc angular lead reached after the late-retreat "
                    "threshold. The stronger tangential lead keeps pulling beyond "
                    "60 degrees without relying on post-release inertia."
                ),
            },
            {
                "name": "--pull_follow_orientation",
                "action": "store_true",
                "default": False,
                "help": (
                    "Follow the handle body's full orientation after unlocking. Disabled by "
                    "default because the spring-loaded WC4 lever can otherwise drag the wrist "
                    "out of the grasp while it rebounds."
                ),
            },
            {
                "name": "--pull_release_angle_deg",
                "type": float,
                "default": 65.0,
                "help": (
                    "Measured pull-side door angle that starts releasing the handle. "
                    "The default keeps a physical grasp beyond the 60 degree task threshold "
                    "so success never depends on post-release inertia."
                ),
            },
            {
                "name": "--pull_release_angle_tolerance_deg",
                "type": float,
                "default": 0.0,
                "help": (
                    "Physical-contact tolerance for the measured release angle. "
                    "The default is zero so the release transition starts as soon "
                    "as the measured door angle reaches pull_release_angle_deg."
                ),
            },
            {
                "name": "--pull_settle_steps",
                "type": int,
                "default": 0,
                "help": "Optional closed-gripper settling frames after reaching the release angle.",
            },
            {
                "name": "--open_hold_steps",
                "type": int,
                "default": 0,
                "help": (
                    "Optional stationary loosening frames before release_handle. "
                    "Disabled by default to avoid a visible pause at the end of pull_door."
                ),
            },
            {
                "name": "--release_handle_steps",
                "type": int,
                "default": 50,
                "help": (
                    "Total release_handle frames, including the final stationary "
                    "home-pose convergence period before pass_through."
                ),
            },
            {
                "name": "--release_handle_motion_steps",
                "type": int,
                "default": 50,
                "help": (
                    "Frames used to open, extract, and retract the arm. Remaining "
                    "release_handle frames hold the arm home before pass_through."
                ),
            },
            {
                "name": "--release_joint_home_start_fraction",
                "type": float,
                "default": 0.35,
                "help": (
                    "Fraction of release_handle_motion_steps after which the already "
                    "opened and outward-extracted arm switches to an explicit joint-space "
                    "return to its initial home configuration. This guarantees that the "
                    "arm is home before pass_through instead of relying on slow Cartesian IK."
                ),
            },
            {
                "name": "--release_gripper_open_start_fraction",
                "type": float,
                "default": 0.0,
                "help": (
                    "Fraction of release_handle_motion_steps for which the gripper "
                    "remains fully closed while the arm starts extracting outward. "
                    "The default starts opening immediately at the measured-angle release event."
                ),
            },
            {
                "name": "--release_gripper_open_fraction",
                "type": float,
                "default": 0.04,
                "help": (
                    "Fraction of release_handle_motion_steps used to open the gripper "
                    "after release_gripper_open_start_fraction. The default opens "
                    "quickly over about four control frames."
                ),
            },
            {
                "name": "--release_outward_fraction",
                "type": float,
                "default": 0.08,
                "help": (
                    "Fraction of release_handle used for the immediate outward extraction. "
                    "The remaining time is kept for the collision-avoiding lateral retraction."
                ),
            },
            {
                "name": "--release_outward_distance",
                "type": float,
                "default": 0.14,
                "help": "Extra pull-direction clearance before sweeping the released arm laterally.",
            },
            {
                "name": "--release_extra_outward_distance",
                "type": float,
                "default": 0.20,
                "help": (
                    "Additional outward travel after the gripper has cleared the handle, "
                    "before beginning the outside retraction arc."
                ),
            },
            {
                "name": "--release_extra_outward_end_fraction",
                "type": float,
                "default": 0.22,
                "help": (
                    "Fraction of release_handle at which the additional outward movement "
                    "finishes and the outside retraction arc begins."
                ),
            },
            {
                "name": "--release_outside_arc_bulge",
                "type": float,
                "default": 0.10,
                "help": (
                    "Outward bulge of the quadratic retraction arc, keeping the arm on "
                    "the pull side of the handle while it folds laterally."
                ),
            },
            {
                "name": "--release_lateral_distance",
                "type": float,
                "default": 0.30,
                "help": "Maximum world-Y arm retraction distance after releasing the handle.",
            },
            {
                "name": "--release_lift_distance",
                "type": float,
                "default": 0.06,
                "help": "Small upward clearance added during post-release arm retraction.",
            },
            {
                "name": "--pass_through_steps",
                "type": int,
                "default": 360,
                "help": "Smooth A2W base traversal duration after the arm clears the handle.",
            },
            {"name": "--return_home_steps", "type": int, "default": 150},
            {"name": "--return_home_target_chase_alpha", "type": float, "default": 0.08},
            {
                "name": "--ee_command_max_step",
                "type": float,
                "default": 0.025,
                "help": "Maximum per-sim-step EE position command delta for DoorTwin skill mode; <=0 disables command smoothing.",
            },
            {"name": "--hold_steps", "type": int, "default": 300},
            {"name": "--pregrasp_offset", "type": float, "default": 0.15},
            {"name": "--grasp_offset", "type": float, "default": 0.0},
            {"name": "--grasp_x_offset", "type": float, "default": -0.015},
            {"name": "--grasp_z_offset", "type": float, "default": -0.03},
            {
                "name": "--wc4_pregrasp_z_offset",
                "type": float,
                "default": 0.0,
                "help": "Extra Z offset applied only to the wc4 pregrasp point.",
            },
            {
                "name": "--wc4_grasp_z_offset",
                "type": float,
                "default": 0.0,
                "help": "Extra Z offset applied only to the wc4 grasp point; rotate/push points inherit it.",
            },
            {"name": "--handle_rotate_right_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_down_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_angle", "type": float, "default": 1.05},
            {"name": "--handle_rotate_direction_sign", "type": float, "default": -1.0},
            {"name": "--door_push_distance", "type": float, "default": 1.10},
            {
                "name": "--door_pull_distance",
                "type": float,
                "default": 0.60,
                "help": "Maximum scripted EE displacement along the live pull tangent.",
            },
            {
                "name": "--pull_contact_bias",
                "type": float,
                "default": 0.04,
                "help": "Outward bias beyond the live handle contact pose while pulling.",
            },
            {"name": "--no_ikpush_env_randomization", "action": "store_true"},
            {"name": "--ikpush_door_x_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_door_y_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_door_wall_x_offset_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_robot_x_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_robot_x_rand_min", "type": float, "default": -0.70},
            {"name": "--ikpush_robot_x_rand_max", "type": float, "default": 0.0},
            {"name": "--ikpush_robot_y_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_robot_y_rand_min", "type": float, "default": -0.10},
            {"name": "--ikpush_robot_y_rand_max", "type": float, "default": 0.04},
            {"name": "--ikpush_robot_z_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_robot_pitch_rand_min", "type": float, "default": 0.0},
            {"name": "--ikpush_robot_pitch_rand_max", "type": float, "default": math.radians(5.0)},
            {"name": "--ikpush_robot_yaw_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_pregrasp_offset_rand", "type": float, "default": 0.025},
            {"name": "--ikpush_grasp_x_offset_rand", "type": float, "default": 0.012},
            {"name": "--ikpush_grasp_z_offset_rand", "type": float, "default": 0.012},
            {"name": "--ikpush_handle_rotate_angle_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_door_push_distance_rand", "type": float, "default": 0.06},
            {"name": "--ikpush_door_joint_friction_rand", "type": float, "default": 0.08},
            {"name": "--ikpush_door_joint_damping_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_handle_joint_friction_rand", "type": float, "default": 0.005},
            {"name": "--ikpush_handle_joint_damping_rand", "type": float, "default": 0.005},
            {"name": "--ikpush_handle_spring_stiffness_rand", "type": float, "default": 0.05},
            {"name": "--ikpush_handle_spring_damping_rand", "type": float, "default": 0.01},
            {
                "name": "--ikpush_door_open_resistance_prob",
                "type": float,
                "default": 0.50,
                "help": "Fraction of envs with nonzero door-closing spring resistance.",
            },
            {"name": "--ikpush_door_open_resistance_min", "type": float, "default": 0.10},
            {"name": "--ikpush_door_open_resistance_max", "type": float, "default": 0.30},
            {"name": "--lever_step_size", "type": float, "default": 0.06},
            {"name": "--push_contact_bias", "type": float, "default": 0.025},
            {"name": "--handle_follow_push_ratio", "type": float, "default": 0.45},
            {"name": "--door_freeze_blend_start_ratio", "type": float, "default": 0.82},
            {"name": "--door_freeze_target_ratio", "type": float, "default": 0.94},
            {"name": "--enable_base_door_collision_check", "dest": "enable_base_door_collision_check", "action": "store_true", "default": False},
            {"name": "--no_base_door_collision_check", "dest": "enable_base_door_collision_check", "action": "store_false"},
            {"name": "--enable_collision_physx_check", "dest": "enable_collision_physx_check", "action": "store_true", "default": False},
            {"name": "--enable_collision_geom_check", "dest": "enable_collision_geom_check", "action": "store_true", "default": False},
            {
                "name": "--no_collision_geom_check",
                "dest": "disable_collision_geom_check",
                "action": "store_true",
                "default": False,
                "help": "Disable the cheap geometric base-door collision gate even when DoorTwin logging is enabled.",
            },
            {"name": "--base_door_collision_distance", "type": float, "default": 0.04},
            {"name": "--base_collision_front_extent", "type": float, "default": 0.55},
            {"name": "--base_collision_rear_extent", "type": float, "default": 0.65},
            {"name": "--base_collision_half_width", "type": float, "default": 0.24},
            {"name": "--rigid_contact_geom_gate", "type": float, "default": 0.16},
            {"name": "--collision_log_interval", "type": int, "default": 30},
            {
                "name": "--allow_post_unlock_geom_contact",
                "dest": "allow_post_unlock_geom_contact",
                "action": "store_true",
                "default": True,
                "help": (
                    "Treat geometry-only base/door proximity after handle unlock as "
                    "a tolerated contact. PhysX rigid/frame contacts and pre-unlock collisions remain blocking."
                ),
            },
            {
                "name": "--no_allow_post_unlock_geom_contact",
                "dest": "allow_post_unlock_geom_contact",
                "action": "store_false",
                "help": "Restore strict behavior where every geometric base/door proximity is blocking.",
            },
            {
                "name": "--push_follow_orientation",
                "action": "store_true",
                "help": "After the door unlocks, also keep the end-effector orientation fixed relative to the handle. By default push contact is position-only.",
            },
            {
                "name": "--unidoor_style_push",
                "action": "store_true",
                "default": True,
                "help": "During push, step from current EE along the handle push direction.",
            },
            {"name": "--no_unidoor_style_push", "dest": "unidoor_style_push", "action": "store_false"},
            {"name": "--gripper_open", "type": float, "default": -1.5707963267948966},
            {"name": "--gripper_closed", "type": float, "default": 0.0},
            {"name": "--gripper_close_ratio", "type": float, "default": 0.8},
            {"name": "--gripper_open_stage_ratio", "type": float, "default": 0.25},
            {"name": "--gripper_loosen_steps", "type": int, "default": 120},
            {"name": "--handle_spring_stiffness", "type": float, "default": 0.5},
            {"name": "--handle_spring_damping", "type": float, "default": 0.1},
            {"name": "--handle_unlock_ratio", "type": float, "default": 40.0 / 45.0},
            {"name": "--door_open_resistance", "type": float, "default": 0.0},
            {"name": "--door_open_damping", "type": float, "default": 0.0},
            {
                "name": "--wc4_disable_door_open_resistance",
                "dest": "wc4_disable_door_open_resistance",
                "action": "store_true",
                "default": True,
                "help": (
                    "Disable the WC4 hinge restoring torque in the pull script, "
                    "including the per-environment randomized restoring torque."
                ),
            },
            {
                "name": "--wc4_enable_door_open_resistance",
                "dest": "wc4_disable_door_open_resistance",
                "action": "store_false",
                "help": "Restore the legacy randomized WC4 hinge closing resistance.",
            },
            {"name": "--door_lock_force", "type": float, "default": 0.0},
            {"name": "--door_joint_friction", "type": float, "default": 0.},
            {"name": "--door_joint_damping", "type": float, "default": 0.},
            {"name": "--handle_joint_friction", "type": float, "default": 0.05},
            {"name": "--handle_joint_damping", "type": float, "default": 0.05},
            {
                "name": "--door_auto_open_force",
                "type": float,
                "default": 0.0,
                "help": (
                    "Optional hinge torque enabled only after the handle unlock "
                    "threshold. Disabled by default so the door must be opened "
                    "through physical robot-door interaction."
                ),
            },
            {"name": "--door_auto_open_sign", "type": float, "default": 1.0},
            {"name": "--door_auto_open_target_ratio", "type": float, "default": 0.95},
            {"name": "--door_vhacd_resolution", "type": int, "default": 100000},
            {"name": "--forward_ee_roll", "type": float, "default": math.pi / 2},
            {"name": "--forward_ee_pitch", "type": float, "default": 0.0},
            {"name": "--gripper_red_axis_rot", "type": float, "default": -math.pi / 2},
            {"name": "--ik_pos_gain", "type": float, "default": 1.0},
            {"name": "--ik_rot_gain", "type": float, "default": 0.7},
            {"name": "--ik_rot_weight", "type": float, "default": 0.1},
            {"name": "--ik_damping", "type": float, "default": 0.08},
            {"name": "--ik_max_step", "type": float, "default": 0.06},
            {"name": "--ik_pos_tolerance", "type": float, "default": 0.015},
            {"name": "--ik_rot_tolerance", "type": float, "default": 0.08},
            {"name": "--ik_position_only", "action": "store_true"},
            {"name": "--ik_include_gripper", "action": "store_true"},
            {"name": "--ik_ee_link", "type": str, "default": base_ik.EE_GRIPPER_LINK},
            {
                "name": "--arm_ik_solver",
                "type": str,
                "default": "gym_jacobian",
                "help": "Arm IK backend: gym_jacobian (original) or tracik (absolute IK plus quintic trajectory).",
            },
            {"name": "--tracik_urdf", "type": str, "default": str(DEFAULT_Z1_TRACIK_URDF)},
            {"name": "--tracik_ee_link", "type": str, "default": DEFAULT_Z1_TRACIK_EE_LINK},
            {"name": "--tracik_timeout", "type": float, "default": 0.005},
            {"name": "--tracik_epsilon", "type": float, "default": 3.0e-3},
            {"name": "--tracik_solver_type", "type": str, "default": "Speed"},
            {"name": "--tracik_max_restarts", "type": int, "default": 30},
            {
                "name": "--tracik_branch_continuity",
                "dest": "tracik_branch_continuity",
                "action": "store_true",
                "default": True,
                "help": (
                    "Reject discontinuous TRAC-IK branches and search deterministic alternate seeds. "
                    "This keeps EE-command policies on the demonstrated Z1 joint branch."
                ),
            },
            {
                "name": "--no_tracik_branch_continuity",
                "dest": "tracik_branch_continuity",
                "action": "store_false",
            },
            {
                "name": "--tracik_branch_lateral_deadband",
                "type": float,
                "default": 0.0,
                "help": "Enable the negative-lateral joint1 branch guard when target y is below -deadband.",
            },
            {
                "name": "--tracik_branch_forward_min",
                "type": float,
                "default": 0.20,
                "help": "Minimum base-frame target x for the handle-approach joint1 branch guard.",
            },
            {
                "name": "--tracik_branch_joint1_upper",
                "type": float,
                "default": 0.05,
                "help": "Maximum accepted joint1 angle while the negative-lateral branch guard is active.",
            },
            {
                "name": "--tracik_branch_candidate_seeds",
                "type": int,
                "default": 8,
                "help": "Maximum deterministic/random seed attempts after TRAC-IK returns a guarded branch.",
            },
            {
                "name": "--tracik_branch_position_fallback",
                "dest": "tracik_branch_position_fallback",
                "action": "store_true",
                "default": True,
                "help": (
                    "When the policy EE target only yields a guarded joint1 branch, advance a "
                    "local FK reference and solve its full pose with TRAC-IK until direct "
                    "full-pose continuity returns."
                ),
            },
            {
                "name": "--no_tracik_branch_position_fallback",
                "dest": "tracik_branch_position_fallback",
                "action": "store_false",
            },
            {
                "name": "--tracik_branch_reference_iterations",
                "type": int,
                "default": 1,
                "help": "FK-based local DLS updates per TRAC-IK command while tracking the guarded branch.",
            },
            {
                "name": "--tracik_branch_reference_joint_step",
                "type": float,
                "default": 0.06,
                "help": "Maximum per-joint branch-reference DLS update in radians.",
            },
            {
                "name": "--tracik_branch_reference_damping",
                "type": float,
                "default": 0.08,
            },
            {
                "name": "--tracik_branch_reference_rotation_weight",
                "type": float,
                "default": 0.04,
                "help": "Soft orientation weight for the portable branch-reference DLS.",
            },
            {
                "name": "--tracik_branch_reference_fd_epsilon",
                "type": float,
                "default": 1.0e-4,
            },
            {
                "name": "--tracik_branch_reference_reset_distance",
                "type": float,
                "default": 1.5,
                "help": "Reset branch reference to measured q if their max joint distance exceeds this value.",
            },
            {
                "name": "--tracik_branch_joint1_weight",
                "type": float,
                "default": 6.0,
                "help": "Joint1 weight when selecting the candidate nearest measured q and the last accepted q.",
            },
            {
                "name": "--tracik_gym_guided_seed",
                "action": "store_true",
                "help": (
                    "Use one full-pose Gym Jacobian DLS step from the measured current "
                    "joints as every TRAC-IK seed. Forces solver_type=Distance and "
                    "max_restarts=1."
                ),
            },
            {
                "name": "--tracik_local_servo",
                "action": "store_true",
                "help": (
                    "Use a lightweight TRAC-IK local servo: seed from measured q_current, "
                    "reject large joint jumps, and shrink the EE target if needed. "
                    "Forces solver_type=Distance, max_restarts=1, and disables heavy "
                    "branch-continuity candidate search."
                ),
            },
            {
                "name": "--tracik_local_servo_branch_anchor",
                "action": "store_true",
                "help": (
                    "Also clamp local-servo seed/solution to the geometric joint1 "
                    "handle-side branch. Experimental; off by default because it can "
                    "over-constrain ACT targets."
                ),
            },
            {
                "name": "--tracik_local_servo_max_joint_step",
                "type": float,
                "default": 0.25,
                "help": "Maximum accepted per-command joint delta from measured q_current in local-servo mode.",
            },
            {
                "name": "--tracik_local_servo_max_q1_step",
                "type": float,
                "default": 0.18,
                "help": "Maximum accepted joint1 delta from measured q_current in local-servo mode.",
            },
            {
                "name": "--tracik_initial_seed_q",
                "type": str,
                "default": "",
                "help": (
                    "Optional comma-separated joint1..joint6 seed used only for the first "
                    "TRAC-IK solve after controller reset. The trajectory still starts "
                    "from the current commanded joints."
                ),
            },
            {
                "name": "--tracik_fixed_seed_q",
                "type": str,
                "default": "",
                "help": (
                    "Optional comma-separated joint1..joint6 seed used for every TRAC-IK "
                    "solve, retry, and reachability-projection solve. This overrides "
                    "--tracik_initial_seed_q."
                ),
            },
            {"name": "--tracik_initial_segment_duration", "type": float, "default": 2.0},
            {"name": "--tracik_projection_iterations", "type": int, "default": 6},
            {"name": "--tracik_projection_min_alpha", "type": float, "default": 1.0e-3},
            {"name": "--no_tracik_unreachable_projection", "action": "store_true"},
            {"name": "--tracik_command_hz", "type": float, "default": 25.0},
            {"name": "--tracik_max_joint_speed", "type": float, "default": 3.0},
            {"name": "--tracik_max_joint_acceleration", "type": float, "default": 80.0},
            {"name": "--tracik_waypoint_velocity_alpha", "type": float, "default": 0.5},
            {"name": "--tracik_max_segment_duration", "type": float, "default": 3.0},
            {"name": "--tracik_max_waypoint_step", "type": float, "default": 0.20},
            {
                "name": "--tracik_mount_correction",
                "type": float,
                "nargs": 3,
                "default": list(DEFAULT_Z1_TRACIK_MOUNT_CORRECTION),
                "help": "Standalone Z1 URDF base translation minus the simulated A2W Z1 mount translation.",
            },
            {"name": "--stiffness", "type": float, "default": 80.0},
            {"name": "--damping", "type": float, "default": 8.0},
            {"name": "--speed_scale", "type": float, "default": 0.6},
            {"name": "--range_scale", "type": float, "default": 0.75},
            {"name": "--joint_filter", "type": str, "default": ""},
            {"name": "--single_asset", "action": "store_true"},
            {"name": "--flip_visual_attachments", "action": "store_true"},
            {"name": "--disable_arm_visual_flip", "action": "store_true"},
            {"name": "--base_visual_flip", "action": "store_true"},
            {"name": "--no_disable_gravity", "action": "store_true"},
            {"name": "--disable_self_collisions", "action": "store_true"},
            {"name": "--print_collision_summary", "action": "store_true"},
            {"name": "--log_interval", "type": int, "default": 60},
            {"name": "--draw_ik_target", "dest": "draw_ik_target", "action": "store_true", "default": True},
            {"name": "--no_draw_ik_target", "dest": "draw_ik_target", "action": "store_false"},
            {"name": "--draw_camera_axes", "dest": "draw_camera_axes", "action": "store_true", "default": True},
            {"name": "--no_draw_camera_axes", "dest": "draw_camera_axes", "action": "store_false"},
            {"name": "--draw_scripted_trajectory", "dest": "draw_scripted_trajectory", "action": "store_true", "default": False},
            {"name": "--no_draw_scripted_trajectory", "dest": "draw_scripted_trajectory", "action": "store_false"},
            {"name": "--trajectory_point_radius", "type": float, "default": 0.018},
            {"name": "--trajectory_point_samples", "type": int, "default": 16},
            {"name": "--enable_wrist_camera", "dest": "enable_wrist_camera", "action": "store_true", "default": True},
            {"name": "--no_enable_wrist_camera", "dest": "enable_wrist_camera", "action": "store_false"},
            {"name": "--enable_front_camera", "dest": "enable_front_camera", "action": "store_true", "default": True},
            {"name": "--no_enable_front_camera", "dest": "enable_front_camera", "action": "store_false"},
            {"name": "--show_camera_images", "dest": "show_camera_images", "action": "store_true", "default": True},
            {"name": "--no_show_camera_images", "dest": "show_camera_images", "action": "store_false"},
            {"name": "--show_camera_masks", "action": "store_true", "default": False},
            {"name": "--show_seg", "action": "store_true"},
            {"name": "--no_show_seg", "action": "store_true"},
            {"name": "--rgb", "action": "store_true", "help": "Show RGB+mask camera previews instead of full depth+mask."},
            {"name": "--depth_only", "dest": "depth_only", "action": "store_true", "default": True, "help": "Record/use only wrist/front depth images, without handle mask images."},
            {"name": "--no_depth_only", "dest": "depth_only", "action": "store_false", "help": "Use legacy depth+mask image inputs."},
            {"name": "--camera_rgb", "action": "store_true"},
            {"name": "--camera_depth", "action": "store_true"},
            {"name": "--no_camera_depth", "action": "store_true"},
            {"name": "--camera_seg", "action": "store_true"},
            {"name": "--no_camera_seg", "action": "store_true"},
            {"name": "--handle_seg_id", "type": int, "default": 2},
            {"name": "--camera_depth_clip_lower", "type": float, "default": 0.2},
            {"name": "--camera_depth_clip_far", "type": float, "default": 1.5},
            {
                "name": "--camera_intrinsics_mode",
                "type": str,
                "choices": ["legacy", "real_k_remap"],
                "default": "legacy",
                "help": "legacy keeps the original FOV pipeline; real_k_remap renders wide then remaps to calibrated K.",
            },
            {
                "name": "--camera_intrinsics_config",
                "type": str,
                "default": str(dc.DEFAULT_REAL_CAMERA_INTRINSICS_CONFIG),
            },
            {"name": "--camera_render_horizontal_fov_deg", "type": float, "default": 60.0},
            {"name": "--camera_display_scale", "type": int, "default": 1},
            {"name": "--camera_display_interval", "type": int, "default": 1},
            {"name": "--dump_initial_depth_dir", "type": str, "default": ""},
            {"name": "--dump_initial_depth_frames", "type": int, "default": 0},
            {"name": "--camera_axis_scale", "type": float, "default": 0.10},
            {"name": "--camera_axis_thickness", "type": float, "default": 0.004},
            {
                "name": "--wrist_camera_yaw_deg",
                "type": float,
                "default": float(dc.DEFAULT_WRIST_CAMERA_CFG["rotation_deg"][0]),
            },
            {
                "name": "--wrist_camera_pitch_deg",
                "type": float,
                "default": float(dc.DEFAULT_WRIST_CAMERA_CFG["rotation_deg"][1]),
            },
            {
                "name": "--wrist_camera_roll_deg",
                "type": float,
                "default": float(dc.DEFAULT_WRIST_CAMERA_CFG["rotation_deg"][2]),
            },
            {
                "name": "--front_camera_yaw_deg",
                "type": float,
                "default": float(dc.DEFAULT_FRONT_CAMERA_CFG["rotation_deg"][0]),
            },
            {
                "name": "--front_camera_pitch_deg",
                "type": float,
                "default": float(dc.DEFAULT_FRONT_CAMERA_CFG["rotation_deg"][1]),
            },
            {
                "name": "--front_camera_roll_deg",
                "type": float,
                "default": float(dc.DEFAULT_FRONT_CAMERA_CFG["rotation_deg"][2]),
            },
            *dc.depth_aug_custom_parameters(),
            {
                "name": "--ee_pose_frame",
                "type": str,
                "default": "robot_base_full",
                "help": (
                    "Frame used to record/decode 10D EE pose state/action. "
                    "'base' keeps the legacy yaw-only base frame; "
                    "'robot_base_full' uses the true robot base frame including pitch."
                ),
            },
            {"name": "--record_dp_dataset", "action": "store_true"},
            {
                "name": "--record_end_signal",
                "action": "store_true",
                "help": "Save a per-frame end signal that is 1 in return_home/hold_home and 0 otherwise.",
            },
            {
                "name": "--end_signal_positive_phases",
                "type": str,
                "default": "return_home,hold_home",
            },
            {
                "name": "--record_camera_pose",
                "action": "store_true",
                "help": "When recording raw DP data, save front/wrist camera optical-frame poses in robot base frame.",
            },
            {
                "name": "--record_handle_bbox",
                "action": "store_true",
                "help": (
                    "When recording raw DP data, save front/wrist handle bbox supervision "
                    "from rendered segmentation masks for offline handle-latent targets."
                ),
            },
            {
                "name": "--record_gripper_handle_contact",
                "action": "store_true",
                "help": (
                    "When recording raw DP data, save per-frame gripperStator/gripperMover "
                    "vs handle rigid-contact scores for grasp-quality filtering."
                ),
            },
            {
                "name": "--filter_gripper_handle_contact",
                "action": "store_true",
                "help": (
                    "Only save successful raw episodes whose close/rotate phases contain enough "
                    "gripper-handle contact frames."
                ),
            },
            {
                "name": "--gripper_handle_contact_gripper_bodies",
                "type": str,
                "default": "gripperStator,gripperMover",
                "help": "Comma-separated arm rigid body names treated as the two gripper contact sides.",
            },
            {
                "name": "--gripper_handle_contact_handle_bodies",
                "type": str,
                "default": "",
                "help": "Optional comma-separated door rigid body names treated as handles; empty uses the selected door handle body.",
            },
            {"name": "--gripper_handle_contact_score_threshold", "type": float, "default": 1.0e-6},
            {"name": "--filter_gripper_handle_contact_min_frames", "type": int, "default": 5},
            {
                "name": "--filter_gripper_handle_contact_require_both",
                "dest": "filter_gripper_handle_contact_require_both",
                "action": "store_true",
                "default": True,
            },
            {
                "name": "--no_filter_gripper_handle_contact_require_both",
                "dest": "filter_gripper_handle_contact_require_both",
                "action": "store_false",
            },
            {
                "name": "--filter_gripper_handle_contact_phase_names",
                "type": str,
                "default": "close_gripper,rotate_handle",
            },
            {"name": "--debug_gripper_handle_contact", "action": "store_true"},
            {"name": "--dp_raw_root", "type": str, "default": str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp")},
            {"name": "--dp_task", "type": str, "default": "pull lever door open"},
            {"name": "--dp_record_env_id", "type": int, "default": 0},
            {"name": "--dp_record_all_envs", "action": "store_true"},
            {"name": "--no_dp_record_all_envs", "action": "store_true"},
            {"name": "--dp_fps", "type": int, "default": 25},
            {"name": "--camera_fps", "type": float, "default": 25.0},
            {"name": "--dp_record_state_mode", "type": str, "default": "full"},
            {"name": "--dp_policy_checkpoint", "type": str, "default": ""},
            {
                "name": "--expert_action_replay_raw_episode",
                "type": str,
                "default": "",
                "help": (
                    "Replace the scripted trajectory with the fixed action sequence from one raw episode. "
                    "Only actions are replayed; simulator state is not restored, so normal environment/domain "
                    "randomization remains active."
                ),
            },
            {"name": "--dp_control_env_id", "type": int, "default": 0},
            {"name": "--dp_control_all_envs", "action": "store_true"},
            {"name": "--no_dp_control_all_envs", "action": "store_true"},
            {"name": "--dp_inference_steps", "type": int, "default": 10},
            {"name": "--dp_noise_scheduler_type", "type": str, "default": "DDIM"},
            {"name": "--dp_action_horizon", "type": int, "default": -1},
            {
                "name": "--dp3_draw_point_cloud",
                "action": "store_true",
                "help": (
                    "Draw the selected point-cloud policy observation. Dual-fused ACT uses green Front "
                    "points and magenta Wrist points."
                ),
            },
            {"name": "--dp3_point_cloud_env_id", "type": int, "default": 0},
            {"name": "--dp_temporal_ensemble", "action": "store_true"},
            {"name": "--dp_temporal_prefetch_actions", "type": int, "default": 3},
            {"name": "--dp_temporal_old_weight", "type": float, "default": 0.3},
            {"name": "--dp_temporal_new_weight", "type": float, "default": 0.7},
            {"name": "--dp_end_signal_monitor", "action": "store_true"},
            {"name": "--dp_end_signal_threshold", "type": float, "default": 0.8},
            {"name": "--dp_end_signal_consecutive_steps", "type": int, "default": 10},
            {"name": "--dp_log_path", "type": str, "default": ""},
            {"name": "--dp_log_interval", "type": int, "default": 25},
            {
                "name": "--dp_log_replay_snapshot",
                "action": "store_true",
                "help": "Include replay-style root/DOF/door snapshot fields in each DP policy JSONL log record.",
            },
            {"name": "--no_dp_print", "dest": "dp_print", "action": "store_false", "default": True},
            {
                "name": "--dp_gripper_latch",
                "action": "store_true",
                "help": "Runtime-only diagnostic: after the DP gripper command closes past a threshold, prevent reopen until the door is open enough.",
            },
            {"name": "--dp_gripper_latch_close_threshold", "type": float, "default": -0.45},
            {"name": "--dp_gripper_latch_release_door_deg", "type": float, "default": 75.0},
            {"name": "--keyframe_loss_weight", "type": float, "default": 8.0},
            {"name": "--keyframe_loss_radius", "type": int, "default": 3},
            {"name": "--no_keyframe_loss_weights", "action": "store_true"},
            {"name": "--dp_warmstart", "action": "store_true"},
            {"name": "--dp_warmstart_raw_episode", "type": str, "default": ""},
            {"name": "--dp_warmstart_step", "type": int, "default": -1},
            {"name": "--dp_warmstart_expert_obs", "dest": "dp_warmstart_expert_obs", "action": "store_true", "default": True},
            {"name": "--no_dp_warmstart_expert_obs", "dest": "dp_warmstart_expert_obs", "action": "store_false"},
            {
                "name": "--recovery_batch_manifest",
                "type": str,
                "default": "",
                "help": "Run a simulator-verification batch produced by dp/recovery/verify_recovery_candidates.py.",
            },
            {"name": "--recovery_result_json", "type": str, "default": ""},
            {"name": "--recovery_raw_root", "type": str, "default": ""},
            {"name": "--recovery_contact_min_frames", "type": int, "default": 5},
            {"name": "--pass_open_angle_deg", "type": float, "default": 60.0},
            {
                "name": "--pull_record_traversal_distance_m",
                "type": float,
                "default": 0.8,
                "help": (
                    "Pull-only raw-recording success threshold: signed base distance beyond "
                    "the door plane. This does not affect the push-door recorder."
                ),
            },
            {"name": "--no_preview_trajectory_at_spawn", "action": "store_true"},
            {
                "name": "--skill_program_json",
                "type": str,
                "default": "",
                "help": "Door Digital Twin skill program JSON path, or 'auto' to materialize defaults.",
            },
            {
                "name": "--door_twin_log_dir",
                "type": str,
                "default": "",
                "help": "Directory for Door Digital Twin rollout reports and keyframe images.",
            },
            {
                "name": "--save_failed_rollouts",
                "action": "store_true",
                "help": "Keep lightweight failed-rollout traces in Door Digital Twin reports.",
            },
            {
                "name": "--dump_keyframe_images",
                "action": "store_true",
                "help": "Dump multi-view RGB/depth/mask images at major Door Digital Twin phase transitions.",
            },
            {
                "name": "--door_twin_camera_views",
                "type": str,
                "default": ",".join(DOOR_TWIN_DEFAULT_CAMERA_VIEWS),
                "help": "Comma-separated Door Twin keyframe views; observer_* and handle_closeup views are fixed in the world.",
            },
            {
                "name": "--door_twin_asset_probe",
                "action": "store_true",
                "help": "Benchmark-only: bypass handle locking so a fixed torque can verify that the door hinge moves in PhysX.",
            },
            {
                "name": "--door_twin_side_camera_yaw_deg",
                "type": float,
                "default": 30.0,
                "help": "Yaw offset in degrees for the front_left/front_right Door Twin cameras.",
            },
            {"name": "--door_twin_observer_distance", "type": float, "default": 1.8},
            {"name": "--door_twin_observer_lateral", "type": float, "default": 1.0},
            {"name": "--door_twin_observer_height", "type": float, "default": 1.45},
            {"name": "--door_twin_handle_closeup_distance", "type": float, "default": 0.55},
            {"name": "--door_twin_handle_closeup_lateral", "type": float, "default": 0.25},
            {"name": "--door_twin_handle_closeup_height_offset", "type": float, "default": 0.16},
            {
                "name": "--capture_stage_screenshots",
                "action": "store_true",
                "help": (
                    "Capture clean offscreen RGB screenshots from the legacy viewer viewpoint at several "
                    "moments in each scripted phase. This is independent of policy cameras and keyframe dumps."
                ),
            },
            {
                "name": "--stage_screenshot_dir",
                "type": str,
                "default": str(REPO_ROOT / "figures" / "a2w_b1_stage_screenshots"),
            },
            {"name": "--stage_screenshot_width", "type": int, "default": 2560},
            {"name": "--stage_screenshot_height", "type": int, "default": 1440},
            {"name": "--stage_screenshot_horizontal_fov_deg", "type": float, "default": 60.0},
            {"name": "--stage_screenshot_eye_x_offset", "type": float, "default": 1.9},
            {"name": "--stage_screenshot_eye_y_offset", "type": float, "default": 3.2},
            {"name": "--stage_screenshot_eye_z", "type": float, "default": 1.8},
            {"name": "--stage_screenshot_target_x_offset", "type": float, "default": 0.3},
            {"name": "--stage_screenshot_target_y_offset", "type": float, "default": 0.0},
            {"name": "--stage_screenshot_target_z", "type": float, "default": 0.8},
            {"name": "--stage_screenshot_frames_per_phase", "type": int, "default": 5},
            {
                "name": "--stage_screenshot_phases",
                "type": str,
                "default": ",".join(DEFAULT_STAGE_SCREENSHOT_PHASES),
            },
            {
                "name": "--stage_screenshot_hold_steps",
                "type": int,
                "default": 100,
                "help": "Expected hold_home duration used only to spread screenshot samples.",
            },
        ],
    )

    argv_list = sys.argv[1:]
    apply_a2w_float_ik_config_defaults(args, argv_list)
    normalize_a2w_runtime_paths(args)

    # gymutil's wrapper does not preserve default=True for store_true custom args,
    # so keep these visualization helpers on by default and let --no_* flags opt out.
    argv = set(argv_list)
    config_defaults = getattr(args, "_a2w_float_ik_config_defaults", {})
    default_true_flags = (
        ("draw_ik_target", "--draw_ik_target", "--no_draw_ik_target"),
        ("draw_camera_axes", "--draw_camera_axes", "--no_draw_camera_axes"),
        ("enable_wrist_camera", "--enable_wrist_camera", "--no_enable_wrist_camera"),
        ("enable_front_camera", "--enable_front_camera", "--no_enable_front_camera"),
        ("show_camera_images", "--show_camera_images", "--no_show_camera_images"),
    )
    for attr, positive_flag, negative_flag in default_true_flags:
        if negative_flag in argv:
            setattr(args, attr, False)
        elif positive_flag in argv:
            setattr(args, attr, True)
        else:
            setattr(args, attr, bool(config_defaults.get(attr, True)))

    if "--no_show_seg" in argv:
        args.show_camera_images = False
    elif "--show_seg" in argv:
        args.show_camera_images = True
    args.camera_rgb = bool(args.camera_rgb or args.rgb)
    args.camera_depth = bool((args.camera_depth or not args.no_camera_depth) and not args.rgb)
    if bool(getattr(args, "rgb", False)):
        args.depth_only = False
    args.camera_seg = bool(args.camera_seg or not args.no_camera_seg)
    if "--no_tracik_branch_continuity" in argv:
        args.tracik_branch_continuity = False
    elif "--tracik_branch_continuity" in argv:
        args.tracik_branch_continuity = True
    else:
        args.tracik_branch_continuity = bool(
            config_defaults.get("tracik_branch_continuity", True)
        )
    if "--no_tracik_branch_position_fallback" in argv:
        args.tracik_branch_position_fallback = False
    elif "--tracik_branch_position_fallback" in argv:
        args.tracik_branch_position_fallback = True
    else:
        args.tracik_branch_position_fallback = bool(
            config_defaults.get("tracik_branch_position_fallback", True)
        )
    if "--no_dp_record_all_envs" in argv:
        args.dp_record_all_envs = False
    elif "--dp_record_all_envs" in argv:
        args.dp_record_all_envs = True
    else:
        args.dp_record_all_envs = bool(config_defaults.get("dp_record_all_envs", not bool(args.no_dp_record_all_envs)))
    if "--no_dp_control_all_envs" in argv:
        args.dp_control_all_envs = False
    elif "--dp_control_all_envs" in argv:
        args.dp_control_all_envs = True
    else:
        args.dp_control_all_envs = bool(config_defaults.get("dp_control_all_envs", not bool(args.no_dp_control_all_envs)))
    args.dp_print = False if "--no_dp_print" in argv else bool(config_defaults.get("dp_print", True))
    if "--enable_base_door_collision_check" in argv:
        args.enable_base_door_collision_check = "--no_base_door_collision_check" not in argv
    elif "--no_base_door_collision_check" in argv:
        args.enable_base_door_collision_check = False
    else:
        args.enable_base_door_collision_check = bool(config_defaults.get("enable_base_door_collision_check", False))
    args.enable_collision_physx_check = args.enable_base_door_collision_check or bool(
        config_defaults.get("enable_collision_physx_check", False)
    ) or "--enable_collision_physx_check" in argv
    args.disable_collision_geom_check = bool(
        getattr(args, "disable_collision_geom_check", False) or "--no_collision_geom_check" in argv
    )
    if args.disable_collision_geom_check:
        args.enable_collision_geom_check = False
    else:
        args.enable_collision_geom_check = args.enable_base_door_collision_check or bool(
            config_defaults.get("enable_collision_geom_check", False)
        ) or "--enable_collision_geom_check" in argv
    if "--no_allow_post_unlock_geom_contact" in argv:
        args.allow_post_unlock_geom_contact = False
    elif "--allow_post_unlock_geom_contact" in argv:
        args.allow_post_unlock_geom_contact = True
    else:
        # gymutil's paired store_true/store_false custom parameters do not
        # reliably preserve the first declaration's default. Resolve this
        # default explicitly so DoorTwin uses the documented relaxed policy.
        args.allow_post_unlock_geom_contact = bool(
            config_defaults.get("allow_post_unlock_geom_contact", True)
        )
    args.dp_action_horizon = None if int(args.dp_action_horizon) < 0 else int(args.dp_action_horizon)
    args.arm_ik_solver = str(args.arm_ik_solver).strip().lower()
    if args.arm_ik_solver not in ("gym_jacobian", "tracik"):
        raise ValueError("--arm_ik_solver must be gym_jacobian or tracik.")
    if args.arm_ik_solver == "tracik" and not cli_attr_was_set(argv_list, "steps"):
        args.steps = 960
    if args.arm_ik_solver == "tracik" and not cli_attr_was_set(argv_list, "initial_hold_steps"):
        args.initial_hold_steps = 100
    if str(args.tracik_solver_type) not in ("Speed", "Distance", "Manip1", "Manip2"):
        raise ValueError("--tracik_solver_type must be Speed, Distance, Manip1, or Manip2.")
    if float(args.tracik_command_hz) <= 0.0:
        raise ValueError("--tracik_command_hz must be positive.")
    if not 0.0 <= float(args.tracik_waypoint_velocity_alpha) <= 1.0:
        raise ValueError("--tracik_waypoint_velocity_alpha must be in [0, 1].")
    if int(args.tracik_max_restarts) <= 0:
        raise ValueError("--tracik_max_restarts must be positive.")
    if float(args.tracik_branch_lateral_deadband) < 0.0:
        raise ValueError("--tracik_branch_lateral_deadband must be non-negative.")
    if float(args.tracik_branch_forward_min) < 0.0:
        raise ValueError("--tracik_branch_forward_min must be non-negative.")
    if int(args.tracik_branch_candidate_seeds) <= 0:
        raise ValueError("--tracik_branch_candidate_seeds must be positive.")
    if float(args.tracik_branch_joint1_weight) < 1.0:
        raise ValueError("--tracik_branch_joint1_weight must be at least 1.")
    if int(args.tracik_branch_reference_iterations) <= 0:
        raise ValueError("--tracik_branch_reference_iterations must be positive.")
    if float(args.tracik_branch_reference_joint_step) <= 0.0:
        raise ValueError("--tracik_branch_reference_joint_step must be positive.")
    if float(args.tracik_branch_reference_damping) <= 0.0:
        raise ValueError("--tracik_branch_reference_damping must be positive.")
    if float(args.tracik_branch_reference_rotation_weight) < 0.0:
        raise ValueError("--tracik_branch_reference_rotation_weight must be non-negative.")
    if float(args.tracik_branch_reference_fd_epsilon) <= 0.0:
        raise ValueError("--tracik_branch_reference_fd_epsilon must be positive.")
    raw_initial_seed = getattr(args, "tracik_initial_seed_q", "")
    if isinstance(raw_initial_seed, (list, tuple, np.ndarray)):
        initial_seed_values = [float(value) for value in raw_initial_seed]
    else:
        initial_seed_text = str(raw_initial_seed or "").strip()
        initial_seed_values = (
            [float(value) for value in initial_seed_text.replace(",", " ").split()]
            if initial_seed_text
            else []
        )
    if initial_seed_values and len(initial_seed_values) != 6:
        raise ValueError("--tracik_initial_seed_q must contain exactly 6 joint values.")
    args.tracik_initial_seed_q = initial_seed_values or None
    if args.tracik_initial_seed_q is not None and args.arm_ik_solver != "tracik":
        raise ValueError("--tracik_initial_seed_q requires --arm_ik_solver tracik.")
    raw_fixed_seed = getattr(args, "tracik_fixed_seed_q", "")
    if isinstance(raw_fixed_seed, (list, tuple, np.ndarray)):
        fixed_seed_values = [float(value) for value in raw_fixed_seed]
    else:
        fixed_seed_text = str(raw_fixed_seed or "").strip()
        fixed_seed_values = (
            [float(value) for value in fixed_seed_text.replace(",", " ").split()]
            if fixed_seed_text
            else []
        )
    if fixed_seed_values and len(fixed_seed_values) != 6:
        raise ValueError("--tracik_fixed_seed_q must contain exactly 6 joint values.")
    args.tracik_fixed_seed_q = fixed_seed_values or None
    if args.tracik_fixed_seed_q is not None and args.arm_ik_solver != "tracik":
        raise ValueError("--tracik_fixed_seed_q requires --arm_ik_solver tracik.")
    if bool(args.tracik_gym_guided_seed):
        if args.arm_ik_solver != "tracik":
            raise ValueError("--tracik_gym_guided_seed requires --arm_ik_solver tracik.")
        if args.tracik_initial_seed_q is not None or args.tracik_fixed_seed_q is not None:
            raise ValueError(
                "--tracik_gym_guided_seed cannot be combined with "
                "--tracik_initial_seed_q or --tracik_fixed_seed_q."
            )
        args.tracik_solver_type = "Distance"
        args.tracik_max_restarts = 1
    if bool(getattr(args, "tracik_local_servo", False)):
        if args.arm_ik_solver != "tracik":
            raise ValueError("--tracik_local_servo requires --arm_ik_solver tracik.")
        if float(args.tracik_local_servo_max_joint_step) <= 0.0:
            raise ValueError("--tracik_local_servo_max_joint_step must be positive.")
        if float(args.tracik_local_servo_max_q1_step) <= 0.0:
            raise ValueError("--tracik_local_servo_max_q1_step must be positive.")
        args.tracik_solver_type = "Distance"
        args.tracik_max_restarts = 1
        args.tracik_branch_continuity = False
    if float(args.tracik_initial_segment_duration) <= 0.0:
        raise ValueError("--tracik_initial_segment_duration must be positive.")
    if int(args.tracik_projection_iterations) < 0:
        raise ValueError("--tracik_projection_iterations must be non-negative.")
    if not 0.0 <= float(args.tracik_projection_min_alpha) <= 1.0:
        raise ValueError("--tracik_projection_min_alpha must be in [0, 1].")
    if args.arm_ik_solver == "tracik" and str(args.ik_ee_link) != str(args.tracik_ee_link):
        raise ValueError("--ik_ee_link and --tracik_ee_link must match in TRAC-IK mode.")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if not args.dp_record_all_envs and (args.dp_record_env_id < 0 or args.dp_record_env_id >= args.num_envs):
        raise ValueError("--dp_record_env_id must be in [0, num_envs - 1].")
    external_action_source = bool(args.dp_policy_checkpoint or args.expert_action_replay_raw_episode)
    if external_action_source and (args.dp_control_env_id < 0 or args.dp_control_env_id >= args.num_envs):
        raise ValueError("--dp_control_env_id must be in [0, num_envs - 1].")
    if args.dp_policy_checkpoint and args.dp_warmstart and args.dp_control_all_envs and args.num_envs > 1:
        raise ValueError("--dp_warmstart currently supports a single controlled env; add --no_dp_control_all_envs.")
    if args.record_dp_dataset and args.dp_policy_checkpoint:
        raise ValueError("--record_dp_dataset and --dp_policy_checkpoint are separate modes; run recording or policy play, not both.")
    if args.record_dp_dataset and args.expert_action_replay_raw_episode:
        raise ValueError(
            "--record_dp_dataset and --expert_action_replay_raw_episode are separate modes; "
            "run recording or expert replay, not both."
        )
    if args.dp_policy_checkpoint and args.expert_action_replay_raw_episode:
        raise ValueError(
            "--dp_policy_checkpoint and --expert_action_replay_raw_episode are mutually exclusive."
        )
    warmstart_params = [
        bool(args.dp_warmstart_raw_episode),
        int(args.dp_warmstart_step) >= 0,
        "--dp_warmstart_expert_obs" in argv or "--no_dp_warmstart_expert_obs" in argv,
    ]
    if not args.dp_warmstart and any(warmstart_params):
        raise ValueError("Warm-start options require --dp_warmstart.")
    if args.dp_warmstart:
        if not args.dp_policy_checkpoint:
            raise ValueError("--dp_warmstart requires --dp_policy_checkpoint.")
        if not args.dp_warmstart_raw_episode:
            raise ValueError("--dp_warmstart requires --dp_warmstart_raw_episode.")
        if int(args.dp_warmstart_step) < 0:
            raise ValueError("--dp_warmstart requires non-negative --dp_warmstart_step.")
    dc.apply_depth_aug_config_defaults(args, sys.argv[1:])
    args._explicit_cli_flags = set(sys.argv[1:])

    if args.headless and args.show_camera_images:
        print(
            "⚠️📷 Headless mode disables OpenCV camera preview windows; run without --headless to view camera images.",
            flush=True,
        )
        args.show_camera_images = False

    args.ik_demo = True
    args.ik_target_pose = ""
    args.ik_demo_offset = "0 0 0"
    args.ik_keep_base_motion = False
    args.disable_base_motion = True
    args.zero_pose_seconds = 0.0
    args.zero_pose_only = False
    args.show_axis = False
    args.base_motion_amplitude = 0.0
    args.base_motion_yaw = 0.0
    args.base_motion_period = 1.0
    # WC4 multiplies this by -1, yielding the negative hinge range that swings
    # toward the robot. The handle-to-unlock rule itself remains unchanged.
    args.door_motion_sign = 1.0
    args.pass_through_door = not bool(args.no_pass_through_door)
    configure_door_twin_args(args)
    return args


def create_tracik_solver_if_requested(args):
    if str(getattr(args, "arm_ik_solver", "gym_jacobian")) != "tracik":
        print("Arm IK backend: gym_jacobian (original Isaac Gym Jacobian DLS).", flush=True)
        return None
    try:
        from ik_solvers.z1_tracik_kinematics import Z1TracIKKinematics
    except Exception as exc:
        raise RuntimeError(
            "TRAC-IK mode requires the b1z1 environment with trac_ik installed."
        ) from exc
    solver = Z1TracIKKinematics(
        args.tracik_urdf,
        ee_link=args.tracik_ee_link,
        timeout=args.tracik_timeout,
        epsilon=args.tracik_epsilon,
        solver_type=args.tracik_solver_type,
    )
    args._tracik_solver = solver
    print(
        "Arm IK backend: tracik + online quintic "
        f"solver_type={str(args.tracik_solver_type)} "
        f"max_restarts={int(args.tracik_max_restarts)} "
        f"command_hz={float(args.tracik_command_hz):.1f} "
        f"max_speed={float(args.tracik_max_joint_speed):.2f}rad/s "
        f"max_acceleration={float(args.tracik_max_joint_acceleration):.2f}rad/s^2 "
        f"initial_segment={float(args.tracik_initial_segment_duration):.2f}s "
        f"initial_seed_q={getattr(args, 'tracik_initial_seed_q', None)} "
        f"fixed_seed_q={getattr(args, 'tracik_fixed_seed_q', None)} "
        f"gym_guided_seed={bool(getattr(args, 'tracik_gym_guided_seed', False))} "
        f"local_servo={bool(getattr(args, 'tracik_local_servo', False))} "
        f"local_servo_joint_step={float(getattr(args, 'tracik_local_servo_max_joint_step', 0.0)):.3f}rad "
        f"local_servo_q1_step={float(getattr(args, 'tracik_local_servo_max_q1_step', 0.0)):.3f}rad "
        f"branch_continuity={bool(getattr(args, 'tracik_branch_continuity', True))} "
        f"branch_y_deadband={float(args.tracik_branch_lateral_deadband):.3f}m "
        f"branch_q1_upper={float(args.tracik_branch_joint1_upper):.3f}rad "
        f"branch_candidate_seeds={int(args.tracik_branch_candidate_seeds)} "
        f"branch_position_fallback={bool(args.tracik_branch_position_fallback)} "
        f"branch_reference_iterations={int(args.tracik_branch_reference_iterations)} "
        f"waypoint_velocity_alpha={float(args.tracik_waypoint_velocity_alpha):.2f} "
        f"unreachable_projection={not bool(args.no_tracik_unreachable_projection)} "
        f"projection_iterations={int(args.tracik_projection_iterations)} "
        f"mount_correction={np.asarray(args.tracik_mount_correction, dtype=np.float64).tolist()}",
        flush=True,
    )
    return solver


# Shared door/IK helpers live in door_common; keep local names to avoid changing push control code.
smoothstep = dc.smoothstep
lerp = dc.lerp
quat_nlerp = dc.quat_nlerp
chase_target_to_current_ee = dc.chase_target_to_current_ee
normalize = dc.normalize
quat_apply = dc.quat_apply
quat_from_angle_axis = dc.quat_from_angle_axis
quat_axis = dc.quat_axis
forward_ee_quat = dc.forward_ee_quat
load_door_specs = dc.load_door_specs
load_door_assets = dc.load_door_assets
load_door_asset = dc.load_door_asset
robot_y_for_door = dc.robot_y_for_door
create_env_actors = dc.create_env_actors
create_parallel_env_actors = dc.create_parallel_env_actors


def configure_pull_contact_friction(gym, env, arm_actor, door_actor, args):
    """Increase contact friction without applying any non-physical door torque."""

    actor_settings = (
        (arm_actor, float(args.pull_gripper_contact_friction), "arm/gripper"),
        (door_actor, float(args.pull_handle_contact_friction), "door/handle"),
    )
    for actor, requested_friction, label in actor_settings:
        shape_props = gym.get_actor_rigid_shape_properties(env, actor)
        for prop in shape_props:
            prop.friction = max(float(prop.friction), requested_friction)
            if hasattr(prop, "restitution"):
                prop.restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, actor, shape_props)
        if int(getattr(args, "parallel_env_id", 0)) == 0:
            applied = min((float(prop.friction) for prop in shape_props), default=0.0)
            print(
                f"[A2WPull] {label} contact friction >= {applied:.2f}",
                flush=True,
            )


def load_a2wz1_float_robot_assets(gym, sim, args, temp_root: Path):
    split_root, base_file, arm_file = a2w_ik.build_a2wz1_split_asset_root(
        args.asset_root,
        args.asset_file,
        temp_root,
    )
    base_asset = base_ik.load_asset_with_visual_flip(
        gym,
        sim,
        split_root,
        base_file,
        args,
        False,
        "A2W base visual actor",
    )
    arm_asset = base_ik.load_asset_with_visual_flip(
        gym,
        sim,
        split_root,
        arm_file,
        args,
        not bool(getattr(args, "disable_arm_visual_flip", False)),
        "Z1 arm articulated actor",
    )
    return base_asset, arm_asset


def configure_a2w_base_visual_dofs(gym, base_asset, args):
    if base_asset is None:
        return None
    num_dofs = gym.get_asset_dof_count(base_asset)
    if num_dofs <= 0:
        return None

    dof_names = list(gym.get_asset_dof_names(base_asset))
    dof_props = gym.get_asset_dof_properties(base_asset)
    dof_props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    dof_props["stiffness"].fill(float(args.stiffness))
    dof_props["damping"].fill(float(args.damping))

    lower = np.asarray(dof_props["lower"], dtype=np.float32)
    upper = np.asarray(dof_props["upper"], dtype=np.float32)
    has_limits = np.asarray(dof_props["hasLimits"], dtype=bool)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    positions = dof_states["pos"]
    default_leg_pos = getattr(a2w_ik, "A2W_DEFAULT_LEG_POS", {})
    for idx, name in enumerate(dof_names):
        value = float(default_leg_pos.get(name, 0.0))
        if bool(has_limits[idx]) and float(lower[idx]) < float(upper[idx]):
            value = float(np.clip(value, float(lower[idx]), float(upper[idx])))
        positions[idx] = value
    print(
        "A2W float-base visual actor DOFs initialized: "
        + ", ".join(f"{name}={positions[idx]:.3f}" for idx, name in enumerate(dof_names)),
        flush=True,
    )
    return dof_props, dof_states


def apply_a2w_base_visual_dofs(gym, env, actor_handles, base_dof_data):
    if base_dof_data is None or len(actor_handles) <= 1:
        return
    base_actor = actor_handles[0]
    base_dof_props, base_dof_states = base_dof_data
    env_base_states = base_dof_states.copy()
    gym.set_actor_dof_properties(env, base_actor, base_dof_props)
    gym.set_actor_dof_states(env, base_actor, env_base_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, base_actor, env_base_states["pos"])


@dataclass
class ParallelEnvState:
    index: int
    args: object
    env: object
    arm_actor: int
    actor_handles: list
    door: DoorRuntime
    door_actor: int
    camera_handles: dict
    ik_state: object
    dof_positions: np.ndarray
    home_positions: np.ndarray
    base_start: np.ndarray
    base_stop: np.ndarray
    base_push: np.ndarray
    base_traverse: np.ndarray
    yaw_start: float
    yaw_push: float
    yaw_traverse: float
    traj: dict
    dp_recorder: object = None
    dp_record_success: bool = False
    dp_record_warned_no_camera: bool = False
    dp_record_sim_steps: int = 0
    dp_record_prev_base_xy: object = None
    dp_record_prev_yaw: object = None
    prev_base_xy: object = None
    prev_yaw: object = None
    last_dp_action: object = None
    last_dp_policy_output: object = None
    last_phase: str = "init"
    last_handle_goal: object = None
    last_door_pos: object = None
    last_target_pos: object = None
    last_target_quat: object = None
    last_gripper: float = 0.0
    dp_gripper_latch_active: bool = False
    dp_gripper_latch_value: object = None
    base_door_collision_detected: bool = False
    base_door_collision_log_step: int = -10**9
    door_twin_tracker: object = None
    tracik_controller: object = None
    tracik_joint_indices: object = None
    tracik_command_q: object = None
    tracik_command_pos_world: object = None
    tracik_command_quat_world: object = None
    dp_end_probability: float = 0.0
    dp_end_consecutive_count: int = 0
    dp_end_triggered: bool = False
    dp_first_end_trigger_step: object = None
    dp_door_angle_at_end_trigger: object = None
    dp_phase_at_end_trigger: object = None
    last_dp_interaction_state: object = None
    stage_screenshot_camera: object = None
    stage_screenshot_phase: str = ""
    stage_screenshot_phase_step: int = -1
    stage_screenshot_next_index: int = 0
    stage_screenshot_pending: object = None


@dataclass
class ExpertActionReplay:
    path: Path
    actions: np.ndarray
    states: np.ndarray
    action_names: tuple[str, ...]
    action_frame: str
    fps: float
    door_asset_name: str
    warned_end: bool = False


def _npz_scalar_text(data, key, default=""):
    if key not in data:
        return str(default)
    value = np.asarray(data[key])
    if value.size == 0:
        return str(default)
    return str(value.reshape(-1)[0])


def load_expert_action_replay(args):
    path = Path(args.expert_action_replay_raw_episode).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expert action replay episode does not exist: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "action" not in data:
            raise KeyError(f"Expert action replay episode has no 'action' array: {path}")
        actions = np.asarray(data["action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 10:
            raise ValueError(
                f"Expert action replay requires action shape [T, 10], got {actions.shape} from {path}."
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"Expert action replay contains non-finite actions: {path}")
        states = np.asarray(data["state"], dtype=np.float32) if "state" in data else np.empty((0, 0), np.float32)
        if "action_names" in data:
            action_names = tuple(str(name) for name in np.asarray(data["action_names"]).reshape(-1).tolist())
        else:
            action_names = tuple(ACTION_NAMES or ())
        if len(action_names) != actions.shape[1]:
            raise ValueError(
                f"Expert action replay action_names has {len(action_names)} entries but action has "
                f"{actions.shape[1]} dimensions: {path}"
            )
        action_frame = dc.normalize_float_dp_pose_frame(_npz_scalar_text(data, "action_frame", "base"))
        if action_frame not in ("world", "base") and not dc.is_full_base_pose_frame(action_frame):
            raise ValueError(
                f"Unsupported expert action_frame={action_frame!r}; expected world, base, or robot_base_full."
            )
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else float(args.dp_fps)
        door_asset_name = _npz_scalar_text(data, "door_asset_name", "unknown")
    if fps <= 0.0:
        raise ValueError(f"Expert action replay fps must be positive, got {fps} from {path}.")
    rounded_fps = int(round(fps))
    if not math.isclose(fps, float(rounded_fps), rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"This play path currently requires integer replay fps, got {fps} from {path}.")
    if int(args.dp_fps) != rounded_fps:
        print(
            f"Expert replay overrides --dp_fps {args.dp_fps} with source episode fps={rounded_fps}.",
            flush=True,
        )
        args.dp_fps = rounded_fps
    requested_door = str(getattr(args, "door_name", "") or "")
    if requested_door and door_asset_name not in ("", "unknown", requested_door):
        print(
            f"Warning: replay source door={door_asset_name!r}, current environment door={requested_door!r}.",
            flush=True,
        )
    return ExpertActionReplay(
        path=path,
        actions=actions.copy(),
        states=states.copy(),
        action_names=action_names,
        action_frame=action_frame,
        fps=fps,
        door_asset_name=door_asset_name,
    )


def collect_expert_replay_actions(gym, env_states, replay, controlled_env_ids, step, args, dt):
    stride = dc.float_dp_policy_sample_stride(args, dt)
    frame_index = int(step) // int(stride)
    if frame_index >= len(replay.actions):
        frame_index = len(replay.actions) - 1
        if not replay.warned_end:
            print(
                f"Expert action replay reached its final frame at sim step {step}; holding frame {frame_index}.",
                flush=True,
            )
            replay.warned_end = True
    action = replay.actions[frame_index]
    source_state = (
        replay.states[frame_index]
        if replay.states.ndim == 2 and frame_index < len(replay.states)
        else np.empty((0,), dtype=np.float32)
    )
    inputs_by_env = {}
    actions_by_env = {}
    for st in env_states:
        if st.index not in controlled_env_ids:
            continue
        base_xy_current = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
        yaw_current = float(st.traj.get("yaw", st.yaw_start))
        handle_pos, handle_quat = get_body_pose(gym, st.env, st.door_actor, st.door.handle_body_index)
        handle_goal = quat_apply(handle_quat, st.door.handle_goal_offset) + handle_pos
        ee_pos, ee_quat = current_ee_pose_from_refreshed_tensors(st.ik_state)
        inputs_by_env[st.index] = {
            "base_xy_current": base_xy_current,
            "yaw_current": yaw_current,
            "handle_goal": np.asarray(handle_goal, dtype=np.float32),
            "ee_pos": np.asarray(ee_pos, dtype=np.float32),
            "ee_quat": np.asarray(ee_quat, dtype=np.float32),
            "dp_state": np.asarray(source_state, dtype=np.float32),
        }
        actions_by_env[st.index] = np.asarray(action, dtype=np.float32).copy()
    return inputs_by_env, actions_by_env


def door_twin_profile_for_args(args):
    program = getattr(args, "door_twin_skill_program", None)
    if program is None:
        return None
    profile = getattr(args, "_door_twin_skill_profile", None)
    if profile is None:
        profile = profile_from_program(program, args)
        setattr(args, "_door_twin_skill_profile", profile)
    return profile


def door_twin_legacy_replay_for_args(args):
    program = getattr(args, "door_twin_skill_program", None)
    if program is None:
        return False
    return str(program.metadata.get("execution_mode", "")).strip().lower() == "legacy_replay"


def door_twin_tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def door_twin_camera_view_names(args):
    raw = str(getattr(args, "door_twin_camera_views", "") or "")
    requested = [name.strip().lower() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(DOOR_TWIN_CAMERA_VIEWS))
    if unknown:
        raise ValueError(
            f"Unsupported --door_twin_camera_views value(s): {unknown}; "
            f"choose from {list(DOOR_TWIN_CAMERA_VIEWS)}"
        )
    return tuple(dict.fromkeys(requested))


def move_to_approach_yaw_delta(args):
    explicit = getattr(args, "move_to_approach_yaw_delta", None)
    if explicit is not None:
        return float(explicit)
    vyaw = float(getattr(args, "move_to_approach_vyaw", 0.0))
    return vyaw * float(getattr(args, "walk_steps", 0)) * float(getattr(args, "sim_dt", 0.02))


def door_twin_separate_traverse_for_args(args):
    profile = door_twin_profile_for_args(args)
    return bool(profile is not None and profile.traverse_required and not door_twin_legacy_replay_for_args(args))


def compute_base_push_and_traverse_targets(args, base_stop, heading):
    if not door_twin_separate_traverse_for_args(args):
        base_push = compute_base_push_target(args, base_stop, heading)
        return base_push, base_push

    heading = np.asarray(heading, dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm > 1.0e-6:
        heading = heading / heading_norm
    push_distance = max(0.0, float(getattr(args, "push_base_distance", 0.0)))
    base_push = np.asarray(base_stop, dtype=np.float32) + heading * push_distance
    traverse_distance = getattr(args, "traverse_distance", None)
    if traverse_distance is None:
        base_traverse = dc.compute_base_pass_target(args, heading)
    else:
        base_traverse = base_push + heading * max(0.0, float(traverse_distance))
    requested_progress = float(np.dot(np.asarray(base_traverse, dtype=np.float32) - base_stop, heading))
    physical_pass_target = dc.compute_base_pass_target(args, heading)
    physical_progress = float(np.dot(np.asarray(physical_pass_target, dtype=np.float32) - base_stop, heading))
    minimum_progress = max(
        MIN_DOOR_TWIN_FORWARD_DISTANCE_M,
        float(getattr(args, "door_twin_min_forward_distance", MIN_DOOR_TWIN_FORWARD_DISTANCE_M)),
    )
    final_progress = max(requested_progress, physical_progress, minimum_progress)
    base_traverse = np.asarray(base_stop, dtype=np.float32) + heading * final_progress
    return base_push.astype(np.float32), np.asarray(base_traverse, dtype=np.float32)


def init_door_twin_tracker(st):
    if not bool(getattr(st.args, "door_twin_enabled", False)):
        return None
    door_spec = DoorTwinSpec.from_runtime(st.door).to_dict()
    profile = door_twin_profile_for_args(st.args)
    require_traverse = bool(profile.traverse_required) if profile is not None else False
    camera_required = bool(
        getattr(st.args, "record_dp_dataset", False)
        or getattr(st.args, "dump_keyframe_images", False)
    )
    handle_lower = float(st.door.dof_lower[1]) if len(st.door.dof_lower) > 1 else 0.0
    configured_handle_rest = getattr(st.door, "handle_rest_angle", None)
    handle_rest = (
        handle_lower
        if configured_handle_rest is None
        else float(configured_handle_rest)
    )
    tracker = RolloutTracker(
        st.index,
        st.door.spec.get("name", f"door_{st.index}"),
        door_spec,
        None if getattr(st.args, "door_twin_skill_program", None) is None else st.args.door_twin_skill_program.to_dict(),
        pass_open_angle_deg=float(getattr(st.args, "pass_open_angle_deg", 80.0)),
        door_motion_sign=float(getattr(st.args, "door_motion_sign", -1.0)),
        handle_lower=handle_lower,
        handle_unlock_threshold=float(getattr(st.door, "handle_unlock_threshold", 0.0)),
        handle_rest_angle=handle_rest,
        handle_unlock_direction_sign=float(
            getattr(st.door, "handle_unlock_direction_sign", 1.0)
        ),
        require_traverse=require_traverse,
        camera_required=camera_required,
        save_trace=bool(getattr(st.args, "save_failed_rollouts", False)),
        base_start=st.base_start,
        base_push=st.base_traverse if require_traverse else st.base_push,
        pass_plane_point=[float(st.args.door_x), float(st.args.door_y)],
        pass_direction=(st.base_traverse - st.base_stop) if require_traverse else None,
        robot_rear_offset=float(getattr(st.args, "robot_rear_offset", 0.65)),
        sim_dt=float(getattr(st.args, "sim_dt", 0.02)),
    )
    st.door_twin_tracker = tracker
    tracker.add_artifact(
        "camera_views",
        {
            "requested": list(door_twin_camera_view_names(st.args))
            if getattr(st.args, "dump_keyframe_images", False)
            else [],
            "created": sorted(st.camera_handles),
        },
    )
    return tracker


def update_door_twin_tracker(st, step, door_pos_record):
    tracker = getattr(st, "door_twin_tracker", None)
    if tracker is None:
        return
    ee_pos, _ee_quat = current_ee_pose_from_refreshed_tensors(st.ik_state)
    camera_required = bool(getattr(st.args, "record_dp_dataset", False) or getattr(st.args, "dump_keyframe_images", False))
    camera_available = not camera_required
    if getattr(st.args, "record_dp_dataset", False):
        camera_available = bool(getattr(getattr(st, "dp_recorder", None), "frame_count", 0) > 0)
    elif getattr(st.args, "dump_keyframe_images", False):
        camera_available = bool(getattr(st, "_door_twin_dumped_keyframes", set()))
    base_xy = st.traj.get("base_xy", st.base_start)
    base_yaw = float(st.traj.get("yaw", st.yaw_start))
    base_vx, base_vyaw = base_command_from_targets(
        base_xy,
        base_yaw,
        st.prev_base_xy,
        st.prev_yaw,
        float(getattr(st.args, "sim_dt", 0.02)),
    )
    # DoorTwin's arm-limit diagnostic must only inspect the six Z1 arm joints.
    # ``st.dof_positions`` also contains jointGripper, whose open command is
    # intentionally equal to its upper limit and would otherwise make every
    # pre-grasp frame look like an arm joint-limit violation.
    all_dof_positions = np.asarray(st.dof_positions)
    all_lower = door_twin_tensor_to_numpy(getattr(st.ik_state, "lower", None))
    all_upper = door_twin_tensor_to_numpy(getattr(st.ik_state, "upper", None))
    arm_indices = getattr(st, "tracik_joint_indices", None)
    if arm_indices is None:
        arm_indices = np.arange(min(6, all_dof_positions.size), dtype=np.int64)
    else:
        arm_indices = np.asarray(arm_indices, dtype=np.int64)
    tracker.update(
        step=int(step),
        phase=str(st.last_phase),
        door_pos=door_pos_record,
        handle_goal=st.last_handle_goal,
        target_pos=st.last_target_pos,
        ee_pos=ee_pos,
        ee_tracking_error=float(getattr(st.ik_state, "last_pos_error", 0.0)),
        base_xy=base_xy,
        base_vx=base_vx,
        base_vyaw=base_vyaw,
        base_collision=bool(getattr(st, "base_door_collision_detected", False)),
        camera_available=camera_available,
        dof_positions=all_dof_positions[arm_indices],
        lower=None if all_lower is None else np.asarray(all_lower)[arm_indices],
        upper=None if all_upper is None else np.asarray(all_upper)[arm_indices],
    )


def door_twin_keyframe_dump_due(st):
    if not bool(getattr(st.args, "dump_keyframe_images", False)):
        return False
    if not str(getattr(st.args, "door_twin_log_dir", "") or "").strip():
        return False
    if not st.camera_handles:
        return False
    phase = str(getattr(st, "last_phase", ""))
    if phase not in DOOR_TWIN_KEYFRAME_PHASES:
        return False
    dumped = getattr(st, "_door_twin_dumped_keyframes", set())
    return phase not in dumped


def maybe_dump_door_twin_keyframe_images(gym, sim, st, step):
    if not door_twin_keyframe_dump_due(st):
        return False
    if cv2 is None:
        print("DoorTwin keyframe image dump skipped: cv2 is not available.", flush=True)
        return False
    out_dir = Path(st.args.door_twin_log_dir).expanduser() / "keyframes" / f"env_{int(st.index):04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = capture_dp_camera_images_from_rendered(gym, sim, st.env, st.camera_handles, st.args)
    width = int(dc.DEFAULT_FRONT_CAMERA_CFG.get("resolution", dc.DEPTH_CAMERA_RESOLUTION)[0])
    height = int(dc.DEFAULT_FRONT_CAMERA_CFG.get("resolution", dc.DEPTH_CAMERA_RESOLUTION)[1])
    for camera_name, camera_handle in sorted(st.camera_handles.items()):
        rgb_raw = gym.get_camera_image(sim, st.env, camera_handle, gymapi.IMAGE_COLOR)
        if rgb_raw is not None:
            images[f"{camera_name}_rgb"] = camera_color_to_rgb(rgb_raw, height, width)
    phase = str(st.last_phase)
    records = []
    montage_inputs = []
    for name, image in sorted(images.items()):
        array = np.asarray(image)
        valid_pixels = int(np.count_nonzero(array))
        if array.ndim == 3 and array.shape[-1] >= 3:
            bgr = array[..., :3]
            if "rgb" in name:
                bgr = bgr[..., ::-1]
        elif array.ndim == 2:
            bgr = array
        else:
            continue
        filename = f"step_{int(step):05d}_{phase}_{name}.png"
        path = out_dir / filename
        if cv2.imwrite(str(path), np.ascontiguousarray(bgr)):
            record = {
                "step": int(step),
                "phase": phase,
                "image": name,
                "path": str(path),
                "shape": list(array.shape),
                "nonzero_pixels": valid_pixels,
                "valid": bool(valid_pixels > 0),
            }
            records.append(record)
            if name.endswith("_rgb"):
                labeled = np.ascontiguousarray(bgr.copy())
                cv2.putText(
                    labeled,
                    name[:-4],
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                montage_inputs.append(labeled)
    if montage_inputs:
        tile_height = min(image.shape[0] for image in montage_inputs)
        normalized = [
            cv2.resize(image, (int(round(image.shape[1] * tile_height / image.shape[0])), tile_height))
            for image in montage_inputs
        ]
        columns = 2
        tile_width = max(image.shape[1] for image in normalized)
        blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        rows = []
        for offset in range(0, len(normalized), columns):
            row_tiles = normalized[offset : offset + columns]
            row_tiles += [blank] * (columns - len(row_tiles))
            padded = [
                cv2.copyMakeBorder(
                    image,
                    0,
                    0,
                    0,
                    tile_width - image.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
                for image in row_tiles
            ]
            rows.append(cv2.hconcat(padded))
        montage = cv2.vconcat(rows)
        montage_path = out_dir / f"step_{int(step):05d}_{phase}_montage.png"
        if cv2.imwrite(str(montage_path), montage):
            records.append(
                {
                    "step": int(step),
                    "phase": phase,
                    "image": "multiview_montage",
                    "views": [name for name in sorted(st.camera_handles)],
                    "path": str(montage_path),
                    "shape": list(montage.shape),
                    "nonzero_pixels": int(np.count_nonzero(montage)),
                    "valid": bool(np.count_nonzero(montage) > 0),
                }
            )
    if records:
        dumped = set(getattr(st, "_door_twin_dumped_keyframes", set()))
        dumped.add(phase)
        st._door_twin_dumped_keyframes = dumped
        tracker = getattr(st, "door_twin_tracker", None)
        if tracker is not None:
            for record in records:
                tracker.add_artifact("keyframe", record)
            tracker.mark_camera_available(any(record.get("valid", False) for record in records))
        manifest_path = out_dir / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    return False


def stage_screenshot_phase_names(args):
    names = tuple(
        name.strip()
        for name in str(getattr(args, "stage_screenshot_phases", "") or "").split(",")
        if name.strip()
    )
    unknown = sorted(set(names) - set(DP_PHASE_NAMES))
    if unknown:
        raise ValueError(
            f"Unsupported --stage_screenshot_phases value(s): {unknown}; "
            f"expected a subset of {DP_PHASE_NAMES}."
        )
    return names


def create_stage_screenshot_camera(gym, env, args):
    width = int(getattr(args, "stage_screenshot_width", 2560))
    height = int(getattr(args, "stage_screenshot_height", 1440))
    horizontal_fov = float(getattr(args, "stage_screenshot_horizontal_fov_deg", 60.0))
    if width <= 0 or height <= 0:
        raise ValueError("Stage screenshot width and height must be positive.")
    if not 1.0 <= horizontal_fov < 179.0:
        raise ValueError("--stage_screenshot_horizontal_fov_deg must be in [1, 179).")

    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    props.horizontal_fov = horizontal_fov
    camera = gym.create_camera_sensor(env, props)
    if camera < 0:
        raise RuntimeError("Failed to create the offscreen stage screenshot camera.")

    # Keep this exactly aligned with dc.setup_viewer(), which is the long-standing
    # A2W+B1 door-play viewpoint used by the interactive viewer. The offsets are
    # configurable so paper screenshots can pull back toward the robot spawn side.
    eye = (
        float(args.door_x + float(getattr(args, "stage_screenshot_eye_x_offset", 1.9))),
        float(args.door_y + float(getattr(args, "stage_screenshot_eye_y_offset", 3.2))),
        float(getattr(args, "stage_screenshot_eye_z", 1.8)),
    )
    target = (
        float(args.door_x + float(getattr(args, "stage_screenshot_target_x_offset", 0.3))),
        float(args.door_y + float(getattr(args, "stage_screenshot_target_y_offset", 0.0))),
        float(getattr(args, "stage_screenshot_target_z", 0.8)),
    )
    gym.set_camera_location(
        camera,
        env,
        gymapi.Vec3(*eye),
        gymapi.Vec3(*target),
    )
    print(
        "Stage screenshot camera enabled: "
        f"{width}x{height} fov={horizontal_fov:.1f} "
        f"eye={tuple(round(value, 4) for value in eye)} "
        f"target={tuple(round(value, 4) for value in target)}",
        flush=True,
    )
    return camera


def stage_screenshot_expected_phase_steps(st, phase):
    attr_by_phase = {
        "walk": "walk_steps",
        "initial_hold": "initial_hold_steps",
        "grasp": "grasp_steps",
        "grasp_hold": "grasp_hold_steps",
        "close_gripper": "gripper_close_steps",
        "rotate_handle": "handle_rotate_steps",
        "pull_door": "door_pull_steps",
        "open_hold": "open_hold_steps",
        "release_handle": "release_handle_steps",
        "pass_through": "pass_through_steps",
        "return_home": "return_home_steps",
        "hold_home": "stage_screenshot_hold_steps",
    }
    attr = attr_by_phase.get(str(phase))
    return max(1, int(getattr(st.args, attr, 1))) if attr else 1


def stage_screenshot_target_steps(st, phase):
    count = max(1, int(getattr(st.args, "stage_screenshot_frames_per_phase", 5)))
    duration = stage_screenshot_expected_phase_steps(st, phase)
    fractions = (
        np.asarray([0.5], dtype=np.float64)
        if count == 1
        else np.linspace(0.08, 0.92, count, dtype=np.float64)
    )
    targets = [int(round(float(fraction) * max(0, duration - 1))) for fraction in fractions]
    return tuple(sorted(set(targets)))


def update_stage_screenshot_due(st):
    if getattr(st, "stage_screenshot_camera", None) is None:
        return False
    phase = str(getattr(st, "last_phase", ""))
    if phase != st.stage_screenshot_phase:
        st.stage_screenshot_phase = phase
        st.stage_screenshot_phase_step = 0
        st.stage_screenshot_next_index = 0
        st.stage_screenshot_pending = None
    else:
        st.stage_screenshot_phase_step += 1

    if phase not in stage_screenshot_phase_names(st.args):
        return False
    targets = stage_screenshot_target_steps(st, phase)
    index = int(st.stage_screenshot_next_index)
    if index >= len(targets) or st.stage_screenshot_phase_step < targets[index]:
        return False
    st.stage_screenshot_pending = {
        "phase": phase,
        "phase_index": index,
        "phase_step": int(st.stage_screenshot_phase_step),
        "target_phase_step": int(targets[index]),
    }
    st.stage_screenshot_next_index = index + 1
    return True


def save_stage_screenshot(gym, sim, st, global_step):
    pending = getattr(st, "stage_screenshot_pending", None)
    camera = getattr(st, "stage_screenshot_camera", None)
    if pending is None or camera is None:
        return None
    st.stage_screenshot_pending = None

    width = int(getattr(st.args, "stage_screenshot_width", 2560))
    height = int(getattr(st.args, "stage_screenshot_height", 1440))
    raw = gym.get_camera_image(sim, st.env, camera, gymapi.IMAGE_COLOR)
    if raw is None:
        print(f"Stage screenshot unavailable at step {global_step}.", flush=True)
        return None
    rgb = np.ascontiguousarray(camera_color_to_rgb(raw, height, width))
    if rgb.shape[:2] != (height, width):
        raise RuntimeError(
            f"Unexpected stage screenshot shape {rgb.shape}; expected ({height}, {width}, 3)."
        )

    door_name = str(st.door.spec.get("name", getattr(st.args, "door_name", "door")))
    phase = str(pending["phase"])
    out_dir = Path(st.args.stage_screenshot_dir).expanduser().resolve() / door_name / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{door_name}_{phase}_{int(pending['phase_index']):02d}_"
        f"phase{int(pending['phase_step']):04d}_global{int(global_step):05d}_"
        f"{width}x{height}.png"
    )
    path = out_dir / filename
    if cv2 is None:
        raise RuntimeError("--capture_stage_screenshots requires OpenCV (cv2).")
    if not cv2.imwrite(str(path), rgb[..., ::-1]):
        raise RuntimeError(f"Failed to write stage screenshot: {path}")

    record = {
        "door": door_name,
        "phase": phase,
        "phase_index": int(pending["phase_index"]),
        "phase_step": int(pending["phase_step"]),
        "target_phase_step": int(pending["target_phase_step"]),
        "global_step": int(global_step),
        "width": width,
        "height": height,
        "horizontal_fov_deg": float(
            getattr(st.args, "stage_screenshot_horizontal_fov_deg", 60.0)
        ),
        "viewer_eye": [
            float(
                st.args.door_x
                + float(getattr(st.args, "stage_screenshot_eye_x_offset", 1.9))
            ),
            float(
                st.args.door_y
                + float(getattr(st.args, "stage_screenshot_eye_y_offset", 3.2))
            ),
            float(getattr(st.args, "stage_screenshot_eye_z", 1.8)),
        ],
        "viewer_target": [
            float(
                st.args.door_x
                + float(getattr(st.args, "stage_screenshot_target_x_offset", 0.3))
            ),
            float(
                st.args.door_y
                + float(getattr(st.args, "stage_screenshot_target_y_offset", 0.0))
            ),
            float(getattr(st.args, "stage_screenshot_target_z", 0.8)),
        ],
        "path": str(path),
    }
    manifest_path = Path(st.args.stage_screenshot_dir).expanduser().resolve() / door_name / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved stage screenshot: {path}", flush=True)
    return path


def snapshot_raw_dp_episodes(env_states):
    snapshots = {}
    for st in env_states:
        recorder = getattr(st, "dp_recorder", None)
        raw_root = getattr(recorder, "raw_root", None)
        if raw_root is None:
            continue
        snapshots[int(st.index)] = set(Path(raw_root).glob("episode_*.npz"))
    return snapshots


def attach_new_expert_trajectory_artifacts(env_states, snapshots):
    for st in env_states:
        tracker = getattr(st, "door_twin_tracker", None)
        recorder = getattr(st, "dp_recorder", None)
        raw_root = getattr(recorder, "raw_root", None)
        if tracker is None or raw_root is None:
            continue
        before = snapshots.get(int(st.index), set())
        for path in sorted(set(Path(raw_root).glob("episode_*.npz")) - before):
            tracker.add_artifact(
                "expert_trajectory",
                {
                    "path": str(path),
                    "format": "door_dp_raw_npz_v1",
                    "size_bytes": int(path.stat().st_size),
                },
            )


def apply_dp_gripper_latch(st: ParallelEnvState, gripper: float, door_pos) -> float:
    """Prevent a DP-controlled gripper from reopening after it has closed.

    This is a runtime-only diagnostic/safety shim.  It is disabled by default and
    only applies when --dp_gripper_latch is set.  The gripper convention here is
    open ~= -1.57 and more closed is larger, so a latched command is the max of
    all post-close commands until the door has opened enough to release it.
    """
    if not bool(getattr(st.args, "dp_gripper_latch", False)):
        return float(gripper)

    release_deg = float(getattr(st.args, "dp_gripper_latch_release_door_deg", 75.0))
    door_deg = 0.0
    if door_pos is not None:
        arr = np.asarray(door_pos, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            door_deg = abs(float(arr[0])) * 180.0 / math.pi
    if door_deg >= release_deg:
        st.dp_gripper_latch_active = False
        st.dp_gripper_latch_value = None
        return float(gripper)

    threshold = float(getattr(st.args, "dp_gripper_latch_close_threshold", -0.45))
    value = float(gripper)
    if value >= threshold or bool(st.dp_gripper_latch_active):
        previous = st.dp_gripper_latch_value
        latched = value if previous is None else max(float(previous), value)
        if latched >= threshold:
            st.dp_gripper_latch_active = True
            st.dp_gripper_latch_value = float(latched)
            return float(latched)
    return value


clone_door_runtime = dc.clone_door_runtime
resolve_seed = dc.resolve_seed
seed_for_env = dc.seed_for_env
sample_with_half_range = dc.sample_with_half_range
sample_with_offset_range = dc.sample_with_offset_range


IKPUSH_DEFAULT_ENV_RANGES = {
    "door_joint_friction": (0.05, 0.30),
    "door_joint_damping": (0.05, 0.30),
    "handle_joint_friction": (0.045, 0.055),
    "handle_joint_damping": (0.045, 0.055),
    "handle_spring_stiffness": (0.45, 0.55),
    "handle_spring_damping": (0.09, 0.11),
}


IKPUSH_ARG_FLAGS = {
    "robot_y": "--robot_y",
    "robot_yaw": "--robot_yaw",
    "pregrasp_offset": "--pregrasp_offset",
    "grasp_x_offset": "--grasp_x_offset",
    "grasp_z_offset": "--grasp_z_offset",
    "door_push_distance": "--door_push_distance",
    "handle_rotate_angle": "--handle_rotate_angle",
    "door_wall_x_offset": "--door_wall_x_offset",
    "door_joint_friction": "--door_joint_friction",
    "door_joint_damping": "--door_joint_damping",
    "handle_joint_friction": "--handle_joint_friction",
    "handle_joint_damping": "--handle_joint_damping",
    "handle_spring_stiffness": "--handle_spring_stiffness",
    "handle_spring_damping": "--handle_spring_damping",
}



def sample_env_value(rng, args, attr, half_attr, lower=None, upper=None):
    explicit_flags = getattr(args, "_explicit_cli_flags", set())
    default_range = IKPUSH_DEFAULT_ENV_RANGES.get(attr)
    flag = IKPUSH_ARG_FLAGS.get(attr)
    if default_range is not None and flag not in explicit_flags:
        value = float(rng.uniform(float(default_range[0]), float(default_range[1])))
    else:
        value = sample_with_half_range(rng, getattr(args, attr), getattr(args, half_attr), lower=lower, upper=upper)
    if lower is not None:
        value = max(float(lower), value)
    if upper is not None:
        value = min(float(upper), value)
    return value


def make_env_args(args, env_index, door_name=""):
    env_args = SimpleNamespace(**vars(args))
    env_args._camera_axis_local_poses = {}
    env_seed = seed_for_env(args, env_index)
    rng = np.random.default_rng(env_seed)
    env_args.env_seed = env_seed
    env_args.parallel_env_id = int(env_index)
    enabled = not bool(getattr(args, "no_ikpush_env_randomization", False))

    sampled = {
        "seed": int(getattr(args, "seed", 0)),
        "env_seed": int(env_seed),
        "enabled": bool(enabled),
    }

    def set_sampled(attr, half_attr, lower=None, upper=None):
        base_value = getattr(args, attr)
        value = (
            sample_env_value(rng, args, attr, half_attr, lower=lower, upper=upper)
            if enabled
            else float(base_value)
        )
        setattr(env_args, attr, value)
        sampled[attr] = value

    def set_sampled_offset_range(attr, min_attr, max_attr, legacy_half_attr=None, lower=None, upper=None):
        base_value = getattr(args, attr)
        use_legacy_half_range = (
            legacy_half_attr is not None
            and dc.cli_flag_was_set(args, f"--{legacy_half_attr}")
            and not dc.cli_flag_was_set(args, f"--{min_attr}")
            and not dc.cli_flag_was_set(args, f"--{max_attr}")
        )
        if enabled:
            if use_legacy_half_range:
                value = sample_env_value(rng, args, attr, legacy_half_attr, lower=lower, upper=upper)
            else:
                value = sample_with_offset_range(
                    rng,
                    base_value,
                    getattr(args, min_attr),
                    getattr(args, max_attr),
                    lower=lower,
                    upper=upper,
                )
        else:
            value = float(base_value)
        setattr(env_args, attr, value)
        sampled[attr] = value

    def set_fixed(attr):
        value = float(getattr(args, attr))
        setattr(env_args, attr, value)
        sampled[attr] = value

    set_sampled("door_x", "ikpush_door_x_rand")
    set_sampled("door_y", "ikpush_door_y_rand")
    set_sampled("door_wall_x_offset", "ikpush_door_wall_x_offset_rand")
    set_sampled_offset_range(
        "robot_x",
        "ikpush_robot_x_rand_min",
        "ikpush_robot_x_rand_max",
        legacy_half_attr="ikpush_robot_x_rand",
    )
    set_sampled_offset_range(
        "robot_y",
        "ikpush_robot_y_rand_min",
        "ikpush_robot_y_rand_max",
        legacy_half_attr="ikpush_robot_y_rand",
    )
    set_sampled("robot_z", "ikpush_robot_z_rand")
    set_sampled_offset_range(
        "robot_pitch",
        "ikpush_robot_pitch_rand_min",
        "ikpush_robot_pitch_rand_max",
    )
    set_sampled("robot_yaw", "ikpush_robot_yaw_rand")
    set_fixed("pregrasp_offset")
    set_fixed("grasp_x_offset")
    set_fixed("grasp_z_offset")
    set_fixed("wc4_pregrasp_z_offset")
    set_fixed("wc4_grasp_z_offset")
    set_fixed("handle_rotate_angle")
    set_fixed("door_push_distance")
    set_sampled("door_joint_friction", "ikpush_door_joint_friction_rand", lower=0.0)
    set_sampled("door_joint_damping", "ikpush_door_joint_damping_rand", lower=0.0)
    set_sampled("handle_joint_friction", "ikpush_handle_joint_friction_rand", lower=0.0)
    set_sampled("handle_joint_damping", "ikpush_handle_joint_damping_rand", lower=0.0)
    set_sampled("handle_spring_stiffness", "ikpush_handle_spring_stiffness_rand", lower=0.0)
    set_sampled("handle_spring_damping", "ikpush_handle_spring_damping_rand", lower=0.0)

    resistance_probability = min(
        1.0,
        max(0.0, float(getattr(args, "ikpush_door_open_resistance_prob", 0.50))),
    )
    selected_count = int(math.floor(resistance_probability * int(args.num_envs) + 0.5))
    if enabled and resistance_probability > 0.0 and selected_count == 0:
        selected_count = 1
    if not enabled:
        selected_count = 0
    selected_count = min(int(args.num_envs), max(0, selected_count))
    resistance_selected = False
    if selected_count > 0:
        selection_rng = np.random.default_rng(
            np.random.SeedSequence([int(getattr(args, "seed", 0)) & 0xFFFFFFFF, 0xD00A5E])
        )
        selected_env_ids = selection_rng.choice(
            int(args.num_envs),
            size=selected_count,
            replace=False,
        )
        resistance_selected = int(env_index) in set(int(value) for value in selected_env_ids.tolist())

    resistance_min = max(0.0, float(getattr(args, "ikpush_door_open_resistance_min", 0.10)))
    resistance_max = max(0.0, float(getattr(args, "ikpush_door_open_resistance_max", 0.30)))
    if resistance_max < resistance_min:
        resistance_min, resistance_max = resistance_max, resistance_min
    if enabled:
        env_args.door_open_resistance = (
            float(rng.uniform(resistance_min, resistance_max))
            if resistance_selected
            else 0.0
        )
    else:
        env_args.door_open_resistance = float(args.door_open_resistance)
    requested_door_name = str(getattr(args, "door_name", "") or "").strip().lower()
    runtime_door_name = str(door_name or "").strip().lower()
    wc4_resistance_disabled = (
        (requested_door_name == "wc4" or runtime_door_name == "wc4")
        and not dc.cli_flag_was_set(args, "--wc4_enable_door_open_resistance")
    )
    if wc4_resistance_disabled:
        env_args.door_open_resistance = 0.0
        resistance_selected = False
        resistance_probability = 0.0
        selected_count = 0
    sampled["door_open_resistance_selection_mode"] = "fixed_env_subset"
    sampled["door_open_resistance_probability"] = float(resistance_probability)
    sampled["door_open_resistance_selected_env_count"] = int(selected_count)
    sampled["door_open_resistance_selected"] = bool(resistance_selected)
    sampled["door_open_resistance_range"] = [float(resistance_min), float(resistance_max)]
    sampled["door_open_resistance"] = float(env_args.door_open_resistance)
    sampled["wc4_door_open_resistance_disabled"] = bool(wc4_resistance_disabled)

    depth_noise_selected = dc.configure_depth_noise_for_env(env_args)
    sampled["depth_noise_selection_mode"] = str(env_args.depth_noise_selection_mode)
    sampled["depth_noise_env_probability"] = float(env_args.depth_noise_env_probability)
    sampled["depth_noise_selected_env_count"] = int(env_args.depth_noise_selected_env_count)
    sampled["depth_noise_env_selected"] = bool(depth_noise_selected)

    env_args.ikpush_randomization_json = json.dumps(sampled, sort_keys=True)
    return env_args


set_robot_base_pose = dc.set_robot_base_pose
compute_base_push_target = dc.compute_base_push_target
get_body_pose = dc.get_body_pose
get_actor_body_index = dc.get_actor_body_index
gym_quat_to_np = dc.gym_quat_to_np
draw_local_camera_axes = dc.draw_local_camera_axes
make_camera_properties = dc.make_camera_properties


def draw_dp3_point_cloud(gym, viewer, st, point_cloud_base, pointcloud_mode="single_front"):
    """Draw base-frame XYZ; dual-fused Front/Wrist halves use different colors."""
    points = np.asarray(point_cloud_base, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return
    base_actor = st.actor_handles[0] if len(st.actor_handles) > 1 else st.arm_actor
    base_pos, base_quat = get_body_pose(gym, st.env, base_actor, 0)
    qx, qy, qz, qw = base_ik.normalize_quat(np.asarray(base_quat, dtype=np.float32))
    rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    world = points @ rotation.T + np.asarray(base_pos, dtype=np.float32).reshape(1, 3)
    vertices = np.stack((world, world + np.asarray([0.0, 0.0, 0.006], dtype=np.float32)), axis=1)
    colors = np.broadcast_to(np.asarray([0.1, 0.9, 1.0], dtype=np.float32), (len(world), 3)).copy()
    if str(pointcloud_mode) == "dual_fused":
        split = len(world) // 2
        colors[:split] = np.asarray([0.1, 1.0, 0.25], dtype=np.float32)  # Front: green.
        colors[split:] = np.asarray([1.0, 0.2, 0.9], dtype=np.float32)   # Wrist: magenta.
    gym.add_lines(viewer, st.env, len(world), vertices, colors)


def auto_enable_pointcloud_checkpoint_cameras(args):
    """Enable the depth sensors declared by a packaged point-cloud Door checkpoint."""
    checkpoint_text = str(getattr(args, "dp_policy_checkpoint", "") or "").strip()
    if not checkpoint_text:
        return
    checkpoint = Path(checkpoint_text).expanduser()
    metadata_candidates = []
    if checkpoint.is_dir():
        metadata_candidates.append(checkpoint / "door_policy_meta.json")
    else:
        metadata_candidates.append(checkpoint.with_suffix("") / "door_policy_meta.json")
        metadata_candidates.append(checkpoint.parent / "door_policy_meta.json")
    metadata_path = next((path for path in metadata_candidates if path.is_file()), None)
    if metadata_path is None:
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read Door checkpoint metadata {metadata_path}: {exc}") from exc
    config = metadata.get("policy_config", metadata.get("config", {}))
    if not bool(config.get("pointcloud_conditioning", config.get("point_cloud_conditioning", False))):
        return
    mode = str(config.get("pointcloud_mode", "") or "").strip()
    if not mode:
        mode = {
            "front": "single_front",
            "wrist": "single_wrist",
            "front,wrist": "dual_fused",
        }.get(str(config.get("point_cloud_views", "front")), "single_front")
    needs_front = mode in ("single_front", "dual_view", "dual_fused")
    needs_wrist = mode in ("single_wrist", "dual_view", "dual_fused")
    changed = []
    if needs_front and not bool(getattr(args, "enable_front_camera", False)):
        args.enable_front_camera = True
        changed.append("Front")
    if needs_wrist and not bool(getattr(args, "enable_wrist_camera", False)):
        args.enable_wrist_camera = True
        changed.append("Wrist")
    if changed:
        print(
            f"Point-cloud checkpoint automatically enabled {' + '.join(changed)} depth camera(s): mode={mode}",
            flush=True,
        )


def tracik_target_pose_in_standalone_base(gym, st):
    root_pos, root_quat = get_body_pose(gym, st.env, st.arm_actor, 0)
    target_pos_world = np.asarray(st.ik_state.target_pos_np, dtype=np.float32).reshape(3)
    root_quat_inv = base_ik.quat_conjugate(root_quat)
    target_pos_local = quat_apply(root_quat_inv, target_pos_world - root_pos)
    target_pos_local = target_pos_local + np.asarray(
        st.args.tracik_mount_correction,
        dtype=np.float32,
    )
    if st.ik_state.target_quat_np is None:
        return target_pos_local.astype(np.float64), True
    target_quat_world = base_ik.normalize_quat(
        np.asarray(st.ik_state.target_quat_np, dtype=np.float32)
    )
    target_quat_local = base_ik.normalize_quat(
        base_ik.quat_multiply(root_quat_inv, target_quat_world)
    )
    return np.concatenate([target_pos_local, target_quat_local]).astype(np.float64), False


def store_tracik_command_pose_world(gym, st, q_command):
    """Cache the smoothed TRAC-IK joint command and its EE FK in world coordinates."""
    controller = getattr(st, "tracik_controller", None)
    if controller is None:
        return
    q_command = np.asarray(q_command, dtype=np.float64).reshape(6)
    solver_pose = np.asarray(controller.solver.fk(q_command), dtype=np.float64).reshape(7)
    root_pos, root_quat = get_body_pose(gym, st.env, st.arm_actor, 0)
    # tracik_target_pose_in_standalone_base() adds this correction before IK;
    # invert it here so the FK pose is expressed in the simulated arm-root frame.
    local_pos = solver_pose[:3].astype(np.float32) - np.asarray(
        st.args.tracik_mount_correction,
        dtype=np.float32,
    )
    world_pos = quat_apply(root_quat, local_pos) + np.asarray(root_pos, dtype=np.float32)
    world_quat = base_ik.normalize_quat(
        base_ik.quat_multiply(
            np.asarray(root_quat, dtype=np.float32),
            np.asarray(solver_pose[3:7], dtype=np.float32),
        )
    )
    st.tracik_command_q = q_command.astype(np.float32)
    st.tracik_command_pos_world = np.asarray(world_pos, dtype=np.float32).copy()
    st.tracik_command_quat_world = np.asarray(world_quat, dtype=np.float32).copy()


def reset_tracik_controller_to_targets(gym, st, step, dt):
    controller = getattr(st, "tracik_controller", None)
    if controller is None:
        return
    controller.reset(
        np.asarray(st.dof_positions, dtype=np.float64)[st.tracik_joint_indices],
        float(step) * float(dt),
    )
    store_tracik_command_pose_world(
        gym,
        st,
        np.asarray(st.dof_positions, dtype=np.float64)[st.tracik_joint_indices],
    )


def gym_dls_candidate_for_tracik(gym, st, position_only=False):
    """Return one Gym-Jacobian DLS step from the measured arm joints.

    This intentionally mirrors door_common.update_arm_ik_targets_for_env(),
    including its orientation weighting, but returns a seed candidate instead
    of directly commanding the simulated arm.
    """
    ik_state = st.ik_state
    torch = ik_state.torch
    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    pos_err = ik_state.target_pos - eef_state[:3]
    jacobian_env_idx = (
        int(st.index)
        if ik_state.jacobian.ndim >= 4 and ik_state.jacobian.shape[0] > int(st.index)
        else 0
    )
    j_eef = ik_state.jacobian[jacobian_env_idx, ik_state.eef_jacobian_index, :, :]
    j_control = j_eef[:, ik_state.control_indices]
    if bool(position_only) or ik_state.target_quat is None:
        task_j = j_control[:3, :]
        task_err = float(st.args.ik_pos_gain) * pos_err
    else:
        eef_quat = eef_state[3:7]
        orn_err = base_ik.torch_orientation_error(
            torch,
            ik_state.target_quat,
            eef_quat,
        )
        dpose = torch.cat(
            (
                float(st.args.ik_pos_gain) * pos_err,
                float(st.args.ik_rot_gain) * orn_err,
            ),
            dim=0,
        )
        weights = torch.tensor(
            [
                1.0,
                1.0,
                1.0,
                float(st.args.ik_rot_weight),
                float(st.args.ik_rot_weight),
                float(st.args.ik_rot_weight),
            ],
            dtype=torch.float32,
            device=j_control.device,
        )
        task_j = j_control * weights.view(6, 1)
        task_err = dpose * weights
    j_t = torch.transpose(task_j, 0, 1)
    damping = max(1.0e-6, float(st.args.ik_damping))
    lhs = task_j @ j_t + torch.eye(
        task_j.shape[0],
        dtype=torch.float32,
        device=task_j.device,
    ) * (damping * damping)
    delta = j_t @ torch.linalg.solve(
        lhs,
        task_err.unsqueeze(-1),
    ).squeeze(-1)
    max_step = float(st.args.ik_max_step) * float(st.tracik_controller.command_stride)
    delta = torch.clamp(delta, -max_step, max_step)

    actor_states = gym.get_actor_dof_states(st.env, st.arm_actor, gymapi.STATE_ALL)
    current_q = torch.as_tensor(
        actor_states["pos"][st.tracik_joint_indices],
        dtype=torch.float32,
        device=task_j.device,
    )
    lower = ik_state.lower[ik_state.control_indices]
    upper = ik_state.upper[ik_state.control_indices]
    candidate = torch.max(torch.min(current_q + delta, upper), lower)
    return candidate.detach().cpu().numpy().astype(np.float64)


def update_tracik_arm_targets_for_env(gym, st, step, dt):
    controller = st.tracik_controller
    if controller is None:
        raise RuntimeError("TRAC-IK controller was not initialized for this environment.")
    target_pose, position_only = tracik_target_pose_in_standalone_base(gym, st)
    nearest_position_solution = None
    nearest_position_solve_ms = 0.0
    solve_due = (
        controller.last_solve_step is None
        or int(step) - int(controller.last_solve_step) >= controller.command_stride
    )
    if position_only and solve_due:
        nearest_start_ns = time.perf_counter_ns()
        nearest_position_solution = gym_dls_candidate_for_tracik(
            gym,
            st,
            position_only=True,
        )
        nearest_position_solve_ms = (time.perf_counter_ns() - nearest_start_ns) * 1.0e-6
    gym_guided_seed = None
    if (
        bool(getattr(st.args, "tracik_gym_guided_seed", False))
        or bool(getattr(st.args, "tracik_local_servo", False))
    ) and solve_due:
        guided_start_ns = time.perf_counter_ns()
        gym_guided_seed = gym_dls_candidate_for_tracik(
            gym,
            st,
            position_only=position_only,
        )
        nearest_position_solve_ms += (time.perf_counter_ns() - guided_start_ns) * 1.0e-6
    measured_q = None
    if solve_due and (
        bool(getattr(st.args, "tracik_branch_continuity", True))
        or bool(getattr(st.args, "tracik_local_servo", False))
    ):
        actor_states = gym.get_actor_dof_states(st.env, st.arm_actor, gymapi.STATE_ALL)
        measured_q = np.asarray(
            actor_states["pos"][st.tracik_joint_indices],
            dtype=np.float64,
        ).reshape(6)
    q_command, _qd_command, _qdd_command = controller.update(
        step,
        target_pose,
        position_only=position_only,
        nearest_position_solution=nearest_position_solution,
        nearest_position_solve_ms=nearest_position_solve_ms,
        solve_seed_override=gym_guided_seed,
        measured_q=measured_q,
        initial_move=st.last_phase == "initial_hold",
    )
    st.dof_positions[st.tracik_joint_indices] = np.asarray(q_command, dtype=np.float32)
    store_tracik_command_pose_world(gym, st, q_command)

    eef_state = st.ik_state.rb_states[st.ik_state.eef_body_sim_index]
    current_pos = eef_state[:3].detach().cpu().numpy().astype(np.float32)
    current_quat = base_ik.normalize_quat(
        eef_state[3:7].detach().cpu().numpy().astype(np.float32)
    )
    st.ik_state.current_pos_np = current_pos.copy()
    st.ik_state.current_quat_np = current_quat.copy()
    st.ik_state.last_pos_error = float(
        np.linalg.norm(np.asarray(st.ik_state.target_pos_np, dtype=np.float32) - current_pos)
    )
    if st.ik_state.target_quat_np is None:
        st.ik_state.last_rot_error = None
    else:
        target_quat = base_ik.normalize_quat(st.ik_state.target_quat_np)
        quat_dot = float(np.clip(abs(np.dot(target_quat, current_quat)), 0.0, 1.0))
        st.ik_state.last_rot_error = float(2.0 * math.acos(quat_dot))


def local_camera_transform_from_pose(local_pos, local_quat):
    return gymapi.Transform(
        gymapi.Vec3(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])),
        gymapi.Quat(float(local_quat[0]), float(local_quat[1]), float(local_quat[2]), float(local_quat[3])),
    )


def local_camera_transform_from_cfg(camera_cfg, local_rot_override=None, args=None):
    local_pos = np.asarray(camera_cfg.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
    local_rot = dc.camera_rotation_radians_from_cfg(camera_cfg)
    if local_rot_override is not None:
        local_rot = list(local_rot_override)
    if args is not None:
        local_pos, local_rot = dc.jitter_camera_pose_for_args(local_pos, local_rot, args)
    local_quat = gym_quat_to_np(gymapi.Quat.from_euler_zyx(*local_rot))
    local_quat = base_ik.normalize_quat(local_quat)
    return local_pos, local_quat, local_camera_transform_from_pose(local_pos, local_quat)


def camera_axis_pose_cache(args):
    cache = getattr(args, "_camera_axis_local_poses", None)
    if cache is None:
        cache = {}
        setattr(args, "_camera_axis_local_poses", cache)
    return cache


def cached_local_camera_transform_from_cfg(camera_name, camera_cfg, local_rot_override=None, args=None):
    if args is not None and camera_name:
        cache = camera_axis_pose_cache(args)
        cached = cache.get(camera_name)
        if cached is not None:
            local_pos = np.asarray(cached["local_pos"], dtype=np.float32).copy()
            local_quat = base_ik.normalize_quat(np.asarray(cached["local_quat"], dtype=np.float32)).astype(np.float32)
            return local_pos, local_quat, local_camera_transform_from_pose(local_pos, local_quat)
    local_pos, local_quat, local_transform = local_camera_transform_from_cfg(
        camera_cfg,
        local_rot_override,
        args=args,
    )
    if args is not None and camera_name:
        camera_axis_pose_cache(args)[camera_name] = {
            "local_pos": np.asarray(local_pos, dtype=np.float32).copy(),
            "local_quat": np.asarray(local_quat, dtype=np.float32).copy(),
        }
    return local_pos, local_quat, local_transform


def attach_camera_to_actor_body_cached(
    gym,
    env,
    actor,
    body_name,
    camera_cfg,
    local_rot_override=None,
    args=None,
    camera_name=None,
):
    body_handle = gym.find_actor_rigid_body_handle(env, actor, body_name)
    if body_handle < 0:
        return None
    _local_pos, _local_quat, local_transform = cached_local_camera_transform_from_cfg(
        camera_name,
        camera_cfg,
        local_rot_override,
        args=args,
    )
    camera_handle = gym.create_camera_sensor(env, make_camera_properties(camera_cfg))
    if camera_handle < 0:
        return None
    gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    return camera_handle


def attach_camera_to_actor_root_body(gym, env, actor, camera_cfg, local_rot_override=None, args=None, camera_name=None):
    root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
    if int(root_handle) < 0:
        return None
    _local_pos, _local_quat, local_transform = cached_local_camera_transform_from_cfg(
        camera_name,
        camera_cfg,
        local_rot_override,
        args=args,
    )
    camera_handle = gym.create_camera_sensor(env, make_camera_properties(camera_cfg))
    if camera_handle < 0:
        return None
    gym.attach_camera_to_body(camera_handle, env, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    return camera_handle


def draw_root_camera_axes(gym, viewer, env, actor, camera_cfg, local_rot_override, args, camera_name="front"):
    root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
    if int(root_handle) < 0:
        return False
    local_pos, local_quat, _local_transform = cached_local_camera_transform_from_cfg(
        camera_name,
        camera_cfg,
        local_rot_override,
        args=args,
    )
    root_transform = gym.get_rigid_transform(env, root_handle)
    root_pos = np.array([root_transform.p.x, root_transform.p.y, root_transform.p.z], dtype=np.float32)
    root_quat = np.array([root_transform.r.x, root_transform.r.y, root_transform.r.z, root_transform.r.w], dtype=np.float32)
    root_quat = base_ik.normalize_quat(root_quat)
    camera_pos = root_pos + quat_apply(root_quat, local_pos)
    camera_quat = base_ik.quat_multiply(root_quat, local_quat)
    pose = gymapi.Transform(
        gymapi.Vec3(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])),
        gymapi.Quat(float(camera_quat[0]), float(camera_quat[1]), float(camera_quat[2]), float(camera_quat[3])),
    )
    axes_geom = ThickAxesGeometry(scale=args.camera_axis_scale, thickness=args.camera_axis_thickness, pose=pose)
    gymutil.draw_lines(axes_geom, gym, viewer, env, gymapi.Transform())
    return True


def draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, args):
    wrist_rot = dc.wrist_camera_rotation_radians_from_args(args)
    wrist_pos, wrist_quat, _wrist_transform = cached_local_camera_transform_from_cfg(
        "wrist",
        dc.DEFAULT_WRIST_CAMERA_CFG,
        wrist_rot,
        args=args,
    )
    draw_local_camera_axes(
        gym,
        viewer,
        env,
        arm_actor,
        "link06",
        wrist_pos,
        wrist_quat,
        args.camera_axis_scale,
        args.camera_axis_thickness,
    )

    front_rot = [
        math.radians(float(args.front_camera_yaw_deg)),
        math.radians(float(args.front_camera_pitch_deg)),
        math.radians(float(args.front_camera_roll_deg)),
    ]
    base_actor = actor_handles[0] if len(actor_handles) > 1 else arm_actor
    draw_root_camera_axes(gym, viewer, env, base_actor, dc.DEFAULT_FRONT_CAMERA_CFG, front_rot, args, camera_name="front")


def door_twin_world_from_asset_local(args, door, local_xyz):
    local = np.asarray(local_xyz, dtype=np.float32) * float(door.actor_scale)
    yaw = float(door.actor_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    rotated_xy = np.asarray(
        [c * local[0] - s * local[1], s * local[0] + c * local[1]],
        dtype=np.float32,
    )
    actor_xy = np.asarray(
        [
            float(args.door_x) + float(door.actor_position_offset[0]),
            float(args.door_y) + float(door.actor_position_offset[1]),
        ],
        dtype=np.float32,
    )
    return np.asarray(
        [
            actor_xy[0] + rotated_xy[0],
            actor_xy[1] + rotated_xy[1],
            float(getattr(args, "door_z_offset", 0.0))
            + float(door.actor_position_offset[2])
            + float(local[2]),
        ],
        dtype=np.float32,
    )


def door_twin_observer_target(args, door):
    target_local = door.spec.get("observer_target_local")
    if target_local is not None:
        return door_twin_world_from_asset_local(args, door, target_local)
    local_goal = np.asarray(
        door.handle_bounding.get("goal_pos", [0.0, 0.0, 0.9]),
        dtype=np.float32,
    )
    return door_twin_world_from_asset_local(args, door, local_goal)


def door_twin_handle_closeup_target(args, door):
    target_local = door.spec.get("handle_closeup_target_local")
    if target_local is not None:
        return door_twin_world_from_asset_local(args, door, target_local)
    observer_target = door_twin_observer_target(args, door).copy()
    if float(observer_target[2]) < 0.25:
        observer_target[2] = float(getattr(args, "robot_z", 0.5)) + 0.55
    return observer_target


def create_door_twin_observer_camera(gym, env, args, door, name):
    heading = np.asarray(
        [math.cos(float(args.robot_yaw)), math.sin(float(args.robot_yaw))],
        dtype=np.float32,
    )
    lateral = np.asarray([-heading[1], heading[0]], dtype=np.float32)
    if name == "handle_closeup":
        target = door_twin_handle_closeup_target(args, door)
        distance = float(getattr(args, "door_twin_handle_closeup_distance", 0.55))
        lateral_distance = float(getattr(args, "door_twin_handle_closeup_lateral", 0.25))
        height_offset = float(getattr(args, "door_twin_handle_closeup_height_offset", 0.16))
        position = np.asarray(
            [
                target[0] - heading[0] * distance + lateral[0] * lateral_distance,
                target[1] - heading[1] * distance + lateral[1] * lateral_distance,
                target[2] + height_offset,
            ],
            dtype=np.float32,
        )
    else:
        target = door_twin_observer_target(args, door)
        distance = float(getattr(args, "door_twin_observer_distance", 1.8))
        lateral_distance = float(getattr(args, "door_twin_observer_lateral", 1.0))
        height = float(getattr(args, "door_twin_observer_height", 1.45))
    if name == "overhead":
        position = np.asarray(
            [target[0] - heading[0] * 0.35, target[1] - heading[1] * 0.35, target[2] + 2.0],
            dtype=np.float32,
        )
    elif name != "handle_closeup":
        side = 1.0 if name == "observer_left" else -1.0
        position = np.asarray(
            [
                target[0] - heading[0] * distance + side * lateral[0] * lateral_distance,
                target[1] - heading[1] * distance + side * lateral[1] * lateral_distance,
                height,
            ],
            dtype=np.float32,
        )
    camera = gym.create_camera_sensor(env, make_camera_properties(dc.DEFAULT_FRONT_CAMERA_CFG))
    if camera < 0:
        return None
    gym.set_camera_location(
        camera,
        env,
        gymapi.Vec3(float(position[0]), float(position[1]), float(position[2])),
        gymapi.Vec3(float(target[0]), float(target[1]), float(target[2])),
    )
    return camera


def create_low_level_cameras(gym, env, arm_actor, actor_handles, door, args):
    cameras = {}
    door_twin_views = (
        set(door_twin_camera_view_names(args))
        if bool(getattr(args, "dump_keyframe_images", False))
        else set()
    )
    regular_camera_use = bool(
        args.show_camera_images
        or args.record_dp_dataset
        or args.dp_policy_checkpoint
        or str(getattr(args, "recovery_batch_manifest", "") or "").strip()
    )
    if args.enable_wrist_camera and (regular_camera_use or "wrist" in door_twin_views):
        wrist_rot = dc.wrist_camera_rotation_radians_from_args(args)
        wrist_camera_cfg = dc.camera_cfg_for_args("wrist", args)
        wrist_camera = attach_camera_to_actor_body_cached(
            gym,
            env,
            arm_actor,
            "link06",
            wrist_camera_cfg,
            wrist_rot,
            args=args,
            camera_name="wrist",
        )
        if wrist_camera is None:
            print("⚠️📷 Wrist camera sensor creation failed; wrist camera image display is disabled.", flush=True)
        else:
            cameras["wrist"] = wrist_camera
            print(f"Wrist camera sensor enabled: handle={wrist_camera} body=link06")

    if args.enable_front_camera and (regular_camera_use or "front" in door_twin_views):
        front_rot = [
            math.radians(float(args.front_camera_yaw_deg)),
            math.radians(float(args.front_camera_pitch_deg)),
            math.radians(float(args.front_camera_roll_deg)),
        ]
        base_actor = actor_handles[0] if len(actor_handles) > 1 else arm_actor
        front_camera_cfg = dc.camera_cfg_for_args("front", args)
        front_camera = attach_camera_to_actor_root_body(
            gym,
            env,
            base_actor,
            front_camera_cfg,
            front_rot,
            args=args,
            camera_name="front",
        )
        if front_camera is None:
            print("⚠️📷 Front camera sensor creation failed; front camera image display is disabled.", flush=True)
        else:
            cameras["front"] = front_camera
            print(f"Front camera sensor enabled: handle={front_camera} body=root")
    base_actor = actor_handles[0] if len(actor_handles) > 1 else arm_actor
    side_yaw = float(getattr(args, "door_twin_side_camera_yaw_deg", 30.0))
    for name, yaw_offset in (("front_left", side_yaw), ("front_right", -side_yaw)):
        if name not in door_twin_views:
            continue
        side_rot = [
            math.radians(float(args.front_camera_yaw_deg) + yaw_offset),
            math.radians(float(args.front_camera_pitch_deg)),
            math.radians(float(args.front_camera_roll_deg)),
        ]
        camera = attach_camera_to_actor_root_body(
            gym,
            env,
            base_actor,
            dc.DEFAULT_FRONT_CAMERA_CFG,
            side_rot,
            args=args,
            camera_name=name,
        )
        if camera is None:
            print(f"DoorTwin {name} camera sensor creation failed.", flush=True)
        else:
            cameras[name] = camera
            print(f"DoorTwin {name} camera sensor enabled: handle={camera} body=root")
    for name in ("observer_left", "observer_right", "overhead", "handle_closeup"):
        if name not in door_twin_views:
            continue
        camera = create_door_twin_observer_camera(gym, env, args, door, name)
        if camera is None:
            print(f"DoorTwin {name} camera sensor creation failed.", flush=True)
        else:
            cameras[name] = camera
            print(f"DoorTwin {name} camera sensor enabled: handle={camera} frame=world")
    if args.show_camera_images:
        if cv2 is None:
            print("⚠️📷 cv2 is not available; camera image windows are disabled.", flush=True)
        elif not cameras:
            print("⚠️📷 No camera sensors were created; camera image windows are disabled.", flush=True)
        else:
            show_camera_masks = bool(getattr(args, "show_camera_masks", False))
            pair_names = (
                ", ".join(f"{name}_rgb" + (f"/{name}_mask" if show_camera_masks else "") for name in cameras.keys())
                if args.rgb
                else ", ".join(
                    f"{name}_mask/{name}_full_depth" if show_camera_masks else f"{name}_full_depth"
                    for name in cameras.keys()
                )
            )
            print("Camera image windows enabled:", pair_names)
    return cameras


camera_image_to_array = dc.camera_image_to_array
camera_color_to_rgb = dc.camera_color_to_rgb
show_camera_handle_images = dc.show_camera_handle_images
mask_to_rgb = dc.mask_to_rgb
depth_to_rgb = dc.depth_to_rgb
capture_dp_camera_images = dc.capture_dp_camera_images
capture_dp_camera_images_from_rendered = dc.capture_dp_camera_images_from_rendered
dp_image_inputs_from_cpu_cameras = dc.dp_image_inputs_from_cpu_cameras
get_actor_dof_state = dc.get_actor_dof_state
wrap_to_pi = dc.wrap_to_pi
current_ee_pose = dc.current_ee_pose
current_ee_pose_from_refreshed_tensors = dc.current_ee_pose_from_refreshed_tensors
update_arm_ik_targets_for_env = dc.update_arm_ik_targets_for_env
map_float_dofs_to_dp = dc.map_float_dofs_to_dp
base_command_from_targets = dc.base_command_from_targets
target_quat_for_dp = dc.target_quat_for_dp
make_last_low_action_from_dp = dc.make_last_low_action_from_dp
make_float_dp_state = dc.make_float_dp_state
base_position = dc.base_position
world_pos_to_base = dc.world_pos_to_base
base_pos_to_world = dc.base_pos_to_world
world_quat_to_base = dc.world_quat_to_base
base_quat_to_world = dc.base_quat_to_world
normalize_float_dp_pose_frame = dc.normalize_float_dp_pose_frame
is_full_base_pose_frame = dc.is_full_base_pose_frame
make_float_dp_action = dc.make_float_dp_action
apply_float_dp_action = dc.apply_float_dp_action
apply_float_dp_joint_action9 = dc.apply_float_dp_joint_action9
set_a2w_joint_targets_from_action = dc.set_a2w_joint_targets_from_action
float_dp_action_is_a2w_joint9 = dc.float_dp_action_is_a2w_joint9
make_float_dp_policy_log_record = dc.make_float_dp_policy_log_record
print_float_dp_policy_log_record = dc.print_float_dp_policy_log_record
make_float_replay_snapshot = dc.make_float_replay_snapshot


def scalar_to_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def scalar_to_int(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return int(arr.item())
    return int(arr.reshape(-1)[0])


def raw_vision_mode_from_data(data):
    if "vision_mode" in data.files:
        mode = scalar_to_str(data["vision_mode"]).lower()
        if normalize_vision_mode is not None:
            return normalize_vision_mode(mode)
        return mode
    if "wrist_rgb" in data.files or "front_rgb" in data.files:
        return "rgb"
    return "depth"


def raw_action_frame_from_data(data):
    for key in ("action_frame", "action_pose_frame", "target_pose_frame"):
        if key in data.files:
            return scalar_to_str(data[key]).lower()
    return "world"


def raw_ikpush_state_version_from_data(data):
    if "ikpush_state_version" in data.files:
        return scalar_to_str(data["ikpush_state_version"])
    return "legacy"


def normalize_config_path_for_match(path_value):
    value = scalar_to_str(path_value)
    candidates = [Path(value).expanduser(), HIGH_LEVEL_ROOT / value, REPO_ROOT / value]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return value


def yaw_from_quat_xyzw(quat):
    q = base_ik.normalize_quat(np.asarray(quat, dtype=np.float32))
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pitch_from_quat_xyzw(quat):
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return 0.0
    x, y, z, w = [float(v) for v in q[:4]]
    value = 2.0 * (w * y - z * x)
    return math.asin(float(np.clip(value, -1.0, 1.0)))


def set_actor_root_from_state(gym, env, actor, root_state):
    state = np.asarray(root_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 7:
        raise ValueError(f"Root state must have at least 7 values, got shape {state.shape}")
    root_handle = gym.get_actor_root_rigid_body_handle(env, actor)
    transform = gymapi.Transform()
    transform.p = gymapi.Vec3(float(state[0]), float(state[1]), float(state[2]))
    transform.r = gymapi.Quat(float(state[3]), float(state[4]), float(state[5]), float(state[6]))
    gym.set_rigid_transform(env, root_handle, transform)


def require_warmstart_field(data, key, step):
    if key not in data.files:
        raise KeyError(f"Warm-start raw episode is missing required field {key!r}.")
    arr = np.asarray(data[key])
    if arr.ndim <= 0 or arr.shape[0] <= int(step):
        raise ValueError(f"Warm-start field {key!r} has shape {arr.shape}, cannot read step {step}.")
    return arr


def validate_warmstart_raw(data, args, controller, st):
    expected_vision_mode = "rgb" if args.rgb else ("depth_only" if bool(getattr(args, "depth_only", False)) else "depth")
    raw_vision = raw_vision_mode_from_data(data)
    if raw_vision != expected_vision_mode:
        raise ValueError(
            f"Warm-start raw episode vision_mode={raw_vision!r}, but ikpush play is running {expected_vision_mode!r}."
        )
    raw_frame = raw_action_frame_from_data(data)
    ckpt_frame = str(getattr(controller, "action_frame", "world")).lower()
    if raw_frame != ckpt_frame:
        raise ValueError(f"Warm-start raw action_frame={raw_frame!r}, checkpoint action_frame={ckpt_frame!r}.")
    raw_state_version = raw_ikpush_state_version_from_data(data)
    ckpt_state_version = str(controller.config.get("ikpush_state_version", "legacy"))
    if raw_state_version != ckpt_state_version:
        raise ValueError(
            f"Warm-start raw ikpush_state_version={raw_state_version!r}, "
            f"checkpoint ikpush_state_version={ckpt_state_version!r}."
        )
    if "door_cfg" in data.files:
        raw_cfg = normalize_config_path_for_match(data["door_cfg"])
        play_cfg = normalize_config_path_for_match(args.door_cfg)
        if raw_cfg != play_cfg:
            raise ValueError(f"Warm-start raw door_cfg={raw_cfg!r}, play door_cfg={play_cfg!r}.")
    if "door_asset_name" in data.files:
        raw_name = scalar_to_str(data["door_asset_name"])
        play_name = str(st.door.spec.get("name", ""))
        if raw_name and play_name and raw_name != play_name:
            raise ValueError(f"Warm-start raw door_asset_name={raw_name!r}, play door_asset_name={play_name!r}.")
    elif "door_asset_index" in data.files:
        raw_index = scalar_to_int(data["door_asset_index"])
        play_index = int(getattr(st.door, "asset_index", -1))
        if play_index >= 0 and raw_index != play_index:
            raise ValueError(f"Warm-start raw door_asset_index={raw_index}, play door_asset_index={play_index}.")

    step = int(args.dp_warmstart_step)
    for key in (
        "replay_root_state",
        "replay_dof_pos",
        "replay_dof_vel",
        "replay_door_root_state",
        "replay_door_dof_pos",
        "replay_door_dof_vel",
    ):
        require_warmstart_field(data, key, step)
    if args.dp_warmstart_expert_obs:
        if raw_image_keys_for_vision_mode is None:
            raise RuntimeError("Warm-start expert observation prefill requires door_dp_common.raw_image_keys_for_vision_mode.")
        missing = [key for key in raw_image_keys_for_vision_mode(expected_vision_mode) if key not in data.files]
        if missing:
            raise KeyError(f"Warm-start expert observation prefill is missing raw image fields: {missing}")


def raw_dp_dofs_to_actor_dofs(raw_pos, raw_vel, dof_names, fallback_pos):
    pos_out = np.asarray(fallback_pos, dtype=np.float32).copy()
    vel_out = np.zeros_like(pos_out, dtype=np.float32)
    raw_pos = np.asarray(raw_pos, dtype=np.float32).reshape(-1)
    raw_vel = np.asarray(raw_vel, dtype=np.float32).reshape(-1)
    if raw_pos.shape[0] == len(dof_names):
        pos_out[:] = raw_pos[: len(pos_out)]
        vel_out[:] = raw_vel[: len(vel_out)]
        return pos_out, vel_out
    if raw_pos.shape[0] != DP_NUM_DOFS:
        raise ValueError(f"Expected replay_dof_pos length {DP_NUM_DOFS} or {len(dof_names)}, got {raw_pos.shape[0]}.")
    for src_idx, name in enumerate(dof_names):
        dp_idx = FLOAT_ARM_TO_DP_DOF.get(name)
        if dp_idx is None:
            continue
        pos_out[src_idx] = float(raw_pos[dp_idx])
        if dp_idx < raw_vel.shape[0]:
            vel_out[src_idx] = float(raw_vel[dp_idx])
    return pos_out, vel_out


def apply_warmstart_state(gym, sim, st, data, step, dof_names):
    root_state = require_warmstart_field(data, "replay_root_state", step)[step].astype(np.float32)
    for actor in st.actor_handles:
        set_actor_root_from_state(gym, st.env, actor, root_state)

    raw_dof_pos = require_warmstart_field(data, "replay_dof_pos", step)[step]
    raw_dof_vel = require_warmstart_field(data, "replay_dof_vel", step)[step]
    actor_dof_pos, actor_dof_vel = raw_dp_dofs_to_actor_dofs(raw_dof_pos, raw_dof_vel, dof_names, st.dof_positions)
    arm_states = gym.get_actor_dof_states(st.env, st.arm_actor, gymapi.STATE_ALL)
    if len(arm_states) != len(actor_dof_pos):
        raise ValueError(f"Arm DOF count mismatch: actor={len(arm_states)} warmstart={len(actor_dof_pos)}")
    arm_states["pos"][:] = actor_dof_pos
    arm_states["vel"][:] = actor_dof_vel
    gym.set_actor_dof_states(st.env, st.arm_actor, arm_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(st.env, st.arm_actor, actor_dof_pos)
    st.dof_positions[:] = actor_dof_pos

    door_root_state = require_warmstart_field(data, "replay_door_root_state", step)[step].astype(np.float32)
    set_actor_root_from_state(gym, st.env, st.door_actor, door_root_state)

    door_pos = require_warmstart_field(data, "replay_door_dof_pos", step)[step].astype(np.float32)
    door_vel = require_warmstart_field(data, "replay_door_dof_vel", step)[step].astype(np.float32)
    door_states = gym.get_actor_dof_states(st.env, st.door_actor, gymapi.STATE_ALL)
    n = min(len(door_states), len(door_pos))
    door_states["pos"][:n] = door_pos[:n]
    door_states["vel"][:n] = door_vel[:n]
    gym.set_actor_dof_states(st.env, st.door_actor, door_states, gymapi.STATE_ALL)
    if "replay_door_open_stage" in data.files:
        saved_open_stage = np.asarray(data["replay_door_open_stage"][int(step)], dtype=np.float32).reshape(-1)
        st.door.open_stage = bool(saved_open_stage.size and saved_open_stage[0] > 0.5)
    else:
        # Legacy snapshots did not store the internal latch state. Infer it
        # conservatively: tiny PhysX hinge drift must not unlock the door.
        closed_angle = dc.closed_hinge_angle(st.door, st.args)
        hinge_departure = abs(float(door_pos[0]) - float(closed_angle)) if n >= 1 else 0.0
        st.door.open_stage = bool(
            hinge_departure >= math.radians(2.0)
            or (n >= 2 and dc.handle_is_unlocked(st.door, float(door_pos[1])))
        )
    if not st.door.open_stage:
        # Remove the small numerical hinge displacement stored in a locked
        # failure snapshot before the recovery branch starts.
        dc.enforce_locked_door_hinge(gym, st.env, st.door_actor, st.door, st.args)

    yaw = yaw_from_quat_xyzw(root_state[3:7])
    st.traj["base_xy"] = np.asarray(root_state[:2], dtype=np.float32).copy()
    st.traj["yaw"] = float(yaw)
    st.base_start = np.asarray(root_state[:2], dtype=np.float32).copy()
    st.yaw_start = float(yaw)
    if int(step) > 0 and "replay_root_state" in data.files:
        prev_root = np.asarray(data["replay_root_state"][int(step) - 1], dtype=np.float32)
        st.prev_base_xy = prev_root[:2].copy()
        st.prev_yaw = yaw_from_quat_xyzw(prev_root[3:7])
    else:
        st.prev_base_xy = np.asarray(root_state[:2], dtype=np.float32).copy()
        st.prev_yaw = float(yaw)
    if "action" in data.files:
        action_index = max(0, int(step) - 1)
        st.last_dp_action = np.asarray(data["action"][action_index], dtype=np.float32).copy()
    else:
        st.last_dp_action = np.zeros(10, dtype=np.float32)
    if "replay_ee_pos" in data.files:
        st.last_target_pos = np.asarray(data["replay_ee_pos"][int(step)], dtype=np.float32).copy()
    if "replay_ee_quat" in data.files:
        st.last_target_quat = base_ik.normalize_quat(np.asarray(data["replay_ee_quat"][int(step)], dtype=np.float32))

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)


def prefill_dp_controller_from_expert_obs(controller, data, step, vision_mode, env_id=None):
    if raw_image_keys_for_vision_mode is None:
        raise RuntimeError("Expert observation warm-start requires door_dp_common.raw_image_keys_for_vision_mode.")
    image_keys = raw_image_keys_for_vision_mode(vision_mode)
    if env_id is None:
        controller.obs_buffer.clear()
        controller.action_queue.clear()
    else:
        controller.reset_envs([int(env_id)])
    start = max(0, int(step) - int(controller.obs_horizon) + 1)
    for idx in range(start, int(step) + 1):
        if bool(getattr(controller, "config", {}).get("plucker_conditioning", False)):
            if "front_camera_pose_base" not in data.files or "wrist_camera_pose_base" not in data.files:
                raise ValueError("Plücker warm-start requires raw front/wrist camera pose arrays.")
            front_pose = np.asarray(data["front_camera_pose_base"][idx], dtype=np.float32)
            wrist_pose = np.asarray(data["wrist_camera_pose_base"][idx], dtype=np.float32)
        else:
            front_pose = None
            wrist_pose = None
        if vision_mode == "depth_only":
            wrist_depth = image_to_three_channel_uint8(data[image_keys[0]][idx])
            front_depth = image_to_three_channel_uint8(data[image_keys[1]][idx])
            args = (
                np.asarray(data["state"][idx], dtype=np.float32),
                np.zeros_like(wrist_depth),
                wrist_depth,
                None,
                front_depth,
                front_pose,
                wrist_pose,
            )
        else:
            args = (
                np.asarray(data["state"][idx], dtype=np.float32),
                np.asarray(data[image_keys[0]][idx], dtype=np.uint8),
                np.asarray(data[image_keys[1]][idx], dtype=np.uint8),
                np.asarray(data[image_keys[2]][idx], dtype=np.uint8),
                np.asarray(data[image_keys[3]][idx], dtype=np.uint8),
                front_pose,
                wrist_pose,
            )
        if env_id is None:
            controller.append_observation(*args)
        else:
            controller.append_observation_for_env(int(env_id), *args)


def apply_dp_warmstart_if_requested(gym, sim, args, controller, st, dof_names):
    if not args.dp_warmstart:
        return None
    raw_path = Path(args.dp_warmstart_raw_episode).expanduser()
    if not raw_path.is_absolute():
        raw_path = (Path.cwd() / raw_path).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Warm-start raw episode not found: {raw_path}")
    data = np.load(raw_path, allow_pickle=True)
    step = int(args.dp_warmstart_step)
    validate_warmstart_raw(data, args, controller, st)
    apply_warmstart_state(gym, sim, st, data, step, dof_names)
    vision_mode = "rgb" if args.rgb else ("depth_only" if bool(getattr(args, "depth_only", False)) else "depth")
    if args.dp_warmstart_expert_obs:
        prefill_dp_controller_from_expert_obs(controller, data, step, vision_mode, env_id=st.index)
    print(
        f"DP warm-start loaded raw={raw_path} step={step} "
        f"expert_obs_prefill={bool(args.dp_warmstart_expert_obs)} "
        f"base_xy={np.round(st.traj['base_xy'], 4).tolist()} yaw={float(st.traj['yaw']):.4f} "
        f"door_open_stage={bool(st.door.open_stage)}",
        flush=True,
    )
    return data


door_hinge_open_ratio = dc.door_hinge_open_ratio
compute_door_efforts = dc.compute_door_efforts
set_ik_target = dc.set_ik_target
update_arm_ik_targets = dc.update_arm_ik_targets
refresh_current_ee_pose = dc.refresh_current_ee_pose


def smooth_door_twin_ee_command(args, traj, phase, target_pos):
    """Limit DoorTwin skill EE command jumps before recording/replay action export."""

    if phase == "walk" or door_twin_legacy_replay_for_args(args):
        return np.asarray(target_pos, dtype=np.float32).copy()
    if door_twin_profile_for_args(args) is None:
        return np.asarray(target_pos, dtype=np.float32).copy()
    max_step = float(getattr(args, "ee_command_max_step", 0.0) or 0.0)
    if max_step <= 0.0 or "last_target_pos" not in traj:
        return np.asarray(target_pos, dtype=np.float32).copy()

    previous = np.asarray(traj["last_target_pos"], dtype=np.float32)
    target = np.asarray(target_pos, dtype=np.float32).copy()
    delta = target - previous
    distance = float(np.linalg.norm(delta))
    if distance <= max_step or distance <= 1.0e-8:
        return target

    traj["ee_command_smoothing"] = {
        "phase": str(phase),
        "requested_delta": distance,
        "max_step": max_step,
    }
    return previous + delta * (max_step / distance)


def trajectory_targets(
    step,
    args,
    door,
    gym,
    env,
    door_actor,
    ik_state,
    base_start,
    base_stop,
    base_push,
    base_traverse,
    yaw_start,
    yaw_push,
    yaw_traverse,
    traj,
):
    handle_pos, handle_quat = get_body_pose(gym, env, door_actor, door.handle_body_index)
    handle_goal = quat_apply(handle_quat, door.handle_goal_offset) + handle_pos
    base_xy_current = traj.get("base_xy", base_start)
    approach_dir = np.array([base_xy_current[0], base_xy_current[1], args.robot_z], dtype=np.float32) - handle_goal
    approach_dir[2] = 0.0
    approach_dir = normalize(approach_dir)
    if np.linalg.norm(approach_dir) < 1.0e-5:
        approach_dir = np.array([math.cos(yaw_start), math.sin(yaw_start), 0.0], dtype=np.float32)

    skill_profile = door_twin_profile_for_args(args)
    legacy_skill_replay = door_twin_legacy_replay_for_args(args)
    if skill_profile is not None and not legacy_skill_replay:
        skill_points = compute_skill_waypoints(handle_goal, approach_dir, skill_profile)
        pregrasp = skill_points.pregrasp
        grasp = skill_points.grasp
        rotate_pos = skill_points.rotate
    else:
        pregrasp = handle_goal + approach_dir * args.pregrasp_offset
        grasp = handle_goal + approach_dir * args.grasp_offset
        pregrasp[0] += args.grasp_x_offset
        pregrasp[2] += args.grasp_z_offset
        if dc.door_asset_family(door) == "wc4":
            pregrasp[2] += float(getattr(args, "wc4_pregrasp_z_offset", 0.0))
        grasp[0] += args.grasp_x_offset
        grasp[2] += args.grasp_z_offset
        if dc.door_asset_family(door) == "wc4":
            grasp[2] += float(getattr(args, "wc4_grasp_z_offset", 0.0))
    yaw_stop = yaw_start if legacy_skill_replay else yaw_start + move_to_approach_yaw_delta(args)
    goal_quat = forward_ee_quat(args, yaw_stop)
    if skill_profile is not None and not legacy_skill_replay and abs(float(skill_profile.ee_roll_offset)) > 1.0e-6:
        goal_quat = base_ik.quat_multiply(
            goal_quat,
            quat_from_angle_axis(
                float(skill_profile.ee_roll_offset),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
        )

    if skill_profile is None or legacy_skill_replay:
        rotate_offset = np.zeros(3, dtype=np.float32)
        rotate_offset[1] = args.handle_rotate_right_distance
        rotate_offset[2] = -args.handle_rotate_down_distance
        rotate_pos = grasp + rotate_offset
    pull_dir = quat_axis(handle_quat, axis=2)
    pull_dir[2] = 0.0
    fallback_pull_dir = approach_dir.copy()
    pull_dir = normalize(pull_dir)
    if np.linalg.norm(pull_dir) < 1.0e-5:
        pull_dir = fallback_pull_dir
    if float(np.dot(pull_dir, approach_dir)) < 0.0:
        pull_dir = -pull_dir
    # Keep the original pre-push cache construction byte-for-byte compatible;
    # the pull-specific waypoints are installed only after rotate_handle.
    push_dir = -pull_dir
    push_pos = rotate_pos + push_dir * args.door_push_distance
    pull_pos = rotate_pos + pull_dir * args.door_pull_distance
    base_pull = np.asarray(
        traj.get(
            "base_pull",
            np.asarray(base_stop, dtype=np.float32)
            + pull_dir[:2] * float(args.pull_base_retreat_distance),
        ),
        dtype=np.float32,
    )
    base_release = np.asarray(
        traj.get(
            "base_release",
            np.asarray(base_stop, dtype=np.float32)
            + pull_dir[:2] * float(args.release_base_retreat_distance),
        ),
        dtype=np.float32,
    )

    walk_end = args.walk_steps
    initial_end = walk_end + args.initial_hold_steps
    grasp_end = initial_end + args.grasp_steps
    grasp_hold_end = grasp_end + args.grasp_hold_steps
    close_end = grasp_hold_end + args.gripper_close_steps
    rotate_end = close_end + args.handle_rotate_steps
    nominal_pull_end = rotate_end + max(1, int(args.door_pull_steps))
    # Release is event driven: never leave pull_door merely because the nominal
    # duration elapsed.  The first frame at/above pull_release_angle_deg stores
    # pull_complete_step, after which the remaining phase boundaries are fixed.
    pull_end = int(traj.get("pull_complete_step", 2**31 - 1))
    open_hold_end = pull_end + max(0, int(args.open_hold_steps))
    release_end = open_hold_end + max(1, int(args.release_handle_steps))
    pass_end = release_end + max(1, int(args.pass_through_steps))
    return_home_end = pass_end + max(0, int(args.return_home_steps))

    gripper_closed = args.gripper_open + (args.gripper_closed - args.gripper_open) * args.gripper_close_ratio
    gripper_open_stage = args.gripper_open + (
        args.gripper_closed - args.gripper_open
    ) * args.gripper_open_stage_ratio
    target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
    target_quat = None if args.ik_position_only else goal_quat.copy()
    gripper = args.gripper_open
    base_xy = base_start.copy()
    yaw = yaw_start
    phase = "walk"
    if "home_ee_base_pos" not in traj and ik_state.current_pos_np is not None:
        home_base_xy = traj.get("base_xy", base_start)
        home_yaw = float(traj.get("yaw", yaw_start))
        traj["home_ee_base_pos"] = world_pos_to_base(
            ik_state.current_pos_np,
            home_base_xy,
            args.robot_z,
            home_yaw,
        )
        if ik_state.current_quat_np is not None:
            traj["home_ee_base_quat"] = world_quat_to_base(ik_state.current_quat_np, home_yaw)

    if step < walk_end:
        walk_alpha = float(np.clip((step + 1) / max(1, args.walk_steps), 0.0, 1.0))
        t = (
            walk_alpha
            if skill_profile is not None
            and not legacy_skill_replay
            and "approach" in skill_profile.base_moves
            else smoothstep(walk_alpha)
        )
        base_xy = lerp(base_start, base_stop, t)
        yaw = float(
            lerp(
                np.array([yaw_start], dtype=np.float32),
                np.array([yaw_stop], dtype=np.float32),
                t,
            )[0]
        )
        target_pos = ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
        target_quat = None if args.ik_position_only else ik_state.target_quat_np
    else:
        if "pregrasp" not in traj:
            traj["pregrasp"] = pregrasp.copy()
            traj["grasp"] = grasp.copy()
            traj["rotate"] = rotate_pos.copy()
            traj["push"] = push_pos.copy()
            traj["goal_quat"] = goal_quat.copy()
            traj["push_dir"] = push_dir.copy()
            traj["approach_dir"] = approach_dir.copy()
            traj["initial_hold_start_pos"] = (
                ik_state.current_pos_np.copy() if ik_state.current_pos_np is not None else pregrasp.copy()
            )
            if not args.ik_position_only:
                start_quat = ik_state.target_quat_np
                if start_quat is None:
                    start_quat = ik_state.current_quat_np if ik_state.current_quat_np is not None else goal_quat
                traj["initial_hold_start_quat"] = base_ik.normalize_quat(start_quat).astype(np.float32)

        base_xy = base_stop.copy()
        yaw = yaw_stop
        target_pos = traj["pregrasp"].copy()
        target_quat = None if args.ik_position_only else traj["goal_quat"].copy()
        phase = "initial_hold"

        if step < initial_end:
            initial_step = step - walk_end
            move_steps = min(max(1, int(args.initial_hold_move_steps)), max(1, int(args.initial_hold_steps)))
            if initial_step < move_steps:
                t = smoothstep((initial_step + 1) / move_steps)
                target_pos = lerp(traj["initial_hold_start_pos"], traj["pregrasp"], t)
                if not args.ik_position_only:
                    target_quat = quat_nlerp(traj["initial_hold_start_quat"], traj["goal_quat"], t)
        elif step < grasp_end:
            t = smoothstep((step - initial_end + 1) / max(1, args.grasp_steps))
            target_pos = lerp(traj["pregrasp"], traj["grasp"], t)
            phase = "grasp"
        elif step < grasp_hold_end:
            target_pos = traj["grasp"].copy()
            phase = "grasp_hold"
        elif step < close_end:
            t = smoothstep((step - grasp_hold_end + 1) / max(1, args.gripper_close_steps))
            target_pos = traj["grasp"].copy()
            gripper = args.gripper_open + (gripper_closed - args.gripper_open) * t
            phase = "close_gripper"
        elif step < rotate_end:
            t = smoothstep((step - close_end + 1) / max(1, args.handle_rotate_steps))
            target_pos = lerp(traj["grasp"], traj["rotate"], t)
            target_quat = None if args.ik_position_only else base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(
                    float(args.handle_rotate_direction_sign)
                    * t
                    * float(args.handle_rotate_angle),
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                ),
            )
            gripper = gripper_closed
            phase = "rotate_handle"
        elif step < pull_end:
            if "pull" not in traj:
                traj["pull"] = pull_pos.copy()
                traj["pull_dir"] = pull_dir.copy()
                # Retreat on one fixed straight line. Recomputing this point
                # from the rotating handle frame would make the A2W base jump
                # sideways as the door opens.
                traj["base_pull"] = (
                    np.asarray(base_stop, dtype=np.float32)
                    + pull_dir[:2] * float(args.pull_base_retreat_distance)
                ).astype(np.float32)
                traj["base_release"] = (
                    np.asarray(base_stop, dtype=np.float32)
                    + pull_dir[:2] * float(args.release_base_retreat_distance)
                ).astype(np.float32)
            pull_step = step - rotate_end
            t = smoothstep((pull_step + 1) / max(1, int(args.door_pull_steps)))
            turned_quat = base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(
                    float(args.handle_rotate_direction_sign)
                    * float(args.handle_rotate_angle),
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                ),
            )
            traj["turned_quat"] = turned_quat.copy()
            if "handle_contact_offset_local" not in traj:
                traj["handle_contact_offset_local"] = quat_apply(
                    base_ik.quat_conjugate(handle_quat),
                    traj["rotate"] - handle_goal,
                )
                traj["handle_contact_quat_local"] = base_ik.quat_multiply(
                    base_ik.quat_conjugate(handle_quat),
                    turned_quat,
                )
            if "pull_arc_hinge_pos" not in traj:
                hinge_pos, _ = get_body_pose(
                    gym,
                    env,
                    door_actor,
                    door.door_body_index,
                )
                current_ee = (
                    ik_state.current_pos_np.copy()
                    if ik_state.current_pos_np is not None
                    else traj["rotate"].copy()
                )
                traj["pull_arc_hinge_pos"] = np.asarray(hinge_pos, dtype=np.float32)
                traj["pull_arc_handle_start"] = np.asarray(handle_goal, dtype=np.float32).copy()
                traj["pull_arc_ee_offset_start"] = (
                    np.asarray(current_ee, dtype=np.float32)
                    - np.asarray(handle_goal, dtype=np.float32)
                )
                traj["pull_arc_quat_start"] = np.asarray(turned_quat, dtype=np.float32).copy()
            door_pos, _ = get_actor_dof_state(gym, env, door_actor)
            door_open_deg = (
                dc.door_open_degrees(door_pos, args) if len(door_pos) > 0 else 0.0
            )
            if (
                door.open_stage
                and (
                    door_open_deg
                    + max(0.0, float(args.pull_release_angle_tolerance_deg))
                    >= float(args.pull_release_angle_deg)
                )
                and "pull_complete_step" not in traj
            ):
                traj["pull_complete_step"] = (
                    int(step) + max(0, int(args.pull_settle_steps)) + 1
                )
                traj["pull_release_angle_observed_deg"] = float(door_open_deg)
                release_begin_step = (
                    traj["pull_complete_step"]
                    + max(0, int(args.open_hold_steps))
                )
                transition_note = (
                    f"open_hold begins at step={traj['pull_complete_step']}, "
                    if int(args.open_hold_steps) > 0
                    else "open_hold disabled, "
                )
                print(
                    f"[A2WPull] env={int(getattr(args, 'parallel_env_id', 0))} "
                    f"door reached {door_open_deg:.1f} deg "
                    f"(target={float(args.pull_release_angle_deg):.1f}, "
                    f"tolerance={float(args.pull_release_angle_tolerance_deg):.1f}) "
                    f"at step={int(step)}; "
                    f"{transition_note}"
                    f"release begins at step={release_begin_step}.",
                    flush=True,
                )

            if door.open_stage:
                # Command the grasp pose on a hinge-centered circular arc.  It
                # stays only a few degrees ahead of the measured door angle,
                # which generates tensile force without pulling the fingers
                # away from the handle.  The lever-down wrist orientation is
                # rotated only by the planned door yaw, so the spring-loaded
                # handle cannot rebound while it is being pulled.
                actual_open_deg = max(0.0, float(door_open_deg))
                lead_ramp = smoothstep(
                    (pull_step + 1)
                    / max(1, int(args.pull_arc_lead_ramp_steps))
                )
                if (
                    actual_open_deg >= float(args.pull_late_retreat_start_angle_deg)
                    and "pull_late_retreat_start_step" not in traj
                ):
                    traj["pull_late_retreat_start_step"] = int(step)
                if (
                    actual_open_deg >= float(args.pull_mid_arc_lead_start_angle_deg)
                    and "pull_mid_arc_lead_start_step" not in traj
                ):
                    traj["pull_mid_arc_lead_start_step"] = int(step)
                mid_pull_t = 0.0
                if "pull_mid_arc_lead_start_step" in traj:
                    mid_pull_t = smoothstep(
                        (
                            int(step)
                            - int(traj["pull_mid_arc_lead_start_step"])
                            + 1
                        )
                        / max(1, int(args.pull_mid_arc_lead_ramp_steps))
                    )
                late_pull_t = 0.0
                if "pull_late_retreat_start_step" in traj:
                    late_pull_t = smoothstep(
                        (
                            int(step)
                            - int(traj["pull_late_retreat_start_step"])
                            + 1
                        )
                        / max(1, int(args.pull_late_retreat_steps))
                    )
                mid_arc_lead_deg = float(
                    lerp(
                        np.array(
                            [float(args.pull_arc_lead_angle_deg)],
                            dtype=np.float32,
                        ),
                        np.array(
                            [float(args.pull_mid_arc_lead_angle_deg)],
                            dtype=np.float32,
                        ),
                        mid_pull_t,
                    )[0]
                )
                arc_lead_deg = float(
                    lerp(
                        np.array(
                            [mid_arc_lead_deg],
                            dtype=np.float32,
                        ),
                        np.array(
                            [float(args.pull_late_arc_lead_angle_deg)],
                            dtype=np.float32,
                        ),
                        late_pull_t,
                    )[0]
                )
                desired_open_deg = min(
                    # Do not cap the physical grasp target exactly at the
                    # release threshold.  With compliance/contact error that
                    # makes the real hinge settle just below the threshold.
                    # Keep a small circular-arc lead until the *measured*
                    # hinge reaches pull_release_angle_deg; release is still
                    # gated exclusively by the measured door angle above.
                    float(args.pull_release_angle_deg) + arc_lead_deg,
                    actual_open_deg
                    + arc_lead_deg * float(lead_ramp),
                )
                planned_yaw = (
                    float(args.door_motion_sign)
                    * math.radians(desired_open_deg)
                )
                hinge_pos = np.asarray(traj["pull_arc_hinge_pos"], dtype=np.float32)
                handle_start = np.asarray(
                    traj["pull_arc_handle_start"],
                    dtype=np.float32,
                )
                radial = handle_start - hinge_pos
                desired_handle = hinge_pos.copy()
                desired_handle[:2] += dc.rotate_xy(radial[:2], planned_yaw)
                desired_handle[2] += radial[2]
                ee_offset = np.asarray(
                    traj["pull_arc_ee_offset_start"],
                    dtype=np.float32,
                ).copy()
                desired_ee_offset = ee_offset.copy()
                desired_ee_offset[:2] = dc.rotate_xy(
                    ee_offset[:2],
                    planned_yaw,
                )
                target_pos = desired_handle + desired_ee_offset
                # Keep an unclipped geometric reference for the base.  The EE
                # target below is intentionally bounded relative to the
                # measured EE for stable Jacobian IK, but using that bounded
                # target to drive the base creates a feedback dead-zone: once
                # the arm reaches its lateral workspace limit, the reference
                # itself stops moving and the base never catches up.  The base
                # should instead follow the true hinge-centred handle arc.
                base_arc_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
                max_target_lead = float(args.pull_target_max_distance)
                current_ee = ik_state.current_pos_np
                if max_target_lead > 0.0 and current_ee is not None:
                    current_ee = np.asarray(current_ee, dtype=np.float32)
                    target_delta = np.asarray(target_pos, dtype=np.float32) - current_ee
                    target_delta_norm = float(np.linalg.norm(target_delta))
                    if target_delta_norm > max_target_lead:
                        target_pos = (
                            current_ee
                            + target_delta
                            * (max_target_lead / max(target_delta_norm, 1.0e-9))
                        ).astype(np.float32)
                if args.ik_position_only:
                    target_quat = None
                else:
                    target_quat = base_ik.quat_multiply(
                        quat_from_angle_axis(
                            planned_yaw,
                            np.array([0.0, 0.0, 1.0], dtype=np.float32),
                        ),
                        traj["pull_arc_quat_start"],
                    )
                # Retreat only as the door actually opens.  A time-driven base
                # retreat used to drag the arm away even while the hinge was
                # stationary, breaking the grasp.
                traj["pull_max_open_deg"] = max(
                    float(traj.get("pull_max_open_deg", 0.0)),
                    actual_open_deg,
                )
                base_progress = np.clip(
                    (
                        float(traj["pull_max_open_deg"])
                        + max(0.0, float(args.pull_base_progress_lead_deg))
                        * smoothstep(
                            (pull_step + 1)
                            / max(1, int(args.pull_base_retreat_steps))
                        )
                    )
                    / max(
                        1.0e-6,
                        float(args.pull_base_progress_reference_angle_deg),
                    ),
                    0.0,
                    1.0,
                )
                base_t = smoothstep(float(base_progress))
            else:
                # The hinge is still locked: maintain the fully rotated handle
                # pose while applying only a bounded outward pull.
                target_pos = lerp(
                    traj["rotate"],
                    traj["pull"],
                    min(t, 0.20),
                )
                target_quat = None if args.ik_position_only else turned_quat.copy()
                base_t = 0.0
            base_xy = lerp(base_stop, base_pull, base_t)
            late_retreat_t = 0.0
            if "pull_late_retreat_start_step" in traj:
                late_retreat_t = smoothstep(
                    (int(step) - int(traj["pull_late_retreat_start_step"]) + 1)
                    / max(1, int(args.pull_late_retreat_steps))
                )
            base_xy = (
                np.asarray(base_xy, dtype=np.float32)
                + np.asarray(traj["pull_dir"], dtype=np.float32)[:2]
                * float(args.pull_late_retreat_distance)
                * float(late_retreat_t)
            )
            lateral_dir = np.array(
                [-float(traj["pull_dir"][1]), float(traj["pull_dir"][0])],
                dtype=np.float32,
            )
            lateral_norm = float(np.linalg.norm(lateral_dir))
            if lateral_norm > 1.0e-9:
                lateral_dir /= lateral_norm
            if "pull_late_retreat_start_step" in traj:
                if "pull_lateral_follow_target_start" not in traj:
                    traj["pull_lateral_follow_target_start"] = np.asarray(
                        base_arc_target_pos,
                        dtype=np.float32,
                    ).copy()
                target_sweep = float(
                    np.dot(
                        np.asarray(base_arc_target_pos, dtype=np.float32)[:2]
                        - np.asarray(
                            traj["pull_lateral_follow_target_start"],
                            dtype=np.float32,
                        )[:2],
                        lateral_dir,
                    )
                )
                late_lateral_start = float(
                    args.pull_late_base_lateral_boost_start_deg
                )
                late_lateral_full = max(
                    late_lateral_start + 1.0e-6,
                    float(args.pull_late_base_lateral_boost_full_deg),
                )
                late_lateral_t = smoothstep(
                    float(
                        np.clip(
                            (float(door_open_deg) - late_lateral_start)
                            / (late_lateral_full - late_lateral_start),
                            0.0,
                            1.0,
                        )
                    )
                )
                lateral_follow_ratio = float(
                    lerp(
                        np.array(
                            [float(args.pull_high_angle_base_lateral_follow_ratio)],
                            dtype=np.float32,
                        ),
                        np.array(
                            [float(args.pull_late_base_lateral_follow_ratio)],
                            dtype=np.float32,
                        ),
                        late_lateral_t,
                    )[0]
                )
                lateral_follow = float(
                    np.clip(
                        target_sweep
                        * lateral_follow_ratio,
                        -float(args.pull_high_angle_base_lateral_max_distance),
                        float(args.pull_high_angle_base_lateral_max_distance),
                    )
                )
                base_xy = (
                    np.asarray(base_xy, dtype=np.float32)
                    + lateral_dir * lateral_follow
                )
            yaw = float(yaw_stop)
            if (
                "pull_complete_step" in traj
                and "pull_release_base_xy" not in traj
            ):
                traj["pull_release_base_xy"] = np.asarray(
                    base_xy,
                    dtype=np.float32,
                ).copy()
                traj["pull_release_yaw"] = float(yaw)
            tighten_t = smoothstep(
                (pull_step + 1)
                / max(1, int(args.pull_gripper_tighten_steps))
            )
            gripper = float(
                lerp(
                    np.array([gripper_closed], dtype=np.float32),
                    np.array([args.gripper_closed], dtype=np.float32),
                    tighten_t,
                )[0]
            )
            phase = "pull_door"
        elif step < open_hold_end:
            t = smoothstep(
                (step - pull_end + 1) / max(1, int(args.open_hold_steps))
            )
            # Keep the exact final circular-pull command while loosening.
            # Recomputing the target from the rebounding lever pose caused a
            # discontinuous arm command immediately after reaching the release
            # angle.
            target_pos = np.asarray(
                traj.get("last_target_pos", traj["pull"]),
                dtype=np.float32,
            ).copy()
            if args.ik_position_only:
                target_quat = None
            else:
                previous_quat = traj.get("last_target_quat")
                target_quat = (
                    np.asarray(previous_quat, dtype=np.float32).copy()
                    if previous_quat is not None
                    else traj["turned_quat"].copy()
                )
            base_xy = np.asarray(
                traj.get("pull_release_base_xy", base_pull),
                dtype=np.float32,
            ).copy()
            yaw = float(traj.get("pull_release_yaw", yaw_stop))
            # pull_door has already tightened to the fully closed command.
            gripper = float(
                lerp(
                    np.array([args.gripper_closed], dtype=np.float32),
                    np.array([gripper_open_stage], dtype=np.float32),
                    t,
                )[0]
            )
            phase = "open_hold"
        elif step < release_end:
            release_t = float(
                np.clip(
                    (step - open_hold_end + 1)
                    / max(
                        1,
                        min(
                            int(args.release_handle_motion_steps),
                            int(args.release_handle_steps),
                        ),
                    ),
                    0.0,
                    1.0,
                )
            )
            if "release_start_pos" not in traj:
                release_start = (
                    ik_state.current_pos_np.copy()
                    if ik_state.current_pos_np is not None
                    else np.asarray(traj["last_target_pos"], dtype=np.float32).copy()
                )
                live_pull_dir = dc.handle_open_tangent_dir(
                    gym,
                    env,
                    door_actor,
                    door,
                    handle_goal,
                    traj["pull_dir"],
                    args,
                )
                if float(np.dot(live_pull_dir, traj["approach_dir"])) < 0.0:
                    live_pull_dir = -live_pull_dir
                release_out = (
                    release_start
                    + live_pull_dir * float(args.release_outward_distance)
                )
                release_outer = (
                    release_out
                    + live_pull_dir * float(args.release_extra_outward_distance)
                )
                home_world = None
                if "home_ee_base_pos" in traj:
                    home_world = base_pos_to_world(
                        traj["home_ee_base_pos"],
                        base_release,
                        args.robot_z,
                        yaw_stop,
                    )
                release_retract = release_outer.copy()
                if home_world is not None:
                    # The outside arc now finishes at the actual arm home
                    # position before pass_through.  Keeping the old partially
                    # extended endpoint attached to the moving base swept the
                    # forearm back through the handle during traversal.
                    release_retract = np.asarray(home_world, dtype=np.float32).copy()
                else:
                    lateral_delta = math.copysign(
                        float(args.release_lateral_distance),
                        float(base_pull[1] - release_outer[1])
                        if abs(float(base_pull[1] - release_outer[1])) > 1.0e-6
                        else 1.0,
                    )
                    release_retract[1] += lateral_delta
                    release_retract[2] += float(args.release_lift_distance)
                release_arc_control = (
                    release_outer
                    + live_pull_dir * float(args.release_outside_arc_bulge)
                    + 0.5 * (release_retract - release_outer)
                )
                traj["release_start_pos"] = release_start
                traj["release_out_pos"] = release_out
                traj["release_outer_pos"] = release_outer
                traj["release_arc_control_pos"] = release_arc_control
                traj["release_retract_pos"] = release_retract
                traj["release_retract_base_pos"] = world_pos_to_base(
                    release_retract,
                    base_release,
                    args.robot_z,
                    yaw_stop,
                )
                if not args.ik_position_only:
                    release_quat = (
                        ik_state.current_quat_np.copy()
                        if ik_state.current_quat_np is not None
                        else traj.get("last_target_quat")
                    )
                    if release_quat is not None:
                        release_quat = base_ik.normalize_quat(release_quat).astype(np.float32)
                        traj["release_quat"] = release_quat
                        release_retract_quat = release_quat
                        if "home_ee_base_quat" in traj:
                            release_retract_quat = base_quat_to_world(
                                traj["home_ee_base_quat"],
                                yaw_stop,
                            )
                        traj["release_retract_quat"] = (
                            base_ik.normalize_quat(release_retract_quat)
                            .astype(np.float32)
                        )
                        traj["release_retract_base_quat"] = world_quat_to_base(
                            traj["release_retract_quat"],
                            yaw_stop,
                        )

            outward_fraction = float(
                np.clip(args.release_outward_fraction, 1.0e-3, 0.95)
            )
            outer_end_fraction = float(
                np.clip(
                    args.release_extra_outward_end_fraction,
                    outward_fraction + 1.0e-3,
                    0.95,
                )
            )
            gripper_open_start_fraction = float(
                np.clip(args.release_gripper_open_start_fraction, 0.0, 0.95)
            )
            gripper_open_fraction = float(
                np.clip(args.release_gripper_open_fraction, 1.0e-3, 1.0)
            )
            if release_t < outward_fraction:
                motion_t = smoothstep(release_t / outward_fraction)
                target_pos = lerp(
                    traj["release_start_pos"],
                    traj["release_out_pos"],
                    motion_t,
                )
            elif release_t < outer_end_fraction:
                motion_t = smoothstep(
                    (release_t - outward_fraction)
                    / max(1.0e-6, outer_end_fraction - outward_fraction)
                )
                target_pos = lerp(
                    traj["release_out_pos"],
                    traj["release_outer_pos"],
                    motion_t,
                )
            else:
                motion_t = smoothstep(
                    (release_t - outer_end_fraction)
                    / max(1.0e-6, 1.0 - outer_end_fraction)
                )
                one_minus_t = 1.0 - motion_t
                # Quadratic Bezier arc that first stays outside the handle,
                # then folds laterally.  The outward control point prevents
                # the old inward sweep through the door/handle.
                target_pos = (
                    one_minus_t * one_minus_t * traj["release_outer_pos"]
                    + 2.0
                    * one_minus_t
                    * motion_t
                    * traj["release_arc_control_pos"]
                    + motion_t * motion_t * traj["release_retract_pos"]
                )
            target_quat = None
            if not args.ik_position_only:
                release_quat = traj.get("release_quat", traj.get("turned_quat"))
                retract_quat = traj.get("release_retract_quat", release_quat)
                if (
                    release_quat is not None
                    and retract_quat is not None
                    and release_t >= outer_end_fraction
                ):
                    quat_t = smoothstep(
                        (release_t - outer_end_fraction)
                        / max(1.0e-6, 1.0 - outer_end_fraction)
                    )
                    target_quat = quat_nlerp(
                        release_quat,
                        retract_quat,
                        quat_t,
                    )
                else:
                    target_quat = release_quat
            release_gripper_start = (
                gripper_open_stage
                if int(args.open_hold_steps) > 0
                else float(args.gripper_closed)
            )
            gripper_open_t = float(
                np.clip(
                    (release_t - gripper_open_start_fraction)
                    / gripper_open_fraction,
                    0.0,
                    1.0,
                )
            )
            gripper = float(
                lerp(
                    np.array([release_gripper_start], dtype=np.float32),
                    np.array([args.gripper_open], dtype=np.float32),
                    smoothstep(gripper_open_t),
                )[0]
            )
            release_base_start = np.asarray(
                traj.get("pull_release_base_xy", base_pull),
                dtype=np.float32,
            )
            base_xy = lerp(
                release_base_start,
                base_release,
                smoothstep(release_t),
            )
            yaw = float(yaw_stop)
            joint_home_start = float(
                np.clip(args.release_joint_home_start_fraction, 0.0, 0.95)
            )
            traj["release_joint_home_alpha"] = (
                smoothstep(
                    (release_t - joint_home_start)
                    / max(1.0e-6, 1.0 - joint_home_start)
                )
                if release_t >= joint_home_start
                else 0.0
            )
            phase = "release_handle"
        elif step < pass_end:
            t = smoothstep(
                (step - release_end + 1)
                / max(1, int(args.pass_through_steps))
            )
            base_xy = lerp(base_release, base_traverse, t)
            yaw = float(
                lerp(
                    np.array([yaw_stop], dtype=np.float32),
                    np.array([yaw_traverse], dtype=np.float32),
                    t,
                )[0]
            )
            retract_base_pos = traj.get("release_retract_base_pos")
            target_pos = (
                base_pos_to_world(
                    retract_base_pos,
                    base_xy,
                    args.robot_z,
                    yaw,
                )
                if retract_base_pos is not None
                else traj["release_retract_pos"].copy()
            )
            target_quat = None
            if (
                not args.ik_position_only
                and "release_retract_base_quat" in traj
            ):
                target_quat = base_quat_to_world(
                    traj["release_retract_base_quat"],
                    yaw,
                )
            gripper = args.gripper_open
            phase = "pass_through"
        elif step < return_home_end:
            final_base = base_traverse
            final_yaw = yaw_traverse
            t = smoothstep(
                (step - pass_end + 1) / max(1, args.return_home_steps)
            )
            if "return_home_start_base_xy" not in traj:
                traj["return_home_start_base_xy"] = traj.get("base_xy", final_base).copy()
                traj["return_home_start_yaw"] = float(traj.get("yaw", final_yaw))
                traj["return_home_start_target_pos"] = (
                    traj["last_target_pos"].copy()
                    if "last_target_pos" in traj
                    else (
                        ik_state.current_pos_np.copy()
                        if ik_state.current_pos_np is not None
                        else traj["release_retract_pos"].copy()
                    )
                )
                if not args.ik_position_only:
                    if "last_target_quat" in traj and traj["last_target_quat"] is not None:
                        traj["return_home_start_target_quat"] = traj["last_target_quat"].copy()
                    elif ik_state.current_quat_np is not None:
                        traj["return_home_start_target_quat"] = base_ik.normalize_quat(ik_state.current_quat_np).astype(np.float32)
            base_xy = lerp(traj["return_home_start_base_xy"], final_base, t)
            yaw = float(lerp(
                np.array([traj["return_home_start_yaw"]], dtype=np.float32),
                np.array([final_yaw], dtype=np.float32),
                t,
            )[0])
            home_base_pos = traj.get("home_ee_base_pos")
            return_home_goal_pos = (
                base_pos_to_world(home_base_pos, base_xy, args.robot_z, yaw)
                if home_base_pos is not None
                else traj["release_retract_pos"].copy()
            )
            target_pos = lerp(traj["return_home_start_target_pos"], return_home_goal_pos, t)
            target_quat = None
            if not args.ik_position_only:
                return_home_goal_quat = (
                    base_quat_to_world(traj["home_ee_base_quat"], yaw)
                    if "home_ee_base_quat" in traj
                    else traj.get("goal_quat")
                )
                if "return_home_start_target_quat" in traj and return_home_goal_quat is not None:
                    target_quat = quat_nlerp(traj["return_home_start_target_quat"], return_home_goal_quat, t)
                elif return_home_goal_quat is not None:
                    target_quat = return_home_goal_quat
            gripper = args.gripper_open
            traj["return_home_alpha"] = t
            phase = "return_home"
        else:
            final_base = base_traverse
            final_yaw = yaw_traverse
            home_base_pos = traj.get("home_ee_base_pos")
            fallback_pos = (
                base_pos_to_world(home_base_pos, final_base, args.robot_z, final_yaw)
                if home_base_pos is not None
                else traj.get("release_retract_pos", traj["pull"]).copy()
            )
            fallback_quat = None
            if not args.ik_position_only and "home_ee_base_quat" in traj:
                fallback_quat = base_quat_to_world(traj["home_ee_base_quat"], final_yaw)
            target_pos, target_quat = chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                fallback_pos,
                fallback_quat,
            )
            base_xy = final_base.copy()
            yaw = final_yaw
            gripper = args.gripper_open
            traj["return_home_alpha"] = 1.0
            phase = "hold_home"

    target_pos = smooth_door_twin_ee_command(args, traj, phase, target_pos)
    traj["base_xy"] = base_xy.copy()
    traj["yaw"] = float(yaw)
    traj["last_target_pos"] = np.asarray(target_pos, dtype=np.float32).copy()
    traj["last_target_quat"] = (
        None
        if target_quat is None
        else base_ik.normalize_quat(np.asarray(target_quat, dtype=np.float32)).astype(np.float32)
    )
    traj["last_gripper"] = float(gripper)
    return phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal


def draw_scripted_trajectory(gym, viewer, env, traj, args, current_target_pos=None):
    if not bool(getattr(args, "draw_scripted_trajectory", False)):
        return
    keys = ("pregrasp", "grasp", "rotate", "pull")
    if not all(key in traj for key in keys):
        return

    radius = max(0.004, float(getattr(args, "trajectory_point_radius", 0.018)))
    samples = max(2, int(getattr(args, "trajectory_point_samples", 16)))
    green_sphere = gymutil.WireframeSphereGeometry(
        radius=radius,
        num_lats=8,
        num_lons=8,
        color=(0.0, 1.0, 0.2),
        color2=(0.0, 0.7, 0.2),
    )
    red_sphere = gymutil.WireframeSphereGeometry(
        radius=radius * 1.35,
        num_lats=8,
        num_lons=8,
        color=(1.0, 0.0, 0.0),
        color2=(1.0, 0.0, 0.0),
    )

    points = [np.asarray(traj[key], dtype=np.float32) for key in keys]
    for a, b in zip(points[:-1], points[1:]):
        for i in range(samples):
            u = i / max(1, samples - 1)
            pos = lerp(a, b, u)
            gymutil.draw_lines(green_sphere, gym, viewer, env, base_ik.transform_from_arrays(pos))

    if current_target_pos is not None:
        gymutil.draw_lines(
            red_sphere,
            gym,
            viewer,
            env,
            base_ik.transform_from_arrays(np.asarray(current_target_pos, dtype=np.float32)),
        )


setup_viewer = dc.setup_viewer


def run_demo(
    gym,
    sim,
    env,
    arm_actor,
    actor_handles,
    door,
    door_actor,
    viewer,
    camera_handles,
    args,
    dt,
    dof_names,
    dof_positions,
    defaults,
    ik_state,
):
    num_arm_dofs = len(dof_positions)
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    gripper_idx = dof_dict.get("jointGripper")
    if gripper_idx is not None:
        dof_positions[gripper_idx] = args.gripper_open

    yaw_start, heading, base_start, base_stop = dc.compute_base_walk_targets(args, door)
    dc.configure_dynamic_walk_steps(args, base_start, base_stop)
    base_push, base_traverse = compute_base_push_and_traverse_targets(args, base_stop, heading)
    yaw_push = yaw_start + move_to_approach_yaw_delta(args) + args.push_base_yaw_delta
    yaw_traverse = yaw_push + float(getattr(args, "traverse_yaw_delta", 0.0))
    traj = {"base_xy": base_start.copy()}

    print(
        "base_start:",
        base_start.tolist(),
        "base_stop:",
        base_stop.tolist(),
        "base_push:",
        base_push.tolist(),
        "base_traverse:",
        base_traverse.tolist(),
    )
    print(
        "pass_through_door:",
        bool(args.pass_through_door),
        "rear_offset:",
        float(args.robot_rear_offset),
        "door_pass_clearance:",
        float(args.door_pass_clearance),
    )
    print("Close viewer to exit.")
    start = time.time()
    step = 0
    home_positions = np.asarray(defaults, dtype=np.float32).copy()
    if gripper_idx is not None:
        home_positions[gripper_idx] = np.clip(args.gripper_open, ik_state.lower[gripper_idx].item(), ik_state.upper[gripper_idx].item())

    max_steps = args.steps if args.steps > 0 else 2405
    dp_recorder = None
    single_record_state = None
    if args.record_dp_dataset:
        dc.require_float_dp_recording_deps(args, RawDoorDPRecorder, make_state_feature_names)
        if not camera_handles:
            print(
                "⚠️📷 DP raw recording requested, but no camera sensors were created; episode frames will be discarded.",
                flush=True,
            )
        vision_mode = dc.float_dp_vision_mode(args, normalize_vision_mode)
        dp_recorder = dc.make_float_dp_recorder(
            args,
            door,
            0,
            vision_mode,
            DP_PHASE_NAMES,
            "ikpull",
            IKPULL_STATE_VERSION,
            RawDoorDPRecorder,
            make_state_feature_names,
            randomization_metadata_key="ikpush_randomization",
        )
        dc.print_float_dp_recording_start(args, {0}, vision_mode)
    prev_base_xy = None
    prev_yaw = None
    prev_dp_action = np.zeros(10, dtype=np.float32)
    if dp_recorder is not None:
        single_record_state = SimpleNamespace(
            index=0,
            args=args,
            env=env,
            arm_actor=arm_actor,
            actor_handles=actor_handles,
            door=door,
            door_actor=door_actor,
            camera_handles=camera_handles,
            ik_state=ik_state,
            base_start=base_start,
            yaw_start=yaw_start,
            base_traverse=base_traverse,
            yaw_traverse=yaw_traverse,
            traj=traj,
            dp_recorder=dp_recorder,
            dp_record_success=False,
            dp_record_warned_no_camera=False,
            base_door_collision_detected=False,
            base_door_collision_log_step=-10**9,
            last_dp_action=prev_dp_action.copy(),
        )
    while step < max_steps:
        if not handle_viewer_pause(gym, sim, viewer):
            break

        phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal = trajectory_targets(
            step,
            args,
            door,
            gym,
            env,
            door_actor,
            ik_state,
            base_start,
            base_stop,
            base_push,
            base_traverse,
            yaw_start,
            yaw_push,
            yaw_traverse,
            traj,
        )
        set_robot_base_pose(
            gym,
            env,
            actor_handles,
            base_xy,
            args.robot_z,
            yaw,
            getattr(args, "robot_pitch", 0.0),
        )
        release_joint_home_alpha = (
            float(traj.get("release_joint_home_alpha", 0.0))
            if phase == "release_handle"
            else 0.0
        )
        if phase == "release_handle" and release_joint_home_alpha > 0.0:
            if "release_joint_home_start_dofs" not in traj:
                traj["release_joint_home_start_dofs"] = np.asarray(
                    dof_positions,
                    dtype=np.float32,
                ).copy()
            dof_positions[:] = lerp(
                traj["release_joint_home_start_dofs"],
                home_positions,
                release_joint_home_alpha,
            )
            if gripper_idx is not None:
                dof_positions[gripper_idx] = home_positions[gripper_idx]
            ik_state.last_pos_error = 0.0
        elif phase == "return_home":
            if "return_home_start_dofs" not in traj:
                traj["return_home_start_dofs"] = np.asarray(dof_positions, dtype=np.float32).copy()
            alpha = float(traj.get("return_home_alpha", 0.0))
            dof_positions[:] = lerp(traj["return_home_start_dofs"], home_positions, alpha)
            ik_state.last_pos_error = 0.0
        elif phase in ("walk", "pass_through", "hold_home"):
            dof_positions[:] = home_positions
            ik_state.last_pos_error = 0.0
        else:
            set_ik_target(ik_state, target_pos, target_quat)
            update_arm_ik_targets(gym, sim, dof_positions, ik_state, args, num_arm_dofs)
            if gripper_idx is not None:
                dof_positions[gripper_idx] = np.clip(gripper, ik_state.lower[gripper_idx].item(), ik_state.upper[gripper_idx].item())
        gym.set_actor_dof_position_targets(env, arm_actor, dof_positions)

        dc.enforce_locked_door_hinge(gym, env, door_actor, door, args)
        door_pos, door_vel = get_actor_dof_state(gym, env, door_actor)
        door_efforts = compute_door_efforts(door, door_pos, door_vel, args)
        if len(door_efforts) > 0:
            gym.apply_actor_dof_efforts(env, door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        need_camera_render = bool(camera_handles and (args.show_camera_images or args.record_dp_dataset))
        if viewer is not None and need_camera_render and (
            args.draw_ik_target
            or args.draw_camera_axes
            or args.draw_scripted_trajectory
            or bool(getattr(args, "dp3_draw_point_cloud", False))
        ):
            # Clear viewer-only debug lines before camera rendering so depth/RGB tensors stay clean.
            gym.clear_lines(viewer)
        if viewer is not None or need_camera_render:
            gym.step_graphics(sim)
        if need_camera_render:
            gym.render_all_camera_sensors(sim)
        if args.show_camera_images and camera_handles and step % max(1, args.camera_display_interval) == 0:
            show_camera_handle_images(gym, sim, env, camera_handles, args)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        door_pos_record, door_vel_record = get_actor_dof_state(gym, env, door_actor)

        if single_record_state is not None:
            single_record_state.traj = traj
            single_record_state.prev_base_xy = prev_base_xy
            single_record_state.prev_yaw = prev_yaw
            single_record_state.last_phase = phase
            single_record_state.last_door_pos = door_pos_record
            single_record_state.last_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
            single_record_state.last_target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float32).copy()
            single_record_state.last_gripper = float(gripper)
            dc.monitor_base_door_collision(gym, step, single_record_state)
            dc.record_float_dp_frame(
                gym,
                sim,
                single_record_state,
                dof_names,
                gripper_idx,
                dt,
                DP_PHASE_ID.get(phase, 0),
                door_pos_record,
                door_vel_record,
            )
            prev_dp_action = np.asarray(single_record_state.last_dp_action, dtype=np.float32).copy()

        if viewer is not None:
            if args.draw_ik_target or args.draw_camera_axes or args.draw_scripted_trajectory:
                gym.clear_lines(viewer)
            if args.draw_camera_axes:
                draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, args)
            if args.draw_scripted_trajectory:
                draw_scripted_trajectory(gym, viewer, env, traj, args, target_pos)
            if args.draw_ik_target:
                if phase in ("return_home", "hold_home"):
                    saved_target_pos_np = ik_state.target_pos_np
                    saved_target_quat_np = ik_state.target_quat_np
                    ik_state.target_pos_np = np.asarray(target_pos, dtype=np.float32).copy()
                    ik_state.target_quat_np = (
                        None
                        if target_quat is None
                        else base_ik.normalize_quat(target_quat).astype(np.float32)
                    )
                    try:
                        base_ik.draw_ik_target(gym, viewer, env, ik_state)
                    finally:
                        ik_state.target_pos_np = saved_target_pos_np
                        ik_state.target_quat_np = saved_target_quat_np
                else:
                    base_ik.draw_ik_target(gym, viewer, env, ik_state)
                target_pose = base_ik.transform_from_arrays(handle_goal)
                goal_sphere = gymutil.WireframeSphereGeometry(
                    radius=0.035,
                    num_lats=8,
                    num_lons=8,
                    color=(0.0, 1.0, 0.2),
                    color2=(0.0, 0.7, 0.2),
                )
                gymutil.draw_lines(goal_sphere, gym, viewer, env, target_pose)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"[{step:04d}] phase={phase:14s} "
                f"ik_pos_err={ik_state.last_pos_error:.4f} "
                f"door={math.degrees(float(door_pos[0])) if len(door_pos) else 0.0:.1f}deg "
                f"signed_push={math.degrees(args.door_motion_sign * float(door_pos[0])) if len(door_pos) else 0.0:.1f}deg "
                f"handle={math.degrees(float(door_pos[1])) if len(door_pos) > 1 else 0.0:.1f}deg "
                f"open_stage={door.open_stage}",
                flush=True,
            )
        prev_base_xy = np.asarray(base_xy, dtype=np.float32).copy()
        prev_yaw = float(yaw)
        step += 1

    print(f"Done after {step} steps ({time.time() - start:.2f}s).")
    if single_record_state is not None:
        dc.finish_float_dp_recorders([single_record_state], args)


def initialize_parallel_env_state(
    index,
    env,
    arm_actor,
    actor_handles,
    door,
    door_actor,
    camera_handles,
    ik_state,
    args,
    dof_names,
    dof_positions_template,
    defaults,
    dp_recorder,
):
    dof_positions = np.asarray(dof_positions_template, dtype=np.float32).copy()
    home_positions = np.asarray(defaults, dtype=np.float32).copy()
    gripper_idx = {name: i for i, name in enumerate(dof_names)}.get("jointGripper")
    if gripper_idx is not None:
        dof_positions[gripper_idx] = args.gripper_open
        home_positions[gripper_idx] = np.clip(
            args.gripper_open,
            ik_state.lower[gripper_idx].item(),
            ik_state.upper[gripper_idx].item(),
        )

    yaw_start, heading, base_start, base_stop = dc.compute_base_walk_targets(args, door)
    dc.configure_dynamic_walk_steps(args, base_start, base_stop, env_index=index)
    base_push, base_traverse = compute_base_push_and_traverse_targets(args, base_stop, heading)
    yaw_push = yaw_start + move_to_approach_yaw_delta(args) + args.push_base_yaw_delta
    state = ParallelEnvState(
        index=int(index),
        args=args,
        env=env,
        arm_actor=arm_actor,
        actor_handles=actor_handles,
        door=door,
        door_actor=door_actor,
        camera_handles=camera_handles,
        ik_state=ik_state,
        dof_positions=dof_positions,
        home_positions=home_positions,
        base_start=base_start,
        base_stop=base_stop,
        base_push=base_push,
        base_traverse=base_traverse,
        yaw_start=yaw_start,
        yaw_push=yaw_push,
        yaw_traverse=yaw_push + float(getattr(args, "traverse_yaw_delta", 0.0)),
        traj={"base_xy": base_start.copy()},
        dp_recorder=dp_recorder,
    )
    if str(getattr(args, "arm_ik_solver", "gym_jacobian")) == "tracik":
        joint_indices = np.asarray([dof_names.index(name) for name in Z1_ARM_JOINT_NAMES], dtype=np.int64)
        lower_np = ik_state.lower.detach().cpu().numpy()[joint_indices]
        upper_np = ik_state.upper.detach().cpu().numpy()[joint_indices]
        state.tracik_joint_indices = joint_indices
        state.tracik_controller = TracIKTrajectoryController(
            args._tracik_solver,
            dof_positions[joint_indices],
            lower_np,
            upper_np,
            float(args.sim_dt),
            args,
            seed=int(getattr(args, "env_seed", getattr(args, "seed", 0))),
        )
        if int(index) == 0:
            controller = state.tracik_controller
            print(
                "TRAC-IK simulation cadence: "
                f"sim_hz={1.0 / controller.sim_dt:.1f} "
                f"command_stride={controller.command_stride} "
                f"effective_command_hz={1.0 / controller.command_period:.1f}",
                flush=True,
            )
    init_door_twin_tracker(state)
    return state


def create_parallel_env_states(
    gym,
    sim,
    base_asset,
    arm_asset,
    door_templates,
    dof_props,
    dof_states,
    dof_positions,
    lower,
    upper,
    defaults,
    dof_names,
    args,
    base_dof_data=None,
):
    dc.require_float_dp_recording_deps(args, RawDoorDPRecorder, make_state_feature_names)
    vision_mode = dc.float_dp_vision_mode(args, normalize_vision_mode)

    envs_per_row = max(1, int(math.ceil(math.sqrt(float(args.num_envs)))))
    record_env_ids = dc.float_dp_record_env_ids(args)
    created = []
    for env_index in range(int(args.num_envs)):
        door_template = door_templates[env_index % len(door_templates)]
        door = clone_door_runtime(door_template)
        env_args = make_env_args(
            args,
            env_index,
            str(door_template.spec.get("name", "")),
        )
        env, arm_actor, actor_handles, door_actor, _ = create_parallel_env_actors(
            gym,
            sim,
            base_asset,
            arm_asset,
            door,
            dof_props,
            dof_states,
            env_args,
            env_index,
            envs_per_row,
        )
        configure_pull_contact_friction(
            gym,
            env,
            arm_actor,
            door_actor,
            env_args,
        )
        apply_a2w_base_visual_dofs(gym, env, actor_handles, base_dof_data)
        created.append((env_index, env_args, env, arm_actor, actor_handles, door, door_actor))

    env_states = []
    for env_index, env_args, env, arm_actor, actor_handles, door, door_actor in created:
        camera_handles = {}
        if (
            env_args.show_camera_images
            or env_args.record_dp_dataset
            or env_args.dp_policy_checkpoint
            or str(getattr(env_args, "recovery_batch_manifest", "") or "").strip()
            or env_args.dump_keyframe_images
        ) and (
            env_args.enable_wrist_camera
            or env_args.enable_front_camera
            or any(
                view in ("front_left", "front_right", "observer_left", "observer_right", "overhead", "handle_closeup")
                for view in door_twin_camera_view_names(env_args)
            )
        ):
            camera_handles = create_low_level_cameras(
                gym,
                env,
                arm_actor,
                actor_handles,
                door,
                env_args,
            )
        ik_state = base_ik.setup_ik_controller(gym, sim, env, arm_actor, arm_asset, dof_names, lower, upper, env_args)
        dp_recorder = None
        if env_args.record_dp_dataset and env_index in record_env_ids:
            dp_recorder = dc.make_float_dp_recorder(
                env_args,
                door,
                env_index,
                vision_mode,
                DP_PHASE_NAMES,
                "ikpull",
                IKPULL_STATE_VERSION,
                RawDoorDPRecorder,
                make_state_feature_names,
                randomization_metadata_key="ikpush_randomization",
            )
        state = initialize_parallel_env_state(
                env_index,
                env,
                arm_actor,
                actor_handles,
                door,
                door_actor,
                camera_handles,
                ik_state,
                env_args,
                dof_names,
                dof_positions,
                defaults,
                dp_recorder,
            )
        if env_index == 0 and bool(getattr(env_args, "capture_stage_screenshots", False)):
            state.stage_screenshot_camera = create_stage_screenshot_camera(gym, env, env_args)
        env_states.append(state)
    if args.record_dp_dataset:
        dc.print_float_dp_recording_start(args, record_env_ids, vision_mode)
    shown_randomization = [json.loads(st.args.ikpush_randomization_json) for st in env_states[: min(4, len(env_states))]]
    print(
        f"ikpush per-env randomization seed={int(args.seed)} "
        f"enabled={not bool(args.no_ikpush_env_randomization)} "
        f"door_cycle={[door.spec.get('name', '') for door in door_templates]} "
        f"sample_envs={shown_randomization}",
        flush=True,
    )
    return env_states, vision_mode


def load_recovery_batch_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = list(manifest.get("entries", []))
    if not entries:
        raise ValueError(f"Recovery batch manifest has no entries: {manifest_path}")
    return manifest_path, manifest, entries


def recovery_source_frame_index(data, branch_step):
    if "step" not in data.files:
        return int(branch_step)
    steps = np.asarray(data["step"], dtype=np.int64).reshape(-1)
    exact = np.flatnonzero(steps == int(branch_step))
    if exact.size:
        return int(exact[0])
    return int(np.argmin(np.abs(steps - int(branch_step))))


def apply_recovery_randomization(gym, st, metadata):
    sampled = dict(metadata.get("ikpush_randomization") or {})
    for name in (
        "door_x",
        "door_y",
        "door_wall_x_offset",
        "robot_x",
        "robot_y",
        "robot_yaw",
        "robot_z",
        "robot_pitch",
        "pregrasp_offset",
        "grasp_x_offset",
        "grasp_z_offset",
        "handle_rotate_angle",
        "door_push_distance",
        "door_joint_friction",
        "door_joint_damping",
        "door_open_resistance",
        "handle_joint_friction",
        "handle_joint_damping",
        "handle_spring_stiffness",
        "handle_spring_damping",
    ):
        value = sampled.get(name)
        if value is not None and not isinstance(value, (list, dict)):
            setattr(st.args, name, float(value))
    dc.configure_door_actor_dofs(gym, st.env, st.door_actor, st.door, st.args)


def initialize_recovery_candidate(gym, sim, st, entry, dof_names, raw_root, dt):
    failure_npz = Path(entry["failure_rollout_npz"]).expanduser().resolve()
    metadata_path = Path(entry["metadata_path"]).expanduser().resolve()
    data = np.load(failure_npz, allow_pickle=True)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    params = dict(entry["parameters"])
    diagnosis = dict(entry.get("diagnosis") or {})
    branch_step = int(entry["branch_step"])
    frame_idx = recovery_source_frame_index(data, branch_step)

    apply_recovery_randomization(gym, st, metadata)
    apply_warmstart_state(gym, sim, st, data, frame_idx, dof_names)
    current_ee_pose(gym, sim, st.ik_state)
    root_state = np.asarray(data["replay_root_state"][frame_idx], dtype=np.float32)
    st.args.robot_z = float(root_state[2])
    st.args.robot_pitch = float(pitch_from_quat_xyzw(root_state[3:7]))
    base_xy = root_state[:2].astype(np.float32).copy()
    yaw = yaw_from_quat_xyzw(root_state[3:7])

    handle_pos, handle_quat = get_body_pose(gym, st.env, st.door_actor, st.door.handle_body_index)
    handle_goal = quat_apply(handle_quat, st.door.handle_goal_offset) + handle_pos
    approach = np.asarray([base_xy[0], base_xy[1], st.args.robot_z], dtype=np.float32) - handle_goal
    approach[2] = 0.0
    approach = normalize(approach)
    if float(np.linalg.norm(approach)) < 1.0e-5:
        approach = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    lateral = np.asarray([-approach[1], approach[0]], dtype=np.float32)

    source_grasp_x = float(getattr(st.args, "grasp_x_offset", 0.0))
    source_grasp_z = float(getattr(st.args, "grasp_z_offset", 0.0))
    source_pregrasp_offset = float(getattr(st.args, "pregrasp_offset", 0.15))
    source_handle_rotate_steps = int(getattr(st.args, "handle_rotate_steps", 100))
    source_door_push_steps = int(getattr(st.args, "door_push_steps", 300))
    open_first = bool(params.get("open_gripper_first", True))
    if open_first:
        st.args.walk_steps = 30
        st.args.initial_hold_steps = 30
        st.args.initial_hold_move_steps = 30
        st.args.grasp_steps = 50
        st.args.grasp_hold_steps = 5 if params.get("close_timing") == "after_5_stable_contact_frames" else 0
        st.args.gripper_close_steps = 50
        st.args.pregrasp_offset = source_pregrasp_offset + float(params.get("retreat_distance_m", 0.05))
    else:
        st.args.walk_steps = 0
        st.args.initial_hold_steps = 0
        st.args.initial_hold_move_steps = 0
        st.args.grasp_steps = 0
        st.args.grasp_hold_steps = 0
        st.args.gripper_close_steps = 0
    # Preserve the base-motion cadence used by the original scripted expert.
    # Recovery candidates may alter arm/gripper re-acquisition, but they must
    # not change the base push duration or introduce lateral base jumps.
    st.args.handle_rotate_steps = max(1, source_handle_rotate_steps)
    st.args.door_push_steps = max(1, source_door_push_steps)
    st.args.return_home_steps = 0
    st.args.traverse_steps = 0
    st.args.grasp_x_offset = source_grasp_x + float(params.get("handle_regrasp_x_offset_m", 0.0))
    st.args.grasp_z_offset = source_grasp_z + float(params.get("reapproach_z_offset_m", 0.0))

    adjusted_base = base_xy.copy()
    heading = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    st.base_start = base_xy.copy()
    st.base_stop = adjusted_base.astype(np.float32)
    st.base_push, st.base_traverse = compute_base_push_and_traverse_targets(st.args, st.base_stop, heading)
    st.yaw_start = float(yaw)
    st.yaw_push = float(yaw) + float(getattr(st.args, "push_base_yaw_delta", 0.0))
    st.yaw_traverse = st.yaw_push + float(getattr(st.args, "traverse_yaw_delta", 0.0))
    st.traj = {"base_xy": base_xy.copy(), "yaw": float(yaw)}
    st.last_target_pos = st.ik_state.current_pos_np.copy()
    st.last_target_quat = (
        None if st.ik_state.current_quat_np is None else st.ik_state.current_quat_np.copy()
    )
    st.last_gripper = float(
        np.asarray(data["replay_dof_pos"][frame_idx], dtype=np.float32).reshape(-1)[-1]
    )
    st.last_dp_action = np.asarray(
        data["action"][frame_idx] if "action" in data.files else np.zeros(10), dtype=np.float32
    ).copy()
    st.recovery_entry = entry
    st.recovery_applied_parameters = {
        **params,
        "base_lateral_adjust_m": 0.0,
        "rotate_steps": int(st.args.handle_rotate_steps),
        "push_steps": int(st.args.door_push_steps),
        "preserve_scripted_base_motion": True,
    }
    st.recovery_source_frame_idx = frame_idx
    st.recovery_failure_type = str(diagnosis.get("failure_type", "progress_stagnation"))
    st.recovery_initial_door = np.asarray(data["replay_door_dof_pos"][frame_idx], dtype=np.float32).copy()
    st.recovery_contact_streak = 0
    st.recovery_max_contact_streak = 0
    st.recovery_progress_restart = False
    st.recovery_local_success = False
    st.recovery_final_success = False
    st.recovery_first_local_step = -1
    st.recovery_first_final_step = -1
    st.recovery_max_door_deg = 0.0
    st.recovery_initial_open_stage = bool(st.door.open_stage)
    st.recovery_first_contact_any_step = -1
    st.recovery_first_contact_both_step = -1
    st.recovery_first_unlock_step = -1
    st.recovery_unlock_phase = ""
    st.recovery_handle_deg_at_unlock = 0.0
    st.recovery_max_locked_hinge_drift_deg = 0.0
    st.recovery_premature_unlock = False
    st.recovery_done = False
    st.base_door_collision_detected = False
    st.dp_record_warned_no_camera = False
    st.dp_record_sim_steps = 0
    st.dp_record_prev_base_xy = None
    st.dp_record_prev_yaw = None

    st.args.dp_raw_root = str(raw_root)
    st.args.record_camera_pose = True
    st.args.record_gripper_handle_contact = True
    st.args.filter_gripper_handle_contact = False
    st.args.dp_record_state_mode = "pi05_last_command_state10"
    st.dp_recorder = dc.make_float_dp_recorder(
        st.args,
        st.door,
        st.index,
        dc.float_dp_vision_mode(st.args, normalize_vision_mode),
        DP_PHASE_NAMES,
        "ikpull",
        IKPULL_STATE_VERSION,
        RawDoorDPRecorder,
        make_state_feature_names,
        randomization_metadata_key="ikpush_randomization",
        extra_metadata={
            "aux.is_recovery": 1,
            "recovery_source_failure_rollout": str(failure_npz),
            "recovery_failure_type": st.recovery_failure_type,
            "recovery_t_dev": int(diagnosis.get("t_dev", branch_step)),
            "recovery_t_fail": int(diagnosis.get("t_fail", branch_step)),
            "recovery_t_branch": branch_step,
            "recovery_candidate_id": int(entry["candidate_id"]),
            "recovery_parameters_json": json.dumps(st.recovery_applied_parameters, sort_keys=True),
        },
    )
    st.recovery_total_steps = (
        int(st.args.walk_steps)
        + int(st.args.initial_hold_steps)
        + int(st.args.grasp_steps)
        + int(st.args.grasp_hold_steps)
        + int(st.args.gripper_close_steps)
        + int(st.args.handle_rotate_steps)
        + int(st.args.door_push_steps)
    )
    return data


def recovery_local_success(st, contact_both, door_pos, ee_pos, handle_goal, step):
    if bool(contact_both):
        st.recovery_contact_streak += 1
    else:
        st.recovery_contact_streak = 0
    st.recovery_max_contact_streak = max(st.recovery_max_contact_streak, st.recovery_contact_streak)
    initial = np.asarray(st.recovery_initial_door, dtype=np.float32).reshape(-1)
    current = np.asarray(door_pos, dtype=np.float32).reshape(-1)
    door_delta = abs(float(current[0] - initial[0])) if current.size and initial.size else 0.0
    handle_delta = abs(float(current[1] - initial[1])) if current.size > 1 and initial.size > 1 else 0.0
    if door_delta >= math.radians(2.0) or handle_delta >= math.radians(5.0):
        st.recovery_progress_restart = True

    failure_type = st.recovery_failure_type
    contact_ok = st.recovery_max_contact_streak >= int(st.args.recovery_contact_min_frames)
    if failure_type in {"contact_establishment_failure", "contact_maintenance_failure"}:
        local = contact_ok
    elif failure_type in {"insufficient_interaction", "progress_stagnation"}:
        local = bool(st.recovery_progress_restart)
    elif failure_type == "geometric_misalignment":
        local = contact_ok or float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(handle_goal))) <= 0.06
    elif failure_type == "kinematic_infeasibility":
        local = float(getattr(st.ik_state, "last_pos_error", 1.0)) <= 0.05
    elif failure_type == "collision_or_clearance_failure":
        local = not bool(st.base_door_collision_detected)
    else:
        local = contact_ok or bool(st.recovery_progress_restart)
    if local and not st.recovery_local_success:
        st.recovery_local_success = True
        st.recovery_first_local_step = int(step)
    return bool(st.recovery_local_success)


def run_parallel_recovery_batch(gym, sim, env_states, viewer, args, dt, dof_names, manifest, entries):
    if len(env_states) != len(entries):
        raise ValueError(f"Recovery env count {len(env_states)} != candidate count {len(entries)}")
    raw_root = Path(args.recovery_raw_root).expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.recovery_result_json).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    gripper_idx = {name: i for i, name in enumerate(dof_names)}.get("jointGripper")
    num_arm_dofs = len(env_states[0].dof_positions)
    for st, entry in zip(env_states, entries):
        initialize_recovery_candidate(gym, sim, st, entry, dof_names, raw_root, dt)

    max_steps = max(int(st.recovery_total_steps) for st in env_states)
    print(f"Recovery verification batch: candidates={len(env_states)} steps={max_steps}", flush=True)
    for step in range(max_steps):
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        for st in env_states:
            if step >= int(st.recovery_total_steps):
                st.recovery_done = True
                continue
            current_ee_pose_from_refreshed_tensors(st.ik_state)
            local_step = step
            phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal = trajectory_targets(
                local_step,
                st.args,
                st.door,
                gym,
                st.env,
                st.door_actor,
                st.ik_state,
                st.base_start,
                st.base_stop,
                st.base_push,
                st.base_traverse,
                st.yaw_start,
                st.yaw_push,
                st.yaw_traverse,
                st.traj,
            )
            st.last_phase = phase
            st.traj["base_xy"] = np.asarray(base_xy, dtype=np.float32).copy()
            st.traj["yaw"] = float(yaw)
            st.last_handle_goal = np.asarray(handle_goal, dtype=np.float32).copy()
            st.last_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
            st.last_target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float32).copy()
            st.last_gripper = float(gripper)
            set_robot_base_pose(
                gym,
                st.env,
                st.actor_handles,
                base_xy,
                st.args.robot_z,
                yaw,
                getattr(st.args, "robot_pitch", 0.0),
            )
            set_ik_target(st.ik_state, target_pos, target_quat)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        for st in env_states:
            if bool(st.recovery_done):
                continue
            update_arm_ik_targets_for_env(
                gym,
                st.env,
                st.arm_actor,
                st.index,
                st.dof_positions,
                st.ik_state,
                st.args,
                num_arm_dofs,
            )
            if gripper_idx is not None:
                st.dof_positions[gripper_idx] = np.clip(
                    st.last_gripper,
                    st.ik_state.lower[gripper_idx].item(),
                    st.ik_state.upper[gripper_idx].item(),
                )
            gym.set_actor_dof_position_targets(st.env, st.arm_actor, st.dof_positions)
            dc.enforce_locked_door_hinge(gym, st.env, st.door_actor, st.door, st.args)
            door_pos, door_vel = get_actor_dof_state(gym, st.env, st.door_actor)
            efforts = compute_door_efforts(st.door, door_pos, door_vel, st.args)
            if len(efforts):
                gym.apply_actor_dof_efforts(st.env, st.door_actor, efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            if bool(st.recovery_done):
                continue
            door_pos, door_vel = get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos
            dc.monitor_base_door_collision(gym, step, st)
            contact = dc.gripper_handle_contact_snapshot(gym, st)
            contact_any = bool(np.asarray(contact["gripper_handle_contact_any"]).reshape(-1)[0] > 0.5)
            contact_both = bool(np.asarray(contact["gripper_handle_contact_both"]).reshape(-1)[0] > 0.5)
            if contact_any and st.recovery_first_contact_any_step < 0:
                st.recovery_first_contact_any_step = int(step)
            if contact_both and st.recovery_first_contact_both_step < 0:
                st.recovery_first_contact_both_step = int(step)
            closed_angle = dc.closed_hinge_angle(st.door, st.args)
            locked_hinge_drift_deg = (
                abs(math.degrees(float(door_pos[0]) - float(closed_angle))) if len(door_pos) else 0.0
            )
            if not st.door.open_stage:
                st.recovery_max_locked_hinge_drift_deg = max(
                    float(st.recovery_max_locked_hinge_drift_deg),
                    float(locked_hinge_drift_deg),
                )
            if st.door.open_stage and not st.recovery_initial_open_stage and st.recovery_first_unlock_step < 0:
                st.recovery_first_unlock_step = int(step)
                st.recovery_unlock_phase = str(st.last_phase)
                st.recovery_handle_deg_at_unlock = (
                    math.degrees(max(0.0, dc.handle_unlock_progress(st.door, float(door_pos[1]))))
                    if len(door_pos) > 1
                    else 0.0
                )
                if (
                    st.recovery_first_contact_any_step < 0
                    or st.recovery_unlock_phase not in {"rotate_handle", "pull_door"}
                    or st.recovery_max_locked_hinge_drift_deg > 0.5
                ):
                    st.recovery_premature_unlock = True
            ee_pos, _ee_quat = current_ee_pose_from_refreshed_tensors(st.ik_state)
            recovery_local_success(st, contact_both, door_pos, ee_pos, st.last_handle_goal, step)
            door_deg = abs(math.degrees(float(door_pos[0]))) if len(door_pos) else 0.0
            st.recovery_max_door_deg = max(float(st.recovery_max_door_deg), float(door_deg))
            if (
                door_deg >= float(args.pass_open_angle_deg)
                and not bool(st.base_door_collision_detected)
                and not bool(st.recovery_premature_unlock)
                and st.recovery_local_success
            ):
                if not st.recovery_final_success:
                    st.recovery_first_final_step = int(step)
                st.recovery_final_success = True
            dc.record_float_dp_frame(
                gym,
                sim,
                st,
                dof_names,
                gripper_idx,
                dt,
                DP_PHASE_ID.get(st.last_phase, 0),
                door_pos,
                door_vel,
            )
            st.prev_base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32).copy()
            st.prev_yaw = float(st.traj.get("yaw", st.yaw_start))

        if step % 50 == 0 or step + 1 == max_steps:
            local_count = sum(bool(st.recovery_local_success) for st in env_states)
            final_count = sum(bool(st.recovery_final_success) for st in env_states)
            print(f"[recovery {step:04d}] local={local_count}/{len(env_states)} final={final_count}/{len(env_states)}", flush=True)

    results = []
    for st in env_states:
        st.dp_record_success = bool(
            st.recovery_local_success
            and st.recovery_final_success
            and not bool(st.base_door_collision_detected)
            and not bool(st.recovery_premature_unlock)
        )
        raw_path = None
        if st.dp_record_success and st.dp_recorder.frame_count > 0:
            st.dp_recorder.metadata["recovery_local_success"] = True
            st.dp_recorder.metadata["recovery_final_success"] = True
            st.dp_recorder.metadata["recovery_first_local_step"] = int(st.recovery_first_local_step)
            st.dp_recorder.metadata["recovery_first_final_step"] = int(st.recovery_first_final_step)
            raw_path = st.dp_recorder._next_episode_path()
            st.dp_recorder.save_episode()
        st.dp_recorder.finalize()
        door_pos = np.asarray(st.last_door_pos, dtype=np.float32).reshape(-1)
        result = {
            "candidate_id": int(st.recovery_entry["candidate_id"]),
            "branch_step": int(st.recovery_entry["branch_step"]),
            "failure_rollout_npz": str(st.recovery_entry["failure_rollout_npz"]),
            "parameters": st.recovery_entry["parameters"],
            "applied_parameters": st.recovery_applied_parameters,
            "failure_type": st.recovery_failure_type,
            "local_recovery_success": bool(st.recovery_local_success),
            "final_task_success": bool(st.recovery_final_success),
            "recovery_success": bool(st.dp_record_success),
            "max_contact_streak": int(st.recovery_max_contact_streak),
            "progress_restart": bool(st.recovery_progress_restart),
            "first_local_step": int(st.recovery_first_local_step),
            "first_final_step": int(st.recovery_first_final_step),
            "initial_open_stage": bool(st.recovery_initial_open_stage),
            "first_contact_any_step": int(st.recovery_first_contact_any_step),
            "first_contact_both_step": int(st.recovery_first_contact_both_step),
            "first_unlock_step": int(st.recovery_first_unlock_step),
            "unlock_phase": str(st.recovery_unlock_phase),
            "handle_deg_at_unlock": float(st.recovery_handle_deg_at_unlock),
            "max_locked_hinge_drift_deg": float(st.recovery_max_locked_hinge_drift_deg),
            "premature_unlock": bool(st.recovery_premature_unlock),
            "max_door_deg": float(st.recovery_max_door_deg),
            "final_door_deg": abs(math.degrees(float(door_pos[0]))) if door_pos.size else 0.0,
            "unsafe_or_collision": bool(st.base_door_collision_detected),
            "raw_episode": None if raw_path is None else str(raw_path),
        }
        results.append(result)
    payload = {
        "schema_version": 1,
        "verification_rule": (
            "local_recovery_success AND final_task_success AND "
            "NOT unsafe_or_collision AND NOT premature_unlock"
        ),
        "source_batch_manifest": str(args.recovery_batch_manifest),
        "results": results,
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(
        f"Recovery verification result: {sum(r['recovery_success'] for r in results)}/{len(results)} "
        f"verified -> {result_path}",
        flush=True,
    )


def update_pull_raw_record_success(st, door_pos):
    """Apply the pull-only open-plus-traversal success rule to a raw recorder.

    The shared recorder historically marks success from the instantaneous door
    angle used by push-door tasks.  Pulling is different: the door may swing
    back after a valid release, so we require the *maximum* opening angle and
    signed base traversal beyond the door plane.  Keeping this override in the
    pull entrypoint leaves every push-door rule unchanged.
    """
    if st.dp_recorder is None:
        return

    open_deg = dc.door_open_degrees(door_pos, st.args) if len(door_pos) else 0.0
    st.traj["pull_record_max_open_deg"] = max(
        float(st.traj.get("pull_record_max_open_deg", 0.0)),
        float(open_deg),
    )

    direction = np.asarray(st.base_traverse, dtype=np.float32) - np.asarray(
        st.base_stop, dtype=np.float32
    )
    direction_norm = float(np.linalg.norm(direction))
    traversal_m = float("-inf")
    if direction_norm > 1.0e-8:
        direction = direction / direction_norm
        actor_offset = np.asarray(
            getattr(st.door, "actor_position_offset", (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        door_xy = np.array(
            [
                float(st.args.door_x) + float(actor_offset[0]),
                float(st.args.door_y) + float(actor_offset[1]),
            ],
            dtype=np.float32,
        )
        base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
        traversal_m = float(np.dot(base_xy[:2] - door_xy, direction[:2]))
        st.traj["pull_record_max_traversal_m"] = max(
            float(st.traj.get("pull_record_max_traversal_m", float("-inf"))),
            traversal_m,
        )

    release_angle = float(st.traj.get("pull_release_angle_observed_deg", 0.0))
    max_open = float(st.traj.get("pull_record_max_open_deg", 0.0))
    max_traversal = float(
        st.traj.get("pull_record_max_traversal_m", float("-inf"))
    )
    release_ok = release_angle >= float(st.args.pull_release_angle_deg)
    open_ok = max_open >= float(st.args.pass_open_angle_deg)
    traversal_ok = max_traversal >= float(st.args.pull_record_traversal_distance_m)
    st.dp_record_success = bool(
        release_ok
        and open_ok
        and traversal_ok
        and not bool(getattr(st, "base_door_collision_detected", False))
    )
    st.dp_recorder.metadata.update(
        {
            "pull_success_metric": "max_open_and_signed_base_traversal",
            "pull_release_angle_observed_deg": release_angle,
            "pull_max_open_deg": max_open,
            "pull_max_traversal_m": max_traversal,
            "pull_open_threshold_deg": float(st.args.pass_open_angle_deg),
            "pull_traversal_threshold_m": float(
                st.args.pull_record_traversal_distance_m
            ),
        }
    )


def run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names):
    if not env_states:
        raise RuntimeError("No parallel envs were created.")
    num_arm_dofs = len(env_states[0].dof_positions)
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    gripper_idx = dof_dict.get("jointGripper")
    max_steps = args.steps if args.steps > 0 else 2405
    start = time.time()
    step = 0
    print(
        f"Parallel float_ik run: num_envs={len(env_states)} steps={max_steps} "
        f"recorders={sum(st.dp_recorder is not None for st in env_states)}",
        flush=True,
    )
    first = env_states[0]
    print(
        "base_start:",
        first.base_start.tolist(),
        "base_stop:",
        first.base_stop.tolist(),
        "base_push:",
        first.base_push.tolist(),
        "base_traverse:",
        first.base_traverse.tolist(),
    )
    print(
        "pass_through_door:",
        bool(args.pass_through_door),
        "rear_offset:",
        float(args.robot_rear_offset),
        "door_pass_clearance:",
        float(args.door_pass_clearance),
    )
    print("Close viewer to exit.")
    dp_controller, dp_logger, dp_control_state, _dp_control_env_ids, dp_control_env_id_set = dc.setup_float_dp_policy_controller(
        args,
        env_states,
        DoorDPPolicyController,
        DoorDPJsonlLogger,
        "ikpull",
        IKPULL_STATE_VERSION,
    )
    expert_replay = load_expert_action_replay(args) if args.expert_action_replay_raw_episode else None
    if expert_replay is not None:
        if dp_controller is not None:
            raise RuntimeError("Expert action replay and a learned policy cannot control the same run.")
        dp_control_env_ids = (
            list(range(args.num_envs)) if args.dp_control_all_envs else [int(args.dp_control_env_id)]
        )
        dp_control_env_id_set = set(dp_control_env_ids)
        for env_id in dp_control_env_ids:
            env_states[env_id].dp_action_frame = expert_replay.action_frame
        if args.dp_log_path:
            if DoorDPJsonlLogger is None:
                raise RuntimeError("Expert action replay logging requires high-level/dp/door_dp_common.py.")
            dp_logger = DoorDPJsonlLogger(args.dp_log_path)
            print(f"Expert action replay log: {args.dp_log_path}", flush=True)
        print(
            f"Loaded expert action replay from {expert_replay.path}: "
            f"frames={len(expert_replay.actions)} fps={expert_replay.fps:g} "
            f"action_frame={expert_replay.action_frame} source_door={expert_replay.door_asset_name!r}. "
            "Recorded simulator state will not be restored; current randomized env state is used.",
            flush=True,
        )
        if args.dp_control_all_envs:
            print(f"Expert action replay controls all {len(dp_control_env_ids)} envs.", flush=True)
        else:
            print(
                f"Expert action replay controls only env {args.dp_control_env_id}; other envs remain scripted.",
                flush=True,
            )
    external_action_control = bool(dp_controller is not None or expert_replay is not None)
    if external_action_control:
        apply_dp_warmstart_if_requested(gym, sim, args, dp_controller, dp_control_state, dof_names)
        dp_policy_stride = dc.float_dp_policy_sample_stride(args, dt)
        dp_policy_dt = float(dt) * float(dp_policy_stride)
        for st in env_states:
            st.args.dp_policy_dt = dp_policy_dt
        print(
            f"Door DP policy rate: requested={float(getattr(args, 'dp_fps', 25)):.2f}Hz "
            f"effective={dc.float_dp_policy_effective_fps(args, dt):.2f}Hz "
            f"sim_dt={dt:.4f}s stride={dp_policy_stride} policy_dt={dp_policy_dt:.4f}s",
            flush=True,
        )
    if expert_replay is not None:
        dp_policy_action_names = list(expert_replay.action_names)
        external_action_frame = expert_replay.action_frame
    elif dp_controller is not None:
        dp_policy_action_names = list(getattr(dp_controller, "action_names", ACTION_NAMES or []))
        external_action_frame = getattr(dp_controller, "action_frame", "world")
    else:
        dp_policy_action_names = []
        external_action_frame = "world"
    dp_policy_uses_joint_action = bool(float_dp_action_is_a2w_joint9(dp_policy_action_names))
    if dp_policy_uses_joint_action:
        print("Door DP policy action mode: A2W joint9 (vx, yaw, joint1..joint6, jointGripper).", flush=True)

    while step < max_steps:
        if not handle_viewer_pause(gym, sim, viewer):
            break

        gym.refresh_rigid_body_state_tensor(sim)
        for st in env_states:
            current_ee_pose_from_refreshed_tensors(st.ik_state)

        dp_policy_update_due = bool(
            external_action_control and dc.float_dp_policy_update_due(step, args, dt)
        )
        if dp_policy_update_due:
            if viewer is not None and (
                args.draw_ik_target
                or args.draw_camera_axes
                or args.draw_scripted_trajectory
                or bool(getattr(args, "dp3_draw_point_cloud", False))
            ):
                # Clear viewer-only debug lines before camera rendering so policy observations stay clean.
                gym.clear_lines(viewer)
            if dp_controller is not None:
                gym.step_graphics(sim)
                gym.render_all_camera_sensors(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_jacobian_tensors(sim)

        if dp_policy_update_due and dp_controller is not None:
            dp_policy_inputs_by_env, dp_actions_by_env = dc.collect_float_dp_policy_actions(
                gym,
                sim,
                env_states,
                dof_names,
                gripper_idx,
                dt,
                dp_controller,
                dp_control_env_id_set,
                "ikpull",
            )
            if bool(getattr(args, "dp3_draw_point_cloud", False)) and hasattr(
                dp_controller, "get_last_point_cloud_for_env"
            ):
                selected_env = int(getattr(args, "dp3_point_cloud_env_id", 0))
                point_cloud = dp_controller.get_last_point_cloud_for_env(selected_env)
                if 0 <= selected_env < len(env_states):
                    env_states[selected_env].dp3_debug_point_cloud = point_cloud
                    env_states[selected_env].dp3_debug_point_cloud_mode = str(
                        getattr(dp_controller, "config", {}).get("pointcloud_mode", "single_front")
                    )
        elif dp_policy_update_due and expert_replay is not None:
            dp_policy_inputs_by_env, dp_actions_by_env = collect_expert_replay_actions(
                gym,
                env_states,
                expert_replay,
                dp_control_env_id_set,
                step,
                args,
                dt,
            )
        else:
            dp_policy_inputs_by_env, dp_actions_by_env = {}, {}

        for st in env_states:
            if external_action_control and (st.index in dp_actions_by_env or st.last_dp_action is not None):
                phase = "expert_action_replay" if expert_replay is not None else "dp_policy"
                if st.index in dp_actions_by_env:
                    dp_policy_input = dp_policy_inputs_by_env[st.index]
                    base_xy_current = dp_policy_input["base_xy_current"]
                    yaw_current = dp_policy_input["yaw_current"]
                    handle_goal = dp_policy_input["handle_goal"]
                    ee_pos = dp_policy_input["ee_pos"]
                    ee_quat = dp_policy_input["ee_quat"]
                    dp_state = dp_policy_input["dp_state"]
                    dp_action = dp_actions_by_env[st.index]
                    st.last_dp_policy_output = np.asarray(dp_action, dtype=np.float32).copy()
                    st.last_dp_action = np.asarray(dp_action[:10], dtype=np.float32).copy()
                    st.last_dp_state = None if dp_state is None else np.asarray(dp_state, dtype=np.float32).copy()
                    st.last_dp_ee_pos = None if ee_pos is None else np.asarray(ee_pos, dtype=np.float32).copy()
                    st.last_dp_ee_quat = None if ee_quat is None else np.asarray(ee_quat, dtype=np.float32).copy()
                    st.last_dp_handle_goal = None if handle_goal is None else np.asarray(handle_goal, dtype=np.float32).copy()
                    camera_gates = dp_policy_input.get("camera_gates")
                    interaction_state = dp_policy_input.get("interaction_state")
                    st.last_dp_camera_gates = (
                        None if camera_gates is None else np.asarray(camera_gates, dtype=np.float32).copy()
                    )
                    st.last_dp_interaction_state = (
                        None
                        if interaction_state is None
                        else np.asarray(interaction_state, dtype=np.float32).copy()
                    )
                else:
                    base_xy_current = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
                    yaw_current = float(st.traj.get("yaw", st.yaw_start))
                    handle_goal = getattr(st, "last_dp_handle_goal", st.last_handle_goal)
                    ee_pos = getattr(st, "last_dp_ee_pos", None)
                    ee_quat = getattr(st, "last_dp_ee_quat", None)
                    dp_state = getattr(st, "last_dp_state", None)
                    dp_action = np.asarray(
                        st.last_dp_policy_output
                        if st.last_dp_policy_output is not None
                        else st.last_dp_action,
                        dtype=np.float32,
                    ).copy()
                    camera_gates = getattr(st, "last_dp_camera_gates", None)
                    interaction_state = getattr(st, "last_dp_interaction_state", None)
                if dp_policy_uses_joint_action:
                    base_xy, yaw, joint_targets = apply_float_dp_joint_action9(
                        dp_action,
                        base_xy_current,
                        yaw_current,
                        dt,
                    )
                    set_a2w_joint_targets_from_action(
                        st.dof_positions,
                        dof_names,
                        joint_targets,
                        st.ik_state.lower,
                        st.ik_state.upper,
                    )
                    target_pos = np.asarray(ee_pos if ee_pos is not None else st.last_target_pos, dtype=np.float32)
                    target_quat = None if ee_quat is None else np.asarray(ee_quat, dtype=np.float32)
                    gripper = float(joint_targets.get("jointGripper", st.last_gripper))
                else:
                    base_xy, yaw, target_pos, target_quat, gripper = apply_float_dp_action(
                        dp_action,
                        base_xy_current,
                        st.args.robot_z,
                        yaw_current,
                        dt,
                        action_frame=external_action_frame,
                        base_pitch=float(getattr(st.args, "robot_pitch", 0.0)),
                    )
                st.traj["base_xy"] = np.asarray(base_xy, dtype=np.float32).copy()
                st.traj["yaw"] = float(yaw)
                door_pos_for_log, door_vel_for_log = get_actor_dof_state(gym, st.env, st.door_actor)
                if st.index in dp_actions_by_env:
                    dc.update_float_dp_end_signal_monitor(
                        st,
                        dp_action,
                        step,
                        door_pos_for_log,
                        phase,
                    )
            else:
                phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal = trajectory_targets(
                    step,
                    st.args,
                    st.door,
                    gym,
                    st.env,
                    st.door_actor,
                    st.ik_state,
                    st.base_start,
                    st.base_stop,
                    st.base_push,
                    st.base_traverse,
                    st.yaw_start,
                    st.yaw_push,
                    st.yaw_traverse,
                    st.traj,
                )
                if (
                    str(getattr(st.args, "arm_ik_solver", "gym_jacobian")) == "tracik"
                    and phase == "initial_hold"
                ):
                    target_pos = np.asarray(st.traj["pregrasp"], dtype=np.float32).copy()
                    target_quat = (
                        None
                        if st.args.ik_position_only
                        else np.asarray(st.traj["goal_quat"], dtype=np.float32).copy()
                    )
                dp_action = None
                dp_state = None
                ee_pos = None
                ee_quat = None
                door_pos_for_log = None
                door_vel_for_log = None
            if dp_action is not None:
                gripper = apply_dp_gripper_latch(st, gripper, door_pos_for_log)
                if bool(dp_policy_uses_joint_action) and gripper_idx is not None:
                    st.dof_positions[gripper_idx] = np.clip(
                        float(gripper),
                        st.ik_state.lower[gripper_idx].item(),
                        st.ik_state.upper[gripper_idx].item(),
                    )
            st.last_phase = phase
            st.dp_joint_action_active = bool(dp_action is not None and dp_policy_uses_joint_action)
            st.last_handle_goal = handle_goal
            st.last_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
            st.last_target_quat = None if target_quat is None else np.asarray(target_quat, dtype=np.float32).copy()
            st.last_gripper = float(gripper)
            if dp_action is not None:
                dp_record = make_float_dp_policy_log_record(
                    step,
                    st,
                    dp_action,
                    dp_state,
                    ee_pos,
                    ee_quat,
                    door_pos_for_log,
                    door_vel_for_log,
                    phase,
                    action_names=dp_policy_action_names or ACTION_NAMES,
                    camera_gates=camera_gates,
                    interaction_state=interaction_state,
                    gym=gym,
                    dof_names=dof_names,
                )
                traversal_direction = np.asarray(st.base_traverse, dtype=np.float32) - np.asarray(
                    st.base_stop, dtype=np.float32
                )
                traversal_direction_norm = float(np.linalg.norm(traversal_direction))
                if traversal_direction_norm > 1.0e-6:
                    traversal_direction = traversal_direction / traversal_direction_norm
                actor_offset = np.asarray(
                    getattr(st.door, "actor_position_offset", (0.0, 0.0, 0.0)),
                    dtype=np.float32,
                )
                dp_record["pull_task_geometry"] = {
                    "door_plane_xy": [
                        float(st.args.door_x) + float(actor_offset[0]),
                        float(st.args.door_y) + float(actor_offset[1]),
                    ],
                    "traversal_direction_xy": np.round(traversal_direction, 6).tolist(),
                    "base_start_xy": np.round(np.asarray(st.base_start, dtype=np.float32), 6).tolist(),
                    "base_stop_xy": np.round(np.asarray(st.base_stop, dtype=np.float32), 6).tolist(),
                    "base_traverse_target_xy": np.round(
                        np.asarray(st.base_traverse, dtype=np.float32), 6
                    ).tolist(),
                }
                if dp_logger is not None:
                    dp_logger.write(dp_record)
                if args.dp_print and step % max(1, int(args.dp_log_interval)) == 0:
                    print_float_dp_policy_log_record(dp_record)
            set_robot_base_pose(
                gym,
                st.env,
                st.actor_handles,
                base_xy,
                st.args.robot_z,
                yaw,
                getattr(st.args, "robot_pitch", 0.0),
            )
            release_joint_home_alpha = (
                float(st.traj.get("release_joint_home_alpha", 0.0))
                if phase == "release_handle"
                else 0.0
            )
            if phase == "release_handle" and release_joint_home_alpha > 0.0:
                if "release_joint_home_start_dofs" not in st.traj:
                    st.traj["release_joint_home_start_dofs"] = np.asarray(
                        st.dof_positions,
                        dtype=np.float32,
                    ).copy()
                st.dof_positions[:] = lerp(
                    st.traj["release_joint_home_start_dofs"],
                    st.home_positions,
                    release_joint_home_alpha,
                )
                if gripper_idx is not None:
                    st.dof_positions[gripper_idx] = st.home_positions[gripper_idx]
                st.ik_state.last_pos_error = 0.0
                reset_tracik_controller_to_targets(gym, st, step, dt)
            elif phase == "return_home":
                if "return_home_start_dofs" not in st.traj:
                    st.traj["return_home_start_dofs"] = np.asarray(st.dof_positions, dtype=np.float32).copy()
                alpha = float(st.traj.get("return_home_alpha", 0.0))
                st.dof_positions[:] = lerp(st.traj["return_home_start_dofs"], st.home_positions, alpha)
                st.ik_state.last_pos_error = 0.0
                reset_tracik_controller_to_targets(gym, st, step, dt)
            elif phase in ("walk", "pass_through", "hold_home"):
                st.dof_positions[:] = st.home_positions
                st.ik_state.last_pos_error = 0.0
                reset_tracik_controller_to_targets(gym, st, step, dt)
            elif getattr(st, "dp_joint_action_active", False):
                st.ik_state.last_pos_error = 0.0
                reset_tracik_controller_to_targets(gym, st, step, dt)
            else:
                set_ik_target(st.ik_state, target_pos, target_quat)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            release_joint_home_active = bool(
                st.last_phase == "release_handle"
                and float(st.traj.get("release_joint_home_alpha", 0.0)) > 0.0
            )
            if st.last_phase not in (
                "walk",
                "pass_through",
                "return_home",
                "hold_home",
            ) and not release_joint_home_active:
                if getattr(st, "dp_joint_action_active", False):
                    st.ik_state.last_pos_error = 0.0
                else:
                    if str(getattr(st.args, "arm_ik_solver", "gym_jacobian")) == "tracik":
                        update_tracik_arm_targets_for_env(gym, st, step, dt)
                    else:
                        update_arm_ik_targets_for_env(
                            gym,
                            st.env,
                            st.arm_actor,
                            st.index,
                            st.dof_positions,
                            st.ik_state,
                            st.args,
                            num_arm_dofs,
                        )
                    if gripper_idx is not None:
                        st.dof_positions[gripper_idx] = np.clip(
                            st.last_gripper,
                            st.ik_state.lower[gripper_idx].item(),
                            st.ik_state.upper[gripper_idx].item(),
                        )
            gym.set_actor_dof_position_targets(st.env, st.arm_actor, st.dof_positions)

            dc.enforce_locked_door_hinge(gym, st.env, st.door_actor, st.door, st.args)
            door_pos, door_vel = get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos
            door_efforts = compute_door_efforts(st.door, door_pos, door_vel, st.args)
            if step == 0 and bool(getattr(st.args, "door_twin_asset_probe", False)):
                probe_props = gym.get_actor_dof_properties(st.env, st.door_actor)
                print(
                    "door_twin_asset_probe_effort "
                    f"env={st.index} pos={np.asarray(door_pos).round(6).tolist()} "
                    f"effort={np.asarray(door_efforts).round(6).tolist()} "
                    f"limits={np.asarray(probe_props['effort']).round(6).tolist()} "
                    f"drive={np.asarray(probe_props['driveMode']).tolist()} "
                    f"force={float(getattr(st.args, 'door_auto_open_force', 0.0)):.3f} "
                    f"motion_sign={float(getattr(st.args, 'door_motion_sign', 0.0)):.1f} "
                    f"auto_sign={float(getattr(st.args, 'door_auto_open_sign', 0.0)):.1f}",
                    flush=True,
                )
            if len(door_efforts) > 0:
                gym.apply_actor_dof_efforts(st.env, st.door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        record_camera_due = bool(
            args.record_dp_dataset
            and any(st.camera_handles and dc.float_dp_record_frame_due(st, dt) for st in env_states)
        )
        door_twin_dump_due = bool(any(door_twin_keyframe_dump_due(st) for st in env_states))
        stage_screenshot_due = bool(any(update_stage_screenshot_due(st) for st in env_states))
        regular_camera_render = bool(
            any(st.camera_handles for st in env_states)
            and (args.show_camera_images or record_camera_due or door_twin_dump_due)
        )
        need_camera_render = bool(regular_camera_render or stage_screenshot_due)
        if viewer is not None and need_camera_render and (
            args.draw_ik_target or args.draw_camera_axes or args.draw_scripted_trajectory
        ):
            # Clear viewer-only debug lines before camera rendering so depth/RGB tensors stay clean.
            gym.clear_lines(viewer)
        if viewer is not None or need_camera_render:
            gym.step_graphics(sim)
        if need_camera_render:
            gym.render_all_camera_sensors(sim)
        if args.show_camera_images and env_states[0].camera_handles and step % max(1, args.camera_display_interval) == 0:
            show_camera_handle_images(gym, sim, env_states[0].env, env_states[0].camera_handles, env_states[0].args)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            door_pos_record, door_vel_record = get_actor_dof_state(gym, st.env, st.door_actor)
            st.last_door_pos = door_pos_record
            dc.monitor_base_door_collision(gym, step, st)
            if door_twin_dump_due:
                maybe_dump_door_twin_keyframe_images(gym, sim, st, step)
            if stage_screenshot_due:
                save_stage_screenshot(gym, sim, st, step)
            if st.dp_recorder is not None:
                frame_recorded = dc.record_float_dp_frame(
                    gym,
                    sim,
                    st,
                    dof_names,
                    gripper_idx,
                    dt,
                    DP_PHASE_ID.get(st.last_phase, 0),
                    door_pos_record,
                    door_vel_record,
                )
                update_pull_raw_record_success(st, door_pos_record)
                tracker = getattr(st, "door_twin_tracker", None)
                if tracker is not None and frame_recorded:
                    tracker.mark_camera_available(True)
            update_door_twin_tracker(st, step, door_pos_record)
            st.prev_base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32).copy()
            st.prev_yaw = float(st.traj.get("yaw", st.yaw_start))

        if viewer is not None:
            if (
                args.draw_ik_target
                or args.draw_camera_axes
                or args.draw_scripted_trajectory
                or bool(getattr(args, "dp3_draw_point_cloud", False))
            ):
                gym.clear_lines(viewer)
            for st in env_states[: min(4, len(env_states))]:
                if args.draw_camera_axes:
                    draw_low_level_camera_axes(gym, viewer, st.env, st.arm_actor, st.actor_handles, st.args)
                if args.draw_scripted_trajectory:
                    draw_scripted_trajectory(gym, viewer, st.env, st.traj, st.args, st.last_target_pos)
                if args.draw_ik_target:
                    if st.last_phase in ("return_home", "hold_home") and st.last_target_pos is not None:
                        saved_target_pos_np = st.ik_state.target_pos_np
                        saved_target_quat_np = st.ik_state.target_quat_np
                        st.ik_state.target_pos_np = np.asarray(st.last_target_pos, dtype=np.float32).copy()
                        st.ik_state.target_quat_np = (
                            None
                            if st.last_target_quat is None
                            else base_ik.normalize_quat(st.last_target_quat).astype(np.float32)
                        )
                        try:
                            base_ik.draw_ik_target(gym, viewer, st.env, st.ik_state)
                        finally:
                            st.ik_state.target_pos_np = saved_target_pos_np
                            st.ik_state.target_quat_np = saved_target_quat_np
                    else:
                        base_ik.draw_ik_target(gym, viewer, st.env, st.ik_state)
                    if st.last_handle_goal is not None:
                        target_pose = base_ik.transform_from_arrays(st.last_handle_goal)
                        goal_sphere = gymutil.WireframeSphereGeometry(
                            radius=0.035,
                            num_lats=8,
                            num_lons=8,
                            color=(0.0, 1.0, 0.2),
                            color2=(0.0, 0.7, 0.2),
                        )
                        gymutil.draw_lines(goal_sphere, gym, viewer, st.env, target_pose)
            if bool(getattr(args, "dp3_draw_point_cloud", False)):
                selected_env = int(getattr(args, "dp3_point_cloud_env_id", 0))
                if 0 <= selected_env < len(env_states):
                    point_cloud = getattr(env_states[selected_env], "dp3_debug_point_cloud", None)
                    if point_cloud is not None:
                        draw_dp3_point_cloud(
                            gym,
                            viewer,
                            env_states[selected_env],
                            point_cloud,
                            getattr(
                                env_states[selected_env],
                                "dp3_debug_point_cloud_mode",
                                "single_front",
                            ),
                        )
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            shown = env_states[: min(len(env_states), 4)]
            door_deg = [
                round(math.degrees(float(st.last_door_pos[0])), 1) if st.last_door_pos is not None and len(st.last_door_pos) else 0.0
                for st in shown
            ]
            handle_deg = [
                round(math.degrees(float(st.last_door_pos[1])), 1)
                if st.last_door_pos is not None and len(st.last_door_pos) > 1
                else None
                for st in shown
            ]
            handle_unlocked = [bool(st.door.open_stage) for st in shown]
            phases = [st.last_phase for st in shown]
            successes = sum(st.dp_record_success for st in env_states if st.dp_recorder is not None)
            pull_note = ""
            if shown and shown[0].last_phase in {"pull_door", "open_hold", "release_handle"}:
                pull_state = shown[0]
                pull_note = (
                    f" pull_ee={np.round(pull_state.ik_state.current_pos_np, 3).tolist()}"
                    f" pull_target={np.round(pull_state.last_target_pos, 3).tolist()}"
                    f" handle_goal={np.round(pull_state.last_handle_goal, 3).tolist()}"
                    f" base_xy={np.round(pull_state.traj.get('base_xy', pull_state.base_stop), 3).tolist()}"
                    f" ik_pos_err={float(pull_state.ik_state.last_pos_error):.4f}"
                    f" gripper_cmd={float(pull_state.last_gripper):.3f}"
                )
            tracik_note = ""
            if str(getattr(args, "arm_ik_solver", "gym_jacobian")) == "tracik":
                controller = first.tracik_controller
                tracik_note = (
                    f" tracik_solve={controller.last_solve_ms:.3f}ms"
                    f" failures={controller.failure_count}"
                    f" projections={controller.projection_count}"
                    f" nearest_position={controller.nearest_position_count}"
                    f" branch_guards={controller.branch_guard_count}"
                    f" branch_rejects={controller.branch_reject_count}"
                    f" branch_recoveries={controller.branch_recovery_count}"
                    f" branch_position_fallbacks={controller.branch_position_fallback_count}"
                    f" branch_position_active={controller.last_branch_position_fallback}"
                    f" branch_candidates={controller.last_branch_candidates}"
                    f" local_servo={controller.local_servo_count}"
                    f" local_rejects={controller.local_servo_reject_count}"
                    f" local_projections={controller.local_servo_projection_count}"
                    f" local_alpha={controller.last_local_servo_alpha:.3f}"
                    f" projection_alpha={controller.last_projection_alpha:.3f}"
                    f" projection_ms={controller.last_projection_ms:.3f}"
                    f" initial_plan={controller.initial_plan_active}"
                    f" segment={controller.last_segment_duration * 1000.0:.1f}ms"
                    f" q1_measured={controller.last_measured_q[0]:.3f}"
                    f" q1_command={controller.last_solution[0]:.3f}"
                    f" ik_pos_err={first.ik_state.last_pos_error:.4f}"
                    f" tracik_error={controller.last_error or 'none'}"
                )
                if controller.last_error and controller.last_target_pose is not None:
                    tracik_note += (
                        " tracik_target="
                        f"{np.round(controller.last_target_pose, 5).tolist()}"
                    )
                    if controller.last_candidate is not None:
                        tracik_note += (
                            " tracik_seed_q="
                            f"{np.round(controller.last_solve_seed, 3).tolist()}"
                            " tracik_last_q="
                            f"{np.round(controller.last_solution, 3).tolist()}"
                            " tracik_candidate_q="
                            f"{np.round(controller.last_candidate, 3).tolist()}"
                        )
                    if controller.last_local_servo_reason:
                        tracik_note += f" local_servo_reason={controller.last_local_servo_reason}"
            print(
                f"[{step:04d}] phases={phases} door_deg={door_deg} "
                f"handle_deg={handle_deg} handle_unlocked={handle_unlocked} "
                f"record_success={successes}/{sum(st.dp_recorder is not None for st in env_states)}"
                f"{pull_note}"
                f"{tracik_note}",
                flush=True,
            )
        step += 1

    elapsed = time.time() - start
    release_angles = [
        float(st.traj["pull_release_angle_observed_deg"])
        for st in env_states
        if "pull_release_angle_observed_deg" in st.traj
    ]
    completed_phases = {"return_home", "hold_home"}
    completed_count = sum(st.last_phase in completed_phases for st in env_states)
    max_open_angles = [
        float(st.traj.get("pull_max_open_deg", 0.0)) for st in env_states
    ]
    if release_angles:
        release_stats = (
            f"min/mean/max={min(release_angles):.2f}/"
            f"{float(np.mean(release_angles)):.2f}/{max(release_angles):.2f} deg"
        )
    else:
        release_stats = "min/mean/max=n/a"
    print(
        "[A2WPullValidation] "
        f"released={len(release_angles)}/{len(env_states)} "
        f"release_angle_{release_stats} "
        f"max_open_ge_60={sum(angle >= 60.0 for angle in max_open_angles)}/"
        f"{len(env_states)} completed_traverse={completed_count}/{len(env_states)}",
        flush=True,
    )
    print(f"Done after {step} steps ({elapsed:.2f}s).")
    raw_episode_snapshots = snapshot_raw_dp_episodes(env_states)
    dc.finish_float_dp_recorders(env_states, args)
    attach_new_expert_trajectory_artifacts(env_states, raw_episode_snapshots)
    if str(getattr(args, "door_twin_log_dir", "") or "").strip():
        trackers = [st.door_twin_tracker for st in env_states if getattr(st, "door_twin_tracker", None) is not None]
        if trackers:
            summary = write_rollout_reports(
                trackers,
                args.door_twin_log_dir,
                run_metadata={
                    "mode": "a2w_float_ik",
                    "steps": int(step),
                    "elapsed_s": float(elapsed),
                    "seed": int(getattr(args, "seed", 0)),
                    "skill_program_json": str(getattr(args, "skill_program_json", "") or ""),
                    "record_dp_dataset": bool(getattr(args, "record_dp_dataset", False)),
                },
            )
            print(
                f"DoorTwin rollout summary: {summary.get('success_count', 0)}/{summary.get('num_envs', 0)} "
                f"success -> {summary.get('summary_path')}",
                flush=True,
            )
    if dp_logger is not None:
        dp_logger.close()


def main():
    args = parse_args()
    recovery_manifest = None
    recovery_entries = None
    if str(getattr(args, "recovery_batch_manifest", "") or "").strip():
        if not str(getattr(args, "recovery_result_json", "") or "").strip():
            raise ValueError("--recovery_result_json is required with --recovery_batch_manifest.")
        if not str(getattr(args, "recovery_raw_root", "") or "").strip():
            raise ValueError("--recovery_raw_root is required with --recovery_batch_manifest.")
        _manifest_path, recovery_manifest, recovery_entries = load_recovery_batch_manifest(
            args.recovery_batch_manifest
        )
        args.num_envs = len(recovery_entries)
        args.record_dp_dataset = False
        args.dp_policy_checkpoint = ""
        args.show_camera_images = False
        print(
            f"Recovery verification mode: manifest={args.recovery_batch_manifest} "
            f"candidates={args.num_envs}",
            flush=True,
        )
    auto_enable_pointcloud_checkpoint_cameras(args)
    create_tracik_solver_if_requested(args)
    seed = resolve_seed(args)
    print(f"ikpush seed={seed}", flush=True)
    gym = gymapi.acquire_gym()
    sim, dt = base_ik.create_sim(gym, args)
    args.sim_dt = float(dt)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="a2wz1_float_ik_door_assets_") as temp_dir:
        base_asset, arm_asset = load_a2wz1_float_robot_assets(gym, sim, args, Path(temp_dir))
        door_templates = load_door_assets(gym, sim, args)
        if bool(args.pull_flip_door_actor):
            for door_template in door_templates:
                original_yaw = float(door_template.actor_yaw)
                original_robot_y_offset = float(door_template.robot_y_offset)
                door_template.actor_yaw = math.atan2(
                    math.sin(original_yaw + math.pi),
                    math.cos(original_yaw + math.pi),
                )
                # A pi rotation swaps the handle to the opposite lateral side.
                # Mirror the configured base alignment as well so the unchanged
                # pre-push trajectory still approaches the handle head-on.
                door_template.robot_y_offset = -original_robot_y_offset
                print(
                    f"[A2WPull] pull-side door load: "
                    f"{door_template.spec.get('name', door_template.asset_index)} "
                    f"actor_yaw {original_yaw:.6f} -> {door_template.actor_yaw:.6f}, "
                    f"robot_y_offset {original_robot_y_offset:.4f} -> "
                    f"{door_template.robot_y_offset:.4f}",
                    flush=True,
                )
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "A2W base visual actor", verbose=args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "Z1 arm articulated actor", verbose=args.print_collision_summary)
        base_dof_data = configure_a2w_base_visual_dofs(gym, base_asset, args)
        dof_data = base_ik.configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        if "jointGripper" in dof_names:
            gripper_index = dof_names.index("jointGripper")
            dof_props["stiffness"][gripper_index] = float(args.pull_gripper_stiffness)
            dof_props["damping"][gripper_index] = float(args.pull_gripper_damping)
            dof_props["effort"][gripper_index] = max(
                float(dof_props["effort"][gripper_index]),
                float(args.pull_gripper_effort_limit),
            )
            dof_states["pos"][gripper_index] = args.gripper_open
            dof_positions[gripper_index] = args.gripper_open
            print(
                "[A2WPull] jointGripper drive "
                f"stiffness={float(dof_props['stiffness'][gripper_index]):.1f} "
                f"damping={float(dof_props['damping'][gripper_index]):.1f} "
                f"effort_limit={float(dof_props['effort'][gripper_index]):.1f}",
                flush=True,
            )
        env_states, _vision_mode = create_parallel_env_states(
            gym,
            sim,
            base_asset,
            arm_asset,
            door_templates,
            dof_props,
            dof_states,
            dof_positions,
            lower,
            upper,
            defaults,
            dof_names,
            args,
            base_dof_data=base_dof_data,
        )
        viewer = setup_viewer(gym, sim, args)
        setup_viewer_pause_shortcut(gym, viewer)
        try:
            if recovery_entries is not None:
                run_parallel_recovery_batch(
                    gym,
                    sim,
                    env_states,
                    viewer,
                    args,
                    dt,
                    dof_names,
                    recovery_manifest,
                    recovery_entries,
                )
            else:
                run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names)
        finally:
            if args.show_camera_images and cv2 is not None:
                cv2.destroyAllWindows()
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
