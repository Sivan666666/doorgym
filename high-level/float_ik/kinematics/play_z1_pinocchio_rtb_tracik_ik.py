#!/usr/bin/env python3
"""Play four Z1 arms with the same 6D target pose.

Left to right:
    Isaac Gym Jacobian IK, Pinocchio IK, Robotics Toolbox IK, TRAC-IK.

Press S in the viewer to sample a new reachable target pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FLOAT_IK_DIR = SCRIPT_DIR.parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
for path in (FLOAT_IK_DIR, IK_SOLVERS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from z1_pinocchio_ik import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_JOINT_NAMES,
    DEFAULT_Z1_URDF,
    Z1PinocchioIK,
    normalize_quat_xyzw,
)
from z1_rtb_kinematics import Z1RtbKinematics
from z1_tracik_kinematics import Z1TracIKKinematics

from play_z1_gym_vs_pinocchio_ik import (
    A2W_DEFAULT_LEG_POS,
    build_a2w_base_visual_asset_root,
    make_transform,
    pose_error,
    pose_to_np,
    torch_orientation_error,
    transform_pose,
)


HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "z1"
DEFAULT_ASSET_FILE = "urdf/z1_arm.urdf"
DEFAULT_A2W_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "a2wz1"
DEFAULT_A2W_ASSET_FILE = "urdf/a2wz1.urdf"


def pose_xyzw_to_wxyz(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    q = normalize_quat_xyzw(pose[3:7])
    return np.concatenate([pose[:3], np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--asset_file", type=str, default=DEFAULT_ASSET_FILE)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until viewer close.")
    parser.add_argument("--target_delta", type=float, default=0.35)
    parser.add_argument(
        "--global_targets",
        action="store_true",
        help="Sample each new command from the full reachable joint workspace instead of a local delta.",
    )
    parser.add_argument(
        "--global_seed_mode",
        choices=("home", "middle", "zero", "target", "current"),
        default="home",
        help="IK seed used for offline solvers when --global_targets is enabled.",
    )
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument("--disable_arm_visual_flip", action="store_true")
    parser.add_argument("--stiffness", type=float, default=850.0)
    parser.add_argument("--damping", type=float, default=85.0)
    parser.add_argument("--gym_ik_damping", type=float, default=0.05)
    parser.add_argument("--gym_max_step", type=float, default=0.045)
    parser.add_argument("--ik_pos_gain", type=float, default=1.0)
    parser.add_argument("--ik_rot_gain", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--pin_max_iter", type=int, default=1000)
    parser.add_argument("--pin_restarts", type=int, default=1)
    parser.add_argument("--pin_allow_restarts", action="store_true")
    parser.add_argument("--pin_dt", type=float, default=0.05)
    parser.add_argument("--pin_damping", type=float, default=1.0e-6)
    parser.add_argument("--pin_threshold", type=float, default=1.0e-4)
    parser.add_argument("--rtb_ilimit", type=int, default=80)
    parser.add_argument("--rtb_threshold", type=float, default=1.0e-4)
    parser.add_argument("--tracik_timeout", type=float, default=0.005)
    parser.add_argument("--tracik_epsilon", type=float, default=1.0e-5)
    parser.add_argument("--tracik_solver_type", choices=("Speed", "Distance", "Manip1", "Manip2"), default="Speed")
    parser.add_argument("--tracik_restarts", type=int, default=1)
    parser.add_argument("--tracik_allow_restarts", action="store_true")
    parser.add_argument("--pos_tol", type=float, default=0.012)
    parser.add_argument("--rot_tol", type=float, default=0.055)
    parser.add_argument("--print_interval", type=int, default=60)
    parser.add_argument("--target_sphere_radius", type=float, default=0.035)
    parser.add_argument("--axis_scale", type=float, default=0.13)
    parser.add_argument("--root_z", type=float, default=0.50)
    parser.add_argument("--show_a2w_base", dest="show_a2w_base", action="store_true", default=True)
    parser.add_argument("--no_show_a2w_base", dest="show_a2w_base", action="store_false")
    parser.add_argument("--a2w_asset_root", type=str, default=str(DEFAULT_A2W_ASSET_ROOT))
    parser.add_argument("--a2w_asset_file", type=str, default=DEFAULT_A2W_ASSET_FILE)
    return parser.parse_args()


@dataclass
class VisualSolver:
    key: str
    label: str
    actor_name: str
    color: object
    root_pose: np.ndarray
    solve_fn: Optional[Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]]
    actor: object = None
    base_actor: object = None
    ee_body_sim_index: int = -1
    goal_q6: Optional[np.ndarray] = None
    last_error: tuple = (math.inf, math.inf)
    last_solve_ms: float = math.nan
    last_success: bool = False


class TripleZ1IKPlay:
    def __init__(self, args, pin_ik, rtb_ik, tracik_ik, rng, gymapi, gymtorch, gymutil, torch):
        self.args = args
        self.pin_ik = pin_ik
        self.rtb_ik = rtb_ik
        self.tracik_ik = tracik_ik
        self.rng = rng
        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.gymutil = gymutil
        self.torch = torch
        self.gym = gymapi.acquire_gym()
        self.command_id = 0
        self.command_step = 0
        self.reached_reported = False
        self.target_q6 = np.asarray(args.home_q, dtype=np.float64).copy()
        self.target_local_pose = self.pin_ik.fk(self.target_q6)
        self.temp_dirs = []

        root_z = float(args.root_z)
        self.solvers = [
            VisualSolver(
                key="gym",
                label="GymJacobian",
                actor_name="z1_gym_jacobian",
                color=None,
                root_pose=np.asarray([-1.8, 0.0, root_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                solve_fn=None,
            ),
            VisualSolver(
                key="pinocchio",
                label="Pinocchio",
                actor_name="z1_pinocchio",
                color=None,
                root_pose=np.asarray([-0.6, 0.0, root_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                solve_fn=self._solve_pinocchio,
            ),
            VisualSolver(
                key="rtb",
                label="RTB",
                actor_name="z1_rtb",
                color=None,
                root_pose=np.asarray([0.6, 0.0, root_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                solve_fn=self._solve_rtb,
            ),
            VisualSolver(
                key="tracik",
                label="TRAC-IK",
                actor_name="z1_tracik",
                color=None,
                root_pose=np.asarray([1.8, 0.0, root_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                solve_fn=self._solve_tracik,
            ),
        ]

        self._create_sim()
        self._load_asset_and_actors()
        self._prepare_tensors()
        self._reset_actors()
        self._create_viewer()
        self._build_draw_geometries()

    def _vec3(self, rgb):
        return self.gymapi.Vec3(float(rgb[0]), float(rgb[1]), float(rgb[2]))

    def _solve_pinocchio(self, target_pose_xyzw, seed_q):
        return self.pin_ik.ik(
            target_pose_xyzw,
            seed_joints=seed_q,
            position_only=False,
            max_iter=int(self.args.pin_max_iter),
            max_restarts=int(self.args.pin_restarts),
            dt=float(self.args.pin_dt),
            damping=float(self.args.pin_damping),
            threshold=float(self.args.pin_threshold),
            rotation_weight=float(self.args.rot_weight),
            rng=self.rng,
            strict_seed=not bool(self.args.pin_allow_restarts),
        )

    def _solve_rtb(self, target_pose_xyzw, seed_q):
        mask = [1.0, 1.0, 1.0, float(self.args.rot_weight), float(self.args.rot_weight), float(self.args.rot_weight)]
        return self.rtb_ik.solve_ik(
            pose_xyzw_to_wxyz(target_pose_xyzw),
            seed_joints=seed_q,
            pose_constraint=mask,
            threshold=float(self.args.rtb_threshold),
            ilimit=int(self.args.rtb_ilimit),
        )

    def _solve_tracik(self, target_pose_xyzw, seed_q):
        return self.tracik_ik.solve_ik(
            pose_xyzw_to_wxyz(target_pose_xyzw),
            seed_joints=seed_q,
            max_restarts=max(1, int(self.args.tracik_restarts)),
            strict_seed=not bool(self.args.tracik_allow_restarts),
            position_only=False,
            rng=self.rng,
        )

    def _create_sim(self):
        sim_params = self.gymapi.SimParams()
        sim_params.up_axis = self.gymapi.UP_AXIS_Z
        sim_params.gravity = self.gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.dt = 1.0 / 60.0
        self.sim = self.gym.create_sim(0, 0, self.gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")
        plane_params = self.gymapi.PlaneParams()
        plane_params.normal = self.gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_asset_and_actors(self):
        asset_options = self.gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = not bool(self.args.disable_arm_visual_flip)
        asset_options.thickness = 0.001
        asset_options.armature = 0.01

        asset_root = str(Path(self.args.asset_root).expanduser().resolve())
        asset_file = str(self.args.asset_file)
        print(
            f"Loading Z1 arm asset: root={asset_root}, file={asset_file}, "
            f"flip_visual_attachments={asset_options.flip_visual_attachments}"
        )
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if self.asset is None:
            raise RuntimeError(f"Failed to load asset root={asset_root} file={asset_file}")

        self.base_asset = self._load_a2w_base_asset() if bool(self.args.show_a2w_base) else None
        self.env = self.gym.create_env(
            self.sim,
            self.gymapi.Vec3(-3.0, -1.2, -0.2),
            self.gymapi.Vec3(3.0, 1.2, 1.5),
            1,
        )

        actor_colors = {
            "gym": (0.75, 0.35, 1.0),
            "pinocchio": (0.25, 0.48, 1.0),
            "rtb": (0.10, 0.85, 0.45),
            "tracik": (1.0, 0.48, 0.12),
        }
        base_colors = {
            "gym": (0.40, 0.34, 0.46),
            "pinocchio": (0.34, 0.37, 0.42),
            "rtb": (0.34, 0.42, 0.36),
            "tracik": (0.42, 0.36, 0.30),
        }

        for solver in self.solvers:
            root_tf = make_transform(self.gymapi, solver.root_pose[:3], solver.root_pose[3:7])
            solver.root_pose = pose_to_np(root_tf)
            if self.base_asset is not None:
                solver.base_actor = self.gym.create_actor(
                    self.env,
                    self.base_asset,
                    root_tf,
                    f"a2w_base_{solver.key}",
                    0,
                    0,
                )
                self._paint_actor(solver.base_actor, self._vec3(base_colors[solver.key]))
                self._configure_a2w_base_actor(solver.base_actor)
            solver.actor = self.gym.create_actor(self.env, self.asset, root_tf, solver.actor_name, 0, 0)
            solver.color = self._vec3(actor_colors[solver.key])
            self._paint_actor(solver.actor, solver.color)

        self.dof_names = list(self.gym.get_asset_dof_names(self.asset))
        self.num_dofs = len(self.dof_names)
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        self.dof_props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        self.dof_props["stiffness"].fill(float(self.args.stiffness))
        self.dof_props["damping"].fill(float(self.args.damping))
        for solver in self.solvers:
            self.gym.set_actor_dof_properties(self.env, solver.actor, self.dof_props)

        self.lower = np.asarray(self.dof_props["lower"], dtype=np.float64)
        self.upper = np.asarray(self.dof_props["upper"], dtype=np.float64)
        self.control_indices = np.asarray(
            [self.dof_names.index(name) for name in DEFAULT_Z1_JOINT_NAMES if name in self.dof_names],
            dtype=np.int64,
        )
        if self.control_indices.shape[0] != len(DEFAULT_Z1_JOINT_NAMES):
            raise RuntimeError(f"Missing Z1 control DOFs. Loaded DOFs: {self.dof_names}")

        self.body_names = list(self.gym.get_asset_rigid_body_names(self.asset))
        if self.args.ee_link not in self.body_names:
            raise RuntimeError(f"EE link {self.args.ee_link!r} not in asset bodies: {self.body_names}")
        for solver in self.solvers:
            solver.ee_body_sim_index = self.gym.find_actor_rigid_body_index(
                self.env,
                solver.actor,
                self.args.ee_link,
                self.gymapi.DOMAIN_SIM,
            )

    def _load_a2w_base_asset(self):
        temp_dir = tempfile.TemporaryDirectory(prefix="a2w_base_visual_")
        self.temp_dirs.append(temp_dir)
        split_root, base_file = build_a2w_base_visual_asset_root(
            self.args.a2w_asset_root,
            self.args.a2w_asset_file,
            temp_dir.name,
        )
        asset_options = self.gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = False
        asset_options.thickness = 0.001
        asset_options.armature = 0.01
        print(f"Loading A2W base visual: root={split_root}, file={base_file}")
        asset = self.gym.load_asset(self.sim, str(split_root), base_file, asset_options)
        if asset is None:
            raise RuntimeError(f"Failed to load A2W base visual root={split_root} file={base_file}")
        return asset

    def _configure_a2w_base_actor(self, actor):
        if actor is None or self.base_asset is None:
            return
        num_dofs = self.gym.get_asset_dof_count(self.base_asset)
        if num_dofs <= 0:
            return
        dof_names = list(self.gym.get_asset_dof_names(self.base_asset))
        dof_props = self.gym.get_asset_dof_properties(self.base_asset)
        dof_props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        dof_props["stiffness"].fill(float(self.args.stiffness))
        dof_props["damping"].fill(float(self.args.damping))
        lower = np.asarray(dof_props["lower"], dtype=np.float64)
        upper = np.asarray(dof_props["upper"], dtype=np.float64)
        has_limits = np.asarray(dof_props["hasLimits"], dtype=bool)
        states = np.zeros(num_dofs, dtype=self.gymapi.DofState.dtype)
        for idx, name in enumerate(dof_names):
            value = float(A2W_DEFAULT_LEG_POS.get(name, 0.0))
            if bool(has_limits[idx]) and float(lower[idx]) < float(upper[idx]):
                value = float(np.clip(value, lower[idx], upper[idx]))
            states["pos"][idx] = value
        self.gym.set_actor_dof_properties(self.env, actor, dof_props)
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, states["pos"])

    def _paint_actor(self, actor, color):
        for body_i in range(self.gym.get_actor_rigid_body_count(self.env, actor)):
            self.gym.set_rigid_body_color(self.env, actor, body_i, self.gymapi.MESH_VISUAL, color)

    def _prepare_tensors(self):
        self.gym.prepare_sim(self.sim)
        self.rb_states = self.gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.gym_jacobian = self.gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "z1_gym_jacobian"))
        self.ee_jacobian_index, self.control_jacobian_indices = self._jacobian_mapping(self.args.ee_link)

    def _jacobian_mapping(self, ee_link):
        ee_asset_index = int(self.body_names.index(ee_link))
        jac = self.gym_jacobian[0] if self.gym_jacobian.ndim >= 4 else self.gym_jacobian
        jac_body_dim = int(jac.shape[0])
        jac_col_dim = int(jac.shape[-1])
        if jac_body_dim == len(self.body_names):
            ee_jacobian_index = ee_asset_index
        elif jac_body_dim == len(self.body_names) - 1:
            ee_jacobian_index = ee_asset_index - 1
        else:
            raise RuntimeError(
                f"Cannot map Jacobian body dim={jac_body_dim} to asset bodies={len(self.body_names)}"
            )

        if jac_col_dim == self.num_dofs:
            col_offset = 0
        elif jac_col_dim == self.num_dofs + 6:
            col_offset = 6
        else:
            col_offset = jac_col_dim - self.num_dofs
            if col_offset < 0 or col_offset > 6:
                raise RuntimeError(f"Cannot map Jacobian columns={jac_col_dim}, dofs={self.num_dofs}")
        return ee_jacobian_index, self.control_indices + col_offset

    def _reset_actors(self):
        home_q = np.zeros(self.num_dofs, dtype=np.float64)
        home_q[self.control_indices] = np.asarray(self.args.home_q, dtype=np.float64)
        home_q = np.clip(home_q, self.lower, self.upper)
        if "jointGripper" in self.dof_names:
            home_q[self.dof_names.index("jointGripper")] = 0.0
        for solver in self.solvers:
            self._set_actor_state(solver.actor, home_q)
            solver.goal_q6 = home_q[self.control_indices].copy()
        self._simulate_once()
        self._refresh()
        self.generate_command(force=True)

    def _create_viewer(self):
        self.viewer = None
        if self.args.headless:
            return
        camera_props = self.gymapi.CameraProperties()
        camera_props.width = 1600
        camera_props.height = 900
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer. Re-run with --headless for non-graphical smoke tests.")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_S, "new_command")
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            self.gymapi.Vec3(0.0, -4.6, float(self.args.root_z) + 1.1),
            self.gymapi.Vec3(0.0, 0.0, float(self.args.root_z) + 0.15),
        )

    def _build_draw_geometries(self):
        gymapi = self.gymapi
        gymutil = self.gymutil

        class ThickAxesGeometry(gymutil.LineGeometry):
            def __init__(self, scale=1.0, thickness=0.004):
                offsets = {
                    0: [(0, 0, 0), (0, thickness, 0), (0, -thickness, 0), (0, 0, thickness), (0, 0, -thickness)],
                    1: [(0, 0, 0), (thickness, 0, 0), (-thickness, 0, 0), (0, 0, thickness), (0, 0, -thickness)],
                    2: [(0, 0, 0), (thickness, 0, 0), (-thickness, 0, 0), (0, thickness, 0), (0, -thickness, 0)],
                }
                axis_end = [(scale, 0, 0), (0, scale, 0), (0, 0, scale)]
                axis_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
                verts = np.empty((15, 2), gymapi.Vec3.dtype)
                colors = np.empty(15, gymapi.Vec3.dtype)
                idx = 0
                for axis in range(3):
                    for offset in offsets[axis]:
                        verts[idx][0] = offset
                        verts[idx][1] = tuple(axis_end[axis][j] + offset[j] for j in range(3))
                        colors[idx] = axis_color[axis]
                        idx += 1
                self.verts = verts
                self._colors = colors

            def vertices(self):
                return self.verts

            def colors(self):
                return self._colors

        radius = float(self.args.target_sphere_radius)
        self.target_geom = gymutil.WireframeSphereGeometry(
            radius=radius,
            num_lats=10,
            num_lons=10,
            color=(1.0, 0.82, 0.05),
            color2=(1.0, 0.45, 0.05),
        )
        self.current_geoms = {
            "gym": gymutil.WireframeSphereGeometry(radius=radius * 0.75, num_lats=8, num_lons=8, color=(0.75, 0.35, 1.0)),
            "pinocchio": gymutil.WireframeSphereGeometry(radius=radius * 0.75, num_lats=8, num_lons=8, color=(0.25, 0.48, 1.0)),
            "rtb": gymutil.WireframeSphereGeometry(radius=radius * 0.75, num_lats=8, num_lons=8, color=(0.10, 0.85, 0.45)),
            "tracik": gymutil.WireframeSphereGeometry(radius=radius * 0.75, num_lats=8, num_lons=8, color=(1.0, 0.48, 0.12)),
        }
        self.axes_geom = ThickAxesGeometry(scale=float(self.args.axis_scale), thickness=0.004)

    def _set_actor_state(self, actor, q):
        q = np.asarray(q, dtype=np.float64).reshape(self.num_dofs)
        states = self.gym.get_actor_dof_states(self.env, actor, self.gymapi.STATE_ALL)
        states["pos"][:] = q.astype(np.float32)
        states["vel"][:] = 0.0
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, q.astype(np.float32))

    def _actor_q(self, actor):
        states = self.gym.get_actor_dof_states(self.env, actor, self.gymapi.STATE_ALL)
        return np.asarray(states["pos"], dtype=np.float64).copy()

    def _simulate_once(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def _refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def _ee_pose(self, body_sim_index):
        state = self.rb_states[body_sim_index].detach().cpu().numpy()
        return np.concatenate([np.asarray(state[:3], dtype=np.float64), normalize_quat_xyzw(state[3:7])])

    def _target_pose_for_solver(self, solver):
        return transform_pose(solver.root_pose, self.target_local_pose)

    def _solver_seed_q(self, solver, target_q6=None):
        if not bool(self.args.global_targets) or self.args.global_seed_mode == "current":
            return self._actor_q(solver.actor)[self.control_indices]

        lower6 = self.lower[self.control_indices]
        upper6 = self.upper[self.control_indices]
        margin = np.minimum(0.05, np.maximum(upper6 - lower6, 0.0) * 0.02)
        if self.args.global_seed_mode == "target" and target_q6 is not None:
            seed = np.asarray(target_q6, dtype=np.float64).reshape(6)
        elif self.args.global_seed_mode == "middle":
            seed = 0.5 * (lower6 + upper6)
        elif self.args.global_seed_mode == "zero":
            seed = np.zeros(6, dtype=np.float64)
        else:
            seed = np.asarray(self.args.home_q, dtype=np.float64).reshape(6)
        return np.clip(seed, lower6 + margin, upper6 - margin)

    def _solve_all(self, local_pose, target_q6=None):
        solutions: Dict[str, np.ndarray] = {}
        timings: Dict[str, float] = {}
        for solver in self.solvers:
            if solver.solve_fn is None:
                timings[solver.key] = 0.0
                continue
            seed_q = self._solver_seed_q(solver, target_q6=target_q6)
            start_ns = time.perf_counter_ns()
            result = solver.solve_fn(local_pose, seed_q)
            timings[solver.key] = (time.perf_counter_ns() - start_ns) * 1.0e-6
            if result is None:
                return None, timings
            solutions[solver.key] = np.asarray(result, dtype=np.float64).reshape(-1)[:6]
        return solutions, timings

    def _apply_command(self, local_pose, solutions, timings, label):
        self.command_id += 1
        self.command_step = 0
        self.reached_reported = False
        self.target_local_pose = np.asarray(local_pose, dtype=np.float64).reshape(7)
        for solver in self.solvers:
            if solver.key in solutions:
                solver.goal_q6 = np.asarray(solutions[solver.key], dtype=np.float64).copy()
            else:
                solver.goal_q6 = self._actor_q(solver.actor)[self.control_indices].copy()
            solver.last_solve_ms = float(timings.get(solver.key, math.nan))
            solver.last_success = True

        print(
            f"[command {self.command_id}] {label} local xyz="
            f"({local_pose[0]:+.4f}, {local_pose[1]:+.4f}, {local_pose[2]:+.4f}) "
            f"quat=({local_pose[3]:+.4f}, {local_pose[4]:+.4f}, {local_pose[5]:+.4f}, {local_pose[6]:+.4f})"
        )
        for solver in self.solvers:
            if solver.solve_fn is None:
                print(f"  {solver.label:<11} online Jacobian IK")
                continue
            print(
                f"  {solver.label:<11} solve={solver.last_solve_ms:7.3f} ms "
                f"goal_q={np.array2string(solver.goal_q6, precision=3, suppress_small=True)}"
            )

    def generate_command(self, force=False):
        del force
        lower6 = self.lower[self.control_indices]
        upper6 = self.upper[self.control_indices]
        center = self.target_q6.copy()
        margin = np.minimum(0.05, np.maximum(upper6 - lower6, 0.0) * 0.02)

        for attempt in range(80):
            if bool(self.args.global_targets):
                target_q6 = lower6 + margin + self.rng.random(6) * (upper6 - lower6 - 2.0 * margin)
                label = f"global sample attempt={attempt + 1} seed_mode={self.args.global_seed_mode}"
            else:
                delta = self.rng.uniform(-float(self.args.target_delta), float(self.args.target_delta), size=6)
                delta[5] *= 1.6
                target_q6 = np.clip(center + delta, lower6 + margin, upper6 - margin)
                label = f"sample attempt={attempt + 1}"
            local_pose = self.pin_ik.fk(target_q6)
            solutions, timings = self._solve_all(local_pose, target_q6=target_q6)
            if solutions is not None:
                self.target_q6 = target_q6
                self._apply_command(local_pose, solutions, timings, label)
                return True

        local_pose = self.pin_ik.fk(center)
        solutions, timings = self._solve_all(local_pose, target_q6=center)
        if solutions is None:
            print("No common IK solution found; keeping previous command.")
            return False
        self._apply_command(local_pose, solutions, timings, "fallback previous target")
        return True

    def _update_actors(self):
        for solver in self.solvers:
            if solver.solve_fn is None:
                self._update_gym_jacobian(solver)
                continue
            if solver.goal_q6 is None:
                continue
            current_q = self._actor_q(solver.actor)
            target_q = current_q.copy()
            target_q[self.control_indices] = solver.goal_q6
            target_q = np.clip(target_q, self.lower, self.upper)
            self.gym.set_actor_dof_position_targets(self.env, solver.actor, target_q.astype(np.float32))

    def _update_gym_jacobian(self, solver):
        torch = self.torch
        target_pose = self._target_pose_for_solver(solver)
        target_pos = torch.tensor(target_pose[:3], dtype=torch.float32, device=self.gym_jacobian.device)
        target_quat = torch.tensor(target_pose[3:7], dtype=torch.float32, device=self.gym_jacobian.device)

        eef_state = self.rb_states[solver.ee_body_sim_index]
        eef_pos = eef_state[:3]
        eef_quat = eef_state[3:7]
        pos_err = target_pos - eef_pos
        orn_err = torch_orientation_error(torch, target_quat, eef_quat)

        jac = self.gym_jacobian[0] if self.gym_jacobian.ndim >= 4 else self.gym_jacobian
        j_eef = jac[self.ee_jacobian_index, :, :]
        control_cols = torch.tensor(self.control_jacobian_indices, dtype=torch.long, device=j_eef.device)
        j_control = j_eef[:, control_cols]

        dpose = torch.cat(
            (
                float(self.args.ik_pos_gain) * pos_err,
                float(self.args.ik_rot_gain) * orn_err,
            ),
            dim=0,
        )
        weights = torch.tensor(
            [1.0, 1.0, 1.0, self.args.rot_weight, self.args.rot_weight, self.args.rot_weight],
            dtype=torch.float32,
            device=j_control.device,
        )
        task_j = j_control * weights.view(6, 1)
        task_err = dpose * weights

        j_t = torch.transpose(task_j, 0, 1)
        damping = max(1.0e-6, float(self.args.gym_ik_damping))
        lhs = task_j @ j_t + torch.eye(task_j.shape[0], dtype=torch.float32, device=task_j.device) * (
            damping * damping
        )
        delta = j_t @ torch.linalg.solve(lhs, task_err.unsqueeze(-1)).squeeze(-1)
        delta = torch.clamp(delta, -float(self.args.gym_max_step), float(self.args.gym_max_step))

        current_q = self._actor_q(solver.actor)
        next_q = current_q.copy()
        next_q[self.control_indices] += delta.detach().cpu().numpy().astype(np.float64)
        next_q = np.clip(next_q, self.lower, self.upper)
        self.gym.set_actor_dof_position_targets(self.env, solver.actor, next_q.astype(np.float32))

    def _update_errors(self):
        all_reached = True
        for solver in self.solvers:
            current_pose = self._ee_pose(solver.ee_body_sim_index)
            solver.last_error = pose_error(self._target_pose_for_solver(solver), current_pose)
            all_reached = (
                all_reached
                and solver.last_error[0] <= float(self.args.pos_tol)
                and solver.last_error[1] <= float(self.args.rot_tol)
            )
        if all_reached and not self.reached_reported:
            error_text = " | ".join(
                f"{solver.key} pos={solver.last_error[0]:.4f} rot={solver.last_error[1]:.4f}" for solver in self.solvers
            )
            print(f"[command {self.command_id}] reached: {error_text}")
            self.reached_reported = True

    def _draw(self):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        for solver in self.solvers:
            target_pose = self._target_pose_for_solver(solver)
            current_pose = self._ee_pose(solver.ee_body_sim_index)
            target_tf = make_transform(self.gymapi, target_pose[:3], target_pose[3:7])
            current_tf = make_transform(self.gymapi, current_pose[:3], current_pose[3:7])
            root_tf = make_transform(self.gymapi, solver.root_pose[:3], solver.root_pose[3:7])
            self.gymutil.draw_lines(self.target_geom, self.gym, self.viewer, self.env, target_tf)
            self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, target_tf)
            self.gymutil.draw_lines(self.current_geoms[solver.key], self.gym, self.viewer, self.env, current_tf)
            self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, current_tf)
            if bool(self.args.show_a2w_base):
                self.gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.env, root_tf)
            self._draw_error_line(current_pose, target_pose, solver.color)

    def _draw_error_line(self, current_pose, target_pose, color):
        p1 = self.gymapi.Vec3(float(current_pose[0]), float(current_pose[1]), float(current_pose[2]))
        p2 = self.gymapi.Vec3(float(target_pose[0]), float(target_pose[1]), float(target_pose[2]))
        self.gymutil.draw_line(p1, p2, color, self.gym, self.viewer, self.env)

    def _handle_viewer_events(self):
        if self.viewer is None:
            return False
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "quit" and event.value > 0:
                return True
            if event.action == "new_command" and event.value > 0:
                self._refresh()
                self.generate_command()
        return False

    def run(self):
        print("Viewer controls: S = new shared 6D command, Esc = quit.")
        print("Left to right: purple Gym Jacobian, blue Pinocchio, green Robotics Toolbox, orange TRAC-IK.")
        print("TRAC-IK solves full pose; RTB/Pinocchio use rot_weight in their weighted IK wrappers.")
        if bool(self.args.global_targets):
            print(
                "Command mode: global reachable pose sampling; "
                f"offline IK seed_mode={self.args.global_seed_mode}, "
                f"pin_restarts={int(self.args.pin_restarts)}, "
                f"pin_allow_restarts={bool(self.args.pin_allow_restarts)}."
            )
        else:
            print(f"Command mode: local joint-space delta sampling; target_delta={float(self.args.target_delta):.3f}.")
        step = 0
        max_steps = int(self.args.max_steps)
        while True:
            if self.viewer is not None and self.gym.query_viewer_has_closed(self.viewer):
                break
            if max_steps > 0 and step >= max_steps:
                break
            if self._handle_viewer_events():
                break

            self._refresh()
            self._update_actors()
            self._simulate_once()
            self._refresh()
            self._update_errors()

            if self.args.print_interval > 0 and step % int(self.args.print_interval) == 0:
                error_text = " | ".join(
                    f"{solver.key} pos={solver.last_error[0]:.4f} rot={solver.last_error[1]:.4f}" for solver in self.solvers
                )
                print(f"[step {step:05d}] {error_text}")

            if self.viewer is not None:
                self._draw()
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            step += 1
            self.command_step += 1

        error_text = " | ".join(
            f"{solver.key} pos={solver.last_error[0]:.4f} rot={solver.last_error[1]:.4f}" for solver in self.solvers
        )
        print(f"Final errors: {error_text}")

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    pin_ik = Z1PinocchioIK(args.urdf, ee_link=args.ee_link)
    rtb_ik = Z1RtbKinematics(args.urdf, ee_link=args.ee_link)
    tracik_ik = Z1TracIKKinematics(
        args.urdf,
        ee_link=args.ee_link,
        timeout=float(args.tracik_timeout),
        epsilon=float(args.tracik_epsilon),
        solver_type=args.tracik_solver_type,
    )

    from isaacgym import gymapi, gymtorch, gymutil
    import torch

    app = TripleZ1IKPlay(args, pin_ik, rtb_ik, tracik_ik, rng, gymapi, gymtorch, gymutil, torch)
    try:
        app.run()
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
