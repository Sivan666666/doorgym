#!/usr/bin/env python3
"""Visualize raw and jerk-continuous Z1 joint trajectories in Isaac Gym.

Left:  piecewise-linear interpolation through the IK joint waypoints.
Right: globally smoothed polynomial trajectory through the same waypoints.

Viewer controls:
    S      sample new reachable joint waypoints
    R      replay the current trajectory
    Space  pause/resume
    Esc    quit
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
IK_SOLVERS_DIR = SCRIPT_DIR / "ik_solvers"
if str(IK_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(IK_SOLVERS_DIR))

from joint_trajectory_smoothing import continuity_report, load_waypoints, smooth_joint_waypoints
from ik_solvers.z1_pinocchio_ik import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_JOINT_NAMES,
    DEFAULT_Z1_URDF,
    Z1PinocchioIK,
)
from play_z1_gym_vs_pinocchio_ik import (
    A2W_DEFAULT_LEG_POS,
    build_a2w_base_visual_asset_root,
    make_transform,
)


HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "z1"
DEFAULT_ASSET_FILE = "urdf/z1_arm.urdf"
DEFAULT_A2W_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "a2wz1"
DEFAULT_A2W_ASSET_FILE = "urdf/a2wz1.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--asset_file", type=str, default=DEFAULT_ASSET_FILE)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--input", type=str, default="", help="Optional .npy/.npz/.json IK waypoints [N,6].")
    parser.add_argument("--input_key", type=str, default="q")
    parser.add_argument("--method", choices=("quintic", "quintic_hermite", "septic"), default="quintic")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--segment_time", type=float, default=1.0)
    parser.add_argument("--max_vel", type=float, default=0.0)
    parser.add_argument("--num_waypoints", type=int, default=5)
    parser.add_argument("--waypoint_delta", type=float, default=0.50)
    parser.add_argument("--home_q", type=float, nargs=6, default=(0.0, 1.05, -1.45, 0.75, 0.0, 0.0))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hold_time", type=float, default=0.75)
    parser.add_argument("--no_loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until viewer close.")
    parser.add_argument("--root_z", type=float, default=0.50)
    parser.add_argument("--disable_arm_visual_flip", action="store_true")
    parser.add_argument("--show_a2w_base", dest="show_a2w_base", action="store_true", default=True)
    parser.add_argument("--no_show_a2w_base", dest="show_a2w_base", action="store_false")
    parser.add_argument("--a2w_asset_root", type=str, default=str(DEFAULT_A2W_ASSET_ROOT))
    parser.add_argument("--a2w_asset_file", type=str, default=DEFAULT_A2W_ASSET_FILE)
    return parser.parse_args()


def finite_difference_kinematics(q: np.ndarray, t: np.ndarray):
    edge_order = 2 if len(t) >= 3 else 1
    qd = np.gradient(q, t, axis=0, edge_order=edge_order)
    qdd = np.gradient(qd, t, axis=0, edge_order=edge_order)
    qddd = np.gradient(qdd, t, axis=0, edge_order=edge_order)
    return qd, qdd, qddd


def piecewise_linear_samples(waypoints: np.ndarray, waypoint_t: np.ndarray, sample_t: np.ndarray):
    q = np.column_stack(
        [np.interp(sample_t, waypoint_t, waypoints[:, joint]) for joint in range(waypoints.shape[1])]
    )
    qd, qdd, qddd = finite_difference_kinematics(q, sample_t)
    return {"t": sample_t.copy(), "q": q, "qd": qd, "qdd": qdd, "qddd": qddd}


class Z1TrajectoryPlay:
    def __init__(self, args, pin, gymapi, gymutil):
        self.args = args
        self.gymapi = gymapi
        self.gymutil = gymutil
        self.gym = gymapi.acquire_gym()
        self.rng = np.random.default_rng(int(args.seed))
        self.pin = pin
        self.temp_dirs = []
        self.paused = False
        self.frame = 0
        self.hold_frames = 0
        self.command_id = 0
        self.input_consumed = False
        self.trajectory = None

        self.root_poses = {
            "raw": np.asarray([-0.75, 0.0, float(args.root_z), 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "smooth": np.asarray([0.75, 0.0, float(args.root_z), 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        }
        self.colors = {
            "raw": gymapi.Vec3(1.0, 0.36, 0.10),
            "smooth": gymapi.Vec3(0.12, 0.52, 1.0),
        }

        self._create_sim()
        self._load_assets_and_actors()
        self._create_viewer()
        self._build_geometries()
        self.new_trajectory(use_input=True)

    def _create_sim(self):
        sim_params = self.gymapi.SimParams()
        sim_params.up_axis = self.gymapi.UP_AXIS_Z
        sim_params.gravity = self.gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.dt = float(self.args.dt)
        self.sim = self.gym.create_sim(0, 0, self.gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")
        plane = self.gymapi.PlaneParams()
        plane.normal = self.gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)

    def _arm_asset_options(self):
        options = self.gymapi.AssetOptions()
        options.fix_base_link = True
        options.collapse_fixed_joints = False
        options.disable_gravity = True
        options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_NONE)
        options.use_mesh_materials = True
        options.flip_visual_attachments = not bool(self.args.disable_arm_visual_flip)
        options.thickness = 0.001
        options.armature = 0.01
        return options

    def _load_assets_and_actors(self):
        asset_root = str(Path(self.args.asset_root).expanduser().resolve())
        options = self._arm_asset_options()
        print(
            f"Loading Z1 asset: root={asset_root}, file={self.args.asset_file}, "
            f"flip_visual_attachments={options.flip_visual_attachments}"
        )
        self.asset = self.gym.load_asset(self.sim, asset_root, self.args.asset_file, options)
        if self.asset is None:
            raise RuntimeError(f"Failed to load Z1 asset from {asset_root}/{self.args.asset_file}")

        self.base_asset = self._load_a2w_base_asset() if self.args.show_a2w_base else None
        self.env = self.gym.create_env(
            self.sim,
            self.gymapi.Vec3(-2.0, -1.2, -0.2),
            self.gymapi.Vec3(2.0, 1.2, 1.7),
            1,
        )
        self.actors = {}
        self.base_actors = {}
        for key in ("raw", "smooth"):
            root = self.root_poses[key]
            root_tf = make_transform(self.gymapi, root[:3], root[3:7])
            if self.base_asset is not None:
                base_actor = self.gym.create_actor(self.env, self.base_asset, root_tf, f"a2w_{key}", 0, 0)
                self.base_actors[key] = base_actor
                self._configure_a2w_base(base_actor)
                self._paint(base_actor, self.gymapi.Vec3(0.34, 0.36, 0.39))
            actor = self.gym.create_actor(self.env, self.asset, root_tf, f"z1_{key}_trajectory", 0, 0)
            self.actors[key] = actor
            self._paint(actor, self.colors[key])

        self.dof_names = list(self.gym.get_asset_dof_names(self.asset))
        self.num_dofs = len(self.dof_names)
        props = self.gym.get_asset_dof_properties(self.asset)
        props["driveMode"].fill(int(self.gymapi.DOF_MODE_NONE))
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        for actor in self.actors.values():
            self.gym.set_actor_dof_properties(self.env, actor, props)

        self.lower = np.asarray(props["lower"], dtype=np.float64)
        self.upper = np.asarray(props["upper"], dtype=np.float64)
        self.control_indices = np.asarray(
            [self.dof_names.index(name) for name in DEFAULT_Z1_JOINT_NAMES if name in self.dof_names],
            dtype=np.int64,
        )
        if self.control_indices.size != 6:
            raise RuntimeError(f"Expected six Z1 arm joints, loaded DOFs: {self.dof_names}")
        self.lower6 = self.lower[self.control_indices]
        self.upper6 = self.upper[self.control_indices]

    def _load_a2w_base_asset(self):
        temp_dir = tempfile.TemporaryDirectory(prefix="z1_trajectory_a2w_")
        self.temp_dirs.append(temp_dir)
        root, asset_file = build_a2w_base_visual_asset_root(
            self.args.a2w_asset_root,
            self.args.a2w_asset_file,
            temp_dir.name,
        )
        options = self.gymapi.AssetOptions()
        options.fix_base_link = True
        options.collapse_fixed_joints = False
        options.disable_gravity = True
        options.default_dof_drive_mode = int(self.gymapi.DOF_MODE_POS)
        options.use_mesh_materials = True
        options.flip_visual_attachments = False
        asset = self.gym.load_asset(self.sim, str(root), asset_file, options)
        if asset is None:
            raise RuntimeError(f"Failed to load A2W base asset from {root}/{asset_file}")
        return asset

    def _configure_a2w_base(self, actor):
        count = self.gym.get_asset_dof_count(self.base_asset)
        if count <= 0:
            return
        names = list(self.gym.get_asset_dof_names(self.base_asset))
        props = self.gym.get_asset_dof_properties(self.base_asset)
        props["driveMode"].fill(int(self.gymapi.DOF_MODE_POS))
        props["stiffness"].fill(850.0)
        props["damping"].fill(85.0)
        states = np.zeros(count, dtype=self.gymapi.DofState.dtype)
        lower = np.asarray(props["lower"], dtype=np.float64)
        upper = np.asarray(props["upper"], dtype=np.float64)
        limited = np.asarray(props["hasLimits"], dtype=bool)
        for i, name in enumerate(names):
            value = float(A2W_DEFAULT_LEG_POS.get(name, 0.0))
            if limited[i] and lower[i] < upper[i]:
                value = float(np.clip(value, lower[i], upper[i]))
            states["pos"][i] = value
        self.gym.set_actor_dof_properties(self.env, actor, props)
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, actor, states["pos"])

    def _paint(self, actor, color):
        for body in range(self.gym.get_actor_rigid_body_count(self.env, actor)):
            self.gym.set_rigid_body_color(self.env, actor, body, self.gymapi.MESH_VISUAL, color)

    def _create_viewer(self):
        self.viewer = None
        if self.args.headless:
            return
        camera = self.gymapi.CameraProperties()
        camera.width = 1500
        camera.height = 900
        self.viewer = self.gym.create_viewer(self.sim, camera)
        if self.viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_S, "new")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_R, "replay")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, self.gymapi.KEY_SPACE, "pause")
        self.gym.viewer_camera_look_at(
            self.viewer,
            None,
            self.gymapi.Vec3(0.0, -3.6, float(self.args.root_z) + 1.15),
            self.gymapi.Vec3(0.0, 0.0, float(self.args.root_z) + 0.15),
        )

    def _build_geometries(self):
        self.waypoint_geometries = {
            "raw": self.gymutil.WireframeSphereGeometry(0.025, 8, 8, color=(1.0, 0.36, 0.10)),
            "smooth": self.gymutil.WireframeSphereGeometry(0.025, 8, 8, color=(0.12, 0.52, 1.0)),
        }

    def _sample_waypoints(self):
        count = max(2, int(self.args.num_waypoints))
        margin = np.minimum(0.05, np.maximum(self.upper6 - self.lower6, 0.0) * 0.02)
        lo = self.lower6 + margin
        hi = self.upper6 - margin
        home = np.clip(np.asarray(self.args.home_q, dtype=np.float64), lo, hi)
        delta = max(1.0e-3, float(self.args.waypoint_delta))

        for _ in range(100):
            rows = [home.copy()]
            current = home.copy()
            for _ in range(count - 2):
                current = np.clip(current + self.rng.uniform(-delta, delta, size=6), lo, hi)
                rows.append(current.copy())
            rows.append(home.copy())
            waypoints = np.asarray(rows, dtype=np.float64)
            smooth = smooth_joint_waypoints(
                waypoints,
                dt=float(self.args.dt),
                segment_time=float(self.args.segment_time),
                method=self.args.method,
                max_vel=float(self.args.max_vel) if self.args.max_vel > 0.0 else None,
            )
            if np.all(smooth["q"] >= lo[None, :] - 1.0e-9) and np.all(smooth["q"] <= hi[None, :] + 1.0e-9):
                return waypoints, smooth
        raise RuntimeError("Could not sample a smoothed trajectory inside the Z1 joint limits; reduce --waypoint_delta")

    def _load_input_trajectory(self):
        waypoints = load_waypoints(Path(self.args.input), key=self.args.input_key)
        if waypoints.shape[1] != 6:
            raise ValueError(f"Input IK waypoints must have shape [N,6], got {waypoints.shape}")
        if np.any(waypoints < self.lower6[None, :]) or np.any(waypoints > self.upper6[None, :]):
            raise ValueError("Input IK waypoints exceed the loaded Z1 joint limits")
        smooth = smooth_joint_waypoints(
            waypoints,
            dt=float(self.args.dt),
            segment_time=float(self.args.segment_time),
            method=self.args.method,
            max_vel=float(self.args.max_vel) if self.args.max_vel > 0.0 else None,
        )
        if np.any(smooth["q"] < self.lower6[None, :]) or np.any(smooth["q"] > self.upper6[None, :]):
            raise ValueError("Smoothed input trajectory overshoots the Z1 joint limits")
        return waypoints, smooth

    def new_trajectory(self, use_input=False):
        if use_input and self.args.input and not self.input_consumed:
            waypoints, smooth = self._load_input_trajectory()
            self.input_consumed = True
            source = str(Path(self.args.input).expanduser().resolve())
        else:
            waypoints, smooth = self._sample_waypoints()
            source = "reachable random joint waypoints"

        raw = piecewise_linear_samples(waypoints, smooth["waypoint_t"], smooth["t"])
        self.trajectory = {"waypoints": waypoints, "raw": raw, "smooth": smooth}
        self.path_points = {
            key: np.asarray([self._world_position(key, q) for q in self.trajectory[key]["q"]])
            for key in ("raw", "smooth")
        }
        self.waypoint_points = {
            key: np.asarray([self._world_position(key, q) for q in waypoints])
            for key in ("raw", "smooth")
        }
        self.frame = 0
        self.hold_frames = 0
        self.paused = False
        self.command_id += 1
        self._set_actors_at_frame(0)
        self._print_report(source)

    def _print_report(self, source):
        raw = self.trajectory["raw"]
        smooth = self.trajectory["smooth"]
        jumps = continuity_report(smooth)
        print(f"\n[trajectory {self.command_id}] source={source}")
        print(
            f"  method={self.args.method} waypoints={len(self.trajectory['waypoints'])} "
            f"samples={len(smooth['t'])} duration={smooth['t'][-1]:.3f}s dt={self.args.dt:.4f}s"
        )
        print(
            "  raw linear max_abs: "
            f"vel={np.max(np.abs(raw['qd'])):.4f} "
            f"acc={np.max(np.abs(raw['qdd'])):.4f} jerk={np.max(np.abs(raw['qddd'])):.4f}"
        )
        print(
            "  smooth max_abs:    "
            f"vel={np.max(np.abs(smooth['qd'])):.4f} "
            f"acc={np.max(np.abs(smooth['qdd'])):.4f} jerk={np.max(np.abs(smooth['qddd'])):.4f}"
        )
        print(
            "  smooth knot jumps: "
            f"q={jumps['q']:.3e} qd={jumps['qd']:.3e} "
            f"qdd={jumps['qdd']:.3e} qddd={jumps['qddd']:.3e}"
        )

    def _set_actor_q(self, actor, q6, gripper=None):
        states = self.gym.get_actor_dof_states(self.env, actor, self.gymapi.STATE_ALL)
        states["pos"][:] = 0.0
        states["vel"][:] = 0.0
        states["pos"][self.control_indices] = np.asarray(q6, dtype=np.float32)
        if "jointGripper" in self.dof_names:
            gripper_index = self.dof_names.index("jointGripper")
            gripper_value = 0.0 if gripper is None else float(gripper)
            states["pos"][gripper_index] = float(
                np.clip(gripper_value, self.lower[gripper_index], self.upper[gripper_index])
            )
        self.gym.set_actor_dof_states(self.env, actor, states, self.gymapi.STATE_ALL)

    def _set_actors_at_frame(self, frame):
        frame = int(np.clip(frame, 0, len(self.trajectory["smooth"]["q"]) - 1))
        gripper_trajectories = self.trajectory.get("gripper", {})
        for key in ("raw", "smooth"):
            gripper = None
            if key in gripper_trajectories:
                gripper = gripper_trajectories[key][frame]
            self._set_actor_q(self.actors[key], self.trajectory[key]["q"][frame], gripper=gripper)

    def _world_position(self, key, q6):
        local_pose = self.pin.fk(np.asarray(q6, dtype=np.float64))
        return self.root_poses[key][:3] + local_pose[:3]

    def _draw_polyline(self, points, color):
        stride = max(1, int(np.ceil((len(points) - 1) / 180.0)))
        indices = list(range(0, len(points), stride))
        if indices[-1] != len(points) - 1:
            indices.append(len(points) - 1)
        for first, second in zip(indices[:-1], indices[1:]):
            p0 = points[first]
            p1 = points[second]
            self.gymutil.draw_line(
                self.gymapi.Vec3(float(p0[0]), float(p0[1]), float(p0[2])),
                self.gymapi.Vec3(float(p1[0]), float(p1[1]), float(p1[2])),
                color,
                self.gym,
                self.viewer,
                self.env,
            )

    def _draw(self):
        if self.viewer is None:
            return
        self.gym.clear_lines(self.viewer)
        for key in ("raw", "smooth"):
            self._draw_polyline(self.path_points[key], self.colors[key])
            for pos in self.waypoint_points[key]:
                tf = make_transform(self.gymapi, pos)
                self.gymutil.draw_lines(
                    self.waypoint_geometries[key],
                    self.gym,
                    self.viewer,
                    self.env,
                    tf,
                )

    def _handle_events(self):
        if self.viewer is None:
            return False
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.value <= 0:
                continue
            if event.action == "quit":
                return True
            if event.action == "new":
                self.new_trajectory(use_input=False)
            elif event.action == "replay":
                self.frame = 0
                self.hold_frames = 0
                self.paused = False
                self._set_actors_at_frame(0)
            elif event.action == "pause":
                self.paused = not self.paused
                print("Playback paused." if self.paused else "Playback resumed.")
        return False

    def run(self):
        print(
            getattr(
                self,
                "viewer_controls_text",
                "Viewer controls: S=new waypoints, R=replay, Space=pause/resume, Esc=quit.",
            )
        )
        print(
            getattr(
                self,
                "comparison_text",
                "Left orange: raw piecewise-linear. Right blue: jerk-continuous smoothed trajectory.",
            )
        )
        step = 0
        max_steps = int(self.args.max_steps)
        while True:
            if self.viewer is not None and self.gym.query_viewer_has_closed(self.viewer):
                break
            if max_steps > 0 and step >= max_steps:
                break
            if self._handle_events():
                break

            if not self.paused:
                last_frame = len(self.trajectory["smooth"]["q"]) - 1
                if self.frame < last_frame:
                    self.frame += 1
                    self._set_actors_at_frame(self.frame)
                else:
                    self.hold_frames += 1
                    hold_steps = max(0, int(round(float(self.args.hold_time) / float(self.args.dt))))
                    if self.hold_frames >= hold_steps:
                        if self.args.headless:
                            break
                        if not self.args.no_loop:
                            self.frame = 0
                            self.hold_frames = 0
                            self._set_actors_at_frame(0)

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if self.viewer is not None:
                self._draw()
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            step += 1

        print(f"Playback finished at frame={self.frame}, step={step}.")

    def destroy(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()


def main():
    args = parse_args()
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.segment_time <= 0.0:
        raise ValueError("--segment_time must be positive")

    # Pinocchio and Isaac Gym ship different Boost.Python bindings. Building
    # the Pinocchio model first avoids type-registration conflicts in py3.8.
    pin = Z1PinocchioIK(args.urdf, ee_link=args.ee_link)

    from isaacgym import gymapi, gymutil

    app = Z1TrajectoryPlay(args, pin, gymapi, gymutil)
    try:
        app.run()
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
