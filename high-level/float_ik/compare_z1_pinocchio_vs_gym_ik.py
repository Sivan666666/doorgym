#!/usr/bin/env python3
"""Compare standalone Z1 Pinocchio IK against the original Isaac Gym Jacobian IK."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from z1_pinocchio_ik import (
    DEFAULT_Z1_EE_LINK,
    DEFAULT_Z1_JOINT_NAMES,
    DEFAULT_Z1_URDF,
    Z1PinocchioIK,
    normalize_quat_xyzw,
    orientation_error_xyzw,
)


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "z1"
DEFAULT_ASSET_FILE = "urdf/z1_arm.urdf"
ACTOR_NAME = "z1_arm_articulated"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=str, default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--asset_file", type=str, default=DEFAULT_ASSET_FILE)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_Z1_URDF))
    parser.add_argument("--ee_link", type=str, default=DEFAULT_Z1_EE_LINK)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position_only", action="store_true")
    parser.add_argument("--gym_max_iter", type=int, default=100)
    parser.add_argument("--pin_max_iter", type=int, default=100)
    parser.add_argument("--pin_restarts", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--damping", type=float, default=1.0e-4)
    parser.add_argument("--max_step", type=float, default=0.08)
    parser.add_argument("--rot_weight", type=float, default=0.5)
    parser.add_argument("--target_delta", type=float, default=0.35)
    parser.add_argument("--global_targets", action="store_true")
    parser.add_argument("--pos_tol", type=float, default=1.0e-3)
    parser.add_argument("--rot_tol", type=float, default=2.0e-2)
    parser.add_argument("--assert_fk_pos", type=float, default=5.0e-3)
    parser.add_argument("--assert_pin_pos", type=float, default=2.0e-2)
    parser.add_argument("--assert_gym_pos", type=float, default=math.inf)
    parser.add_argument("--debug_fk", action="store_true")
    return parser.parse_args()


def torch_quat_conjugate(torch, q):
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def torch_quat_multiply(torch, lhs, rhs):
    x1, y1, z1, w1 = lhs.unbind(-1)
    x2, y2, z2, w2 = rhs.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def torch_orientation_error(torch, desired, current):
    delta = torch_quat_multiply(torch, desired, torch_quat_conjugate(torch, current))
    return delta[..., :3] * torch.sign(delta[..., 3:4])


def pose_error(target_pose, achieved_pose):
    target_pose = np.asarray(target_pose, dtype=np.float64)
    achieved_pose = np.asarray(achieved_pose, dtype=np.float64)
    pos_err = float(np.linalg.norm(target_pose[:3] - achieved_pose[:3]))
    rot_err = float(np.linalg.norm(orientation_error_xyzw(target_pose[3:7], achieved_pose[3:7])))
    return pos_err, rot_err


class GymZ1Fixture:
    def __init__(self, args):
        from isaacgym import gymapi, gymtorch
        import torch

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.torch = torch
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.dt = 1.0 / 60.0
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_options.use_mesh_materials = True
        self.asset_root = str(Path(args.asset_root).expanduser().resolve())
        self.asset_file = str(args.asset_file)
        self.asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)
        if self.asset is None:
            raise RuntimeError(f"Failed to load asset root={self.asset_root} file={self.asset_file}")

        self.env = self.gym.create_env(
            self.sim,
            gymapi.Vec3(-1.0, -1.0, -1.0),
            gymapi.Vec3(1.0, 1.0, 1.0),
            1,
        )
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.actor = self.gym.create_actor(self.env, self.asset, pose, ACTOR_NAME, 0, 0)

        self.dof_names = list(self.gym.get_asset_dof_names(self.asset))
        self.num_dofs = len(self.dof_names)
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        self.dof_props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
        self.dof_props["stiffness"].fill(1000.0)
        self.dof_props["damping"].fill(100.0)
        self.gym.set_actor_dof_properties(self.env, self.actor, self.dof_props)

        self.lower = np.asarray(self.dof_props["lower"], dtype=np.float64)
        self.upper = np.asarray(self.dof_props["upper"], dtype=np.float64)
        self.control_indices = np.asarray(
            [self.dof_names.index(name) for name in DEFAULT_Z1_JOINT_NAMES if name in self.dof_names],
            dtype=np.int64,
        )
        if self.control_indices.shape[0] != len(DEFAULT_Z1_JOINT_NAMES):
            raise RuntimeError(f"Missing Z1 control DOFs. Loaded DOFs: {self.dof_names}")

        self.body_names = list(self.gym.get_asset_rigid_body_names(self.asset))
        if args.ee_link not in self.body_names:
            raise RuntimeError(f"EE link {args.ee_link!r} not in asset bodies: {self.body_names}")
        self.ee_body_sim_index = self.gym.find_actor_rigid_body_index(
            self.env,
            self.actor,
            args.ee_link,
            gymapi.DOMAIN_SIM,
        )

        self.gym.prepare_sim(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.jacobian = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, ACTOR_NAME))
        self.ee_jacobian_index, self.control_jacobian_indices = self._jacobian_mapping(args.ee_link)
        self.set_q(np.zeros(self.num_dofs, dtype=np.float64))

    def _jacobian_mapping(self, ee_link):
        ee_asset_index = int(self.body_names.index(ee_link))
        jac = self.jacobian[0] if self.jacobian.ndim >= 4 else self.jacobian
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

    def destroy(self):
        self.gym.destroy_sim(self.sim)

    def set_q(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(self.num_dofs)
        states = self.gym.get_actor_dof_states(self.env, self.actor, self.gymapi.STATE_ALL)
        states["pos"][:] = q.astype(np.float32)
        states["vel"][:] = 0.0
        self.gym.set_actor_dof_states(self.env, self.actor, states, self.gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, q.astype(np.float32))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def ee_pose(self, q=None):
        if q is not None:
            self.set_q(q)
        state = self.rb_states[self.ee_body_sim_index].detach().cpu().numpy()
        return np.concatenate(
            [
                np.asarray(state[:3], dtype=np.float64),
                normalize_quat_xyzw(np.asarray(state[3:7], dtype=np.float64)),
            ]
        )

    def current_q(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        return self.dof_state_tensor[: self.num_dofs, 0].detach().cpu().numpy().astype(np.float64).copy()

    def solve_gym_ik(self, target_pose, seed_q, args):
        torch = self.torch
        target_pose = np.asarray(target_pose, dtype=np.float64)
        target_pos = torch.tensor(target_pose[:3], dtype=torch.float32, device=self.jacobian.device)
        target_quat = torch.tensor(target_pose[3:7], dtype=torch.float32, device=self.jacobian.device)
        target_quat = target_quat / torch.clamp(torch.linalg.norm(target_quat), min=1.0e-8)
        lower = torch.tensor(self.lower, dtype=torch.float32, device=self.jacobian.device)
        upper = torch.tensor(self.upper, dtype=torch.float32, device=self.jacobian.device)
        control_indices = torch.tensor(self.control_indices, dtype=torch.long, device=self.jacobian.device)
        control_jacobian_indices = torch.tensor(
            self.control_jacobian_indices,
            dtype=torch.long,
            device=self.jacobian.device,
        )
        q_np = np.asarray(seed_q, dtype=np.float64).copy()
        self.set_q(q_np)
        q_np = self.current_q()

        for _ in range(max(1, int(args.gym_max_iter))):
            eef_state = self.rb_states[self.ee_body_sim_index]
            eef_pos = eef_state[:3]
            eef_quat = eef_state[3:7]
            pos_err = target_pos - eef_pos
            jac = self.jacobian[0] if self.jacobian.ndim >= 4 else self.jacobian
            j_eef = jac[self.ee_jacobian_index, :, :]
            j_control = j_eef[:, control_jacobian_indices]
            if args.position_only:
                task_j = j_control[:3, :]
                task_err = pos_err
                rot_err_norm = 0.0
            else:
                orn_err = torch_orientation_error(torch, target_quat, eef_quat)
                weights = torch.tensor(
                    [1.0, 1.0, 1.0, args.rot_weight, args.rot_weight, args.rot_weight],
                    dtype=torch.float32,
                    device=self.jacobian.device,
                )
                task_j = j_control * weights.view(6, 1)
                task_err = torch.cat((pos_err, orn_err), dim=0) * weights
                rot_err_norm = float(torch.linalg.norm(orn_err).detach().cpu())
            pos_err_norm = float(torch.linalg.norm(pos_err).detach().cpu())
            if pos_err_norm <= args.pos_tol and rot_err_norm <= args.rot_tol:
                return q_np

            j_t = torch.transpose(task_j, 0, 1)
            lhs = task_j @ j_t + torch.eye(task_j.shape[0], dtype=torch.float32, device=self.jacobian.device) * (
                float(args.damping) ** 2
            )
            delta = j_t @ torch.linalg.solve(lhs, task_err.unsqueeze(-1)).squeeze(-1)
            delta = torch.clamp(delta, -float(args.max_step), float(args.max_step))

            q_tensor = torch.tensor(q_np, dtype=torch.float32, device=self.jacobian.device)
            q_tensor[control_indices] += delta
            q_tensor = torch.max(torch.min(q_tensor, upper), lower)
            q_np = q_tensor.detach().cpu().numpy().astype(np.float64)
            self.set_q(q_np)
            q_np = self.current_q()
        return q_np


def sample_actor_q(fixture, rng):
    q = np.zeros(fixture.num_dofs, dtype=np.float64)
    lower = fixture.lower[fixture.control_indices]
    upper = fixture.upper[fixture.control_indices]
    span = upper - lower
    q[fixture.control_indices] = lower + rng.uniform(0.05, 0.95, size=span.shape) * span
    if "jointGripper" in fixture.dof_names:
        q[fixture.dof_names.index("jointGripper")] = 0.0
    return q


def sample_target_q(fixture, seed_q, rng, target_delta, global_targets):
    if global_targets:
        return sample_actor_q(fixture, rng)
    target_q = np.asarray(seed_q, dtype=np.float64).copy()
    lower = fixture.lower[fixture.control_indices]
    upper = fixture.upper[fixture.control_indices]
    delta = rng.uniform(-float(target_delta), float(target_delta), size=fixture.control_indices.shape[0])
    target_q[fixture.control_indices] = np.clip(target_q[fixture.control_indices] + delta, lower, upper)
    return target_q


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return f"{name}: no successful samples"
    return (
        f"{name}: mean={float(np.mean(values)):.6f} "
        f"p95={float(np.percentile(values, 95)):.6f} "
        f"max={float(np.max(values)):.6f}"
    )


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    pin_ik = Z1PinocchioIK(args.urdf, ee_link=args.ee_link)
    fixture = GymZ1Fixture(args)

    fk_pos_errors = []
    fk_rot_errors = []
    pin_pos_errors = []
    pin_rot_errors = []
    gym_pos_errors = []
    gym_rot_errors = []
    q_l2_diffs = []
    pin_failures = 0

    try:
        if args.debug_fk:
            print("dof_names:", fixture.dof_names, flush=True)
            print("body_names:", fixture.body_names, flush=True)
            print(
                "jacobian_shape:",
                tuple(int(x) for x in fixture.jacobian.shape),
                "ee_jacobian_index:",
                int(fixture.ee_jacobian_index),
                "control_jacobian_indices:",
                fixture.control_jacobian_indices.astype(int).tolist(),
                flush=True,
            )
            debug_qs = [
                np.zeros(6, dtype=np.float64),
                np.asarray([0.4, 0.7, -1.4, 0.5, 0.2, -0.3], dtype=np.float64),
            ]
            for debug_q6 in debug_qs:
                debug_actor_q = np.zeros(fixture.num_dofs, dtype=np.float64)
                debug_actor_q[fixture.control_indices] = debug_q6
                debug_gym_pose = fixture.ee_pose(debug_actor_q)
                debug_pin_pose = pin_ik.fk(debug_q6)
                debug_pos, debug_rot = pose_error(debug_gym_pose, debug_pin_pose)
                print("debug q6:", np.round(debug_q6, 6).tolist(), flush=True)
                print("  gym:", np.round(debug_gym_pose, 6).tolist(), flush=True)
                print("  pin:", np.round(debug_pin_pose, 6).tolist(), flush=True)
                print(f"  fk_err: pos={debug_pos:.6f} rot={debug_rot:.6f}", flush=True)

        for _ in range(max(1, int(args.samples))):
            q_seed = sample_actor_q(fixture, rng)
            q_target = sample_target_q(
                fixture,
                q_seed,
                rng,
                target_delta=float(args.target_delta),
                global_targets=bool(args.global_targets),
            )
            target_pose = fixture.ee_pose(q_target)
            q_target_actual = fixture.current_q()
            target_joints = q_target_actual[fixture.control_indices]
            pin_fk_pose = pin_ik.fk(target_joints)
            fk_pos, fk_rot = pose_error(target_pose, pin_fk_pose)
            fk_pos_errors.append(fk_pos)
            fk_rot_errors.append(fk_rot)

            gym_q = fixture.solve_gym_ik(target_pose, q_seed, args)
            gym_pose = fixture.ee_pose(gym_q)
            gym_pos, gym_rot = pose_error(target_pose, gym_pose)
            gym_pos_errors.append(gym_pos)
            gym_rot_errors.append(gym_rot)

            pin_joints = pin_ik.ik(
                target_pose,
                seed_joints=q_seed[fixture.control_indices],
                position_only=bool(args.position_only),
                max_iter=int(args.pin_max_iter),
                max_restarts=int(args.pin_restarts),
                dt=float(args.dt),
                damping=float(args.damping),
                threshold=float(args.pos_tol),
                rotation_weight=float(args.rot_weight),
                rng=rng,
            )
            if pin_joints is None:
                pin_failures += 1
                continue
            pin_q = q_seed.copy()
            pin_q[fixture.control_indices] = pin_joints
            pin_pose = fixture.ee_pose(pin_q)
            pin_pos, pin_rot = pose_error(target_pose, pin_pose)
            pin_pos_errors.append(pin_pos)
            pin_rot_errors.append(pin_rot)
            q_l2_diffs.append(float(np.linalg.norm(pin_q[fixture.control_indices] - gym_q[fixture.control_indices])))
    finally:
        fixture.destroy()

    print("Z1 Pinocchio vs IsaacGym IK comparison")
    print(f"asset={fixture.asset_root}/{fixture.asset_file}")
    print(f"urdf={Path(args.urdf).expanduser().resolve()}")
    print(f"samples={args.samples} position_only={bool(args.position_only)} pin_failures={pin_failures}")
    print(summarize("FK pos error pin-vs-gym (m)", fk_pos_errors))
    print(summarize("FK rot error pin-vs-gym", fk_rot_errors))
    print(summarize("Gym IK final pos error (m)", gym_pos_errors))
    print(summarize("Gym IK final rot error", gym_rot_errors))
    print(summarize("Pin IK final pos error measured by Gym (m)", pin_pos_errors))
    print(summarize("Pin IK final rot error measured by Gym", pin_rot_errors))
    print(summarize("Pin-vs-Gym solution joint L2 (rad)", q_l2_diffs))

    if fk_pos_errors and float(np.max(fk_pos_errors)) > float(args.assert_fk_pos):
        raise SystemExit(f"FK mismatch too large: max {float(np.max(fk_pos_errors)):.6f} m")
    if gym_pos_errors and float(np.max(gym_pos_errors)) > float(args.assert_gym_pos):
        raise SystemExit(f"Gym IK error too large: max {float(np.max(gym_pos_errors)):.6f} m")
    if pin_pos_errors and float(np.max(pin_pos_errors)) > float(args.assert_pin_pos):
        raise SystemExit(f"Pinocchio IK error too large: max {float(np.max(pin_pos_errors)):.6f} m")
    if pin_failures:
        raise SystemExit(f"Pinocchio IK failed on {pin_failures}/{args.samples} samples")


if __name__ == "__main__":
    main()
