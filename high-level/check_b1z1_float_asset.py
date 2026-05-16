import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

ROOT = Path(__file__).resolve().parents[1]
HIGH_LEVEL_ROOT = ROOT / "high-level"
if str(HIGH_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGH_LEVEL_ROOT))

from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402


DEFAULT_ARM_POSE = {
    "z1_waist": 0.0,
    "z1_shoulder": 1.48,
    "z1_elbow": -0.63,
    "z1_wrist_angle": -0.84,
    "z1_forearm_roll": 0.0,
    "z1_wrist_rotate": 1.57,
    "z1_jointGripper": -0.785,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal b1z1-float asset loading check.")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--collapse_fixed_joints", action="store_true")
    parser.add_argument("--free_base", dest="fix_base_link", action="store_false")
    parser.set_defaults(fix_base_link=True)
    parser.add_argument("--enable_gravity", dest="disable_gravity", action="store_false")
    parser.set_defaults(disable_gravity=True)
    parser.add_argument("--vhacd_resolution", type=int, default=300000)
    parser.add_argument(
        "--asset_root",
        type=str,
        default=str(HIGH_LEVEL_ROOT / "data" / "asset" / "b1z1-float"),
    )
    parser.add_argument("--asset_file", type=str, default="urdf/b1z1.urdf")
    parser.add_argument("--robot_x", type=float, default=0.0)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--robot_z", type=float, default=0.8)
    parser.add_argument("--robot_yaw", type=float, default=0.0)
    parser.add_argument("--print_body_states", action="store_true")
    return parser.parse_args()


def create_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.sim_device.startswith("cuda")
    sim_params.physx.use_gpu = args.sim_device.startswith("cuda")
    sim_params.physx.num_position_iterations = 12
    sim_params.physx.num_velocity_iterations = 4
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.002

    sim_device_id = int(args.sim_device.split(":")[-1]) if args.sim_device.startswith("cuda") else 0
    graphics_id = -1 if args.headless else args.graphics_device_id
    sim = gym.create_sim(sim_device_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)
    return sim


def load_robot(gym, sim, args):
    asset_root = Path(args.asset_root).expanduser()
    if not asset_root.is_absolute():
        asset_root = (ROOT / asset_root).resolve()
    asset_file = args.asset_file

    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = args.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = False
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = args.fix_base_link
    asset_options.disable_gravity = args.disable_gravity
    asset_options.use_mesh_materials = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = args.vhacd_resolution

    robot_asset = gym.load_asset(sim, str(asset_root), asset_file, asset_options)
    if robot_asset is None:
        raise RuntimeError(f"Failed to load {asset_root / asset_file}")

    dof_names = gym.get_asset_dof_names(robot_asset)
    body_names = gym.get_asset_rigid_body_names(robot_asset)
    print("------------------------------------------------------")
    print("asset:", asset_root / asset_file)
    print("collapse_fixed_joints:", args.collapse_fixed_joints)
    print("fix_base_link:", args.fix_base_link)
    print("num_dofs:", gym.get_asset_dof_count(robot_asset))
    print("dof_names:", dof_names)
    print("num_bodies:", gym.get_asset_rigid_body_count(robot_asset))
    print("body_names:", body_names)
    print("------------------------------------------------------")
    return robot_asset


def main():
    args = parse_args()
    os.chdir(ROOT)
    gym = gymapi.acquire_gym()
    sim = create_sim(gym, args)
    robot_asset = load_robot(gym, sim, args)

    env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0.0), gymapi.Vec3(1.5, 1.5, 1.5), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(args.robot_x, args.robot_y, args.robot_z)
    pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, args.robot_yaw)
    actor = gym.create_actor(env, robot_asset, pose, "b1z1_float", 0, 0, 0)

    dof_props = gym.get_actor_dof_properties(env, actor)
    dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
    dof_props["stiffness"][:].fill(80.0)
    dof_props["damping"][:].fill(4.0)
    gym.set_actor_dof_properties(env, actor, dof_props)

    dof_names = gym.get_actor_dof_names(env, actor)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_targets = np.zeros(len(dof_names), dtype=np.float32)
    for i, name in enumerate(dof_names):
        value = float(DEFAULT_ARM_POSE.get(name, 0.0))
        dof_states["pos"][i] = value
        dof_states["vel"][i] = 0.0
        dof_targets[i] = value
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, dof_targets)
    gym.prepare_sim(sim)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    root_states[0, 0] = args.robot_x
    root_states[0, 1] = args.robot_y
    root_states[0, 2] = args.robot_z
    root_states[0, 3:7] = torch.tensor(
        [
            pose.r.x,
            pose.r.y,
            pose.r.z,
            pose.r.w,
        ],
        dtype=root_states.dtype,
        device=root_states.device,
    )
    root_states[0, 7:13] = 0.0
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
    gym.refresh_actor_root_state_tensor(sim)
    print("spawn_pose_xyz:", [float(args.robot_x), float(args.robot_y), float(args.robot_z)])
    print("actual_root_xyz:", root_states[0, :3].detach().cpu().tolist())
    rigid_body_states = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    gym.refresh_rigid_body_state_tensor(sim)
    if args.print_body_states:
        body_names = gym.get_actor_rigid_body_names(env, actor)
        print("body_world_positions:")
        for i, name in enumerate(body_names):
            print(f"  {i:02d} {name}: {rigid_body_states[i, :3].detach().cpu().tolist()}")

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(2.0, -2.0, args.robot_z + 0.7),
            gymapi.Vec3(args.robot_x, args.robot_y, args.robot_z),
        )

    for step in range(args.steps):
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        gym.simulate(sim)
        if not args.sim_device.startswith("cuda"):
            gym.fetch_results(sim, True)
        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        if step == 0:
            print("[step 0] actor created")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
