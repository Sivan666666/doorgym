import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_ROOT = ROOT / "low-level"
if str(LOW_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOW_LEVEL_ROOT))

from isaacgym import gymapi  # noqa: E402
import isaacgym  # noqa: F401,E402


READY_POSE = {
    "joint1": 0.0,
    "joint2": 1.48,
    "joint3": -0.63,
    "joint4": -0.84,
    "joint5": 0.0,
    "joint6": 1.57,
    "jointGripper": -0.785,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plain Isaac Gym asset viewer for b1z1_basearn.urdf.")
    parser.add_argument("--sim_device", type=str, default="cpu")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--asset_root", type=str, default=str(LOW_LEVEL_ROOT / "resources" / "robots" / "b1z1"))
    parser.add_argument("--asset_file", type=str, default="urdf/b1z1_basearn.urdf")
    parser.add_argument("--robot_x", type=float, default=0.0)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--robot_z", type=float, default=0.12)
    parser.add_argument("--robot_yaw", type=float, default=0.0)
    parser.add_argument("--pose", choices=("zero", "ready"), default="zero")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--collapse_fixed_joints", action="store_true")
    parser.add_argument("--free_base", dest="fix_base_link", action="store_false")
    parser.set_defaults(fix_base_link=True)
    parser.add_argument("--enable_gravity", dest="disable_gravity", action="store_false")
    parser.set_defaults(disable_gravity=True)
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--camera_eye", type=float, nargs=3, default=(1.4, -1.5, 0.85))
    parser.add_argument("--camera_target", type=float, nargs=3, default=(0.1, 0.0, 0.28))
    parser.add_argument("--screenshot", type=str, default="")
    parser.add_argument("--print_body_states", action="store_true")
    return parser.parse_args()


def resolve_asset_root(asset_root):
    path = Path(asset_root).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def make_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = args.dt
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.sim_device.startswith("cuda")
    sim_params.physx.use_gpu = args.sim_device.startswith("cuda")

    sim_device_id = int(args.sim_device.split(":")[-1]) if args.sim_device.startswith("cuda") else 0
    graphics_device_id = -1 if args.headless and not args.screenshot else args.graphics_device_id
    sim = gym.create_sim(sim_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    return sim


def load_robot_asset(gym, sim, args):
    asset_root = resolve_asset_root(args.asset_root)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = args.fix_base_link
    asset_options.disable_gravity = args.disable_gravity
    asset_options.collapse_fixed_joints = args.collapse_fixed_joints
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True

    robot_asset = gym.load_asset(sim, str(asset_root), args.asset_file, asset_options)
    if robot_asset is None:
        raise RuntimeError(f"Failed to load {asset_root / args.asset_file}")

    print("------------------------------------------------------")
    print("asset:", asset_root / args.asset_file)
    print("normal_load: plain gym.load_asset + minimal AssetOptions")
    print("fix_base_link:", args.fix_base_link)
    print("disable_gravity:", args.disable_gravity)
    print("collapse_fixed_joints:", args.collapse_fixed_joints)
    print("num_dofs:", gym.get_asset_dof_count(robot_asset))
    print("dof_names:", gym.get_asset_dof_names(robot_asset))
    print("num_bodies:", gym.get_asset_rigid_body_count(robot_asset))
    print("body_names:", gym.get_asset_rigid_body_names(robot_asset))
    print("------------------------------------------------------")
    return robot_asset


def get_initial_dof_positions(dof_names, dof_props, pose_name):
    values = np.zeros(len(dof_names), dtype=np.float32)
    if pose_name == "ready":
        for i, name in enumerate(dof_names):
            values[i] = READY_POSE.get(name, 0.0)
    if len(values) > 0:
        values = np.clip(values, dof_props["lower"], dof_props["upper"])
    return values


def set_actor_dof_positions(gym, env, actor, values):
    if len(values) == 0:
        return
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states["pos"][:] = values
    dof_states["vel"][:] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)


def animated_dof_positions(step, args, base_values, dof_names, dof_props):
    values = base_values.copy()
    if not args.animate:
        return values
    t = step * args.dt
    for i, name in enumerate(dof_names):
        if name not in ("joint1", "joint2", "joint3", "joint4", "joint6", "jointGripper"):
            continue
        values[i] += 0.18 * math.sin(2.0 * math.pi * 0.35 * t + i * 0.4)
    return np.clip(values, dof_props["lower"], dof_props["upper"])


def create_viewer(gym, sim, args):
    if args.headless:
        return None
    props = gymapi.CameraProperties()
    props.width = args.camera_width
    props.height = args.camera_height
    viewer = gym.create_viewer(sim, props)
    if viewer is None:
        raise RuntimeError("Failed to create viewer")
    gym.viewer_camera_look_at(
        viewer,
        None,
        gymapi.Vec3(*args.camera_eye),
        gymapi.Vec3(*args.camera_target),
    )
    return viewer


def create_screenshot_camera(gym, env, args):
    if not args.screenshot:
        return None
    props = gymapi.CameraProperties()
    props.width = args.camera_width
    props.height = args.camera_height
    camera = gym.create_camera_sensor(env, props)
    gym.set_camera_location(
        camera,
        env,
        gymapi.Vec3(*args.camera_eye),
        gymapi.Vec3(*args.camera_target),
    )
    return camera


def write_screenshot(gym, sim, env, camera, output_path):
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    gym.write_camera_image_to_file(sim, env, camera, gymapi.IMAGE_COLOR, str(output))
    print("screenshot:", output)


def main():
    args = parse_args()
    os.chdir(ROOT)

    if args.animate and args.sim_device.startswith("cuda"):
        print("Warning: --animate is intended for this plain CPU viewer. Use --sim_device cpu for animation.")

    gym = gymapi.acquire_gym()
    sim = make_sim(gym, args)
    robot_asset = load_robot_asset(gym, sim, args)

    env = gym.create_env(sim, gymapi.Vec3(-1.5, -1.5, 0.0), gymapi.Vec3(1.5, 1.5, 1.5), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(args.robot_x, args.robot_y, args.robot_z)
    pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, args.robot_yaw)
    actor = gym.create_actor(env, robot_asset, pose, "b1z1_basearn", 0, 0, 0)

    dof_names = gym.get_actor_dof_names(env, actor)
    dof_props = gym.get_actor_dof_properties(env, actor)
    base_dof_positions = get_initial_dof_positions(dof_names, dof_props, args.pose)
    set_actor_dof_positions(gym, env, actor, base_dof_positions)

    if args.print_body_states:
        rigid_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
        print("body_world_positions:")
        for i, name in enumerate(gym.get_actor_rigid_body_names(env, actor)):
            pos = rigid_states["pose"]["p"][i]
            print(f"  {i:02d} {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    viewer = create_viewer(gym, sim, args)
    camera = create_screenshot_camera(gym, env, args)

    print("spawn_pose_xyz:", [args.robot_x, args.robot_y, args.robot_z])
    print("viewer: close the Isaac Gym window to exit.")
    wrote_screenshot = False
    for step in range(max(1, args.steps)):
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        if args.animate and not args.sim_device.startswith("cuda"):
            values = animated_dof_positions(step, args, base_dof_positions, dof_names, dof_props)
            set_actor_dof_positions(gym, env, actor, values)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if viewer is not None or camera is not None:
            gym.step_graphics(sim)
        if camera is not None:
            gym.render_all_camera_sensors(sim)
            if not wrote_screenshot:
                write_screenshot(gym, sim, env, camera, args.screenshot)
                wrote_screenshot = True
                if args.headless:
                    break
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
