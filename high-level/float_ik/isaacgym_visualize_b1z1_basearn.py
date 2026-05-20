#!/usr/bin/env python3
"""Load b1z1_basearn.urdf in Isaac Gym and animate its arm DOFs."""

from __future__ import annotations

import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DOORGYM_ROOT = REPO_ROOT.parent
DEFAULT_ASSET_ROOT = DOORGYM_ROOT / "low-level" / "resources" / "robots" / "b1z1"
DEFAULT_ASSET_FILE = "urdf/b1z1_basearn.urdf"
ARM_ACTOR_NAME = "z1_arm_articulated"
EE_GRIPPER_LINK = "ee_gripper_link"
DEFAULT_IK_DEMO_OFFSET = "0.12 0.16 0.03"
GRIPPER_FIX_MESH_SOURCE = REPO_ROOT / "data" / "asset" / "b1z1-col" / "meshes" / "gripper_fix_90d.obj"
ARM_IK_DOF_NAMES = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}
BASE_LINKS = {"base", "trunk"}
ARM_LINKS = {
    "base",
    "link00",
    "link01",
    "link02",
    "link03",
    "link04",
    "link05",
    "link06",
    "gripperStator",
    EE_GRIPPER_LINK,
    "gripperMover",
}
EE_GRIPPER_JOINT_XML = """
  <joint name="ee_gripper" type="fixed" dont_collapse="true">
    <axis xyz="1 0 0" />
    <origin rpy="0 0 0" xyz="0.135 0 0" />
    <parent link="gripperStator" />
    <child link="ee_gripper_link" />
  </joint>
  <link name="ee_gripper_link">
    <inertial>
      <mass value="0.001" />
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001" />
    </inertial>
  </link>
"""


def import_isaacgym():
    """Import Isaac Gym, falling back to local source trees when needed."""
    candidates = [
        Path("/home/sivan/whole_body/visual_whole_body/third_party/isaacgym/python"),
        REPO_ROOT / "low_level_WBC/third_party/isaacgym/python",
        REPO_ROOT / "arm_pushing_policy/src/isaacgym/python",
        Path("/home/sivan/Downloads/IsaacGym_Preview_4_Package/isaacgym/python"),
    ]

    last_error: BaseException | None = None
    for idx, candidate in enumerate([None, *candidates]):
        if candidate is not None:
            if not candidate.exists():
                continue
            sys.path.insert(0, str(candidate))

        try:
            from isaacgym import gymapi, gymutil

            return gymapi, gymutil
        except BaseException as exc:
            last_error = exc
            for name in list(sys.modules):
                if name == "isaacgym" or name.startswith("isaacgym."):
                    del sys.modules[name]
            if idx == 0:
                continue

    raise SystemExit(
        "Failed to import Isaac Gym. Use a Python 3.6/3.7/3.8 Isaac Gym env, for example:\n"
        "  /home/sivan/miniconda3/envs/b1z1/bin/python "
        "scripts/isaacgym_visualize_b1z1_basearn.py\n"
        f"Last import error: {last_error}"
    )


gymapi, gymutil = import_isaacgym()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_args():
    args = gymutil.parse_arguments(
        description="Visualize b1z1_basearn.urdf in Isaac Gym and sweep each arm joint.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {
                "name": "--asset_root",
                "type": str,
                "default": str(DEFAULT_ASSET_ROOT),
                "help": "Asset root passed to gym.load_asset",
            },
            {
                "name": "--asset_file",
                "type": str,
                "default": DEFAULT_ASSET_FILE,
                "help": "URDF path relative to --asset_root",
            },
            {
                "name": "--steps",
                "type": int,
                "default": 0,
                "help": "Number of sim steps to run. 0 means run until the viewer is closed.",
            },
            {
                "name": "--speed_scale",
                "type": float,
                "default": 0.6,
                "help": "Joint sweep speed multiplier",
            },
            {
                "name": "--stiffness",
                "type": float,
                "default": 80.0,
                "help": "Position drive stiffness for all DOFs",
            },
            {
                "name": "--damping",
                "type": float,
                "default": 8.0,
                "help": "Position drive damping for all DOFs",
            },
            {
                "name": "--range_scale",
                "type": float,
                "default": 0.75,
                "help": "Fraction of each URDF joint range to sweep away from the zero pose",
            },
            {
                "name": "--zero_pose_seconds",
                "type": float,
                "default": 2.0,
                "help": "Seconds to hold the URDF zero pose before animating",
            },
            {
                "name": "--zero_pose_only",
                "action": "store_true",
                "help": "Only show the URDF zero pose; do not animate joints",
            },
            {
                "name": "--joint_filter",
                "type": str,
                "default": "",
                "help": "Only animate DOFs whose names contain this substring",
            },
            {
                "name": "--show_axis",
                "action": "store_true",
                "help": "Draw the currently animated DOF axis in red",
            },
            {
                "name": "--disable_self_collisions",
                "action": "store_true",
                "help": "Use collision filter bit 1, which disables same-filter shape collisions",
            },
            {
                "name": "--print_collision_summary",
                "action": "store_true",
                "help": "Print collision shape count for every rigid body",
            },
            {
                "name": "--disable_base_motion",
                "action": "store_true",
                "help": "Disable the kinematic base pose demo",
            },
            {
                "name": "--base_motion_amplitude",
                "type": float,
                "default": 0.16,
                "help": "XY translation amplitude for the base pose demo, in meters",
            },
            {
                "name": "--base_motion_yaw",
                "type": float,
                "default": 0.28,
                "help": "Yaw amplitude for the base pose demo, in radians",
            },
            {
                "name": "--base_motion_period",
                "type": float,
                "default": 6.0,
                "help": "Base pose demo period, in seconds",
            },
            {
                "name": "--ik_demo",
                "action": "store_true",
                "help": "Move the end effector to an automatically generated nearby IK target",
            },
            {
                "name": "--ik_target_pose",
                "type": str,
                "default": "",
                "help": (
                    "Enable IK and set a world target. Use quoted values: "
                    "'x y z' for position-only, 'x y z roll pitch yaw', or "
                    "'x y z qx qy qz qw'. Angles are radians."
                ),
            },
            {
                "name": "--ik_demo_offset",
                "type": str,
                "default": DEFAULT_IK_DEMO_OFFSET,
                "help": "With --ik_demo and no explicit target, add this xyz world offset to the initial EE pose",
            },
            {
                "name": "--ik_ee_link",
                "type": str,
                "default": EE_GRIPPER_LINK,
                "help": "Rigid body used as the IK end effector",
            },
            {
                "name": "--ik_include_gripper",
                "action": "store_true",
                "help": "Let jointGripper participate in IK. By default only joint1-joint6 are controlled",
            },
            {
                "name": "--ik_position_only",
                "action": "store_true",
                "help": "Ignore target orientation even when --ik_target_pose includes orientation",
            },
            {
                "name": "--ik_keep_base_motion",
                "action": "store_true",
                "help": "Keep the base pose demo running during IK. By default IK disables it for a fixed world target",
            },
            {
                "name": "--ik_pos_gain",
                "type": float,
                "default": 1.0,
                "help": "IK position error gain",
            },
            {
                "name": "--ik_rot_gain",
                "type": float,
                "default": 0.7,
                "help": "IK orientation error gain",
            },
            {
                "name": "--ik_rot_weight",
                "type": float,
                "default": 0.1,
                "help": "Relative weight for orientation rows in full-pose IK; increase for stricter orientation tracking",
            },
            {
                "name": "--ik_damping",
                "type": float,
                "default": 0.08,
                "help": "Damped least-squares IK damping",
            },
            {
                "name": "--ik_max_step",
                "type": float,
                "default": 0.06,
                "help": "Maximum per-frame IK joint target change in radians",
            },
            {
                "name": "--ik_pos_tolerance",
                "type": float,
                "default": 0.015,
                "help": "Position tolerance used for the one-time IK reached message",
            },
            {
                "name": "--ik_rot_tolerance",
                "type": float,
                "default": 0.08,
                "help": "Orientation tolerance used for the one-time IK reached message",
            },
            {
                "name": "--single_asset",
                "action": "store_true",
                "help": "Load the original URDF as one actor instead of split base/arm visual actors",
            },
            {
                "name": "--flip_visual_attachments",
                "action": "store_true",
                "help": "With --single_asset, set AssetOptions.flip_visual_attachments=True globally",
            },
            {
                "name": "--disable_arm_visual_flip",
                "action": "store_true",
                "help": "In split mode, set AssetOptions.flip_visual_attachments=False for the arm actor",
            },
            {
                "name": "--base_visual_flip",
                "action": "store_true",
                "help": "In split mode, set AssetOptions.flip_visual_attachments=True for the base actor",
            },
            {
                "name": "--no_disable_gravity",
                "action": "store_true",
                "help": "Keep gravity enabled on the imported asset",
            },
        ],
    )
    return args


def create_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    # Keep the CPU tensor/API pipeline for this viewer utility.  The default
    # sim device is still cuda:0, so PhysX itself runs on the GPU.
    sim_params.use_gpu_pipeline = False
    args.use_gpu_pipeline = False
    args.pipeline = "CPU"
    if args.use_gpu:
        print(f"Using GPU PhysX on {args.sim_device}; CPU API pipeline for viewer controls.")

    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")
    return sim, sim_params.dt


def find_child(node, tag):
    for child in node:
        if child.tag == tag:
            return child
    return None


def append_ee_gripper_link(output_root):
    existing_links = {
        child.get("name") for child in output_root if child.tag == "link" and child.get("name")
    }
    existing_joints = {
        child.get("name") for child in output_root if child.tag == "joint" and child.get("name")
    }
    if EE_GRIPPER_LINK in existing_links or "ee_gripper" in existing_joints:
        return

    ee_root = ET.fromstring(f"<robot>{EE_GRIPPER_JOINT_XML}</robot>")
    for child in list(ee_root):
        output_root.append(child)


def collision_box(xyz, rpy, size):
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": rpy})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": size})
    return collision


def collision_mesh(xyz, rpy, filename, scale="1 1 1"):
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": rpy})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": filename, "scale": scale})
    return collision


def replace_link_collisions(link_node, collisions):
    for child in list(link_node):
        if child.tag == "collision":
            link_node.remove(child)

    insert_index = 0
    for idx, child in enumerate(list(link_node)):
        if child.tag == "visual":
            insert_index = idx + 1

    for collision in collisions:
        link_node.insert(insert_index, collision)
        insert_index += 1


def align_gripper_collisions_with_flipped_visuals(output_root):
    """Use the locally corrected gripper collision geometry for flipped Z1 visuals."""
    for link in output_root.findall("link"):
        link_name = link.get("name")
        if link_name == "gripperStator":
            replace_link_collisions(
                link,
                [
                    collision_box("0.058 0 -0.028", "0 0 0", "0.08 0.08 0.02"),
                    collision_box("0.11 0 -0.024", "0 -0.2967 0", "0.03 0.08 0.02"),
                    collision_box("0.135 0 -0.02", "0 0 0", "0.03 0.06 0.02"),
                    collision_mesh("0 0 0", "0 0 0", "../meshes/gripper_fix_90d.obj"),
                ],
            )
        elif link_name == "gripperMover":
            replace_link_collisions(
                link,
                [
                    collision_box("0.026 0 0.012", "0 0 0", "0.05 0.08 0.024"),
                    collision_box("0.065 0 0.007", "0 0.384 0", "0.032 0.08 0.022"),
                    collision_box("0.085 0 0", "0 0 0", "0.026 0.06 0.018"),
                ],
            )


def write_filtered_urdf(
    source_urdf,
    output_urdf,
    keep_links,
    add_ee_gripper=False,
    align_gripper_collisions=False,
):
    source_tree = ET.parse(source_urdf)
    source_root = source_tree.getroot()
    output_root = ET.Element(source_root.tag, source_root.attrib)

    for child in source_root:
        if child.tag == "material":
            output_root.append(child)
        elif child.tag == "link" and child.get("name") in keep_links:
            output_root.append(child)
        elif child.tag == "joint":
            parent = find_child(child, "parent")
            child_link = find_child(child, "child")
            if parent is None or child_link is None:
                continue
            parent_name = parent.get("link")
            child_name = child_link.get("link")
            if parent_name in keep_links and child_name in keep_links:
                output_root.append(child)

    if add_ee_gripper:
        append_ee_gripper_link(output_root)
    if align_gripper_collisions:
        align_gripper_collisions_with_flipped_visuals(output_root)

    ET.ElementTree(output_root).write(output_urdf, encoding="utf-8", xml_declaration=True)


def write_augmented_urdf(source_urdf, output_urdf, align_gripper_collisions=False):
    source_tree = ET.parse(source_urdf)
    source_root = source_tree.getroot()
    output_root = ET.Element(source_root.tag, source_root.attrib)
    for child in source_root:
        output_root.append(child)
    append_ee_gripper_link(output_root)
    if align_gripper_collisions:
        align_gripper_collisions_with_flipped_visuals(output_root)
    ET.ElementTree(output_root).write(output_urdf, encoding="utf-8", xml_declaration=True)


def populate_mesh_dir(source_mesh_dir, mesh_dir):
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_mesh_dir.iterdir():
        target_path = mesh_dir / source_path.name
        if target_path.exists():
            continue
        target_path.symlink_to(source_path)

    if GRIPPER_FIX_MESH_SOURCE.exists():
        target_path = mesh_dir / GRIPPER_FIX_MESH_SOURCE.name
        if not target_path.exists():
            target_path.symlink_to(GRIPPER_FIX_MESH_SOURCE)


def build_split_asset_root(source_asset_root, asset_file, temp_root, align_arm_gripper_collisions=False):
    source_urdf = source_asset_root / asset_file
    split_root = temp_root / "b1z1_split_assets"
    split_urdf_dir = split_root / "urdf"
    split_urdf_dir.mkdir(parents=True, exist_ok=True)

    populate_mesh_dir(source_asset_root / "meshes", split_root / "meshes")

    base_file = "urdf/b1_base_visual_only.urdf"
    arm_file = "urdf/z1_arm_visual_flip.urdf"
    write_filtered_urdf(source_urdf, split_root / base_file, BASE_LINKS)
    write_filtered_urdf(
        source_urdf,
        split_root / arm_file,
        ARM_LINKS,
        add_ee_gripper=True,
        align_gripper_collisions=align_arm_gripper_collisions,
    )
    return split_root, base_file, arm_file


def build_augmented_asset_root(source_asset_root, asset_file, temp_root, align_gripper_collisions=False):
    source_urdf = source_asset_root / asset_file
    augmented_root = temp_root / "b1z1_augmented_assets"
    augmented_urdf_dir = augmented_root / "urdf"
    augmented_urdf_dir.mkdir(parents=True, exist_ok=True)

    populate_mesh_dir(source_asset_root / "meshes", augmented_root / "meshes")

    augmented_file = "urdf/b1z1_basearn_with_ee_gripper.urdf"
    write_augmented_urdf(
        source_urdf,
        augmented_root / augmented_file,
        align_gripper_collisions=align_gripper_collisions,
    )
    return augmented_root, augmented_file


def make_asset_options(args, flip_visual_attachments):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = not args.no_disable_gravity
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.use_mesh_materials = True
    asset_options.flip_visual_attachments = flip_visual_attachments
    asset_options.thickness = 0.001
    asset_options.armature = 0.01
    return asset_options


def load_asset_with_visual_flip(gym, sim, asset_root, asset_file, args, flip_visual_attachments, label):
    asset_root = Path(asset_root).expanduser().resolve()
    urdf_path = asset_root / asset_file
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    asset_options = make_asset_options(args, flip_visual_attachments)
    print(
        f"Loading {label}: root={asset_root}, file={asset_file}, "
        f"flip_visual_attachments={flip_visual_attachments}"
    )
    asset = gym.load_asset(sim, str(asset_root), asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"gym.load_asset returned None for {label}")
    return asset


def load_robot_assets(gym, sim, args, temp_root):
    asset_root = Path(args.asset_root).expanduser().resolve()
    asset_file = args.asset_file
    urdf_path = asset_root / asset_file
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    if args.single_asset:
        if ik_requested(args):
            asset_root, asset_file = build_augmented_asset_root(
                asset_root,
                asset_file,
                temp_root,
                align_gripper_collisions=args.flip_visual_attachments,
            )
        robot_asset = load_asset_with_visual_flip(
            gym,
            sim,
            asset_root,
            asset_file,
            args,
            args.flip_visual_attachments,
            "single robot asset",
        )
        return None, robot_asset

    arm_visual_flip = not args.disable_arm_visual_flip
    split_root, base_file, arm_file = build_split_asset_root(
        asset_root,
        asset_file,
        temp_root,
        align_arm_gripper_collisions=arm_visual_flip,
    )
    base_asset = load_asset_with_visual_flip(
        gym,
        sim,
        split_root,
        base_file,
        args,
        args.base_visual_flip,
        "base visual actor",
    )
    arm_asset = load_asset_with_visual_flip(
        gym,
        sim,
        split_root,
        arm_file,
        args,
        arm_visual_flip,
        "arm articulated actor",
    )
    return base_asset, arm_asset


def print_collision_summary(gym, asset, label, *, verbose=False):
    body_names = gym.get_asset_rigid_body_names(asset)
    shape_ranges = gym.get_asset_rigid_body_shape_indices(asset)
    total_shapes = gym.get_asset_rigid_shape_count(asset)
    missing = []

    print(f"{label} collision shapes: {total_shapes}")
    for body_name, shape_range in zip(body_names, shape_ranges):
        count = int(shape_range.count)
        if count == 0:
            missing.append(body_name)
        if verbose or count == 0:
            print(f"  {body_name:16s} collision_shapes={count}")

    if missing:
        print(f"  bodies without collision: {', '.join(missing)}")


def configure_dofs(gym, asset, args):
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

    dof_props["driveMode"].fill(int(gymapi.DOF_MODE_POS))
    dof_props["stiffness"].fill(args.stiffness)
    dof_props["damping"].fill(args.damping)

    lower_limits = np.array(dof_props["lower"], dtype=np.float32)
    upper_limits = np.array(dof_props["upper"], dtype=np.float32)
    has_limits = np.array(dof_props["hasLimits"], dtype=bool)

    for i in range(num_dofs):
        if not has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            else:
                lower_limits[i] = -0.5
                upper_limits[i] = 0.5
        elif lower_limits[i] >= upper_limits[i]:
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi

    defaults = np.zeros(num_dofs, dtype=np.float32)
    for i in range(num_dofs):
        defaults[i] = clamp(0.0, float(lower_limits[i]), float(upper_limits[i]))

    range_scale = clamp(args.range_scale, 0.05, 1.0)
    sweep_lower = defaults - (defaults - lower_limits) * range_scale
    sweep_upper = defaults + (upper_limits - defaults) * range_scale

    speeds = np.zeros(num_dofs, dtype=np.float32)
    for i in range(num_dofs):
        span = float(sweep_upper[i] - sweep_lower[i])
        if dof_types[i] == gymapi.DOF_ROTATION:
            speeds[i] = args.speed_scale * clamp(1.3 * span, 0.25 * math.pi, 2.0 * math.pi)
        else:
            speeds[i] = args.speed_scale * clamp(1.3 * span, 0.05, 2.0)

    print(f"Asset has {num_dofs} DOFs:")
    for i, name in enumerate(dof_names):
        print(
            f"  [{i:02d}] {name:16s} type={gym.get_dof_type_string(dof_types[i]):11s} "
            f"urdf=[{dof_props['lower'][i]: .3f}, {dof_props['upper'][i]: .3f}] "
            f"default={defaults[i]: .3f} sweep=[{sweep_lower[i]: .3f}, {sweep_upper[i]: .3f}]"
        )

    selected = [
        i for i, name in enumerate(dof_names) if not args.joint_filter or args.joint_filter in name
    ]
    if not selected:
        raise RuntimeError(f"No DOFs matched --joint_filter={args.joint_filter!r}")
    selected_names = ", ".join(dof_names[i] for i in selected)
    if ik_requested(args):
        print("Joint sweep DOFs if IK is off:", selected_names)
    else:
        print("Animating DOFs:", selected_names)

    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    dof_positions = dof_states["pos"]
    dof_positions[:] = defaults
    return dof_names, dof_props, dof_states, dof_positions, sweep_lower, sweep_upper, defaults, speeds, selected


def create_env_and_actor(gym, sim, base_asset, arm_asset, dof_props, dof_states, args):
    env = gym.create_env(sim, gymapi.Vec3(-1.2, -1.2, 0.0), gymapi.Vec3(1.2, 1.2, 1.2), 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.35)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handles = []
    collision_filter = 1 if args.disable_self_collisions else 0
    print(f"Actor collision filter={collision_filter} (0 enables same-actor shape collisions).")
    if base_asset is not None:
        base_actor = gym.create_actor(env, base_asset, pose, "b1_base_visual", 0, collision_filter)
        actor_handles.append(base_actor)

    actor = gym.create_actor(env, arm_asset, pose, ARM_ACTOR_NAME, 0, collision_filter)
    actor_handles.append(actor)
    gym.set_actor_dof_properties(env, actor, dof_props)
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, dof_states["pos"])
    return env, actor, actor_handles


def setup_viewer(gym, sim, args):
    if args.headless:
        return None

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer. Try --headless to test loading only.")

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "pause")

    cam_pos = gymapi.Vec3(1.25, -1.10, 0.85)
    cam_target = gymapi.Vec3(0.18, 0.00, 0.36)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    return viewer


def yaw_quat(yaw):
    half = 0.5 * yaw
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32)


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / norm


def rpy_to_quat(roll, pitch, yaw):
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return normalize_quat(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def parse_float_values(text, expected_counts, label):
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) not in expected_counts:
        counts = "/".join(str(count) for count in expected_counts)
        raise ValueError(f"{label} expects {counts} values, got {len(values)}: {text!r}")
    return np.array(values, dtype=np.float32)


def parse_ik_target_pose(text):
    if not text.strip():
        return None, None

    values = parse_float_values(text, {3, 6, 7}, "--ik_target_pose")
    target_pos = values[:3]
    if len(values) == 3:
        return target_pos, None
    if len(values) == 6:
        return target_pos, rpy_to_quat(*values[3:6])
    return target_pos, normalize_quat(values[3:7])


def transform_from_arrays(position, quat=None):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(position[0]), float(position[1]), float(position[2]))
    if quat is None:
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    else:
        pose.r = gymapi.Quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return pose


def torch_quat_conjugate(torch, q):
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def torch_quat_multiply(torch, q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
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
    q_r = torch_quat_multiply(torch, desired, torch_quat_conjugate(torch, current))
    return q_r[..., :3] * torch.sign(q_r[..., 3:4])


def rotate_xy(points, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotated = np.empty_like(points)
    rotated[:, 0] = c * points[:, 0] - s * points[:, 1]
    rotated[:, 1] = s * points[:, 0] + c * points[:, 1]
    rotated[:, 2] = points[:, 2]
    return rotated


def base_motion_pose(args, elapsed):
    if args.disable_base_motion:
        return np.zeros(3, dtype=np.float32), 0.0

    period = max(0.1, args.base_motion_period)
    phase = 2.0 * math.pi * elapsed / period
    amp = args.base_motion_amplitude
    translation = np.array(
        [
            amp * math.sin(phase),
            0.45 * amp * math.sin(0.5 * phase),
            0.0,
        ],
        dtype=np.float32,
    )
    yaw = args.base_motion_yaw * math.sin(phase)
    return translation, yaw


def apply_base_motion_delta(gym, env, actor_handles, prev_translation, prev_yaw, next_translation, next_yaw):
    if not actor_handles:
        return

    delta_yaw = next_yaw - prev_yaw
    rotated_prev_translation = rotate_xy(prev_translation.reshape(1, 3), delta_yaw)[0]
    delta_translation = next_translation - rotated_prev_translation
    delta_quat = yaw_quat(delta_yaw)

    for actor in actor_handles:
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        point = np.array(
            [[states["pose"]["p"]["x"][0], states["pose"]["p"]["y"][0], states["pose"]["p"]["z"][0]]],
            dtype=np.float32,
        )
        moved_point = rotate_xy(point, delta_yaw)[0] + delta_translation

        quat = np.array(
            [
                states["pose"]["r"]["x"][0],
                states["pose"]["r"]["y"][0],
                states["pose"]["r"]["z"][0],
                states["pose"]["r"]["w"][0],
            ],
            dtype=np.float32,
        )
        moved_quat = quat_multiply(delta_quat, quat)

        rigid_handle = gym.get_actor_root_rigid_body_handle(env, actor)
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(float(moved_point[0]), float(moved_point[1]), float(moved_point[2]))
        transform.r = gymapi.Quat(
            float(moved_quat[0]),
            float(moved_quat[1]),
            float(moved_quat[2]),
            float(moved_quat[3]),
        )
        gym.set_rigid_transform(env, rigid_handle, transform)


@dataclass
class IKState:
    torch: object
    jacobian: object
    rb_states: object
    dof_state_tensor: object
    target_pos: object
    target_quat: object | None
    target_pos_np: np.ndarray
    target_quat_np: np.ndarray | None
    eef_body_sim_index: int
    eef_jacobian_index: int
    control_indices: object
    lower: object
    upper: object
    eef_link: str
    control_dof_names: list[str]
    reached_reported: bool = False
    last_pos_error: float = math.inf
    last_rot_error: float | None = None
    current_pos_np: np.ndarray | None = None


def ik_requested(args):
    return args.ik_demo or bool(args.ik_target_pose.strip())


def configure_mode_defaults(args):
    if ik_requested(args) and not args.ik_keep_base_motion and not args.disable_base_motion:
        args.disable_base_motion = True
        print("IK mode: disabling base pose demo so the world target stays fixed relative to the arm.")


def setup_ik_controller(gym, sim, env, actor, asset, dof_names, lower, upper, args):
    if not ik_requested(args):
        return None

    try:
        import torch
        from isaacgym import gymtorch
    except ImportError as exc:
        raise RuntimeError("IK mode needs PyTorch and isaacgym.gymtorch in the active env") from exc

    gym.prepare_sim(sim)
    jacobian = gymtorch.wrap_tensor(gym.acquire_jacobian_tensor(sim, ARM_ACTOR_NAME))
    rb_states = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    dof_state_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    body_names = gym.get_asset_rigid_body_names(asset)
    if args.ik_ee_link not in body_names:
        raise RuntimeError(
            f"--ik_ee_link={args.ik_ee_link!r} is not in the asset. "
            f"Available bodies: {', '.join(body_names)}"
        )

    eef_asset_index = body_names.index(args.ik_ee_link)
    eef_jacobian_index = eef_asset_index - 1
    if eef_jacobian_index < 0 or eef_jacobian_index >= jacobian.shape[1]:
        raise RuntimeError(
            f"Cannot use root body {args.ik_ee_link!r} as an IK target; choose a child link."
        )

    control_indices_np = np.array(
        [
            i
            for i, name in enumerate(dof_names)
            if name in ARM_IK_DOF_NAMES or (args.ik_include_gripper and name == "jointGripper")
        ],
        dtype=np.int64,
    )
    if len(control_indices_np) == 0:
        raise RuntimeError("No DOFs selected for IK control")

    eef_body_sim_index = gym.find_actor_rigid_body_index(
        env, actor, args.ik_ee_link, gymapi.DOMAIN_SIM
    )
    current_pos_np = rb_states[eef_body_sim_index, :3].detach().cpu().numpy().copy()
    current_quat_np = rb_states[eef_body_sim_index, 3:7].detach().cpu().numpy().copy()
    target_pos_np, target_quat_np = parse_ik_target_pose(args.ik_target_pose)
    if target_pos_np is None:
        offset = parse_float_values(args.ik_demo_offset, {3}, "--ik_demo_offset")
        target_pos_np = current_pos_np + offset
        target_quat_np = None

    if args.ik_position_only:
        target_quat_np = None

    device = jacobian.device
    target_pos = torch.tensor(target_pos_np, dtype=torch.float32, device=device)
    target_quat = None
    if target_quat_np is not None:
        target_quat = torch.tensor(target_quat_np, dtype=torch.float32, device=device)
        target_quat = target_quat / torch.clamp(torch.linalg.norm(target_quat), min=1.0e-8)

    control_indices = torch.tensor(control_indices_np, dtype=torch.long, device=device)
    lower_tensor = torch.tensor(lower, dtype=torch.float32, device=device)
    upper_tensor = torch.tensor(upper, dtype=torch.float32, device=device)
    control_dof_names = [dof_names[i] for i in control_indices_np]

    print("IK mode enabled; joint sweep disabled.")
    print(f"  end effector: {args.ik_ee_link}")
    print(f"  controlled DOFs: {', '.join(control_dof_names)}")
    print(
        "  initial ee pose xyz+quat: "
        f"{current_pos_np[0]:.3f}, {current_pos_np[1]:.3f}, {current_pos_np[2]:.3f}, "
        f"{current_quat_np[0]:.3f}, {current_quat_np[1]:.3f}, "
        f"{current_quat_np[2]:.3f}, {current_quat_np[3]:.3f}"
    )
    if target_quat_np is None:
        print(f"  target xyz: {target_pos_np[0]:.3f}, {target_pos_np[1]:.3f}, {target_pos_np[2]:.3f}")
    else:
        print(
            "  target pose xyz+quat: "
            f"{target_pos_np[0]:.3f}, {target_pos_np[1]:.3f}, {target_pos_np[2]:.3f}, "
            f"{target_quat_np[0]:.3f}, {target_quat_np[1]:.3f}, "
            f"{target_quat_np[2]:.3f}, {target_quat_np[3]:.3f}"
        )

    return IKState(
        torch=torch,
        jacobian=jacobian,
        rb_states=rb_states,
        dof_state_tensor=dof_state_tensor,
        target_pos=target_pos,
        target_quat=target_quat,
        target_pos_np=target_pos_np.copy(),
        target_quat_np=None if target_quat_np is None else target_quat_np.copy(),
        eef_body_sim_index=eef_body_sim_index,
        eef_jacobian_index=eef_jacobian_index,
        control_indices=control_indices,
        lower=lower_tensor,
        upper=upper_tensor,
        eef_link=args.ik_ee_link,
        control_dof_names=control_dof_names,
        current_pos_np=current_pos_np.copy(),
    )


def update_ik_targets(gym, sim, dof_positions, ik_state, args):
    torch = ik_state.torch
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    eef_state = ik_state.rb_states[ik_state.eef_body_sim_index]
    eef_pos = eef_state[:3]
    eef_quat = eef_state[3:7]
    pos_err = ik_state.target_pos - eef_pos

    j_eef = ik_state.jacobian[0, ik_state.eef_jacobian_index, :, :]
    j_control = j_eef[:, ik_state.control_indices]

    if ik_state.target_quat is None:
        task_j = j_control[:3, :]
        task_err = args.ik_pos_gain * pos_err
        rot_err_norm = None
    else:
        orn_err = torch_orientation_error(torch, ik_state.target_quat, eef_quat)
        dpose = torch.cat((args.ik_pos_gain * pos_err, args.ik_rot_gain * orn_err), dim=0)
        weights = torch.tensor(
            [1.0, 1.0, 1.0, args.ik_rot_weight, args.ik_rot_weight, args.ik_rot_weight],
            dtype=torch.float32,
            device=j_control.device,
        )
        task_j = j_control * weights.view(6, 1)
        task_err = dpose * weights
        rot_err_norm = float(torch.linalg.norm(orn_err).detach().cpu())

    j_t = torch.transpose(task_j, 0, 1)
    damping = max(1.0e-6, float(args.ik_damping))
    lhs = task_j @ j_t + torch.eye(task_j.shape[0], dtype=torch.float32, device=task_j.device) * (
        damping * damping
    )
    delta = j_t @ torch.linalg.solve(lhs, task_err.unsqueeze(-1)).squeeze(-1)
    max_step = max(1.0e-6, float(args.ik_max_step))
    delta = torch.clamp(delta, -max_step, max_step)

    current_q = ik_state.dof_state_tensor[:, 0].clone()
    next_q = current_q.clone()
    next_q[ik_state.control_indices] += delta
    next_q = torch.max(torch.min(next_q, ik_state.upper), ik_state.lower)
    dof_positions[:] = next_q.detach().cpu().numpy()

    pos_err_norm = float(torch.linalg.norm(pos_err).detach().cpu())
    ik_state.last_pos_error = pos_err_norm
    ik_state.last_rot_error = rot_err_norm
    ik_state.current_pos_np = eef_pos.detach().cpu().numpy().copy()

    if not ik_state.reached_reported:
        reached_pos = pos_err_norm <= args.ik_pos_tolerance
        reached_rot = rot_err_norm is None or rot_err_norm <= args.ik_rot_tolerance
        if reached_pos and reached_rot:
            if rot_err_norm is None:
                print(f"IK target reached: pos_err={pos_err_norm:.4f} m")
            else:
                print(f"IK target reached: pos_err={pos_err_norm:.4f} m, rot_err={rot_err_norm:.4f}")
            ik_state.reached_reported = True


def draw_ik_target(gym, viewer, env, ik_state):
    target_pose = transform_from_arrays(ik_state.target_pos_np, ik_state.target_quat_np)
    target_sphere = gymutil.WireframeSphereGeometry(
        radius=0.03,
        num_lats=8,
        num_lons=8,
        color=(1.0, 0.82, 0.1),
        color2=(1.0, 0.45, 0.1),
    )
    gymutil.draw_lines(target_sphere, gym, viewer, env, target_pose)

    if ik_state.target_quat_np is not None:
        gymutil.draw_lines(gymutil.AxesGeometry(scale=0.12), gym, viewer, env, target_pose)

    if ik_state.current_pos_np is not None:
        current_pose = transform_from_arrays(ik_state.current_pos_np)
        current_sphere = gymutil.WireframeSphereGeometry(
            radius=0.03,
            num_lats=8,
            num_lons=8,
            color=(1.0, 0.0, 0.0),
            color2=(1.0, 0.0, 0.0),
        )
        gymutil.draw_lines(current_sphere, gym, viewer, env, current_pose)

        p1 = gymapi.Vec3(
            float(ik_state.current_pos_np[0]),
            float(ik_state.current_pos_np[1]),
            float(ik_state.current_pos_np[2]),
        )
        p2 = gymapi.Vec3(
            float(ik_state.target_pos_np[0]),
            float(ik_state.target_pos_np[1]),
            float(ik_state.target_pos_np[2]),
        )
        gymutil.draw_line(p1, p2, gymapi.Vec3(1.0, 0.75, 0.0), gym, viewer, env)


def run_viewer(
    gym,
    sim,
    env,
    actor,
    actor_handles,
    viewer,
    args,
    dt,
    dof_names,
    dof_states,
    dof_positions,
    lower,
    upper,
    defaults,
    speeds,
    selected,
    ik_state=None,
):
    anim_seek_lower = 1
    anim_seek_upper = 2
    anim_seek_default = 3
    anim_finished = 4

    anim_state = anim_seek_lower
    selected_index = 0
    paused = False
    step_count = 0
    max_steps = args.steps
    if args.headless and max_steps == 0:
        max_steps = 600

    current_dof = selected[selected_index]
    zero_pose_steps = max(0, int(args.zero_pose_seconds / dt))
    if ik_state is not None:
        if zero_pose_steps > 0:
            print(f"Holding URDF zero pose for {args.zero_pose_seconds:.2f}s before IK motion.")
        else:
            print("Solving IK toward the target pose.")
    elif args.zero_pose_only:
        print("Showing URDF zero pose only. Close the viewer to exit.")
    elif zero_pose_steps > 0:
        print(f"Holding URDF zero pose for {args.zero_pose_seconds:.2f}s before animation.")
        print(f"Next animated DOF {current_dof}: {dof_names[current_dof]}")
    else:
        print(f"Animating DOF {current_dof}: {dof_names[current_dof]}")
    if args.disable_base_motion:
        print("Base pose demo is disabled.")
    else:
        print(
            "Base pose demo enabled: "
            f"XY amplitude={args.base_motion_amplitude:.3f} m, "
            f"yaw amplitude={args.base_motion_yaw:.3f} rad."
        )
    start = time.time()
    motion_time = 0.0
    prev_base_translation, prev_base_yaw = base_motion_pose(args, motion_time)

    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if max_steps > 0 and step_count >= max_steps:
            break

        if viewer is not None:
            for event in gym.query_viewer_action_events(viewer):
                if event.action == "quit" and event.value > 0:
                    return
                if event.action == "pause" and event.value > 0:
                    paused = not paused
                    print("Paused" if paused else "Running")
                if event.action == "reset" and event.value > 0:
                    dof_positions[:] = defaults
                    anim_state = anim_seek_lower
                    selected_index = 0
                    current_dof = selected[selected_index]
                    next_base_translation, next_base_yaw = base_motion_pose(args, 0.0)
                    apply_base_motion_delta(
                        gym,
                        env,
                        actor_handles,
                        prev_base_translation,
                        prev_base_yaw,
                        next_base_translation,
                        next_base_yaw,
                    )
                    prev_base_translation, prev_base_yaw = next_base_translation, next_base_yaw
                    motion_time = 0.0
                    if ik_state is not None:
                        ik_state.reached_reported = False
                        print("Reset. Solving IK toward the target pose.")
                    else:
                        print(f"Reset. Animating DOF {current_dof}: {dof_names[current_dof]}")

        holding_zero_pose = step_count < zero_pose_steps
        if not paused and not args.zero_pose_only and not holding_zero_pose and ik_state is not None:
            update_ik_targets(gym, sim, dof_positions, ik_state, args)
        elif not paused and not args.zero_pose_only and not holding_zero_pose:
            speed = float(speeds[current_dof])
            if anim_state == anim_seek_lower:
                dof_positions[current_dof] -= speed * dt
                if dof_positions[current_dof] <= lower[current_dof]:
                    dof_positions[current_dof] = lower[current_dof]
                    anim_state = anim_seek_upper
            elif anim_state == anim_seek_upper:
                dof_positions[current_dof] += speed * dt
                if dof_positions[current_dof] >= upper[current_dof]:
                    dof_positions[current_dof] = upper[current_dof]
                    anim_state = anim_seek_default
            elif anim_state == anim_seek_default:
                direction = -1.0 if dof_positions[current_dof] > defaults[current_dof] else 1.0
                dof_positions[current_dof] += direction * speed * dt
                reached = (
                    direction < 0.0
                    and dof_positions[current_dof] <= defaults[current_dof]
                    or direction > 0.0
                    and dof_positions[current_dof] >= defaults[current_dof]
                )
                if reached:
                    dof_positions[current_dof] = defaults[current_dof]
                    anim_state = anim_finished
            elif anim_state == anim_finished:
                selected_index = (selected_index + 1) % len(selected)
                current_dof = selected[selected_index]
                anim_state = anim_seek_lower
                print(f"Animating DOF {current_dof}: {dof_names[current_dof]}")
        elif args.zero_pose_only or holding_zero_pose:
            dof_positions[:] = defaults

        gym.set_actor_dof_position_targets(env, actor, dof_positions)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if not paused:
            motion_time += dt
            next_base_translation, next_base_yaw = base_motion_pose(args, motion_time)
            apply_base_motion_delta(
                gym,
                env,
                actor_handles,
                prev_base_translation,
                prev_base_yaw,
                next_base_translation,
                next_base_yaw,
            )
            prev_base_translation, prev_base_yaw = next_base_translation, next_base_yaw

        if viewer is not None:
            if args.show_axis or ik_state is not None:
                gym.clear_lines(viewer)

            if ik_state is not None:
                draw_ik_target(gym, viewer, env, ik_state)

            if args.show_axis:
                dof_handle = gym.get_actor_dof_handle(env, actor, current_dof)
                frame = gym.get_dof_frame(env, dof_handle)
                p1 = frame.origin
                p2 = frame.origin + frame.axis * 0.35
                gymutil.draw_line(p1, p2, gymapi.Vec3(1.0, 0.0, 0.0), gym, viewer, env)

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        step_count += 1

    elapsed = time.time() - start
    if ik_state is not None and math.isfinite(ik_state.last_pos_error):
        if ik_state.last_rot_error is None:
            print(f"Final IK error: pos={ik_state.last_pos_error:.4f} m")
        else:
            print(
                f"Final IK error: pos={ik_state.last_pos_error:.4f} m, "
                f"rot={ik_state.last_rot_error:.4f}"
            )
    print(f"Done after {step_count} steps ({elapsed:.2f}s).")


def main():
    args = parse_args()
    configure_mode_defaults(args)
    gym = gymapi.acquire_gym()
    sim, dt = create_sim(gym, args)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="b1z1_isaacgym_assets_") as temp_dir:
        base_asset, arm_asset = load_robot_assets(gym, sim, args, Path(temp_dir))
        if base_asset is not None:
            print_collision_summary(
                gym, base_asset, "base visual actor", verbose=args.print_collision_summary
            )
        print_collision_summary(
            gym, arm_asset, "arm articulated actor", verbose=args.print_collision_summary
        )
        dof_data = configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        env, actor, actor_handles = create_env_and_actor(
            gym, sim, base_asset, arm_asset, dof_props, dof_states, args
        )
        ik_state = setup_ik_controller(
            gym, sim, env, actor, arm_asset, dof_names, lower, upper, args
        )
        viewer = setup_viewer(gym, sim, args)

        try:
            run_viewer(
                gym,
                sim,
                env,
                actor,
                actor_handles,
                viewer,
                args,
                dt,
                dof_names,
                dof_states,
                dof_positions,
                lower,
                upper,
                defaults,
                speeds,
                selected,
                ik_state=ik_state,
            )
        finally:
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
