#!/usr/bin/env python3
"""Parallel float-base B1Z1 base+arm IK door-push overview video recorder.

This is a video-focused copy of isaacgym_float_ik_b1z1_basearn_push_door_parallel.py.
It keeps the same asset/controller helpers, but defaults to a large env grid and
records the Isaac Gym viewer while moving the viewer camera from env0 to the
diagonal far env after env0's door starts opening.
"""

from __future__ import annotations

import json
import math
import colorsys
import copy
import re
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]

import door_common as dc

try:
    import isaacgym_a2w_ik_push_door_parallel as a2w_ik
except ImportError:
    a2w_ik = None

try:
    import isaacgym_float_ik_a2w_basearn_push_door_parallel as a2w_scripted
except ImportError:
    a2w_scripted = None

base_ik = dc.base_ik
gymapi = dc.gymapi
gymutil = dc.gymutil
DEFAULT_DOOR_CFG = dc.DEFAULT_DOOR_CFG
DP_NUM_DOFS = dc.DP_NUM_DOFS
DP_NUM_ACTIONS = dc.DP_NUM_ACTIONS
B1Z1_DEFAULT_DOF_POS = dc.B1Z1_DEFAULT_DOF_POS
FLOAT_ARM_TO_DP_DOF = dc.FLOAT_ARM_TO_DP_DOF
ThickAxesGeometry = dc.ThickAxesGeometry
DoorRuntime = dc.DoorRuntime

DP_ROOT = HIGH_LEVEL_ROOT / "dp"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

try:
    from door_dp_common import (
        ACTION_NAMES,
        DoorDPJsonlLogger,
        DoorDPPolicyController,
        RawDoorDPRecorder,
        make_state_feature_names,
        normalize_vision_mode,
        raw_image_keys_for_vision_mode,
    )
except ImportError:
    ACTION_NAMES = None
    DoorDPJsonlLogger = None
    DoorDPPolicyController = None
    RawDoorDPRecorder = None
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
    "push_door",
    "return_home",
    "hold_home",
]
DP_PHASE_ID = {name: idx for idx, name in enumerate(DP_PHASE_NAMES)}
IKPUSH_STATE_VERSION = "zero_leg_dof_pos_prev_action_v1"
sorted_asset_entries = dc.sorted_asset_entries
G1_DEFAULT_JOINT_ANGLES = {
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "torso_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

G1_RIGHT_ARM_IK_DOF_NAMES = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}

B1_FULL_BODY_ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "b1z1-col"
B1_FULL_BODY_ASSET_FILE = "urdf/b1z1.urdf"
B1_BODY_VISUAL_LINKS = {
    "base",
    "trunk",
    "imu_link",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
}
B1_LEG_DOF_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
B1_DEFAULT_LEG_JOINT_ANGLES = {
    name: float(B1Z1_DEFAULT_DOF_POS[idx])
    for idx, name in enumerate(B1_LEG_DOF_NAMES)
}

SCRIPTED_TRAJECTORY_STEP_FIELDS = (
    "walk_steps",
    "initial_hold_steps",
    "initial_hold_move_steps",
    "grasp_steps",
    "grasp_hold_steps",
    "gripper_close_steps",
    "handle_rotate_steps",
    "door_push_steps",
    "return_home_steps",
    "hold_steps",
    "gripper_loosen_steps",
)
SCRIPTED_TRAJECTORY_PHASE_FIELDS = (
    "walk_steps",
    "initial_hold_steps",
    "grasp_steps",
    "grasp_hold_steps",
    "gripper_close_steps",
    "handle_rotate_steps",
    "door_push_steps",
    "return_home_steps",
)

ROBOT_BODY_DEFAULT_DOOR_PREFER = {
    "g1": "rec_using-the-reference-image-for-overall-proportion_20260610_152759_469492_cdfe437e",
    "unitree_g1": "rec_using-the-reference-image-for-overall-proportion_20260610_152759_469492_cdfe437e",
    "g1_right_arm": "rec_using-the-reference-image-for-overall-proportion_20260610_152759_469492_cdfe437e",
    "scout_z1": "rec_using-the-reference-image-for-overall-proportion_20260610_193220_721835_ab3c271a",
    "scout_ugv_z1": "rec_using-the-reference-image-for-overall-proportion_20260610_193220_721835_ab3c271a",
    "ugv_z1": "rec_using-the-reference-image-for-overall-proportion_20260610_193220_721835_ab3c271a",
    "scout": "rec_using-the-reference-image-for-overall-proportion_20260610_193220_721835_ab3c271a",
    "scout_mini_z1": "rec_using-the-reference-image-for-overall-proportion_20260610_193220_721835_ab3c271a",
    "a2w_z1": "rec_using-the-reference-image-for-overall-proportion_20260610_161038_309387_2dbe2f90",
    "a2wz1": "rec_using-the-reference-image-for-overall-proportion_20260610_161038_309387_2dbe2f90",
    "a2w": "rec_using-the-reference-image-for-overall-proportion_20260610_161038_309387_2dbe2f90",
}
DOOR_CFG_DEFAULT_PREFER = {
    "b1z1_opendoor_door4.yaml": (
        "rec_using-the-reference-image-for-proportions-and-la_20260614_161003_946126_175e0acb"
    ),
}


def is_g1_robot_body(args) -> bool:
    return str(getattr(args, "robot_body", "b1z1")).lower() in ("g1", "unitree_g1", "g1_right_arm")


def is_scout_robot_body_name(robot_body: str) -> bool:
    return str(robot_body).lower() in ("scout_ugv_z1", "scout_z1", "ugv_z1", "scout", "scout_mini_z1")


def is_a2w_robot_body_name(robot_body: str) -> bool:
    return str(robot_body).lower() in ("a2w_z1", "a2wz1", "a2w")


def apply_scripted_trajectory_speed_scale(args, explicit_cli_flags):
    speed_scale = float(getattr(args, "scripted_trajectory_speed_scale", 1.0))
    if speed_scale <= 0.0:
        raise ValueError("--scripted_trajectory_speed_scale must be greater than 0.")
    if abs(speed_scale - 1.0) <= 1.0e-6:
        return

    changes = []
    for field in SCRIPTED_TRAJECTORY_STEP_FIELDS:
        old_value = int(getattr(args, field))
        new_value = max(1, int(round(old_value / speed_scale)))
        setattr(args, field, new_value)
        changes.append(f"{field}:{old_value}->{new_value}")

    args.initial_hold_move_steps = min(
        int(args.initial_hold_move_steps),
        int(args.initial_hold_steps),
    )
    phase_steps = sum(int(getattr(args, field)) for field in SCRIPTED_TRAJECTORY_PHASE_FIELDS)
    steps_was_set = any(
        token == "--steps" or str(token).startswith("--steps=")
        for token in explicit_cli_flags
    )
    if not steps_was_set:
        old_steps = int(args.steps)
        args.steps = phase_steps
        changes.append(f"steps:{old_steps}->{args.steps}")

    print(
        f"Scripted trajectory speed={speed_scale:.3g}x: " + ", ".join(changes),
        flush=True,
    )


def configure_visual_entrance_placement(args, door, env_index):
    if str(door.spec.get("runtime_mapping", "")) != "first_two_dofs":
        return
    _world_min_x, world_max_x, _world_min_y, _world_max_y = dc.door_world_xy_bounds_from_bbox(args, door)
    robot_y = dc.robot_y_for_door(args, door.handle_bounding, door)
    start_clearance = max(0.0, float(getattr(args, "entrance_robot_start_clearance", 1.0)))
    stop_clearance = max(0.0, float(getattr(args, "entrance_robot_stop_clearance", 0.20)))
    args.robot_x = float(world_max_x) + float(args.robot_front_offset) + start_clearance
    args.stop_distance = max(0.0, float(world_max_x) + stop_clearance - float(args.door_x))
    args.pass_through_door = False
    args.push_base_distance = 0.0
    if int(env_index) < 4:
        print(
            f"visual_entrance_placement env={env_index} door={door.spec.get('name')} "
            f"world_max_x={world_max_x:.3f} robot_x={float(args.robot_x):.3f} "
            f"robot_y={robot_y:.3f} rightmost_handle={door.spec.get('robot_alignment_handle')} "
            f"stop_distance={float(args.stop_distance):.3f}",
            flush=True,
        )


def indent_xml(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def scout_package_mesh_path(package_root: Path, filename: str) -> Path:
    prefix = "package://scout_description/"
    if filename.startswith(prefix):
        return package_root / filename[len(prefix):]
    if filename.startswith("package://"):
        return package_root / "meshes" / Path(filename).name
    path = Path(filename)
    if path.is_absolute():
        return path
    candidate = package_root / path
    if candidate.exists():
        return candidate
    return package_root / "meshes" / path.name


def parse_origin_xyz_rpy(origin_elem):
    def parse_vector(text):
        values = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(text))
        if len(values) < 3:
            return np.zeros(3, dtype=np.float64)
        return np.asarray([float(value) for value in values[:3]], dtype=np.float64)

    if origin_elem is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = parse_vector(origin_elem.get("xyz", "0 0 0"))
    rpy = parse_vector(origin_elem.get("rpy", "0 0 0"))
    return xyz, rpy


def rpy_matrix(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def transform_matrix(xyz, rpy):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rpy_matrix(rpy)
    mat[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return mat


def matrix_to_xyz_rpy(mat):
    rot = np.asarray(mat[:3, :3], dtype=np.float64)
    sy = max(-1.0, min(1.0, float(-rot[2, 0])))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1.0e-8:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rot[0, 1], rot[1, 1])
    return np.asarray(mat[:3, 3], dtype=np.float64), np.array([roll, pitch, yaw], dtype=np.float64)


def origin_from_matrix(mat):
    xyz, rpy = matrix_to_xyz_rpy(mat)
    origin = ET.Element("origin")
    origin.set("xyz", " ".join(f"{float(v):.9g}" for v in xyz))
    origin.set("rpy", " ".join(f"{float(v):.9g}" for v in rpy))
    return origin


def flatten_visual_urdf(
    root: ET.Element,
    root_link_name: str,
    robot_name: str,
    default_joint_angles=None,
    keep_links=None,
) -> ET.Element:
    default_joint_angles = default_joint_angles or {}
    keep_links = None if keep_links is None else set(keep_links)
    parent_joints = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        xyz, rpy = parse_origin_xyz_rpy(joint.find("origin"))
        parent_to_child = transform_matrix(xyz, rpy)
        joint_type = joint.get("type", "fixed")
        if joint_type in ("revolute", "continuous", "prismatic"):
            angle = float(default_joint_angles.get(joint.get("name", ""), 0.0))
            if joint_type == "prismatic":
                axis_elem = joint.find("axis")
                axis = np.fromstring(axis_elem.get("xyz", "1 0 0") if axis_elem is not None else "1 0 0", sep=" ")
                if axis.size != 3:
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                parent_to_child[:3, 3] += axis * angle
            else:
                axis_elem = joint.find("axis")
                axis = np.fromstring(axis_elem.get("xyz", "1 0 0") if axis_elem is not None else "1 0 0", sep=" ")
                if axis.size != 3:
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                joint_rot = np.eye(4, dtype=np.float64)
                joint_rot[:3, :3] = axis_angle_matrix(axis, angle)
                parent_to_child = parent_to_child @ joint_rot
        parent_joints[child_name] = (parent_name, parent_to_child)

    link_tf_cache = {root_link_name: np.eye(4, dtype=np.float64)}

    def link_to_root(link_name):
        if link_name in link_tf_cache:
            return link_tf_cache[link_name]
        parent_name, parent_to_child = parent_joints.get(link_name, (root_link_name, np.eye(4, dtype=np.float64)))
        tf = link_to_root(parent_name) @ parent_to_child
        link_tf_cache[link_name] = tf
        return tf

    flat_root = ET.Element("robot", {"name": robot_name})
    flat_link = ET.SubElement(flat_root, "link", {"name": "base_link"})
    inertial = ET.SubElement(flat_link, "inertial")
    ET.SubElement(inertial, "mass", {"value": "1.0"})
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "0.1", "ixy": "0", "ixz": "0", "iyy": "0.1", "iyz": "0", "izz": "0.1"},
    )

    for link in root.findall("link"):
        link_name = link.get("name", root_link_name)
        if keep_links is not None and link_name not in keep_links:
            continue
        root_to_link = link_to_root(link_name)
        for visual in link.findall("visual"):
            flat_visual = copy.deepcopy(visual)
            old_origin = flat_visual.find("origin")
            old_xyz, old_rpy = parse_origin_xyz_rpy(old_origin)
            root_to_visual = root_to_link @ transform_matrix(old_xyz, old_rpy)
            if old_origin is not None:
                flat_visual.remove(old_origin)
            flat_visual.insert(0, origin_from_matrix(root_to_visual))
            flat_link.append(flat_visual)
    return flat_root


def flatten_scout_visual_urdf(root: ET.Element) -> ET.Element:
    return flatten_visual_urdf(root, "base_link", "scout_mini_visual_flat")


def build_b1_full_body_visual_asset_root(args, temp_root: Path) -> tuple[Path, str]:
    source_root = Path(args.b1_full_body_asset_root).expanduser().resolve()
    source_urdf = source_root / args.b1_full_body_asset_file
    if not source_urdf.exists():
        raise FileNotFoundError(f"B1 full-body URDF not found: {source_urdf}")

    out_root = Path(temp_root) / "b1_full_body_visual_assets"
    out_urdf = out_root / "urdf" / "b1_full_body_visual_flat.urdf"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    base_ik.populate_mesh_dir(source_root / "meshes", out_root / "meshes")

    root = ET.parse(source_urdf).getroot()
    flat_root = flatten_visual_urdf(
        root,
        root_link_name="base",
        robot_name="b1_full_body_visual_flat",
        default_joint_angles=B1_DEFAULT_LEG_JOINT_ANGLES,
        keep_links=B1_BODY_VISUAL_LINKS,
    )
    indent_xml(flat_root)
    ET.ElementTree(flat_root).write(out_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Built B1 full-body video visual asset: {out_urdf}", flush=True)
    return out_root, "urdf/b1_full_body_visual_flat.urdf"


def build_scout_base_asset_root(args, temp_root: Path) -> tuple[Path, str]:
    package_root = Path(args.scout_asset_root).expanduser().resolve()
    source_urdf = package_root / args.scout_urdf_file
    if not source_urdf.exists():
        raise FileNotFoundError(f"Scout URDF not found: {source_urdf}")

    out_root = Path(temp_root) / "scout_ugv_z1_split_assets"
    out_urdf = out_root / "urdf" / "scout_mini_base_visual_flat.urdf"
    out_mesh_dir = out_root / "meshes"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_mesh_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(source_urdf).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        src = scout_package_mesh_path(package_root, filename).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Scout mesh referenced by {source_urdf} not found: {src}")
        dst = out_mesh_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        mesh.set("filename", f"../meshes/{src.name}")

    # The base actor is only a kinematic visual body in this float-IK video
    # script. Flatten Scout's articulated wheel links into one visual link so
    # the existing float-base pose setter can move it like the B1 base actor.
    root = flatten_scout_visual_urdf(root)

    indent_xml(root)
    ET.ElementTree(root).write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_root, "urdf/scout_mini_base_visual_flat.urdf"


def build_g1_body_asset_root(args, temp_root: Path) -> tuple[Path, str]:
    package_root = Path(args.g1_asset_root).expanduser().resolve()
    source_urdf = package_root / args.g1_urdf_file
    if not source_urdf.exists():
        raise FileNotFoundError(f"G1 URDF not found: {source_urdf}")

    out_root = Path(temp_root) / "g1_z1_split_assets"
    out_urdf = out_root / "urdf" / "g1_body_visual_flat.urdf"
    out_mesh_dir = out_root / "meshes"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_mesh_dir.mkdir(parents=True, exist_ok=True)

    root = ET.parse(source_urdf).getroot()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        src = scout_package_mesh_path(package_root, filename).resolve()
        if not src.exists():
            raise FileNotFoundError(f"G1 mesh referenced by {source_urdf} not found: {src}")
        dst = out_mesh_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        mesh.set("filename", f"../meshes/{src.name}")

    root = flatten_visual_urdf(
        root,
        root_link_name=str(getattr(args, "g1_root_link", "pelvis")),
        robot_name="g1_visual_flat",
        default_joint_angles=G1_DEFAULT_JOINT_ANGLES,
    )

    indent_xml(root)
    ET.ElementTree(root).write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_root, "urdf/g1_body_visual_flat.urdf"


def build_a2w_video_asset_root(args, temp_root: Path) -> tuple[Path, str, str]:
    if a2w_ik is None:
        raise RuntimeError("A2W video body requires isaacgym_a2w_ik_push_door_parallel.py to be importable.")
    split_root, base_file, arm_file = a2w_ik.build_a2wz1_split_asset_root(
        args.a2wz1_asset_root,
        args.a2wz1_asset_file,
        temp_root,
    )
    base_urdf = Path(split_root) / base_file
    root = ET.parse(base_urdf).getroot()
    flat_root = flatten_visual_urdf(
        root,
        root_link_name="base_link",
        robot_name="a2w_base_visual_flat",
        default_joint_angles=getattr(a2w_ik, "A2W_DEFAULT_LEG_POS", {}),
    )
    flat_file = "urdf/a2w_base_visual_flat.urdf"
    flat_urdf = Path(split_root) / flat_file
    indent_xml(flat_root)
    ET.ElementTree(flat_root).write(flat_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Built A2W video visual base asset: {flat_urdf}", flush=True)
    return Path(split_root), flat_file, arm_file


def load_video_robot_assets(gym, sim, args, temp_root: Path):
    robot_body = str(getattr(args, "robot_body", "b1z1")).lower()
    if robot_body in ("b1z1", "b1", "b1z1_base"):
        base_root, base_file = build_b1_full_body_visual_asset_root(args, temp_root)
        base_asset = base_ik.load_asset_with_visual_flip(
            gym,
            sim,
            base_root,
            base_file,
            args,
            False,
            "B1 full-body visual actor",
        )
        source_asset_root = Path(args.asset_root).expanduser().resolve()
        split_root, _base_file, arm_file = base_ik.build_split_asset_root(
            source_asset_root,
            args.asset_file,
            temp_root,
            align_arm_gripper_collisions=not bool(getattr(args, "disable_arm_visual_flip", False)),
        )
        arm_asset = base_ik.load_asset_with_visual_flip(
            gym,
            sim,
            split_root,
            arm_file,
            args,
            not bool(getattr(args, "disable_arm_visual_flip", False)),
            "Z1 arm actor",
        )
        return base_asset, arm_asset
    supported_bodies = (
        "a2w",
        "a2w_z1",
        "a2wz1",
        "scout_z1",
        "scout_ugv_z1",
        "ugv_z1",
        "scout",
        "scout_mini_z1",
        "g1",
        "unitree_g1",
        "g1_right_arm",
        "g1_z1",
        "g1_visual_z1",
    )
    if robot_body not in supported_bodies:
        raise ValueError(
            f"Unsupported --robot_body={args.robot_body!r}; expected b1z1, scout_z1, a2w_z1, or g1"
        )

    if robot_body in ("g1", "unitree_g1", "g1_right_arm"):
        g1_root = Path(args.g1_asset_root).expanduser().resolve()
        g1_file = str(args.g1_urdf_file)
        if not (g1_root / g1_file).exists():
            raise FileNotFoundError(f"G1 URDF not found: {g1_root / g1_file}")
        g1_asset = base_ik.load_asset_with_visual_flip(
            gym,
            sim,
            g1_root,
            g1_file,
            args,
            bool(getattr(args, "g1_flip_visual_attachments", False)),
            "Unitree G1 articulated right-arm actor",
        )
        return None, g1_asset

    if is_a2w_robot_body_name(robot_body):
        base_root, base_file, arm_file = build_a2w_video_asset_root(args, temp_root)
        base_flip_visual_attachments = False
        base_label = "A2W base visual actor"
    elif robot_body in ("g1_z1", "g1_visual_z1"):
        base_root, base_file = build_g1_body_asset_root(args, temp_root)
        base_flip_visual_attachments = bool(getattr(args, "g1_flip_visual_attachments", False))
        base_label = "Unitree G1 body visual actor"
    else:
        base_root, base_file = build_scout_base_asset_root(args, temp_root)
        base_flip_visual_attachments = not bool(getattr(args, "no_scout_flip_visual_attachments", False))
        if bool(getattr(args, "scout_flip_visual_attachments", False)):
            base_flip_visual_attachments = True
        base_label = "Scout Mini UGV base visual actor"
    base_asset = base_ik.load_asset_with_visual_flip(
        gym,
        sim,
        base_root,
        base_file,
        args,
        base_flip_visual_attachments,
        base_label,
    )

    if is_a2w_robot_body_name(robot_body):
        split_root = base_root
    else:
        b1_asset_root = Path(args.asset_root).expanduser().resolve()
        split_root, _b1_base_file, arm_file = base_ik.build_split_asset_root(
            b1_asset_root,
            args.asset_file,
            temp_root,
            align_arm_gripper_collisions=not bool(getattr(args, "disable_arm_visual_flip", False)),
        )
    arm_asset = base_ik.load_asset_with_visual_flip(
        gym,
        sim,
        split_root,
        arm_file,
        args,
        not bool(getattr(args, "disable_arm_visual_flip", False)),
        "Z1 arm actor",
    )
    return base_asset, arm_asset


def apply_g1_default_dof_pose(args, dof_names, dof_states, dof_positions, lower, upper, defaults):
    if not is_g1_robot_body(args):
        return
    applied = []
    for idx, name in enumerate(dof_names):
        if name not in G1_DEFAULT_JOINT_ANGLES:
            continue
        value = float(G1_DEFAULT_JOINT_ANGLES[name])
        value = float(np.clip(value, float(lower[idx]), float(upper[idx])))
        defaults[idx] = value
        dof_states["pos"][idx] = value
        dof_positions[idx] = value
        applied.append(name)
    print(
        "G1 right-arm mode: using Unitree G1 articulated actor; "
        f"ee_link={args.ik_ee_link}, controlled_joints={', '.join(sorted(base_ik.ARM_IK_DOF_NAMES))}",
        flush=True,
    )
    if applied:
        print(f"G1 default pose applied to {len(applied)} DOFs.", flush=True)


def parse_args():
    args = gymutil.parse_arguments(
        description="B1Z1 base+arm float IK door-push demo.",
        headless=True,
        no_graphics=True,
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": str(base_ik.DEFAULT_ASSET_ROOT)},
            {"name": "--asset_file", "type": str, "default": base_ik.DEFAULT_ASSET_FILE},
            {
                "name": "--b1_full_body_asset_root",
                "type": str,
                "default": str(B1_FULL_BODY_ASSET_ROOT),
                "help": "Source asset used to build the video-only complete B1 visual actor.",
            },
            {
                "name": "--b1_full_body_asset_file",
                "type": str,
                "default": B1_FULL_BODY_ASSET_FILE,
            },
            {
                "name": "--robot_body",
                "type": str,
                "default": "b1z1",
                "help": "Robot body to use: b1z1, scout_z1, a2w_z1, or g1. For g1, the built-in right arm is used instead of Z1.",
            },
            {
                "name": "--a2wz1_asset_root",
                "type": str,
                "default": str(HIGH_LEVEL_ROOT / "data" / "asset" / "a2wz1"),
                "help": "Root of the built A2W+Z1 asset used by --robot_body a2w_z1.",
            },
            {"name": "--a2wz1_asset_file", "type": str, "default": "urdf/a2wz1.urdf"},
            {
                "name": "--scout_asset_root",
                "type": str,
                "default": "/home/sivan/whole_body/ugv_gazebo_sim/scout/scout_description",
                "help": "Root of the scout_description package used by --robot_body scout_ugv_z1.",
            },
            {"name": "--scout_urdf_file", "type": str, "default": "urdf/scout_mini.urdf"},
            {
                "name": "--scout_flip_visual_attachments",
                "action": "store_true",
                "help": "Set AssetOptions.flip_visual_attachments=True for the Scout Mini base actor.",
            },
            {
                "name": "--no_scout_flip_visual_attachments",
                "action": "store_true",
                "help": "Disable Scout Mini visual attachment flipping for debugging raw URDF import.",
            },
            {
                "name": "--g1_asset_root",
                "type": str,
                "default": "/home/sivan/whole_body/unitree_rl_gym/resources/robots/g1_description",
                "help": "Root of the Unitree G1 description package used by --robot_body g1.",
            },
            {"name": "--g1_urdf_file", "type": str, "default": "g1_29dof.urdf"},
            {"name": "--g1_root_link", "type": str, "default": "pelvis"},
            {"name": "--g1_ee_link", "type": str, "default": "right_rubber_hand"},
            {
                "name": "--g1_right_arm_ik_joints",
                "type": str,
                "default": ",".join(sorted(G1_RIGHT_ARM_IK_DOF_NAMES)),
                "help": "Comma-separated G1 right-arm joints used by IK when --robot_body g1.",
            },
            {
                "name": "--g1_use_orientation_ik",
                "action": "store_true",
                "help": "Use full pose IK for G1. By default G1 right-arm IK is position-only.",
            },
            {
                "name": "--g1_flip_visual_attachments",
                "action": "store_true",
                "help": "Set AssetOptions.flip_visual_attachments=True for the flattened Unitree G1 body visual actor.",
            },
            {"name": "--rl_device", "type": str, "default": "cuda:0"},
            {"name": "--num_envs", "type": int, "default": 1024},
            {
                "name": "--env_spacing",
                "type": float,
                "default": 5.0,
                "help": "Center-to-center spacing of parallel environments.",
            },
            {"name": "--steps", "type": int, "default": 2405},
            {"name": "--seed", "type": int, "default": -1},
            {"name": "--door_cfg", "type": str, "default": str(DEFAULT_DOOR_CFG)},
            {"name": "--door_name", "type": str, "default": ""},
            {"name": "--door_index", "type": int, "default": -1},
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
            {
                "name": "--door_exclude_names",
                "type": str,
                "default": "",
                "help": "Comma-separated extra door asset names to skip during bulk door selection.",
            },
            {
                "name": "--door_include_names",
                "type": str,
                "default": "",
                "help": "Optional comma-separated allowlist used only by this video script.",
            },
            {
                "name": "--allow_unsafe_door_assets",
                "action": "store_true",
                "help": "Allow known unsafe door assets that may crash Isaac Gym mesh cooking.",
            },
            {"name": "--door_actor_scale", "type": float, "default": 1.2},
            {"name": "--door_x", "type": float, "default": 2.5},
            {"name": "--door_y", "type": float, "default": 0.0},
            {"name": "--door_z_offset", "type": float, "default": 0.01},
            {"name": "--no_door_side_walls", "action": "store_true", "help": "Disable the static side-wall actors beside each door."},
            {"name": "--door_wall_height", "type": float, "default": 2.2},
            {"name": "--door_wall_opening_width", "type": float, "default": 0.0, "help": "Wall opening width; <=0 uses the scaled door bounding-box width."},
            {"name": "--door_wall_side_width", "type": float, "default": 1.0},
            {"name": "--door_wall_thickness", "type": float, "default": 0.08},
            {"name": "--door_wall_gap", "type": float, "default": 0.0},
            {"name": "--door_wall_x_offset", "type": float, "default": 0.0},
            {"name": "--door_wall_y_offset", "type": float, "default": 0.0},
            {
                "name": "--door_use_urdf_rgba",
                "action": "store_true",
                "help": "Color generated door links from URDF rgba values instead of mesh materials.",
            },
            {"name": "--robot_x", "type": float, "default": 4.1},
            {"name": "--robot_y", "type": float, "default": 0.0},
            {
                "name": "--robot_y_alignment",
                "type": str,
                "default": "handle",
                "help": "How to place the robot in Y relative to each door: auto, handle, door_center, or door_y. Auto uses an asset's explicit handle alignment when available, otherwise centers generated rec_ doors and keeps legacy doors on door_y.",
            },
            {"name": "--robot_z", "type": float, "default": 0.60},
            {
                "name": "--b1_robot_z",
                "type": float,
                "default": 0.50,
                "help": "Default B1 root height for --robot_body b1z1 when --robot_z is not explicitly set.",
            },
            {
                "name": "--scout_robot_z",
                "type": float,
                "default": 0.22,
                "help": "Default base height for --robot_body scout_ugv_z1 when --robot_z is not explicitly set.",
            },
            {
                "name": "--a2w_robot_z",
                "type": float,
                "default": 0.50,
                "help": "Default base height for --robot_body a2w_z1 when --robot_z is not explicitly set.",
            },
            {
                "name": "--g1_robot_z",
                "type": float,
                "default": 0.80,
                "help": "Default pelvis/root height for --robot_body g1 when --robot_z is not explicitly set.",
            },
            {
                "name": "--g1_robot_front_offset",
                "type": float,
                "default": 0.35,
                "help": "Default front offset for --robot_body g1 when --robot_front_offset is not explicitly set.",
            },
            {
                "name": "--g1_stop_distance",
                "type": float,
                "default": 0.10,
                "help": "Default stop distance for --robot_body g1 when --stop_distance is not explicitly set.",
            },
            {"name": "--robot_yaw", "type": float, "default": math.pi},
            {"name": "--robot_front_offset", "type": float, "default": 0.55},
            {"name": "--robot_rear_offset", "type": float, "default": 0.65},
            {"name": "--stop_distance", "type": float, "default": 0.15},
            {"name": "--entrance_robot_start_clearance", "type": float, "default": 1.0},
            {"name": "--entrance_robot_stop_clearance", "type": float, "default": 0.20},
            {"name": "--push_base_distance", "type": float, "default": 0.35},
            {"name": "--base_push_time_scale", "type": float, "default": 1.35},
            {"name": "--door_pass_clearance", "type": float, "default": 0.55},
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
            {"name": "--return_home_steps", "type": int, "default": 150},
            {"name": "--return_home_target_chase_alpha", "type": float, "default": 0.08},
            {"name": "--hold_steps", "type": int, "default": 300},
            {
                "name": "--scripted_trajectory_speed_scale",
                "type": float,
                "default": 1.0,
                "help": "Scripted trajectory speed multiplier. For example, 1.5 or 2.0 shortens all phase durations while preserving their endpoints.",
            },
            {"name": "--pregrasp_offset", "type": float, "default": 0.15},
            {"name": "--grasp_offset", "type": float, "default": 0.0},
            {"name": "--grasp_x_offset", "type": float, "default": -0.015},
            {"name": "--grasp_z_offset", "type": float, "default": -0.03},
            {"name": "--wc4_pregrasp_z_offset", "type": float, "default": 0.0},
            {"name": "--wc4_grasp_z_offset", "type": float, "default": 0.0},
            {"name": "--handle_rotate_right_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_down_distance", "type": float, "default": 0.03},
            {"name": "--handle_rotate_angle", "type": float, "default": 1.05},
            {"name": "--handle_rotate_direction_sign", "type": float, "default": -1.0},
            {"name": "--door_push_distance", "type": float, "default": 1.10},
            {"name": "--no_ikpush_env_randomization", "action": "store_true"},
            {"name": "--ikpush_door_x_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_door_y_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_door_wall_x_offset_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_robot_x_rand", "type": float, "default": 0.03},
            {"name": "--ikpush_robot_x_rand_min", "type": float, "default": -0.70},
            {"name": "--ikpush_robot_x_rand_max", "type": float, "default": 0.0},
            {"name": "--ikpush_robot_y_rand", "type": float, "default": 0.04},
            {"name": "--ikpush_robot_z_rand", "type": float, "default": 0.03},
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
            {"name": "--lever_step_size", "type": float, "default": 0.06},
            {"name": "--push_contact_bias", "type": float, "default": 0.025},
            {"name": "--handle_follow_push_ratio", "type": float, "default": 0.45},
            {"name": "--door_freeze_blend_start_ratio", "type": float, "default": 0.82},
            {"name": "--door_freeze_target_ratio", "type": float, "default": 0.94},
            {"name": "--enable_base_door_collision_check", "dest": "enable_base_door_collision_check", "action": "store_true", "default": False},
            {"name": "--no_base_door_collision_check", "dest": "enable_base_door_collision_check", "action": "store_false"},
            {"name": "--enable_collision_physx_check", "dest": "enable_collision_physx_check", "action": "store_true", "default": False},
            {"name": "--enable_collision_geom_check", "dest": "enable_collision_geom_check", "action": "store_true", "default": False},
            {"name": "--base_door_collision_distance", "type": float, "default": 0.04},
            {"name": "--base_collision_front_extent", "type": float, "default": 0.55},
            {"name": "--base_collision_rear_extent", "type": float, "default": 0.65},
            {"name": "--base_collision_half_width", "type": float, "default": 0.24},
            {"name": "--rigid_contact_geom_gate", "type": float, "default": 0.16},
            {"name": "--collision_log_interval", "type": int, "default": 30},
            {
                "name": "--push_follow_orientation",
                "action": "store_true",
                "help": "After the door unlocks, also keep the end-effector orientation fixed relative to the handle. By default push contact is position-only.",
            },
            {
                "name": "--unidoor_style_push",
                "action": "store_true",
                "default": False,
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
            {"name": "--door_lock_force", "type": float, "default": 0.0},
            {"name": "--door_joint_friction", "type": float, "default": 0.},
            {"name": "--door_joint_damping", "type": float, "default": 0.},
            {"name": "--handle_joint_friction", "type": float, "default": 0.05},
            {"name": "--handle_joint_damping", "type": float, "default": 0.05},
            {"name": "--door_auto_open_force", "type": float, "default": 0.0},
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
            {"name": "--camera_depth_clip_lower", "type": float, "default": 0.02},
            {"name": "--camera_depth_clip_far", "type": float, "default": 2.0},
            {"name": "--camera_display_scale", "type": int, "default": 1},
            {"name": "--camera_display_interval", "type": int, "default": 1},
            {"name": "--camera_axis_scale", "type": float, "default": 0.10},
            {"name": "--camera_axis_thickness", "type": float, "default": 0.004},
            {"name": "--wrist_camera_yaw_deg", "type": float, "default": -90.0},
            {"name": "--wrist_camera_pitch_deg", "type": float, "default": 0.0},
            {"name": "--wrist_camera_roll_deg", "type": float, "default": -60.0},
            {"name": "--front_camera_yaw_deg", "type": float, "default": 0.0},
            {"name": "--front_camera_pitch_deg", "type": float, "default": -45.0},
            {"name": "--front_camera_roll_deg", "type": float, "default": 0.0},
            *dc.depth_aug_custom_parameters(),
            {"name": "--record_dp_dataset", "action": "store_true"},
            {"name": "--dp_raw_root", "type": str, "default": str(HIGH_LEVEL_ROOT / "data" / "door_dp_raw" / "local_door_dp")},
            {"name": "--dp_task", "type": str, "default": "push lever door open"},
            {"name": "--dp_record_env_id", "type": int, "default": 0},
            {"name": "--dp_record_all_envs", "action": "store_true"},
            {"name": "--no_dp_record_all_envs", "action": "store_true"},
            {"name": "--dp_fps", "type": int, "default": 25},
            {"name": "--camera_fps", "type": float, "default": 25.0},
            {"name": "--dp_record_state_mode", "type": str, "default": "full"},
            {"name": "--dp_policy_checkpoint", "type": str, "default": ""},
            {"name": "--dp_control_env_id", "type": int, "default": 0},
            {"name": "--dp_control_all_envs", "action": "store_true"},
            {"name": "--no_dp_control_all_envs", "action": "store_true"},
            {"name": "--dp_inference_steps", "type": int, "default": 10},
            {"name": "--dp_noise_scheduler_type", "type": str, "default": "DDIM"},
            {"name": "--dp_action_horizon", "type": int, "default": -1},
            {"name": "--dp_log_path", "type": str, "default": ""},
            {"name": "--dp_log_interval", "type": int, "default": 25},
            {"name": "--no_dp_print", "dest": "dp_print", "action": "store_false", "default": True},
            {"name": "--dp_warmstart", "action": "store_true"},
            {"name": "--dp_warmstart_raw_episode", "type": str, "default": ""},
            {"name": "--dp_warmstart_step", "type": int, "default": -1},
            {"name": "--dp_warmstart_expert_obs", "dest": "dp_warmstart_expert_obs", "action": "store_true", "default": True},
            {"name": "--no_dp_warmstart_expert_obs", "dest": "dp_warmstart_expert_obs", "action": "store_false"},
            {"name": "--pass_open_angle_deg", "type": float, "default": 80.0},
            {"name": "--no_preview_trajectory_at_spawn", "action": "store_true"},
            {
                "name": "--video_path",
                "type": str,
                "default": str(
                    HIGH_LEVEL_ROOT
                    / "logs"
                    / "float_ik_videos"
                    / f"ikpush_parallel_1024_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                ),
            },
            {"name": "--no_overview_video", "action": "store_true"},
            {"name": "--video_fps", "type": float, "default": 25.0},
            {"name": "--video_capture_stride", "type": int, "default": 2},
            {
                "name": "--video_capture_start_step",
                "type": int,
                "default": 0,
                "help": "Do not encode viewer frames before this simulation step.",
            },
            {"name": "--video_max_frames", "type": int, "default": -1},
            {"name": "--video_output_width", "type": int, "default": 0},
            {"name": "--video_output_height", "type": int, "default": 0},
            {"name": "--video_open_trigger_deg", "type": float, "default": 5.0},
            {"name": "--video_transition_start_step", "type": int, "default": -1},
            {
                "name": "--video_transition_fallback_step",
                "type": int,
                "default": -1,
                "help": "Optional backup step that starts the overview camera pullback even if env0 door has not opened.",
            },
            {"name": "--video_transition_steps", "type": int, "default": 1200},
            {"name": "--video_initial_height", "type": float, "default": 1.8},
            {"name": "--video_start_robot_x_offset", "type": float, "default": 0.8},
            {"name": "--video_start_y_offset", "type": float, "default": 1.10},
            {"name": "--video_start_target_y_offset", "type": float, "default": 0.0},
            {"name": "--video_far_height", "type": float, "default": 22.0},
            {"name": "--video_far_margin", "type": float, "default": 12.0},
            {"name": "--no_video_colorful_walls", "action": "store_true"},
            {"name": "--video_wall_color_saturation", "type": float, "default": 0.62},
            {"name": "--video_wall_color_value", "type": float, "default": 0.88},
            {
                "name": "--video_randomize_asset_colors",
                "action": "store_true",
                "help": (
                    "Assign one random shared color to the A2W base and Z1 arm in every environment. "
                    "Door assets retain their original materials."
                ),
            },
            {"name": "--video_asset_color_saturation_min", "type": float, "default": 0.42},
            {"name": "--video_asset_color_saturation_max", "type": float, "default": 0.78},
            {"name": "--video_asset_color_value_min", "type": float, "default": 0.68},
            {"name": "--video_asset_color_value_max", "type": float, "default": 0.96},
            {"name": "--video_keep_debug", "action": "store_true"},
            {"name": "--video_keep_low_level_cameras", "action": "store_true"},
        ],
    )

    # gymutil's wrapper does not preserve default=True for store_true custom args,
    # so keep these visualization helpers on by default and let --no_* flags opt out.
    argv = set(sys.argv[1:])
    robot_body = str(getattr(args, "robot_body", "b1z1")).lower()
    if Path(args.door_cfg).name == "b1z1_opendoor_door4.yaml" and not any(
        token == "--env_spacing" or token.startswith("--env_spacing=") for token in sys.argv[1:]
    ):
        args.env_spacing = 10.0
    robot_z_was_set = any(token == "--robot_z" or token.startswith("--robot_z=") for token in sys.argv[1:])
    if robot_body in ("b1z1", "b1", "b1z1_base") and not robot_z_was_set:
        args.robot_z = float(getattr(args, "b1_robot_z", 0.50))
    elif is_scout_robot_body_name(robot_body) and not robot_z_was_set:
        args.robot_z = float(getattr(args, "scout_robot_z", 0.22))
    elif is_a2w_robot_body_name(robot_body) and not robot_z_was_set:
        args.robot_z = float(getattr(args, "a2w_robot_z", 0.50))
    elif is_g1_robot_body(args) and not robot_z_was_set:
        args.robot_z = float(getattr(args, "g1_robot_z", 0.80))
    door_prefer_was_set = any(
        token == "--door_prefer_name" or token.startswith("--door_prefer_name=") for token in sys.argv[1:]
    )
    if (
        not door_prefer_was_set
        and not str(getattr(args, "door_name", "") or "")
        and int(getattr(args, "door_index", -1)) < 0
    ):
        preferred = DOOR_CFG_DEFAULT_PREFER.get(Path(args.door_cfg).name)
        if preferred is None:
            preferred = ROBOT_BODY_DEFAULT_DOOR_PREFER.get(robot_body)
        if preferred:
            args.door_prefer_name = preferred
            print(
                f"Robot body {robot_body!r} uses default env0 preferred door: {args.door_prefer_name}",
                flush=True,
            )
    if is_g1_robot_body(args):
        robot_front_offset_was_set = any(
            token == "--robot_front_offset" or token.startswith("--robot_front_offset=") for token in sys.argv[1:]
        )
        stop_distance_was_set = any(
            token == "--stop_distance" or token.startswith("--stop_distance=") for token in sys.argv[1:]
        )
        if not robot_front_offset_was_set:
            args.robot_front_offset = float(getattr(args, "g1_robot_front_offset", 0.35))
        if not stop_distance_was_set:
            args.stop_distance = float(getattr(args, "g1_stop_distance", 0.10))
        ik_ee_was_set = any(token == "--ik_ee_link" or token.startswith("--ik_ee_link=") for token in sys.argv[1:])
        if not ik_ee_was_set:
            args.ik_ee_link = str(getattr(args, "g1_ee_link", "right_rubber_hand"))
        if not bool(getattr(args, "g1_use_orientation_ik", False)):
            args.ik_position_only = True
        g1_ik_joints = [
            part.strip()
            for part in str(getattr(args, "g1_right_arm_ik_joints", "")).replace(";", ",").split(",")
            if part.strip()
        ]
        if not g1_ik_joints:
            raise ValueError("--g1_right_arm_ik_joints must select at least one joint for --robot_body g1.")
        base_ik.ARM_IK_DOF_NAMES = set(g1_ik_joints)
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
            setattr(args, attr, True)

    if "--no_show_seg" in argv:
        args.show_camera_images = False
    elif "--show_seg" in argv:
        args.show_camera_images = True
    args.camera_rgb = bool(args.camera_rgb or args.rgb)
    args.camera_depth = bool((args.camera_depth or not args.no_camera_depth) and not args.rgb)
    if bool(getattr(args, "rgb", False)):
        args.depth_only = False
    args.camera_seg = bool(args.camera_seg or not args.no_camera_seg)
    args.dp_record_all_envs = not bool(args.no_dp_record_all_envs)
    args.dp_control_all_envs = not bool(args.no_dp_control_all_envs)
    args.dp_print = "--no_dp_print" not in argv
    args.enable_base_door_collision_check = "--enable_base_door_collision_check" in argv and "--no_base_door_collision_check" not in argv
    args.enable_collision_physx_check = args.enable_base_door_collision_check or "--enable_collision_physx_check" in argv
    args.enable_collision_geom_check = args.enable_base_door_collision_check or "--enable_collision_geom_check" in argv
    args.dp_action_horizon = None if int(args.dp_action_horizon) < 0 else int(args.dp_action_horizon)
    args.record_overview_video = bool(args.video_path and not args.no_overview_video)
    if args.record_overview_video and args.headless:
        raise ValueError("Overview video recording uses the Isaac Gym viewer; run without --headless.")
    if args.record_overview_video and cv2 is None:
        raise RuntimeError("OpenCV is required to encode the overview mp4 video.")
    if args.record_overview_video and not args.video_keep_debug:
        args.draw_ik_target = False
        args.draw_camera_axes = False
    if args.record_overview_video and not args.video_keep_low_level_cameras and not (
        args.dp_policy_checkpoint or args.record_dp_dataset
    ):
        args.show_camera_images = False
        args.enable_wrist_camera = False
        args.enable_front_camera = False
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if not 0.0 <= args.video_asset_color_saturation_min <= args.video_asset_color_saturation_max <= 1.0:
        raise ValueError("--video_asset_color_saturation_min/max must satisfy 0 <= min <= max <= 1.")
    if not 0.0 <= args.video_asset_color_value_min <= args.video_asset_color_value_max <= 1.0:
        raise ValueError("--video_asset_color_value_min/max must satisfy 0 <= min <= max <= 1.")
    if not args.dp_record_all_envs and (args.dp_record_env_id < 0 or args.dp_record_env_id >= args.num_envs):
        raise ValueError("--dp_record_env_id must be in [0, num_envs - 1].")
    if args.dp_policy_checkpoint and (args.dp_control_env_id < 0 or args.dp_control_env_id >= args.num_envs):
        raise ValueError("--dp_control_env_id must be in [0, num_envs - 1].")
    if args.dp_policy_checkpoint and args.dp_warmstart and args.dp_control_all_envs and args.num_envs > 1:
        raise ValueError("--dp_warmstart currently supports a single controlled env; add --no_dp_control_all_envs.")
    if args.record_dp_dataset and args.dp_policy_checkpoint:
        raise ValueError("--record_dp_dataset and --dp_policy_checkpoint are separate modes; run recording or policy play, not both.")
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
    args.door_motion_sign = -1.0
    args.pass_through_door = not bool(args.no_pass_through_door)
    apply_scripted_trajectory_speed_scale(args, argv)
    return args


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
    yaw_start: float
    yaw_push: float
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
    last_phase: str = "init"
    last_handle_goal: object = None
    last_door_pos: object = None
    last_target_pos: object = None
    last_target_quat: object = None
    last_gripper: float = 0.0
    base_door_collision_detected: bool = False
    base_door_collision_log_step: int = -10**9


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


def colorful_wall_rgb(env_index, args):
    seed = int(getattr(args, "seed", 0))
    base_hue = (float(seed % 1000003) * 0.618033988749895 + 0.17320508075688773) % 1.0
    hue = (base_hue + float(env_index) * 0.618033988749895) % 1.0
    saturation = min(1.0, max(0.0, float(getattr(args, "video_wall_color_saturation", 0.62))))
    value = min(1.0, max(0.0, float(getattr(args, "video_wall_color_value", 0.88))))
    return colorsys.hsv_to_rgb(hue, saturation, value)


def sample_video_asset_colors(args, env_index):
    """Sample one reproducible robot color shared by the A2W base and Z1 arm."""
    env_seed = seed_for_env(args, env_index)
    rng = np.random.default_rng(np.random.SeedSequence([int(env_seed), 0xC010A]))
    hue = float(rng.uniform(0.0, 1.0))
    saturation_min = float(args.video_asset_color_saturation_min)
    saturation_max = float(args.video_asset_color_saturation_max)
    value_min = float(args.video_asset_color_value_min)
    value_max = float(args.video_asset_color_value_max)

    saturation = float(rng.uniform(saturation_min, saturation_max))
    value = float(rng.uniform(value_min, value_max))
    robot_rgb = tuple(float(channel) for channel in colorsys.hsv_to_rgb(hue, saturation, value))
    return {"robot": robot_rgb}


def set_actor_visual_color(gym, env, actor, rgb, body_indices=None):
    if actor is None:
        return
    if body_indices is None:
        body_indices = range(int(gym.get_actor_rigid_body_count(env, actor)))
    color = gymapi.Vec3(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    for body_index in body_indices:
        gym.set_rigid_body_color(env, actor, int(body_index), gymapi.MESH_VISUAL, color)


def apply_video_asset_colors(gym, env, actor_handles, arm_actor, door_actor, door, env_args, env_index):
    if not bool(getattr(env_args, "video_randomize_asset_colors", False)):
        return
    colors = getattr(env_args, "video_asset_colors", None)
    if not colors:
        colors = sample_video_asset_colors(env_args, env_index)

    base_actor = next((actor for actor in actor_handles if actor != arm_actor), None)
    robot_rgb = colors["robot"]
    set_actor_visual_color(gym, env, base_actor, robot_rgb)
    set_actor_visual_color(gym, env, arm_actor, robot_rgb)
    if int(env_index) < 4:
        rounded = [round(float(channel), 3) for channel in robot_rgb]
        print(
            f"video_robot_color env={env_index} door={door.spec.get('name')} "
            f"shared_a2w_z1_rgb={rounded} door_material=original",
            flush=True,
        )


def make_env_args(args, env_index):
    env_args = SimpleNamespace(**vars(args))
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
    set_sampled("robot_y", "ikpush_robot_y_rand")
    set_sampled("robot_z", "ikpush_robot_z_rand")
    set_sampled("robot_yaw", "ikpush_robot_yaw_rand")
    set_fixed("pregrasp_offset")
    set_fixed("grasp_x_offset")
    set_fixed("grasp_z_offset")
    set_fixed("handle_rotate_angle")
    set_fixed("door_push_distance")
    set_sampled("door_joint_friction", "ikpush_door_joint_friction_rand", lower=0.0)
    set_sampled("door_joint_damping", "ikpush_door_joint_damping_rand", lower=0.0)
    set_sampled("handle_joint_friction", "ikpush_handle_joint_friction_rand", lower=0.0)
    set_sampled("handle_joint_damping", "ikpush_handle_joint_damping_rand", lower=0.0)
    set_sampled("handle_spring_stiffness", "ikpush_handle_spring_stiffness_rand", lower=0.0)
    set_sampled("handle_spring_damping", "ikpush_handle_spring_damping_rand", lower=0.0)

    if not bool(getattr(args, "no_video_colorful_walls", False)):
        wall_r, wall_g, wall_b = colorful_wall_rgb(env_index, args)
        env_args.door_wall_color_r = float(wall_r)
        env_args.door_wall_color_g = float(wall_g)
        env_args.door_wall_color_b = float(wall_b)
        sampled["door_wall_color_rgb"] = [float(wall_r), float(wall_g), float(wall_b)]

    if bool(getattr(args, "video_randomize_asset_colors", False)):
        env_args.video_asset_colors = sample_video_asset_colors(args, env_index)
        sampled["video_asset_colors"] = {
            name: [float(channel) for channel in rgb]
            for name, rgb in env_args.video_asset_colors.items()
        }

    depth_noise_selected = dc.configure_depth_noise_for_env(env_args)
    sampled["depth_noise_selection_mode"] = str(env_args.depth_noise_selection_mode)
    sampled["depth_noise_env_probability"] = float(env_args.depth_noise_env_probability)
    sampled["depth_noise_selected_env_count"] = int(env_args.depth_noise_selected_env_count)
    sampled["depth_noise_env_selected"] = bool(depth_noise_selected)

    env_args.ikpush_randomization_json = json.dumps(sampled, sort_keys=True)
    return env_args


def normalize_video_door_layout(door, args, env_index=0):
    """Make mixed door assets face the same way in overview videos.

    Legacy PartNet numeric URDFs already contain a fixed internal visual
    rotation, so their actor yaw must stay on the normal controller value.  Only
    their bbox used for video walls is authored in the other horizontal axis; add
    a bbox-only +90 deg offset so walls use the visible door width.
    """
    family = dc.door_asset_family(door)
    if family == "partnet_numeric":
        door.spec = dict(door.spec)
        door.spec["wall_opening_axis_override"] = "y"
        door.spec["bounds_yaw_offset_override"] = math.pi / 2.0
        door.spec["video_layout_family"] = family
        door.spec["video_layout_actor_yaw"] = float(door.actor_yaw)
    elif family in ("record_materialization", "wc4"):
        door.spec = dict(door.spec)
        door.spec["wall_opening_axis_override"] = "y"
        door.spec.pop("bounds_yaw_offset_override", None)
        door.spec["video_layout_family"] = family
        door.spec["video_layout_actor_yaw"] = float(door.actor_yaw)
    if int(env_index) < 4:
        print(
            f"video_door_layout env={env_index} door={door.spec.get('name')} "
            f"family={family} actor_yaw={float(door.actor_yaw):.3f} "
            f"wall_axis={door.spec.get('wall_opening_axis_override', '')} "
            f"bounds_yaw_offset={float(door.spec.get('bounds_yaw_offset_override', 0.0) or 0.0):.3f}",
            flush=True,
        )
    return door


def apply_video_door_allowlist(args):
    include_text = str(getattr(args, "door_include_names", "") or "").strip()
    if not include_text:
        return
    include_names = [name.strip() for name in include_text.split(",") if name.strip()]
    include_set = set(include_names)

    cfg_path = Path(args.door_cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as stream:
        cfg = dc.yaml.safe_load(stream)
    asset_cfg = cfg["env"]["asset"]
    train_assets = asset_cfg["trainAssets"]
    load_block = asset_cfg.get("load_block") or next(iter(train_assets))
    configured_names = {
        str(spec.get("name", ""))
        for spec in train_assets[load_block].values()
    }
    missing = [name for name in include_names if name not in configured_names]
    if missing:
        raise ValueError(f"--door_include_names contains names absent from {cfg_path}: {missing}")

    existing_excludes = {
        name.strip()
        for name in str(getattr(args, "door_exclude_names", "") or "").split(",")
        if name.strip()
    }
    excluded = (configured_names - include_set) | existing_excludes
    args.door_exclude_names = ",".join(sorted(excluded))
    args.door_selection = "all"
    args.door_max_unique_assets = len(include_names)
    print(
        f"Video door allowlist: {len(include_names)} validated assets; "
        f"env0 preferred={getattr(args, 'door_prefer_name', '')!r}",
        flush=True,
    )


set_robot_base_pose = dc.set_robot_base_pose
compute_base_push_target = dc.compute_base_push_target
get_body_pose = dc.get_body_pose
get_actor_body_index = dc.get_actor_body_index
gym_quat_to_np = dc.gym_quat_to_np
local_camera_pose_from_cfg = dc.local_camera_pose_from_cfg
draw_local_camera_axes = dc.draw_local_camera_axes
draw_low_level_camera_axes = dc.draw_low_level_camera_axes
make_camera_properties = dc.make_camera_properties
attach_camera_to_actor_body = dc.attach_camera_to_actor_body
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
make_float_dp_action = dc.make_float_dp_action
apply_float_dp_action = dc.apply_float_dp_action
make_float_dp_policy_log_record = dc.make_float_dp_policy_log_record
print_float_dp_policy_log_record = dc.print_float_dp_policy_log_record
make_float_replay_snapshot = dc.make_float_replay_snapshot


def create_low_level_cameras(gym, env, arm_actor, actor_handles, args):
    if not is_g1_robot_body(args):
        return dc.create_low_level_cameras(gym, env, arm_actor, actor_handles, args)

    cameras = {}
    if args.enable_wrist_camera:
        wrist_rot = dc.wrist_camera_rotation_radians_from_args(args)
        wrist_camera = None
        for link_name in (str(getattr(args, "g1_ee_link", "right_rubber_hand")), "right_wrist_yaw_link"):
            wrist_camera = attach_camera_to_actor_body(
                gym, env, arm_actor, link_name, dc.DEFAULT_WRIST_CAMERA_CFG, wrist_rot, args=args
            )
            if wrist_camera is not None:
                print(f"G1 wrist camera attached to {link_name}: handle={wrist_camera}", flush=True)
                break
        if wrist_camera is None:
            print("⚠️📷 G1 wrist camera sensor creation failed.", flush=True)
        else:
            cameras["wrist"] = wrist_camera

    if args.enable_front_camera:
        front_rot = [
            math.radians(float(args.front_camera_yaw_deg)),
            math.radians(float(args.front_camera_pitch_deg)),
            math.radians(float(args.front_camera_roll_deg)),
        ]
        front_camera = None
        for link_name in ("head_link", "torso_link", "pelvis"):
            front_camera = attach_camera_to_actor_body(
                gym, env, arm_actor, link_name, dc.DEFAULT_FRONT_CAMERA_CFG, front_rot, args=args
            )
            if front_camera is not None:
                print(f"G1 front camera attached to {link_name}: handle={front_camera}", flush=True)
                break
        if front_camera is None:
            print("⚠️📷 G1 front camera sensor creation failed.", flush=True)
        else:
            cameras["front"] = front_camera

    if args.show_camera_images:
        if cv2 is None:
            print("⚠️📷 cv2 is not available; camera image windows are disabled.", flush=True)
        elif not cameras:
            print("⚠️📷 No G1 camera sensors were created; camera image windows are disabled.", flush=True)
    return cameras


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
    if n >= 1 and abs(float(door_pos[0]) - float(st.door.dof_upper[0])) > 1.0e-4:
        st.door.open_stage = True
    if n >= 2 and float(door_pos[1] - st.door.dof_lower[1]) >= float(st.door.handle_unlock_threshold):
        st.door.open_stage = True

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
        args = (
            np.asarray(data["state"][idx], dtype=np.float32),
            np.asarray(data[image_keys[0]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[1]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[2]][idx], dtype=np.uint8),
            np.asarray(data[image_keys[3]][idx], dtype=np.uint8),
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


def _legacy_video_trajectory_targets(
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
    yaw_start,
    yaw_push,
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

    pregrasp = handle_goal + approach_dir * args.pregrasp_offset
    grasp = handle_goal + approach_dir * args.grasp_offset
    pregrasp[0] += args.grasp_x_offset
    pregrasp[2] += args.grasp_z_offset
    grasp[0] += args.grasp_x_offset
    grasp[2] += args.grasp_z_offset
    goal_quat = forward_ee_quat(args, yaw_start)

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
    push_dir = -pull_dir
    push_pos = rotate_pos + push_dir * args.door_push_distance

    walk_end = args.walk_steps
    initial_end = walk_end + args.initial_hold_steps
    grasp_end = initial_end + args.grasp_steps
    grasp_hold_end = grasp_end + args.grasp_hold_steps
    close_end = grasp_hold_end + args.gripper_close_steps
    rotate_end = close_end + args.handle_rotate_steps
    push_end = rotate_end + args.door_push_steps
    return_home_end = push_end + args.return_home_steps

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
        t = smoothstep((step + 1) / max(1, args.walk_steps))
        base_xy = lerp(base_start, base_stop, t)
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
                quat_from_angle_axis(-t * args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            gripper = gripper_closed
            phase = "rotate_handle"
        elif is_g1_robot_body(args):
            target_pos = traj["rotate"].copy()
            target_quat = None if args.ik_position_only else base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            base_xy = base_stop.copy()
            yaw = yaw_start
            gripper = gripper_closed
            phase = "push_door"
        elif step < push_end:
            t = smoothstep((step - rotate_end + 1) / max(1, args.door_push_steps))
            base_t = smoothstep((step - rotate_end + 1) / max(1.0, args.door_push_steps * args.base_push_time_scale))
            turned_quat = base_ik.quat_multiply(
                traj["goal_quat"],
                quat_from_angle_axis(-args.handle_rotate_angle, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            )
            if "handle_contact_offset_local" not in traj:
                traj["handle_contact_offset_local"] = quat_apply(
                    base_ik.quat_conjugate(handle_quat),
                    traj["rotate"] - handle_goal,
                )
                traj["handle_contact_quat_local"] = base_ik.quat_multiply(
                    base_ik.quat_conjugate(handle_quat),
                    turned_quat,
                )
            live_push_dir = traj["push_dir"].copy()
            if door.open_stage:
                live_pull_dir = quat_axis(handle_quat, axis=2)
                live_pull_dir[2] = 0.0
                live_pull_dir = normalize(live_pull_dir)
                if np.linalg.norm(live_pull_dir) >= 1.0e-5:
                    if float(np.dot(live_pull_dir, traj["approach_dir"])) < 0.0:
                        live_pull_dir = -live_pull_dir
                    live_push_dir = -live_pull_dir

            follow_handle = (
                door.open_stage
                and t <= args.handle_follow_push_ratio
                and "handle_contact_offset_local" in traj
            )
            freeze_ee_target = False
            door_pos, _ = get_actor_dof_state(gym, env, door_actor)
            door_open_ratio = (
                door_hinge_open_ratio(door, float(door_pos[0]), args) if len(door_pos) > 0 else 0.0
            )
            if "handle_contact_offset_local" in traj:
                handle_contact_offset = quat_apply(handle_quat, traj["handle_contact_offset_local"])
                handle_target_pos = handle_goal + handle_contact_offset + live_push_dir * args.push_contact_bias
            else:
                handle_target_pos = handle_goal.copy()
            if (
                door.open_stage
                and len(door_pos) > 0
                and ik_state.current_pos_np is not None
                and door_open_ratio >= args.door_freeze_target_ratio
            ):
                if "door_open_freeze_target_pos" not in traj:
                    traj["door_open_freeze_target_pos"] = handle_target_pos.copy()
                target_pos = traj["door_open_freeze_target_pos"].copy()
                target_quat = None
                freeze_ee_target = True
            elif follow_handle:
                target_pos = handle_target_pos.copy()
                target_quat = None
                if args.push_follow_orientation and not args.ik_position_only and "handle_contact_quat_local" in traj:
                    target_quat = base_ik.quat_multiply(handle_quat, traj["handle_contact_quat_local"])
            elif door.open_stage and ik_state.current_pos_np is not None:
                target_pos = ik_state.current_pos_np + live_push_dir * args.lever_step_size
                blend_start = min(float(args.door_freeze_blend_start_ratio), float(args.door_freeze_target_ratio) - 1.0e-4)
                if door_open_ratio >= blend_start:
                    blend_t = smoothstep(
                        (door_open_ratio - blend_start)
                        / max(1.0e-4, float(args.door_freeze_target_ratio) - blend_start)
                    )
                    target_pos = lerp(target_pos, handle_target_pos, blend_t)
                target_quat = None
            elif args.unidoor_style_push and ik_state.current_pos_np is not None:
                push_step_pos = ik_state.current_pos_np + traj["push_dir"] * args.lever_step_size
                push_max_pos = traj["rotate"] + traj["push_dir"] * args.door_push_distance
                progress = float(np.dot(push_step_pos - traj["rotate"], traj["push_dir"]))
                target_pos = push_max_pos if progress > args.door_push_distance else push_step_pos
            else:
                target_pos = lerp(traj["rotate"], traj["push"], t)
            if freeze_ee_target:
                target_quat = None
            else:
                target_quat = None if args.ik_position_only else turned_quat
            base_xy = lerp(base_stop, base_push, base_t)
            yaw = float(lerp(np.array([yaw_start], dtype=np.float32), np.array([yaw_push], dtype=np.float32), base_t)[0])
            gripper = gripper_closed
            if door.open_stage:
                if "gripper_loosen_start_step" not in traj:
                    traj["gripper_loosen_start_step"] = step
                loosen_t = smoothstep(
                    (step - traj["gripper_loosen_start_step"] + 1) / max(1, args.gripper_loosen_steps)
                )
                gripper = float(lerp(
                    np.array([gripper_closed], dtype=np.float32),
                    np.array([gripper_open_stage], dtype=np.float32),
                    loosen_t,
                )[0])
            phase = "push_door"
        elif step < return_home_end:
            t = smoothstep((step - push_end + 1) / max(1, args.return_home_steps))
            if "return_home_start_base_xy" not in traj:
                traj["return_home_start_base_xy"] = traj.get("base_xy", base_push).copy()
                traj["return_home_start_yaw"] = float(traj.get("yaw", yaw_push))
            base_xy = lerp(traj["return_home_start_base_xy"], base_push, t)
            yaw = float(lerp(
                np.array([traj["return_home_start_yaw"]], dtype=np.float32),
                np.array([yaw_push], dtype=np.float32),
                t,
            )[0])
            target_pos, target_quat = chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                traj["push"].copy(),
                traj.get("goal_quat"),
            )
            gripper = args.gripper_open
            traj["return_home_alpha"] = t
            phase = "return_home"
        else:
            home_base_pos = traj.get("home_ee_base_pos")
            fallback_pos = (
                base_pos_to_world(home_base_pos, base_push, args.robot_z, yaw_push)
                if home_base_pos is not None
                else traj["push"].copy()
            )
            fallback_quat = None
            if not args.ik_position_only and "home_ee_base_quat" in traj:
                fallback_quat = base_quat_to_world(traj["home_ee_base_quat"], yaw_push)
            target_pos, target_quat = chase_target_to_current_ee(
                traj,
                ik_state,
                args,
                fallback_pos,
                fallback_quat,
            )
            base_xy = base_push.copy()
            yaw = yaw_push
            gripper = args.gripper_open
            traj["return_home_alpha"] = 1.0
            phase = "hold_home"

    traj["base_xy"] = base_xy.copy()
    traj["yaw"] = float(yaw)
    traj["last_target_pos"] = np.asarray(target_pos, dtype=np.float32).copy()
    traj["last_target_quat"] = (
        None
        if target_quat is None
        else base_ik.normalize_quat(np.asarray(target_quat, dtype=np.float32)).astype(np.float32)
    )
    return phase, base_xy, yaw, target_pos, target_quat, gripper, handle_goal


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
    yaw_start,
    yaw_push,
    traj,
):
    """Use the current A2W scripted trajectory instead of a video-local copy."""
    if a2w_scripted is None:
        raise RuntimeError(
            "The video recorder requires "
            "isaacgym_float_ik_a2w_basearn_push_door_parallel.py so both scripts use the same trajectory."
        )
    # The overview recorder does not run a DoorTwin skill program with a
    # separate traverse phase, so the push target is also the final base target.
    return a2w_scripted.trajectory_targets(
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
        base_push,
        yaw_start,
        yaw_push,
        yaw_push,
        traj,
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
    base_push = compute_base_push_target(args, base_stop, heading)
    yaw_push = yaw_start + args.push_base_yaw_delta
    traj = {"base_xy": base_start.copy()}

    print("base_start:", base_start.tolist(), "base_stop:", base_stop.tolist(), "base_push:", base_push.tolist())
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
            "ikpush",
            IKPUSH_STATE_VERSION,
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
            traj=traj,
            dp_recorder=dp_recorder,
            dp_record_success=False,
            dp_record_warned_no_camera=False,
            base_door_collision_detected=False,
            base_door_collision_log_step=-10**9,
            last_dp_action=prev_dp_action.copy(),
        )
    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
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
            yaw_start,
            yaw_push,
            traj,
        )
        set_robot_base_pose(gym, env, actor_handles, base_xy, args.robot_z, yaw)
        if phase == "return_home":
            if "return_home_start_dofs" not in traj:
                traj["return_home_start_dofs"] = np.asarray(dof_positions, dtype=np.float32).copy()
            alpha = float(traj.get("return_home_alpha", 0.0))
            dof_positions[:] = lerp(traj["return_home_start_dofs"], home_positions, alpha)
            ik_state.last_pos_error = 0.0
        elif phase in ("walk", "hold_home"):
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
        if viewer is not None and need_camera_render and (args.draw_ik_target or args.draw_camera_axes):
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
            if args.draw_ik_target or args.draw_camera_axes:
                gym.clear_lines(viewer)
            if args.draw_camera_axes:
                draw_low_level_camera_axes(gym, viewer, env, arm_actor, actor_handles, args)
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
    base_push = compute_base_push_target(args, base_stop, heading)
    return ParallelEnvState(
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
        yaw_start=yaw_start,
        yaw_push=yaw_start + args.push_base_yaw_delta,
        traj={"base_xy": base_start.copy()},
        dp_recorder=dp_recorder,
    )


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
):
    dc.require_float_dp_recording_deps(args, RawDoorDPRecorder, make_state_feature_names)
    vision_mode = dc.float_dp_vision_mode(args, normalize_vision_mode)

    envs_per_row = max(1, int(math.ceil(math.sqrt(float(args.num_envs)))))
    record_env_ids = dc.float_dp_record_env_ids(args)
    created = []
    for env_index in range(int(args.num_envs)):
        door_template = door_templates[env_index % len(door_templates)]
        door = clone_door_runtime(door_template)
        env_args = make_env_args(args, env_index)
        normalize_video_door_layout(door, env_args, env_index=env_index)
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
        apply_video_asset_colors(
            gym,
            env,
            actor_handles,
            arm_actor,
            door_actor,
            door,
            env_args,
            env_index,
        )
        created.append((env_index, env_args, env, arm_actor, actor_handles, door, door_actor))

    camera_handles_by_env = {}
    for env_index, env_args, env, arm_actor, actor_handles, _door, _door_actor in created:
        camera_handles = {}
        if (env_args.show_camera_images or env_args.record_dp_dataset or env_args.dp_policy_checkpoint) and (
            env_args.enable_wrist_camera or env_args.enable_front_camera
        ):
            camera_handles = create_low_level_cameras(gym, env, arm_actor, actor_handles, env_args)
        camera_handles_by_env[env_index] = camera_handles

    # Isaac Gym must finalize the complete parallel scene once, after every
    # environment and actor has been created. Re-preparing the growing scene
    # once per environment can corrupt GPU PhysX state at large env counts.
    gym.prepare_sim(sim)

    env_states = []
    for env_index, env_args, env, arm_actor, actor_handles, door, door_actor in created:
        camera_handles = camera_handles_by_env[env_index]
        ik_state = base_ik.setup_ik_controller(
            gym,
            sim,
            env,
            arm_actor,
            arm_asset,
            dof_names,
            lower,
            upper,
            env_args,
            prepare_sim=False,
        )
        dp_recorder = None
        if env_args.record_dp_dataset and env_index in record_env_ids:
            dp_recorder = dc.make_float_dp_recorder(
                env_args,
                door,
                env_index,
                vision_mode,
                DP_PHASE_NAMES,
                "ikpush",
                IKPUSH_STATE_VERSION,
                RawDoorDPRecorder,
                make_state_feature_names,
                randomization_metadata_key="ikpush_randomization",
            )
        env_states.append(
            initialize_parallel_env_state(
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
        )
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


def smoothstep01(value):
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def vec3_to_np(vec):
    return np.array([float(vec.x), float(vec.y), float(vec.z)], dtype=np.float32)


def np_to_vec3(values):
    arr = np.asarray(values, dtype=np.float32).reshape(3)
    return gymapi.Vec3(float(arr[0]), float(arr[1]), float(arr[2]))


class OverviewVideoRecorder:
    def __init__(self, gym, sim, viewer, env_states, args):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.env_states = env_states
        self.args = args
        self.enabled = bool(getattr(args, "record_overview_video", False))
        self.writer = None
        self.frame_count = 0
        self.trigger_step = None
        self.tmp_dir = None
        self.path = Path(getattr(args, "video_path", ""))
        self.anchors = None
        if not self.enabled:
            return
        if viewer is None:
            raise RuntimeError("Overview video recording requires a non-headless Isaac Gym viewer.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="ikpush_parallel_video_frames_")
        self.anchors = self._compute_camera_anchors()
        print(
            f"Overview video: path={self.path} fps={float(args.video_fps):.2f} "
            f"capture_stride={int(args.video_capture_stride)} "
            f"trigger_deg={float(args.video_open_trigger_deg):.1f} "
            f"transition_steps={int(args.video_transition_steps)}",
            flush=True,
        )

    def _env_door_focus_world(self, st, z=0.8):
        origin = vec3_to_np(self.gym.get_env_origin(st.env))
        alignment = str(getattr(st.args, "robot_y_alignment", "auto")).lower()
        use_generated_center = False
        if alignment in ("auto", "generated_auto", "record_materialization_auto"):
            door_name = str(st.door.spec.get("name", ""))
            door_path = str(st.door.spec.get("path", ""))
            use_generated_center = door_name.startswith("rec_") or "record_materialization" in door_path
        if alignment in ("door_center", "door", "center") or use_generated_center:
            world_min_x, world_max_x, world_min_y, world_max_y = dc.door_world_xy_bounds_from_bbox(st.args, st.door)
            return origin + np.array(
                [
                    0.5 * (world_min_x + world_max_x) + 0.3,
                    0.5 * (world_min_y + world_max_y),
                    float(z),
                ],
                dtype=np.float32,
            )
        return origin + np.array([float(st.args.door_x) + 0.3, float(st.args.door_y), float(z)], dtype=np.float32)

    def _compute_camera_anchors(self):
        first = self.env_states[0]
        far_index = max(0, len(self.env_states) - 1)
        far = self.env_states[far_index]
        start_target = self._env_door_focus_world(first, z=0.8)
        start_target[1] += float(getattr(self.args, "video_start_target_y_offset", 0.0))
        far_target = self._env_door_focus_world(far, z=0.8)
        center_target = 0.5 * (start_target + far_target)
        center_target[2] = 1.1

        start_origin = vec3_to_np(self.gym.get_env_origin(first.env))
        start_pos = start_origin + np.array(
            [
                float(first.args.robot_x) + float(getattr(self.args, "video_start_robot_x_offset", 0.25)),
                float(first.args.robot_y) + float(getattr(self.args, "video_start_y_offset", 1.10)),
                float(self.args.video_initial_height),
            ],
            dtype=np.float32,
        )

        span_xy = far_target[:2] - start_target[:2]
        span_norm = float(np.linalg.norm(span_xy))
        if span_norm < 1.0e-6:
            diag = np.array([1.0, 1.0], dtype=np.float32) / math.sqrt(2.0)
        else:
            diag = span_xy / span_norm
        side = np.array([-diag[1], diag[0]], dtype=np.float32)
        far_margin = float(self.args.video_far_margin)
        end_pos = far_target.copy()
        end_pos[:2] = far_target[:2] + diag * far_margin + side * (0.5 * far_margin)
        end_pos[2] = float(self.args.video_far_height)

        print(
            "Overview camera anchors: "
            f"env0_start_pos={start_pos.round(3).tolist()} "
            f"env0_target={start_target.round(3).tolist()} "
            f"far_env={far_index} far_target={far_target.round(3).tolist()} "
            f"end_pos={end_pos.round(3).tolist()}",
            flush=True,
        )
        return {
            "start_pos": start_pos,
            "start_target": start_target,
            "end_pos": end_pos,
            "end_target": center_target,
        }

    def update_camera(self, step):
        if not self.enabled or self.anchors is None:
            return
        explicit_start = int(getattr(self.args, "video_transition_start_step", -1))
        if self.trigger_step is None and explicit_start >= 0 and step >= explicit_start:
            self.trigger_step = int(step)
            print(f"Overview camera transition started by explicit step={step}", flush=True)
        fallback_start = int(getattr(self.args, "video_transition_fallback_step", -1))
        if self.trigger_step is None and fallback_start >= 0 and step >= fallback_start:
            self.trigger_step = int(step)
            print(f"Overview camera transition started by fallback step={step}", flush=True)
        if self.trigger_step is None:
            door_pos = self.env_states[0].last_door_pos
            door_deg = 0.0
            if door_pos is not None and len(door_pos):
                door_deg = abs(math.degrees(float(door_pos[0])))
            if door_deg >= float(self.args.video_open_trigger_deg):
                self.trigger_step = int(step)
                print(f"Overview camera transition triggered at step={step} door0={door_deg:.1f}deg", flush=True)

        if self.trigger_step is None:
            alpha = 0.0
        else:
            denom = max(1, int(getattr(self.args, "video_transition_steps", 1200)))
            alpha = smoothstep01((int(step) - int(self.trigger_step)) / float(denom))

        pos = (1.0 - alpha) * self.anchors["start_pos"] + alpha * self.anchors["end_pos"]
        target = (1.0 - alpha) * self.anchors["start_target"] + alpha * self.anchors["end_target"]
        self.gym.viewer_camera_look_at(self.viewer, None, np_to_vec3(pos), np_to_vec3(target))

    def capture(self, step):
        if not self.enabled:
            return
        if int(step) < max(0, int(getattr(self.args, "video_capture_start_step", 0))):
            return
        max_frames = int(getattr(self.args, "video_max_frames", -1))
        if max_frames >= 0 and self.frame_count >= max_frames:
            return
        stride = max(1, int(getattr(self.args, "video_capture_stride", 2)))
        if int(step) % stride != 0:
            return
        frame_path = Path(self.tmp_dir.name) / f"frame_{self.frame_count:06d}.png"
        self.gym.write_viewer_image_to_file(self.viewer, str(frame_path))
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read viewer frame written to {frame_path}")
        output_width = int(getattr(self.args, "video_output_width", 0))
        output_height = int(getattr(self.args, "video_output_height", 0))
        if output_width > 0 and output_height > 0:
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, float(self.args.video_fps), (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {self.path}")
            print(f"Overview video frame size: {width}x{height}", flush=True)
        self.writer.write(frame)
        self.frame_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
            self.tmp_dir = None
        if self.enabled:
            print(f"Overview video saved: {self.path} frames={self.frame_count}", flush=True)


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
    print("base_start:", first.base_start.tolist(), "base_stop:", first.base_stop.tolist(), "base_push:", first.base_push.tolist())
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
        "ikpush",
        IKPUSH_STATE_VERSION,
    )
    if dp_controller is not None:
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

    video_recorder = OverviewVideoRecorder(gym, sim, viewer, env_states, args)
    while step < max_steps:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        gym.refresh_rigid_body_state_tensor(sim)
        for st in env_states:
            current_ee_pose_from_refreshed_tensors(st.ik_state)

        dp_policy_update_due = bool(
            dp_controller is not None and dc.float_dp_policy_update_due(step, args, dt)
        )
        if dp_policy_update_due:
            if viewer is not None and (args.draw_ik_target or args.draw_camera_axes):
                # Clear viewer-only debug lines before camera rendering so policy observations stay clean.
                gym.clear_lines(viewer)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_jacobian_tensors(sim)

        if dp_policy_update_due:
            dp_policy_inputs_by_env, dp_actions_by_env = dc.collect_float_dp_policy_actions(
                gym,
                sim,
                env_states,
                dof_names,
                gripper_idx,
                dt,
                dp_controller,
                dp_control_env_id_set,
                "ikpush",
            )
        else:
            dp_policy_inputs_by_env, dp_actions_by_env = {}, {}

        for st in env_states:
            if dp_controller is not None and (st.index in dp_actions_by_env or st.last_dp_action is not None):
                phase = "dp_policy"
                if st.index in dp_actions_by_env:
                    dp_policy_input = dp_policy_inputs_by_env[st.index]
                    base_xy_current = dp_policy_input["base_xy_current"]
                    yaw_current = dp_policy_input["yaw_current"]
                    handle_goal = dp_policy_input["handle_goal"]
                    ee_pos = dp_policy_input["ee_pos"]
                    ee_quat = dp_policy_input["ee_quat"]
                    dp_state = dp_policy_input["dp_state"]
                    dp_action = dp_actions_by_env[st.index]
                    st.last_dp_action = np.asarray(dp_action, dtype=np.float32).copy()
                    st.last_dp_state = None if dp_state is None else np.asarray(dp_state, dtype=np.float32).copy()
                    st.last_dp_ee_pos = None if ee_pos is None else np.asarray(ee_pos, dtype=np.float32).copy()
                    st.last_dp_ee_quat = None if ee_quat is None else np.asarray(ee_quat, dtype=np.float32).copy()
                    st.last_dp_handle_goal = None if handle_goal is None else np.asarray(handle_goal, dtype=np.float32).copy()
                else:
                    base_xy_current = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32)
                    yaw_current = float(st.traj.get("yaw", st.yaw_start))
                    handle_goal = getattr(st, "last_dp_handle_goal", st.last_handle_goal)
                    ee_pos = getattr(st, "last_dp_ee_pos", None)
                    ee_quat = getattr(st, "last_dp_ee_quat", None)
                    dp_state = getattr(st, "last_dp_state", None)
                    dp_action = np.asarray(st.last_dp_action, dtype=np.float32).copy()
                base_xy, yaw, target_pos, target_quat, gripper = apply_float_dp_action(
                    dp_action,
                    base_xy_current,
                    st.args.robot_z,
                    yaw_current,
                    dt,
                    action_frame=getattr(dp_controller, "action_frame", "world"),
                )
                st.traj["base_xy"] = np.asarray(base_xy, dtype=np.float32).copy()
                st.traj["yaw"] = float(yaw)
                door_pos_for_log, _door_vel_for_log = get_actor_dof_state(gym, st.env, st.door_actor)
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
                    st.yaw_start,
                    st.yaw_push,
                    st.traj,
                )
                dp_action = None
                dp_state = None
                ee_pos = None
                ee_quat = None
                door_pos_for_log = None
            st.last_phase = phase
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
                    phase,
                    action_names=ACTION_NAMES,
                )
                if dp_logger is not None:
                    dp_logger.write(dp_record)
                if args.dp_print and step % max(1, int(args.dp_log_interval)) == 0:
                    print_float_dp_policy_log_record(dp_record)
            set_robot_base_pose(gym, st.env, st.actor_handles, base_xy, st.args.robot_z, yaw)
            if phase == "return_home":
                if "return_home_start_dofs" not in st.traj:
                    st.traj["return_home_start_dofs"] = np.asarray(st.dof_positions, dtype=np.float32).copy()
                alpha = float(st.traj.get("return_home_alpha", 0.0))
                st.dof_positions[:] = lerp(st.traj["return_home_start_dofs"], st.home_positions, alpha)
                st.ik_state.last_pos_error = 0.0
            elif phase in ("walk", "hold_home"):
                st.dof_positions[:] = st.home_positions
                st.ik_state.last_pos_error = 0.0
            else:
                set_ik_target(st.ik_state, target_pos, target_quat)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        for st in env_states:
            if st.last_phase not in ("walk", "return_home", "hold_home"):
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
            if len(door_efforts) > 0:
                gym.apply_actor_dof_efforts(st.env, st.door_actor, door_efforts)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        record_camera_due = bool(
            args.record_dp_dataset
            and any(st.camera_handles and dc.float_dp_record_frame_due(st, dt) for st in env_states)
        )
        need_camera_render = bool(any(st.camera_handles for st in env_states) and (args.show_camera_images or record_camera_due))
        if viewer is not None and need_camera_render and (args.draw_ik_target or args.draw_camera_axes):
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
            if st.dp_recorder is not None:
                dc.record_float_dp_frame(
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
            st.prev_base_xy = np.asarray(st.traj.get("base_xy", st.base_start), dtype=np.float32).copy()
            st.prev_yaw = float(st.traj.get("yaw", st.yaw_start))

        if viewer is not None:
            video_recorder.update_camera(step)
            if args.draw_ik_target or args.draw_camera_axes:
                gym.clear_lines(viewer)
            for st in env_states[: min(4, len(env_states))]:
                if args.draw_camera_axes:
                    draw_low_level_camera_axes(gym, viewer, st.env, st.arm_actor, st.actor_handles, st.args)
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
            gym.draw_viewer(viewer, sim, True)
            video_recorder.capture(step)
            gym.sync_frame_time(sim)

        if args.log_interval > 0 and step % args.log_interval == 0:
            shown = env_states[: min(len(env_states), 4)]
            door_deg = [
                round(math.degrees(float(st.last_door_pos[0])), 1) if st.last_door_pos is not None and len(st.last_door_pos) else 0.0
                for st in shown
            ]
            phases = [st.last_phase for st in shown]
            successes = sum(st.dp_record_success for st in env_states if st.dp_recorder is not None)
            print(
                f"[{step:04d}] phases={phases} door_deg={door_deg} "
                f"record_success={successes}/{sum(st.dp_recorder is not None for st in env_states)}",
                flush=True,
            )
        step += 1

    video_recorder.close()
    elapsed = time.time() - start
    print(f"Done after {step} steps ({elapsed:.2f}s).")
    dc.finish_float_dp_recorders(env_states, args)
    if dp_logger is not None:
        dp_logger.close()


def main():
    args = parse_args()
    seed = resolve_seed(args)
    print(f"ikpush seed={seed}", flush=True)
    gym = gymapi.acquire_gym()
    sim, dt = base_ik.create_sim(gym, args)
    args.sim_dt = float(dt)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    with tempfile.TemporaryDirectory(prefix="float_ik_video_assets_") as temp_dir:
        base_asset, arm_asset = load_video_robot_assets(gym, sim, args, Path(temp_dir))
        apply_video_door_allowlist(args)
        door_templates = load_door_assets(gym, sim, args)
        if base_asset is not None:
            base_ik.print_collision_summary(gym, base_asset, "base visual actor", verbose=args.print_collision_summary)
        base_ik.print_collision_summary(gym, arm_asset, "arm articulated actor", verbose=args.print_collision_summary)
        dof_data = base_ik.configure_dofs(gym, arm_asset, args)
        dof_names, dof_props, dof_states, dof_positions, lower, upper, defaults, speeds, selected = dof_data
        apply_g1_default_dof_pose(args, dof_names, dof_states, dof_positions, lower, upper, defaults)
        if "jointGripper" in dof_names:
            dof_states["pos"][dof_names.index("jointGripper")] = args.gripper_open
            dof_positions[dof_names.index("jointGripper")] = args.gripper_open
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
        )
        viewer = setup_viewer(gym, sim, args)
        try:
            run_parallel_demo(gym, sim, env_states, viewer, args, dt, dof_names)
        finally:
            if args.show_camera_images and cv2 is not None:
                cv2.destroyAllWindows()
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
