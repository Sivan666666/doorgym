import argparse
import json
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

import play_b1z1_walk_with_door_asset as base


DEFAULT_AIGC_DOOR_DIR = "/home/sivan/whole_body/door_aigc/reference_export"
DEFAULT_FIXED_AIGC_DOOR_DIR = Path(__file__).resolve().parent / "data" / "asset" / "aigc_reference_export_fixed"


def _parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aigc_door_dir", type=str, default=DEFAULT_AIGC_DOOR_DIR)
    parser.add_argument("--aigc_fixed_door_dir", type=str, default=str(DEFAULT_FIXED_AIGC_DOOR_DIR))
    parser.add_argument(
        "--no_fix_aigc_urdf",
        action="store_true",
        help="Load the AIGC URDF exactly as exported, without repairing link visual origins.",
    )
    wrapper_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return wrapper_args


def _xyz_from_origin(origin_elem):
    return [float(v) for v in origin_elem.attrib.get("xyz", "0 0 0").split()]


def _xyz_from_attr(elem, name, default):
    return [float(v) for v in elem.attrib.get(name, default).split()]


def _aigc_to_unidoor_vec(vec):
    """Map AIGC mesh coordinates to the UniDoorManip door coordinate convention.

    The new export stores the visible door height along mesh Y while the JSON
    metadata and UniDoorManip assets use world/door Z as height. The metadata
    matches this transform: X = -mesh Z, Y = -mesh X, Z = mesh Y.
    """
    x, y, z = vec
    return [-z, -x, y]


def _transform_bounds(bounds):
    mn = bounds["handle_min"]
    mx = bounds["handle_max"]
    corners = np.array(
        [[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])],
        dtype=np.float64,
    )
    transformed = np.array([_aigc_to_unidoor_vec(corner) for corner in corners], dtype=np.float64)
    bounds = dict(bounds)
    bounds["handle_min"] = transformed.min(axis=0).tolist()
    bounds["handle_max"] = transformed.max(axis=0).tolist()
    bounds["goal_pos"] = _aigc_to_unidoor_vec(bounds["goal_pos"])
    return bounds


def _set_origin_xyz(parent_elem, xyz):
    origin = parent_elem.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        parent_elem.insert(0, origin)
    origin.attrib["xyz"] = f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}"


def _set_origin_pose(parent_elem, xyz, rpy=None):
    origin = parent_elem.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        parent_elem.insert(0, origin)
    origin.attrib["xyz"] = f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}"
    if rpy is not None:
        origin.attrib["rpy"] = rpy


def _transform_dae_mesh(src_path, dst_path):
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    tree = ET.parse(src_path)
    root = tree.getroot()
    for source in root.findall(".//c:source", ns):
        source_id = source.attrib.get("id", "").lower()
        if "position" not in source_id and "normal" not in source_id:
            continue
        array = source.find("c:float_array", ns)
        if array is None or not array.text:
            continue
        values = np.fromstring(array.text, sep=" ", dtype=np.float64)
        if values.size % 3 != 0:
            continue
        triples = values.reshape(-1, 3)
        transformed = np.array([_aigc_to_unidoor_vec(v) for v in triples], dtype=np.float64)
        array.text = " ".join(f"{v:.9g}" for v in transformed.reshape(-1))
    tree.write(dst_path, encoding="UTF-8", xml_declaration=True)


def _repair_aigc_urdf(src_dir, fixed_dir):
    """Patch AIGC exports whose movable board mesh is still in the parent frame.

    The AIGC mesh files are exported in a different coordinate convention from
    the JSON metadata: mesh Y is height, while UniDoorManip/metadata use Z as
    height. Keep the original DAE files untouched because Isaac Gym's DAE loader
    is fragile, and apply the coordinate transform through URDF visual/collision
    origins plus transformed joint origins/axes. Then link_1 gets the usual
    inverse hinge offset so the board is assembled at the closed-door pose.
    """
    fixed_dir.mkdir(parents=True, exist_ok=True)
    texture_src = src_dir / "texture_dae"
    texture_dst = fixed_dir / "texture_dae"
    if texture_dst.exists():
        shutil.rmtree(texture_dst)
    shutil.copytree(texture_src, texture_dst)

    shutil.copy2(src_dir / "bounding_box.json", fixed_dir / "bounding_box.json")
    with open(src_dir / "handle_bounding.json", "r", encoding="utf-8") as f:
        handle_bounds = json.load(f)
    with open(fixed_dir / "handle_bounding.json", "w", encoding="utf-8") as f:
        json.dump(_transform_bounds(handle_bounds), f, indent=2)

    tree = ET.parse(src_dir / "mobility.urdf")
    root = tree.getroot()
    joint_0 = root.find("./joint[@name='joint_0']")
    if joint_0 is not None:
        joint_0_origin = joint_0.find("origin")
        if joint_0_origin is None:
            joint_0_origin = ET.Element("origin")
            joint_0.insert(0, joint_0_origin)
        joint_0_origin.attrib["rpy"] = "0 0 0"
        joint_0_origin.attrib["xyz"] = "0 0 0"

    for joint in root.findall("./joint"):
        if joint.attrib.get("name") == "joint_0":
            continue
        origin = joint.find("origin")
        if origin is not None:
            origin.attrib["xyz"] = " ".join(f"{v:.8f}" for v in _aigc_to_unidoor_vec(_xyz_from_origin(origin)))
        axis = joint.find("axis")
        if axis is not None:
            axis_vec = np.asarray(_aigc_to_unidoor_vec(_xyz_from_attr(axis, "xyz", "0 0 0")), dtype=np.float64)
            norm = np.linalg.norm(axis_vec)
            if norm > 1e-8:
                axis_vec /= norm
            axis.attrib["xyz"] = " ".join(f"{v:.8f}" for v in axis_vec.tolist())

    joint_1 = root.find("./joint[@name='joint_1']")
    if joint_1 is None:
        raise ValueError("AIGC URDF does not contain joint_1")
    joint_origin = joint_1.find("origin")
    if joint_origin is None:
        raise ValueError("AIGC URDF joint_1 does not contain an origin")

    hinge_xyz = _xyz_from_origin(joint_origin)
    board_mesh_offset = [-hinge_xyz[0], -hinge_xyz[1], -hinge_xyz[2]]
    mesh_rpy = "1.570796326794897 0 -1.570796326794897"
    for link_name, xyz in (("link_0", [0.0, 0.0, 0.0]), ("link_1", board_mesh_offset), ("link_2", [0.0, 0.0, 0.0])):
        link = root.find(f"./link[@name='{link_name}']")
        if link is None:
            raise ValueError(f"AIGC URDF does not contain {link_name}")
        for tag in ("visual", "collision"):
            for elem in link.findall(tag):
                _set_origin_pose(elem, xyz, mesh_rpy)

    tree.write(fixed_dir / "mobility.urdf", encoding="UTF-8", xml_declaration=True)
    return fixed_dir


def _load_single_aigc_door_runtime(cfg_path):
    source_door_dir = Path(WRAPPER_ARGS.aigc_door_dir).expanduser().resolve()
    if WRAPPER_ARGS.no_fix_aigc_urdf:
        door_dir = source_door_dir
    else:
        fixed_door_dir = Path(WRAPPER_ARGS.aigc_fixed_door_dir).expanduser().resolve()
        door_dir = _repair_aigc_urdf(source_door_dir, fixed_door_dir)
    required_files = {
        "path": door_dir / "mobility.urdf",
        "bounding_box": door_dir / "bounding_box.json",
        "handle_bounding": door_dir / "handle_bounding.json",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AIGC door asset is missing required files: {missing}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("env", {})

    with open(required_files["bounding_box"], "r", encoding="utf-8") as f:
        door_bounding = json.load(f)
    with open(required_files["handle_bounding"], "r", encoding="utf-8") as f:
        handle_bounding = json.load(f)

    spec = {
        "bounding_box": f"{door_dir.name}/bounding_box.json",
        "handle_bounding": f"{door_dir.name}/handle_bounding.json",
        "name": door_dir.name,
        "path": f"{door_dir.name}/mobility.urdf",
    }
    return {
        "asset_root": str(door_dir.parent),
        "asset_file_door": "",
        "door_asset_specs": [spec],
        "door_asset_names": [door_dir.name],
        "door_bounding_data": [door_bounding],
        "handle_bounding_data": [handle_bounding],
        "door_lock_force": float(env_cfg.get("doorLockForce", 150.0)),
        "door_open_resistance": float(env_cfg.get("doorOpenResistance", 3.0)),
        "door_open_damping": float(env_cfg.get("doorOpenDamping", 0.5)),
        "handle_unlock_ratio": float(env_cfg.get("handleOpenThresholdRatio", 0.65)),
        "handle_spring_stiffness": float(env_cfg.get("handleSpringStiffness", 40.0)),
        "handle_spring_damping": float(env_cfg.get("handleSpringDamping", 2.0)),
        "door_joint_friction": env_cfg.get("doorJointFriction", [6.0, 18.0]),
        "door_joint_damping": env_cfg.get("doorJointDamping", [3.0, 10.0]),
        "door_joint_effort": env_cfg.get("doorJointEffort", [200.0, 200.0]),
    }


WRAPPER_ARGS = _parse_wrapper_args()
base._load_door_runtime = _load_single_aigc_door_runtime


if __name__ == "__main__":
    print(f"Using AIGC door asset: {Path(WRAPPER_ARGS.aigc_door_dir).expanduser().resolve()}")
    if WRAPPER_ARGS.no_fix_aigc_urdf:
        print("AIGC URDF repair: disabled")
    else:
        print(f"AIGC URDF repair: enabled -> {Path(WRAPPER_ARGS.aigc_fixed_door_dir).expanduser().resolve()}")
    base.main()
