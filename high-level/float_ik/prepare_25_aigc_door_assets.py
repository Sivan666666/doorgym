#!/usr/bin/env python3
"""Audit and prepare generated door URDFs for the scripted door controller.

The source assets are never edited in place. ``audit`` only analyzes assets.
``generate`` copies each asset to a clean output directory, canonicalizes the
movable handle link without changing its physical geometry, writes bounds and
metadata, and creates a multi-door Isaac Gym YAML file.

The canonical handle frame follows the existing door-controller convention:

* local Z points from the door toward the handle/robot side;
* for lever handles, local X follows the lever and points toward its long end;
* visual and collision origins receive the inverse frame transform, preserving
  their world poses, dimensions, and joint motion exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


HIGH_LEVEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "25_aigc_doors"
DEFAULT_OUTPUT_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "door_set" / "aigc_25"
DEFAULT_YAML_OUTPUT = (
    HIGH_LEVEL_ROOT / "experiments" / "isaacgym" / "b1z1_opendoor_25_aigc.yaml"
)
DEFAULT_TEMPLATE_YAML = (
    HIGH_LEVEL_ROOT / "experiments" / "isaacgym" / "b1z1_opendoor_single_door4.yaml"
)
DEFAULT_SOURCE_PATTERN = "rec_using-the-reference-image-*"
ASSET_ROOT = HIGH_LEVEL_ROOT / "data" / "asset"
DOOR_SET_ROOT = ASSET_ROOT / "door_set"
EPS = 1.0e-9


@dataclass
class GeometryRecord:
    link_name: str
    tag: str
    index: int
    points: np.ndarray


@dataclass
class PreparedDoor:
    source_dir: Path
    clean_name: str
    tree: ET.ElementTree
    door_joint_name: str
    handle_joint_name: str
    door_body_name: str
    handle_body_name: str
    handle_type: str
    bounding_box: dict
    handle_bounding: dict
    actor_yaw_offset: float
    actor_position_offset: list[float]
    door_motion_sign_multiplier: float
    ready: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def report(self) -> dict:
        return {
            "source": str(self.source_dir),
            "name": self.clean_name,
            "handle_type": self.handle_type,
            "ready_for_scripted_gt": self.ready,
            "door_joint_name": self.door_joint_name,
            "handle_joint_name": self.handle_joint_name,
            "door_body_name": self.door_body_name,
            "handle_body_name": self.handle_body_name,
            "actor_scale": 1.0,
            "actor_yaw_offset": self.actor_yaw_offset,
            "actor_position_offset": self.actor_position_offset,
            "door_motion_sign_multiplier": self.door_motion_sign_multiplier,
            "bounding_box": self.bounding_box,
            "handle_bounding": self.handle_bounding,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or prepare generated doors for the scripted GT controller."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("audit", "generate"),
        default="audit",
        help="audit is read-only; generate writes copied, canonicalized assets and YAML.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--source-pattern",
        default=DEFAULT_SOURCE_PATTERN,
        help=(
            "Directory glob under source-root. The default selects the 25 single-reference "
            "doors and excludes the older two-reference asset."
        ),
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Ignore source-pattern and include every source directory containing model.urdf.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yaml-output", type=Path, default=DEFAULT_YAML_OUTPUT)
    parser.add_argument("--template-yaml", type=Path, default=DEFAULT_TEMPLATE_YAML)
    parser.add_argument(
        "--existing-yaml",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prepend the active asset block from an existing door YAML. Repeat this option "
            "to combine existing door sets with generated assets."
        ),
    )
    parser.add_argument(
        "--block-name",
        default="aigc_25",
        help="Name of the trainAssets block written to the generated YAML.",
    )
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--name-prefix", default="aigc_door")
    parser.add_argument(
        "--handle-clearance",
        type=float,
        default=0.025,
        help="Meters beyond the outer handle surface used for goal_pos.",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Include non-lever or otherwise review-required assets in the generated YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output-root or YAML file in generate mode.",
    )
    return parser.parse_args()


def parse_vec(text: str | None, default: Iterable[float]) -> np.ndarray:
    if text is None:
        return np.asarray(tuple(default), dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.size != 3:
        raise ValueError(f"Expected three values, got {text!r}")
    return values


def format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def skew_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < EPS or abs(angle) < EPS:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    return (
        skew_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)
        @ skew_axis_angle(np.array([0.0, 1.0, 0.0]), pitch)
        @ skew_axis_angle(np.array([1.0, 0.0, 0.0]), roll)
    )


def matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    value = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(value)
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def origin_transform(parent: ET.Element) -> np.ndarray:
    origin = parent.find("origin")
    if origin is None:
        return np.eye(4, dtype=np.float64)
    xyz = parse_vec(origin.get("xyz"), (0.0, 0.0, 0.0))
    rpy = parse_vec(origin.get("rpy"), (0.0, 0.0, 0.0))
    return make_transform(rpy_matrix(rpy), xyz)


def set_origin_transform(parent: ET.Element, transform: np.ndarray) -> None:
    origin = parent.find("origin")
    if origin is None:
        origin = ET.Element("origin")
        parent.insert(0, origin)
    origin.set("xyz", format_vec(transform[:3, 3]))
    origin.set("rpy", format_vec(matrix_rpy(transform[:3, :3])))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def joint_parent(joint: ET.Element) -> str:
    elem = joint.find("parent")
    if elem is None or not elem.get("link"):
        raise ValueError(f"Joint {joint.get('name')!r} has no parent link")
    return elem.get("link")


def joint_child(joint: ET.Element) -> str:
    elem = joint.find("child")
    if elem is None or not elem.get("link"):
        raise ValueError(f"Joint {joint.get('name')!r} has no child link")
    return elem.get("link")


def joint_axis(joint: ET.Element) -> np.ndarray:
    axis = joint.find("axis")
    value = parse_vec(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0))
    norm = np.linalg.norm(value)
    if norm < EPS:
        raise ValueError(f"Joint {joint.get('name')!r} has a zero axis")
    return value / norm


def set_joint_axis(joint: ET.Element, axis_value: np.ndarray) -> None:
    axis = joint.find("axis")
    if axis is None:
        axis = ET.Element("axis")
        child = joint.find("child")
        insert_at = list(joint).index(child) + 1 if child is not None else len(joint)
        joint.insert(insert_at, axis)
    axis.set("xyz", format_vec(axis_value))


def links_and_joints(root: ET.Element) -> tuple[dict[str, ET.Element], list[ET.Element]]:
    links = {link.get("name"): link for link in root.findall("link")}
    if None in links:
        raise ValueError("A link is missing its name")
    return links, list(root.findall("joint"))


def zero_link_transforms(
    root: ET.Element, joint_positions: dict[str, float] | None = None
) -> dict[str, np.ndarray]:
    links, joints = links_and_joints(root)
    child_links = {joint_child(joint) for joint in joints}
    root_links = [name for name in links if name not in child_links]
    if len(root_links) != 1:
        raise ValueError(f"Expected one root link, found {root_links}")

    transforms = {root_links[0]: np.eye(4, dtype=np.float64)}
    pending = list(joints)
    positions = joint_positions or {}
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint_parent(joint)
            if parent not in transforms:
                continue
            motion = np.eye(4, dtype=np.float64)
            if joint.get("type") in ("revolute", "continuous"):
                angle = float(positions.get(joint.get("name", ""), 0.0))
                motion[:3, :3] = skew_axis_angle(joint_axis(joint), angle)
            transforms[joint_child(joint)] = transforms[parent] @ origin_transform(joint) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            names = [joint.get("name") for joint in pending]
            raise ValueError(f"Disconnected or cyclic joint graph: {names}")
    return transforms


def box_points(size: np.ndarray) -> np.ndarray:
    half = 0.5 * size
    return np.array(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ],
        dtype=np.float64,
    )


def cylinder_points(radius: float, length: float, samples: int = 128) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    points = []
    for z in (-0.5 * length, 0.5 * length):
        points.extend((radius * math.cos(a), radius * math.sin(a), z) for a in angles)
    return np.asarray(points, dtype=np.float64)


def sphere_points(radius: float, samples: int = 32) -> np.ndarray:
    points = []
    for theta in np.linspace(0.0, math.pi, samples):
        ring_radius = radius * math.sin(theta)
        z = radius * math.cos(theta)
        for phi in np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False):
            points.append((ring_radius * math.cos(phi), ring_radius * math.sin(phi), z))
    return np.asarray(points, dtype=np.float64)


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            values = np.fromstring(line[2:], sep=" ", dtype=np.float64)
            if values.size >= 3:
                vertices.append(values[:3])
    if not vertices:
        raise ValueError(f"OBJ mesh contains no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def resolve_mesh_path(asset_dir: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
    path = Path(filename)
    if path.is_absolute():
        return path
    return (asset_dir / path).resolve()


def geometry_points(geometry: ET.Element, asset_dir: Path) -> np.ndarray:
    box = geometry.find("box")
    if box is not None:
        return box_points(parse_vec(box.get("size"), (0.0, 0.0, 0.0)))
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return cylinder_points(float(cylinder.get("radius")), float(cylinder.get("length")))
    sphere = geometry.find("sphere")
    if sphere is not None:
        return sphere_points(float(sphere.get("radius")))
    mesh = geometry.find("mesh")
    if mesh is not None:
        path = resolve_mesh_path(asset_dir, mesh.get("filename", ""))
        if path.suffix.lower() != ".obj":
            raise ValueError(f"Only OBJ meshes are supported for automatic bounds: {path}")
        points = read_obj_vertices(path)
        scale = parse_vec(mesh.get("scale"), (1.0, 1.0, 1.0))
        return points * scale
    raise ValueError("Unsupported or empty URDF geometry")


def collect_geometry_records(
    root: ET.Element, asset_dir: Path, tag: str
) -> list[GeometryRecord]:
    links, _ = links_and_joints(root)
    records = []
    for link_name, link in links.items():
        for index, element in enumerate(link.findall(tag)):
            geometry = element.find("geometry")
            if geometry is None:
                continue
            points = transform_points(origin_transform(element), geometry_points(geometry, asset_dir))
            records.append(GeometryRecord(link_name, tag, index, points))
    return records


def records_in_root(
    root: ET.Element,
    records: list[GeometryRecord],
    joint_positions: dict[str, float] | None = None,
) -> list[np.ndarray]:
    transforms = zero_link_transforms(root, joint_positions)
    return [transform_points(transforms[record.link_name], record.points) for record in records]


def points_for_link(records: list[GeometryRecord], link_name: str) -> np.ndarray:
    selected = [record.points for record in records if record.link_name == link_name]
    if not selected:
        raise ValueError(f"Link {link_name!r} has no visual geometry")
    return np.concatenate(selected, axis=0)


def descendant_links(root: ET.Element, starting_link: str) -> set[str]:
    children_by_parent: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        children_by_parent.setdefault(joint_parent(joint), []).append(joint_child(joint))
    descendants = {starting_link}
    pending = [starting_link]
    while pending:
        parent = pending.pop()
        for child in children_by_parent.get(parent, []):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    return descendants


def bounds_dict(points: np.ndarray) -> dict:
    return {
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
    }


def find_movable_joints(root: ET.Element) -> tuple[ET.Element, ET.Element]:
    movable = [
        joint
        for joint in root.findall("joint")
        if joint.get("type") in ("revolute", "continuous")
    ]
    if len(movable) != 2:
        raise ValueError(f"Expected exactly two movable joints, found {len(movable)}")

    door_candidates = [
        joint
        for joint in movable
        if "door" in (joint.get("name") or "").lower()
        or "panel" in joint_child(joint).lower()
    ]
    if len(door_candidates) != 1:
        raise ValueError(
            f"Could not uniquely identify the door hinge: {[joint.get('name') for joint in movable]}"
        )
    door_joint = door_candidates[0]
    handle_joint = movable[1] if movable[0] is door_joint else movable[0]
    return door_joint, handle_joint


def classify_handle(handle_link: ET.Element) -> str:
    link_name = handle_link.get("name", "")
    if "条形" in link_name:
        return "lever"
    if "圆形" in link_name:
        return "knob"

    text_parts = [link_name]
    for visual in handle_link.findall("visual"):
        text_parts.append(visual.get("name", ""))
        mesh = visual.find("geometry/mesh")
        if mesh is not None:
            text_parts.append(mesh.get("filename", ""))
    text = " ".join(text_parts).lower()
    if "knob" in text:
        return "knob"
    if "lever" in text or "grip" in text or "bar" in text:
        return "lever"
    return "lever"


def projected_principal_axis(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0)
    projector = np.eye(3) - np.outer(normal, normal)
    covariance = projector @ (centered.T @ centered / max(1, len(points))) @ projector
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    primary = vectors[:, order[0]]
    primary -= normal * float(np.dot(primary, normal))
    primary /= max(np.linalg.norm(primary), EPS)
    denominator = max(float(values[order[1]]), EPS)
    return primary, float(values[order[0]] / denominator)


def canonical_handle_basis(
    root: ET.Element,
    handle_joint: ET.Element,
    visual_records: list[GeometryRecord],
    panel_link_name: str,
    handle_link_name: str,
    handle_type: str,
) -> tuple[np.ndarray, dict]:
    transforms = zero_link_transforms(root)
    handle_points = points_for_link(visual_records, handle_link_name)
    panel_points = points_for_link(visual_records, panel_link_name)
    handle_transform = transforms[handle_link_name]
    handle_center_root = transform_points(handle_transform, handle_points.mean(axis=0, keepdims=True))[0]
    panel_root_points = transform_points(transforms[panel_link_name], panel_points)
    panel_center_root = panel_root_points.mean(axis=0)

    axis_old = joint_axis(handle_joint)
    axis_root = handle_transform[:3, :3] @ axis_old
    outward_score = float(np.dot(handle_center_root - panel_center_root, axis_root))
    if abs(outward_score) < 1.0e-5:
        outward_score = float(
            np.dot(handle_transform[:3, 3] - panel_center_root, axis_root)
        )
    z_axis = axis_old if outward_score >= 0.0 else -axis_old

    if handle_type == "lever":
        x_axis, anisotropy = projected_principal_axis(handle_points, z_axis)
        if float(np.dot(handle_points.mean(axis=0), x_axis)) > 0.0:
            x_axis = -x_axis
    else:
        preferred_root = transforms[panel_link_name][:3, :3] @ np.array([1.0, 0.0, 0.0])
        x_axis = handle_transform[:3, :3].T @ preferred_root
        x_axis -= z_axis * float(np.dot(x_axis, z_axis))
        if np.linalg.norm(x_axis) < 1.0e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
            x_axis -= z_axis * float(np.dot(x_axis, z_axis))
        x_axis /= max(np.linalg.norm(x_axis), EPS)
        anisotropy = 1.0

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), EPS)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), EPS)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    metrics = {
        "handle_axis_source": joint_axis(handle_joint).tolist(),
        "handle_outward_score_m": outward_score,
        "handle_planar_anisotropy": anisotropy,
        "canonical_basis_columns_in_source_frame": basis.tolist(),
    }
    return basis, metrics


def reframe_handle_link(
    root: ET.Element,
    handle_joint: ET.Element,
    handle_link_name: str,
    basis: np.ndarray,
) -> None:
    links, joints = links_and_joints(root)
    handle_link = links[handle_link_name]
    basis_transform = make_transform(basis)
    inverse_basis = np.linalg.inv(basis_transform)

    set_origin_transform(handle_joint, origin_transform(handle_joint) @ basis_transform)
    set_joint_axis(handle_joint, basis.T @ joint_axis(handle_joint))

    for tag in ("visual", "collision", "inertial"):
        for element in handle_link.findall(tag):
            set_origin_transform(element, inverse_basis @ origin_transform(element))

    for joint in joints:
        if joint is handle_joint:
            continue
        if joint_parent(joint) == handle_link_name:
            set_origin_transform(joint, inverse_basis @ origin_transform(joint))


def rename_link(root: ET.Element, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    links, joints = links_and_joints(root)
    if new_name in links:
        raise ValueError(f"Cannot rename {old_name!r}; link {new_name!r} already exists")
    links[old_name].set("name", new_name)
    for joint in joints:
        for tag in ("parent", "child"):
            element = joint.find(tag)
            if element is not None and element.get("link") == old_name:
                element.set("link", new_name)


def geometry_pose_error(
    before_root: ET.Element,
    after_root: ET.Element,
    asset_dir: Path,
    before_handle_name: str,
    after_handle_name: str,
    handle_joint_before: str,
    handle_joint_after: str,
    sample_angle: float,
    tag: str,
) -> float:
    before_records = collect_geometry_records(before_root, asset_dir, tag)
    after_records = collect_geometry_records(after_root, asset_dir, tag)
    if len(before_records) != len(after_records):
        return math.inf

    positions_before = {handle_joint_before: sample_angle}
    positions_after = {handle_joint_after: sample_angle}
    before_world = records_in_root(before_root, before_records, positions_before)
    after_world = records_in_root(after_root, after_records, positions_after)
    max_error = 0.0
    for before_record, after_record, points_before, points_after in zip(
        before_records, after_records, before_world, after_world
    ):
        expected_link = (
            after_handle_name if before_record.link_name == before_handle_name else before_record.link_name
        )
        if expected_link != after_record.link_name or points_before.shape != points_after.shape:
            return math.inf
        max_error = max(max_error, float(np.max(np.abs(points_before - points_after))))
    return max_error


def rounded_bounds(bounds: dict) -> dict:
    return {
        key: [round(float(value), 9) for value in bounds[key]]
        for key in ("min", "max")
    }


def prepare_one(
    source_dir: Path,
    clean_name: str,
    handle_clearance: float,
) -> PreparedDoor:
    source_urdf = source_dir / "model.urdf"
    tree = ET.parse(source_urdf)
    original_root = copy.deepcopy(tree.getroot())
    root = tree.getroot()
    warnings: list[str] = []
    errors: list[str] = []

    door_joint, handle_joint = find_movable_joints(root)
    door_link_name = joint_child(door_joint)
    handle_link_name = joint_child(handle_joint)
    links, _ = links_and_joints(root)
    handle_type = classify_handle(links[handle_link_name])
    visual_records = collect_geometry_records(root, source_dir, "visual")
    collision_records = collect_geometry_records(root, source_dir, "collision")
    if not collision_records:
        warnings.append("Asset has no collision geometry.")
    if len(visual_records) != len(collision_records):
        warnings.append(
            f"Visual/collision element counts differ: {len(visual_records)} vs {len(collision_records)}."
        )

    basis, basis_metrics = canonical_handle_basis(
        root,
        handle_joint,
        visual_records,
        door_link_name,
        handle_link_name,
        handle_type,
    )
    old_handle_joint_name = handle_joint.get("name") or "handle_joint"
    reframe_handle_link(root, handle_joint, handle_link_name, basis)

    canonical_handle_name = "lever_handle" if handle_type == "lever" else "door_knob"
    rename_link(root, handle_link_name, canonical_handle_name)
    handle_joint.set("name", "handle_spindle_joint")

    canonical_visual_records = collect_geometry_records(root, source_dir, "visual")
    canonical_handle_points = points_for_link(canonical_visual_records, canonical_handle_name)
    handle_bounds = bounds_dict(canonical_handle_points)
    handle_min = np.asarray(handle_bounds["min"], dtype=np.float64)
    handle_max = np.asarray(handle_bounds["max"], dtype=np.float64)
    goal_pos = np.array(
        [
            0.5 * (handle_min[0] + handle_max[0]),
            0.5 * (handle_min[1] + handle_max[1]),
            handle_max[2] + float(handle_clearance),
        ],
        dtype=np.float64,
    )
    handle_bounding = {
        "handle_min": [round(float(value), 9) for value in handle_min],
        "handle_max": [round(float(value), 9) for value in handle_max],
        "goal_pos": [round(float(value), 9) for value in goal_pos],
    }

    root_visual_points = np.concatenate(
        records_in_root(root, canonical_visual_records), axis=0
    )
    bounding_box = rounded_bounds(bounds_dict(root_visual_points))
    canonical_transforms = zero_link_transforms(root)
    panel_root_points = transform_points(
        canonical_transforms[door_link_name],
        points_for_link(canonical_visual_records, door_link_name),
    )
    moving_door_links = descendant_links(root, door_link_name)
    frame_point_groups = [
        transform_points(canonical_transforms[record.link_name], record.points)
        for record in canonical_visual_records
        if record.link_name not in moving_door_links
    ]
    if not frame_point_groups:
        raise ValueError("Fixed door-frame assembly has no visual geometry")
    frame_root_points = np.concatenate(frame_point_groups, axis=0)
    panel_extents = np.ptp(panel_root_points, axis=0)
    frame_extents = np.ptp(frame_root_points, axis=0)
    panel_frame_width_ratio = float(
        panel_extents[0] / max(frame_extents[0], EPS)
    )
    structural_review = panel_frame_width_ratio > 1.2
    if structural_review:
        warnings.append(
            "Door panel is wider than the fixed frame by a factor of "
            f"{panel_frame_width_ratio:.2f}; source geometry requires review."
        )

    canonical_handle_joint = root.find("./joint[@name='handle_spindle_joint']")
    if canonical_handle_joint is None:
        raise RuntimeError("Internal error: canonical handle joint was not created")
    handle_limit = canonical_handle_joint.find("limit")
    sample_angle = 0.3
    if handle_limit is not None:
        lower = float(handle_limit.get("lower", "0"))
        upper = float(handle_limit.get("upper", "0.6"))
        sample_angle = lower + 0.5 * (upper - lower)
    else:
        lower = 0.0
        upper = 0.6
    effective_handle_range = max(1.0e-4, min(upper, math.pi / 4.0) - lower)
    handle_rotate_target = 0.97 * effective_handle_range
    handle_contact_radius = max(abs(float(goal_pos[0])), 0.03)
    handle_rotate_right_distance = handle_contact_radius * (
        1.0 - math.cos(handle_rotate_target)
    )
    handle_rotate_down_distance = handle_contact_radius * math.sin(
        handle_rotate_target
    )

    max_visual_error = geometry_pose_error(
        original_root,
        root,
        source_dir,
        handle_link_name,
        canonical_handle_name,
        old_handle_joint_name,
        "handle_spindle_joint",
        sample_angle,
        "visual",
    )
    max_collision_error = geometry_pose_error(
        original_root,
        root,
        source_dir,
        handle_link_name,
        canonical_handle_name,
        old_handle_joint_name,
        "handle_spindle_joint",
        sample_angle,
        "collision",
    )
    if max_visual_error > 1.0e-7:
        errors.append(f"Handle reframe moved visual geometry by {max_visual_error:.3e} m.")
    if max_collision_error > 1.0e-7:
        errors.append(f"Handle reframe moved collision geometry by {max_collision_error:.3e} m.")

    extents = np.asarray(bounding_box["max"]) - np.asarray(bounding_box["min"])
    if extents[2] < 1.5 or extents[2] > 3.0:
        warnings.append(f"Unusual unscaled door height: {extents[2]:.3f} m.")
    if extents[0] < 0.5 or extents[0] > 2.0:
        warnings.append(f"Unusual unscaled door width: {extents[0]:.3f} m.")
    if handle_type != "lever":
        warnings.append(
            "Round knob detected: the current scripted GT trajectory presses a lever and "
            "requires controller-level review for this asset."
        )
    low_lever_confidence = (
        handle_type == "lever"
        and basis_metrics["handle_planar_anisotropy"] < 1.5
    )
    if low_lever_confidence:
        warnings.append(
            "Lever principal direction has low geometric confidence; inspect the canonical frame."
        )

    panel_records = collect_geometry_records(root, source_dir, "visual")
    panel_points = points_for_link(panel_records, door_link_name)
    panel_center = panel_points.mean(axis=0)
    # Total actor yaw is pi + offset = pi/2. This maps the generated door's
    # local -Y front normal toward the robot at +X.
    actor_yaw_offset = -0.5 * math.pi
    actor_position_offset = [
        round(float(panel_center[1]), 9),
        round(float(-0.5 * (bounding_box["min"][0] + bounding_box["max"][0])), 9),
        0.0,
    ]

    compile_report_path = source_dir / "compile_report.json"
    compile_status = None
    compile_warnings = []
    if compile_report_path.is_file():
        compile_report = json.loads(compile_report_path.read_text(encoding="utf-8"))
        compile_status = compile_report.get("status")
        compile_warnings = compile_report.get("warnings", [])
        if compile_status != "success":
            errors.append(f"Source compile report status is {compile_status!r}.")
        if compile_warnings:
            warnings.append(f"Source compile report has {len(compile_warnings)} warning(s).")
    else:
        warnings.append("Source compile_report.json is missing.")

    ready = (
        not errors
        and handle_type == "lever"
        and not low_lever_confidence
        and not structural_review
    )
    metrics = {
        **basis_metrics,
        "source_compile_status": compile_status,
        "source_compile_warnings": compile_warnings,
        "visual_elements": len(visual_records),
        "collision_elements": len(collision_records),
        "unscaled_extents_m": extents.tolist(),
        "door_panel_extents_m": panel_extents.tolist(),
        "fixed_frame_extents_m": frame_extents.tolist(),
        "door_panel_to_frame_width_ratio": panel_frame_width_ratio,
        "handle_reframe_sample_angle_rad": sample_angle,
        "handle_joint_lower_rad": lower,
        "handle_joint_upper_rad": upper,
        "handle_contact_radius_m": handle_contact_radius,
        "controller_overrides": {
            "handle_rotate_right_distance": handle_rotate_right_distance,
            "handle_rotate_down_distance": handle_rotate_down_distance,
            "handle_rotate_angle": handle_rotate_target,
            "handle_unlock_ratio": 0.2,
        },
        "visual_preservation_max_error_m": max_visual_error,
        "collision_preservation_max_error_m": max_collision_error,
    }
    return PreparedDoor(
        source_dir=source_dir,
        clean_name=clean_name,
        tree=tree,
        door_joint_name=door_joint.get("name") or "door_hinge_joint",
        handle_joint_name="handle_spindle_joint",
        door_body_name=door_link_name,
        handle_body_name=canonical_handle_name,
        handle_type=handle_type,
        bounding_box=bounding_box,
        handle_bounding=handle_bounding,
        actor_yaw_offset=actor_yaw_offset,
        actor_position_offset=actor_position_offset,
        door_motion_sign_multiplier=-1.0,
        ready=ready,
        warnings=warnings,
        errors=errors,
        metrics=metrics,
    )


def discover_sources(args: argparse.Namespace) -> list[Path]:
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    candidates = source_root.iterdir() if args.include_all else source_root.glob(args.source_pattern)
    sources = sorted(path for path in candidates if path.is_dir() and (path / "model.urdf").is_file())
    if not sources:
        raise RuntimeError(
            f"No model.urdf files matched {source_root / args.source_pattern}"
        )
    return sources


def prepare_all(args: argparse.Namespace) -> list[PreparedDoor]:
    prepared = []
    for index, source_dir in enumerate(discover_sources(args)):
        clean_name = f"{args.name_prefix}_{index:02d}"
        try:
            prepared.append(
                prepare_one(source_dir, clean_name, args.handle_clearance)
            )
        except Exception as exc:
            prepared.append(
                PreparedDoor(
                    source_dir=source_dir,
                    clean_name=clean_name,
                    tree=ET.parse(source_dir / "model.urdf"),
                    door_joint_name="",
                    handle_joint_name="",
                    door_body_name="",
                    handle_body_name="",
                    handle_type="unknown",
                    bounding_box={},
                    handle_bounding={},
                    actor_yaw_offset=0.0,
                    actor_position_offset=[0.0, 0.0, 0.0],
                    door_motion_sign_multiplier=1.0,
                    ready=False,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            )
    return prepared


def report_document(prepared: list[PreparedDoor], args: argparse.Namespace) -> dict:
    ready = sum(door.ready for door in prepared)
    review = sum(not door.ready and not door.errors for door in prepared)
    failed = sum(bool(door.errors) for door in prepared)
    return {
        "schema_version": 1,
        "mode": args.mode,
        "source_root": str(args.source_root.expanduser().resolve()),
        "source_pattern": "*" if args.include_all else args.source_pattern,
        "source_count": len(prepared),
        "ready_count": ready,
        "review_count": review,
        "failed_count": failed,
        "policy": {
            "source_assets_modified": False,
            "actor_scale": 1.0,
            "collision_geometry_resized": False,
            "handle_clearance_m": float(args.handle_clearance),
            "review_assets_in_yaml": bool(args.include_review),
        },
        "doors": [door.report() for door in prepared],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indentation = "\n" + level * "  "
    child_indentation = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indentation
        for child in children:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_indentation
        children[-1].tail = indentation
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def copy_prepared_asset(door: PreparedDoor, destination: Path) -> None:
    shutil.copytree(door.source_dir, destination)
    output_urdf = destination / "model.urdf"
    indent_xml(door.tree.getroot())
    door.tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    write_json(destination / "bounding_box.json", door.bounding_box)
    write_json(destination / "handle_bounding.json", door.handle_bounding)
    write_json(
        destination / "compatibility_meta.json",
        {
            "schema_version": 1,
            "source_directory": door.source_dir.name,
            "source_urdf": "model.urdf",
            "prepared_name": door.clean_name,
            "handle_type": door.handle_type,
            "ready_for_scripted_gt": door.ready,
            "physical_policy": {
                "actor_scale": 1.0,
                "visual_geometry_resized": False,
                "collision_geometry_resized": False,
                "source_asset_modified": False,
            },
            "warnings": door.warnings,
            "metrics": door.metrics,
        },
    )


def output_asset_layout(output_root: Path) -> tuple[str, str, Path]:
    try:
        relative_dir = output_root.relative_to(DOOR_SET_ROOT)
    except ValueError:
        return str(output_root.parent), f"{output_root.name}/", Path()
    return "data/asset", "door_set/", relative_dir


def generated_spec(door: PreparedDoor, output_root: Path) -> dict:
    _, _, relative_dir = output_asset_layout(output_root)
    prefix = (relative_dir / door.clean_name).as_posix()
    spec = {
        "bounding_box": f"{prefix}/bounding_box.json",
        "handle_bounding": f"{prefix}/handle_bounding.json",
        "name": door.clean_name,
        "path": f"{prefix}/model.urdf",
        # Explicitly override the float_ik CLI default (1.2). Generated assets
        # are authored in meters and must retain their original dimensions.
        "actor_scale": 1.0,
        "actor_yaw_offset": float(door.actor_yaw_offset),
        "actor_position_offset": [float(value) for value in door.actor_position_offset],
        "robot_y_offset": 0.0,
        "door_motion_sign_multiplier": float(door.door_motion_sign_multiplier),
        "door_body_name": door.door_body_name,
        "handle_body_name": door.handle_body_name,
        "door_dof_name": door.door_joint_name,
        "handle_dof_name": door.handle_joint_name,
    }
    controller_overrides = door.metrics.get("controller_overrides")
    if controller_overrides:
        spec["controller_overrides"] = {
            key: round(float(value), 9)
            for key, value in controller_overrides.items()
        }
    return spec


def selected_prepared_doors(
    prepared: list[PreparedDoor], include_review: bool
) -> list[PreparedDoor]:
    return [
        door
        for door in prepared
        if not door.errors and (door.ready or include_review)
    ]


def active_asset_specs(config_path: Path) -> list[dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    asset = config.get("env", {}).get("asset", {})
    load_block = asset.get("load_block")
    train_assets = asset.get("trainAssets", {})
    if load_block not in train_assets:
        raise ValueError(
            f"{config_path} does not contain active trainAssets block {load_block!r}"
        )
    block = train_assets[load_block]
    try:
        ordered_keys = sorted(block, key=lambda key: int(key))
    except (TypeError, ValueError):
        ordered_keys = list(block)
    return [copy.deepcopy(block[key]) for key in ordered_keys]


def write_yaml(prepared: list[PreparedDoor], args: argparse.Namespace) -> None:
    template_path = args.template_yaml.expanduser().resolve()
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    env = config.setdefault("env", {})
    asset = env.setdefault("asset", {})
    asset_root, asset_file_door, _ = output_asset_layout(
        args.output_root.expanduser().resolve()
    )
    asset["assetRoot"] = asset_root
    asset["assetFileDoor"] = asset_file_door
    asset["float_ik_include_all"] = True
    block_name = str(args.block_name)
    asset["load_block"] = block_name

    selected = selected_prepared_doors(prepared, args.include_review)
    if not selected:
        raise RuntimeError("No prepared doors are eligible for the generated YAML")
    existing_specs = []
    for config_path in args.existing_yaml:
        existing_specs.extend(active_asset_specs(config_path.expanduser().resolve()))
    combined_specs = existing_specs + [
        generated_spec(door, args.output_root.expanduser().resolve())
        for door in selected
    ]
    asset["trainAssets"] = {
        block_name: {
            str(index): spec
            for index, spec in enumerate(combined_specs)
        }
    }
    args.yaml_output.parent.mkdir(parents=True, exist_ok=True)
    args.yaml_output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def generate(prepared: list[PreparedDoor], args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    yaml_output = args.yaml_output.expanduser().resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    if yaml_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"YAML output already exists: {yaml_output}; pass --overwrite to replace it"
        )

    output_root.mkdir(parents=True)
    for door in selected_prepared_doors(prepared, args.include_review):
        copy_prepared_asset(door, output_root / door.clean_name)
    write_yaml(prepared, args)


def print_summary(report: dict) -> None:
    print(
        "AIGC door audit: "
        f"total={report['source_count']} ready={report['ready_count']} "
        f"review={report['review_count']} failed={report['failed_count']}"
    )
    for index, door in enumerate(report["doors"]):
        status = "READY" if door["ready_for_scripted_gt"] else ("FAILED" if door["errors"] else "REVIEW")
        extents = door.get("metrics", {}).get("unscaled_extents_m")
        extent_text = ""
        if extents:
            extent_text = " size=" + "x".join(f"{value:.3f}" for value in extents)
        print(
            f"[{index:02d}] {status:6s} {door['name']} type={door['handle_type']}{extent_text} "
            f"source={Path(door['source']).name}"
        )
        for message in door["errors"]:
            print(f"     error: {message}")
        for message in door["warnings"]:
            print(f"     warning: {message}")


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.yaml_output = args.yaml_output.expanduser().resolve()
    args.template_yaml = args.template_yaml.expanduser().resolve()
    if args.report_output is not None:
        args.report_output = args.report_output.expanduser().resolve()

    prepared = prepare_all(args)
    report = report_document(prepared, args)
    print_summary(report)

    report_output = args.report_output
    if report_output is None and args.mode == "generate":
        report_output = args.output_root.parent / f"{args.output_root.name}_compatibility_report.json"
    if report_output is not None:
        write_json(report_output, report)
        print(f"Report: {report_output}")

    if args.mode == "generate":
        generate(prepared, args)
        print(f"Prepared assets: {args.output_root}")
        print(f"Generated YAML: {args.yaml_output}")

    if report["failed_count"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
