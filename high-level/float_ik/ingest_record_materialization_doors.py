#!/usr/bin/env python3
"""Ingest generated record_materialization doors into the DoorGym asset layout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]
DEFAULT_SRC = Path("/home/sivan/Downloads/record_materialization")
DEFAULT_OUT_CFG = HIGH_LEVEL_ROOT / "experiments" / "isaacgym" / "b1z1_opendoor_record_materialization.yaml"
DEFAULT_ASSET_SUBDIR = "door_set/record_materialization"
HANDLE_NAME_HINTS = ("handle", "lever", "knob", "spindle")
DOOR_PANEL_NAMES = ("door_panel", "panel", "door")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC))
    parser.add_argument("--out_cfg", type=str, default=str(DEFAULT_OUT_CFG))
    parser.add_argument("--asset_subdir", type=str, default=DEFAULT_ASSET_SUBDIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--actor_scale", type=float, default=1.0)
    parser.add_argument("--actor_yaw_offset", type=float, default=-math.pi / 2.0)
    parser.add_argument("--robot_y_offset", type=float, default=0.18)
    parser.add_argument("--summary", type=str, default="")
    return parser.parse_args()


def parse_vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def rot_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def tf_from_origin(node: ET.Element | None) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    if node is None:
        return mat
    xyz = parse_vec(node.get("xyz"))
    rpy = parse_vec(node.get("rpy"))
    mat[:3, :3] = rot_from_rpy(rpy)
    mat[:3, 3] = xyz
    return mat


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (hom @ mat.T)[:, :3]


def yaw_matrix(yaw: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    mat[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mat


def transform_vector_yaw(yaw: float, vec: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    x, y, z = [float(v) for v in vec]
    return np.asarray([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def resolve_mesh_path(filename: str, src_dir: Path) -> Path:
    raw = filename.replace("package://", "")
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [
        src_dir / raw,
        src_dir / "assets" / "meshes" / path.name,
        src_dir / "meshes" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) >= 4:
                    verts.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return np.asarray(verts, dtype=np.float64)


def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    verts = []
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            offset = 84
            for _ in range(tri_count):
                offset += 12
                for _v in range(3):
                    verts.append(struct.unpack("<fff", data[offset : offset + 12]))
                    offset += 12
                offset += 2
            return np.asarray(verts, dtype=np.float64)
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0] == "vertex":
            verts.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return np.asarray(verts, dtype=np.float64)


def geometry_points(geometry: ET.Element, src_dir: Path) -> tuple[np.ndarray, list[str]]:
    warnings = []
    box = geometry.find("box")
    if box is not None and box.get("size"):
        sx, sy, sz = parse_vec(box.get("size"))
        corners = [[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (-sz / 2, sz / 2)]
        return np.asarray(corners, dtype=np.float64), warnings
    cyl = geometry.find("cylinder")
    if cyl is not None:
        radius = float(cyl.get("radius", "0"))
        length = float(cyl.get("length", "0"))
        pts = []
        for z in (-length / 2, length / 2):
            for i in range(16):
                a = 2.0 * math.pi * i / 16.0
                pts.append([radius * math.cos(a), radius * math.sin(a), z])
        return np.asarray(pts, dtype=np.float64), warnings
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius", "0"))
        corners = [[x, y, z] for x in (-radius, radius) for y in (-radius, radius) for z in (-radius, radius)]
        return np.asarray(corners, dtype=np.float64), warnings
    mesh = geometry.find("mesh")
    if mesh is not None and mesh.get("filename"):
        mesh_path = resolve_mesh_path(mesh.get("filename", ""), src_dir)
        if not mesh_path.exists():
            return np.zeros((0, 3), dtype=np.float64), [f"missing mesh: {mesh.get('filename')}"]
        suffix = mesh_path.suffix.lower()
        if suffix == ".obj":
            pts = read_obj_vertices(mesh_path)
        elif suffix == ".stl":
            pts = read_stl_vertices(mesh_path)
        else:
            pts = np.zeros((0, 3), dtype=np.float64)
            warnings.append(f"unsupported mesh format: {mesh_path.name}")
        scale = parse_vec(mesh.get("scale"), default=(1.0, 1.0, 1.0))
        return pts * scale.reshape(1, 3), warnings
    return np.zeros((0, 3), dtype=np.float64), ["unsupported geometry"]


def child_link(joint: ET.Element) -> str:
    child = joint.find("child")
    return "" if child is None else str(child.get("link", ""))


def parent_link(joint: ET.Element) -> str:
    parent = joint.find("parent")
    return "" if parent is None else str(parent.get("link", ""))


def joint_axis(joint: ET.Element) -> np.ndarray:
    axis = joint.find("axis")
    return parse_vec(axis.get("xyz") if axis is not None else "0 0 1")


def link_transforms(root: ET.Element) -> dict[str, np.ndarray]:
    links = {link.get("name") for link in root.findall("link") if link.get("name")}
    children = {child_link(j) for j in root.findall("joint") if child_link(j)}
    roots = sorted(links - children) or sorted(links)
    transforms = {name: np.eye(4, dtype=np.float64) for name in roots}
    pending = list(root.findall("joint"))
    changed = True
    while pending and changed:
        changed = False
        next_pending = []
        for joint in pending:
            parent = parent_link(joint)
            child = child_link(joint)
            if parent in transforms and child:
                transforms[child] = transforms[parent] @ tf_from_origin(joint.find("origin"))
                changed = True
            else:
                next_pending.append(joint)
        pending = next_pending
    for link in links:
        transforms.setdefault(link, np.eye(4, dtype=np.float64))
    return transforms


def collect_link_points(
    root: ET.Element,
    src_dir: Path,
    link_name: str | None = None,
    world: bool = True,
    transforms: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    warnings = []
    transforms = transforms or link_transforms(root)
    all_points = []
    links = root.findall("link")
    if link_name:
        links = [link for link in links if link.get("name") == link_name]
    for link in links:
        link_tf = transforms.get(link.get("name", ""), np.eye(4, dtype=np.float64)) if world else np.eye(4)
        for geom_parent in list(link.findall("visual")) + list(link.findall("collision")):
            geometry = geom_parent.find("geometry")
            if geometry is None:
                continue
            pts, geom_warnings = geometry_points(geometry, src_dir)
            warnings.extend(geom_warnings)
            if pts.size == 0:
                continue
            geom_tf = tf_from_origin(geom_parent.find("origin"))
            all_points.append(transform_points(link_tf @ geom_tf, pts))
    if not all_points:
        return np.zeros((0, 3), dtype=np.float64), warnings
    return np.concatenate(all_points, axis=0), warnings


def bbox_dict(points: np.ndarray) -> dict:
    if points.size == 0:
        raise ValueError("empty points for bbox")
    return {
        "min": points.min(axis=0).round(6).tolist(),
        "max": points.max(axis=0).round(6).tolist(),
    }


def trimmed_handle_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 20:
        return points
    centered = points - np.median(points, axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, points.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    proj = centered @ axis
    lo, hi = np.quantile(proj, [0.15, 0.85])
    mask = (proj >= lo) & (proj <= hi)
    if int(mask.sum()) >= 8:
        return points[mask]
    return points


def choose_door_joint(root: ET.Element) -> ET.Element | None:
    joints = [j for j in root.findall("joint") if j.get("type") in ("revolute", "continuous")]
    for joint in joints:
        name = str(joint.get("name", "")).lower()
        if "hinge" in name and "door" in name:
            return joint
    for joint in joints:
        child = child_link(joint).lower()
        if any(token in child for token in DOOR_PANEL_NAMES):
            return joint
    return joints[0] if joints else None


def choose_handle_joint(root: ET.Element, door_joint: ET.Element | None) -> ET.Element | None:
    joints = [j for j in root.findall("joint") if j.get("type") in ("revolute", "continuous")]
    if door_joint is not None:
        joints = [j for j in joints if j is not door_joint]
    for joint in joints:
        text = " ".join([str(joint.get("name", "")), parent_link(joint), child_link(joint)]).lower()
        if any(hint in text for hint in HANDLE_NAME_HINTS):
            return joint
    return joints[0] if joints else None


def choose_actor_yaw_and_classify(
    base_yaw: float,
    door_axis_z: float,
    handle_axis_local: np.ndarray,
    handle_center_world: np.ndarray,
    door_points: np.ndarray,
) -> dict:
    """Pick the yaw that puts the handle on the robot-facing side.

    Door scenes in this repo place the robot on the +X side of the closed door,
    looking roughly toward -X.  In the generated door assets, the grabbable
    handle face should therefore be on the lower-X side of the door panel after
    the asset yaw is applied.  If not, rotate the whole asset by pi around Z.
    """

    candidates = []
    for yaw in (float(base_yaw), float(base_yaw) + math.pi):
        yaw_tf = yaw_matrix(yaw)
        door_pts = transform_points(yaw_tf, door_points)
        handle_xyz = transform_points(yaw_tf, handle_center_world.reshape(1, 3))[0]
        door_center = 0.5 * (door_pts.min(axis=0) + door_pts.max(axis=0))
        handle_axis_world = transform_vector_yaw(yaw, handle_axis_local)
        front_delta = float(handle_xyz[0] - door_center[0])
        side_delta = float(handle_xyz[1] - door_center[1])
        candidates.append(
            {
                "actor_yaw_offset": float(yaw),
                "front_delta": front_delta,
                "side_delta": side_delta,
                "handle_center_after_yaw": handle_xyz,
                "door_center_after_yaw": door_center,
                "handle_axis_after_yaw": handle_axis_world,
            }
        )

    selected = min(candidates, key=lambda item: item["front_delta"])
    original = candidates[0]
    side = "right" if selected["side_delta"] >= 0.0 else "left"
    door_motion_sign_multiplier = -1.0 if door_axis_z >= 0.0 else 1.0
    handle_axis_x = float(selected["handle_axis_after_yaw"][0])
    if abs(handle_axis_x) < 1.0e-5:
        handle_rotate_direction = "unknown"
    elif handle_axis_x > 0.0:
        handle_rotate_direction = "positive_counterclockwise_from_robot"
    else:
        handle_rotate_direction = "positive_clockwise_from_robot"

    return {
        "variant": f"push_{side}",
        "handle_side": side,
        "actor_yaw_offset": float(selected["actor_yaw_offset"]),
        "door_motion_sign_multiplier": door_motion_sign_multiplier,
        "handle_was_behind": bool(original["front_delta"] > selected["front_delta"] + 1.0e-5),
        "handle_front_delta_before_yaw_fix": float(original["front_delta"]),
        "handle_front_delta_after_yaw_fix": float(selected["front_delta"]),
        "handle_side_delta_after_yaw_fix": float(selected["side_delta"]),
        "handle_center_after_yaw": selected["handle_center_after_yaw"].round(6).tolist(),
        "door_center_after_yaw": selected["door_center_after_yaw"].round(6).tolist(),
        "handle_axis_after_yaw": selected["handle_axis_after_yaw"].round(6).tolist(),
        "handle_rotate_direction": handle_rotate_direction,
    }


def copy_asset(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_dir} exists; pass --overwrite")
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("door.zip"))


def analyze_and_write(src_urdf: Path, dst_dir: Path, args: argparse.Namespace) -> dict:
    src_dir = src_urdf.parent
    root = ET.parse(src_urdf).getroot()
    transforms = link_transforms(root)
    all_points, warnings = collect_link_points(root, src_dir, world=True, transforms=transforms)
    if all_points.size == 0:
        raise RuntimeError("no supported geometry points found")
    door_joint = choose_door_joint(root)
    handle_joint = choose_handle_joint(root, door_joint)
    if door_joint is None or handle_joint is None:
        raise RuntimeError("missing door or handle revolute joint")

    door_link = child_link(door_joint)
    handle_link = child_link(handle_joint)
    door_points_world, door_warnings = collect_link_points(
        root, src_dir, link_name=door_link, world=True, transforms=transforms
    )
    warnings.extend(door_warnings)
    if door_points_world.size == 0:
        door_points_world = all_points
        warnings.append(f"door link {door_link!r} has no supported geometry; using all geometry for orientation")
    handle_points_local, handle_warnings = collect_link_points(
        root, src_dir, link_name=handle_link, world=False, transforms=transforms
    )
    warnings.extend(handle_warnings)
    if handle_points_local.size == 0:
        handle_points_local = np.zeros((1, 3), dtype=np.float64)
        warnings.append(f"handle link {handle_link!r} has no supported geometry; using link origin")
    handle_trim = trimmed_handle_points(handle_points_local)
    handle_goal = np.median(handle_trim, axis=0)
    handle_tf = transforms.get(handle_link, np.eye(4, dtype=np.float64))
    handle_center_world = transform_points(handle_tf, handle_goal.reshape(1, 3))[0]
    classification = choose_actor_yaw_and_classify(
        float(args.actor_yaw_offset),
        float(joint_axis(door_joint)[2]),
        joint_axis(handle_joint),
        handle_center_world,
        door_points_world,
    )

    bounding = bbox_dict(all_points)
    handle_bounding = {
        "handle_min": handle_trim.min(axis=0).round(6).tolist(),
        "handle_max": handle_trim.max(axis=0).round(6).tolist(),
        "goal_pos": handle_goal.round(6).tolist(),
    }
    (dst_dir / "bounding_box.json").write_text(json.dumps(bounding, indent=2) + "\n", encoding="utf-8")
    (dst_dir / "handle_bounding.json").write_text(json.dumps(handle_bounding, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "name": dst_dir.name,
        "variant": classification["variant"],
        "door_dof_name": door_joint.get("name"),
        "handle_dof_name": handle_joint.get("name"),
        "door_body_name": door_link,
        "handle_body_name": handle_link,
        "door_axis": joint_axis(door_joint).round(6).tolist(),
        "handle_axis": joint_axis(handle_joint).round(6).tolist(),
        "actor_yaw_offset": float(classification["actor_yaw_offset"]),
        "door_motion_sign_multiplier": classification["door_motion_sign_multiplier"],
        "handle_center_world": handle_center_world.round(6).tolist(),
        "handle_side": classification["handle_side"],
        "handle_was_behind": classification["handle_was_behind"],
        "handle_front_delta_before_yaw_fix": classification["handle_front_delta_before_yaw_fix"],
        "handle_front_delta_after_yaw_fix": classification["handle_front_delta_after_yaw_fix"],
        "handle_side_delta_after_yaw_fix": classification["handle_side_delta_after_yaw_fix"],
        "handle_center_after_yaw": classification["handle_center_after_yaw"],
        "door_center_after_yaw": classification["door_center_after_yaw"],
        "handle_axis_after_yaw": classification["handle_axis_after_yaw"],
        "handle_rotate_direction": classification["handle_rotate_direction"],
        "warnings": sorted(set(warnings)),
    }


def make_cfg(entries: list[dict], args: argparse.Namespace) -> dict:
    block = {}
    for idx, item in enumerate(entries):
        rel_dir = f"{args.asset_subdir.rstrip('/')}/{item['name']}"
        block[str(idx)] = {
            "name": item["name"],
            "path": f"{rel_dir}/model.urdf",
            "bounding_box": f"{rel_dir}/bounding_box.json",
            "handle_bounding": f"{rel_dir}/handle_bounding.json",
            "actor_scale": float(args.actor_scale),
            "actor_yaw_offset": float(item["actor_yaw_offset"]),
            "actor_position_offset": [0.0, 0.0, 0.0],
            "robot_y_offset": float(args.robot_y_offset),
            "door_motion_sign_multiplier": float(item["door_motion_sign_multiplier"]),
            "door_body_name": item["door_body_name"],
            "handle_body_name": item["handle_body_name"],
            "door_dof_name": item["door_dof_name"],
            "handle_dof_name": item["handle_dof_name"],
            "generated_variant": item["variant"],
            "handle_side": item["handle_side"],
            "handle_was_behind": bool(item["handle_was_behind"]),
            "handle_rotate_direction": item["handle_rotate_direction"],
        }
    return {
        "env": {
            "asset": {
                "assetRoot": "data/asset",
                "assetFileDoor": "",
                "load_block": "record_materialization",
                "trainAssets": {"record_materialization": block},
            }
        }
    }


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out_cfg = Path(args.out_cfg).expanduser()
    if not out_cfg.is_absolute():
        out_cfg = (REPO_ROOT / out_cfg).resolve()
    asset_root = HIGH_LEVEL_ROOT / "data" / "asset"
    asset_subdir = Path(args.asset_subdir)
    out_asset_root = asset_root / asset_subdir
    out_asset_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    accepted = []
    for src_urdf in sorted(src.rglob("model.urdf")):
        name = src_urdf.parent.name
        dst_dir = out_asset_root / name
        record = {"name": name, "src": str(src_urdf.parent), "dst": str(dst_dir)}
        try:
            copy_asset(src_urdf.parent, dst_dir, args.overwrite)
            result = analyze_and_write(src_urdf, dst_dir, args)
            record.update(result)
            accepted.append(record)
        except Exception as exc:
            record.update({"status": "rejected", "reason": str(exc)})
        summaries.append(record)

    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(accepted, args)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary_path = Path(args.summary).expanduser() if args.summary else out_cfg.with_suffix(".summary.json")
    if not summary_path.is_absolute():
        summary_path = (REPO_ROOT / summary_path).resolve()
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    variants = {}
    for item in accepted:
        variants[item["variant"]] = variants.get(item["variant"], 0) + 1
    print(f"Ingested {len(accepted)}/{len(summaries)} generated doors.")
    print(f"Wrote cfg: {out_cfg}")
    print(f"Wrote summary: {summary_path}")
    print("Variants:", variants)


if __name__ == "__main__":
    main()
