#!/usr/bin/env python3
"""Build the persistent full A2W+Z1 URDF used as the a2wpush source asset."""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_A2W_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "a2w"
DEFAULT_A2W_FILE = "a2_wheel.urdf"
DEFAULT_Z1_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "z1"
DEFAULT_Z1_FILE = "urdf/z1_arm.urdf"
DEFAULT_OUT_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "a2wz1"
DEFAULT_OUT_FILE = "urdf/a2wz1.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-level/data/asset/a2wz1 from A2W and Z1 source URDFs.")
    parser.add_argument("--a2w_root", type=str, default=str(DEFAULT_A2W_ROOT))
    parser.add_argument("--a2w_file", type=str, default=DEFAULT_A2W_FILE)
    parser.add_argument("--z1_root", type=str, default=str(DEFAULT_Z1_ROOT))
    parser.add_argument("--z1_file", type=str, default=DEFAULT_Z1_FILE)
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE)
    parser.add_argument("--copy_meshes", action="store_true", help="Copy mesh files instead of creating relative symlinks.")
    return parser.parse_args()


def find_child(node: ET.Element, tag: str) -> ET.Element | None:
    for child in node:
        if child.tag == tag:
            return child
    return None


def indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def link_or_copy_mesh(source: Path, target: Path, copy_meshes: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Referenced mesh not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy_meshes:
        shutil.copy2(source, target)
        return
    rel_source = os.path.relpath(source, target.parent)
    target.symlink_to(rel_source)


def rewrite_meshes(root: ET.Element, source_root: Path, out_mesh_dir: Path, copy_meshes: bool) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_name = Path(filename).name
        source_path = (source_root / filename).resolve()
        if not source_path.exists():
            source_path = (source_root / "meshes" / mesh_name).resolve()
        link_or_copy_mesh(source_path, out_mesh_dir / mesh_name, copy_meshes)
        mesh.set("filename", f"../meshes/{mesh_name}")


def rewrite_mujoco_meshdir(root: ET.Element) -> None:
    for compiler in root.iter("compiler"):
        if "meshdir" in compiler.attrib:
            compiler.set("meshdir", "../meshes")


def build_mount_joint() -> ET.Element:
    joint = ET.Element("joint", {"name": "a2w_z1_mount_joint", "type": "fixed", "dont_collapse": "true"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": "base_link"})
    ET.SubElement(joint, "child", {"link": "base"})
    return joint


def build_asset(args: argparse.Namespace) -> Path:
    a2w_root = Path(args.a2w_root).expanduser().resolve()
    z1_root = Path(args.z1_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_urdf = out_root / args.out_file
    out_mesh_dir = out_root / "meshes"

    a2w_urdf = a2w_root / args.a2w_file
    z1_urdf = z1_root / args.z1_file
    if not a2w_urdf.exists():
        raise FileNotFoundError(f"A2W URDF not found: {a2w_urdf}")
    if not z1_urdf.exists():
        raise FileNotFoundError(f"Z1 URDF not found: {z1_urdf}")

    if out_mesh_dir.exists():
        shutil.rmtree(out_mesh_dir)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)

    a2w_root_xml = ET.parse(a2w_urdf).getroot()
    z1_root_xml = ET.parse(z1_urdf).getroot()
    output_root = ET.Element("robot", {"name": "a2wz1"})

    for child in list(a2w_root_xml):
        copied = deepcopy(child)
        rewrite_mujoco_meshdir(copied)
        output_root.append(copied)
        rewrite_meshes(copied, a2w_root, out_mesh_dir, args.copy_meshes)

    output_root.append(build_mount_joint())

    for child in list(z1_root_xml):
        copied = deepcopy(child)
        rewrite_mujoco_meshdir(copied)
        output_root.append(copied)
        rewrite_meshes(copied, z1_root, out_mesh_dir, args.copy_meshes)

    indent(output_root)
    ET.ElementTree(output_root).write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_urdf


def main() -> None:
    out_urdf = build_asset(parse_args())
    print(f"Wrote {out_urdf}")


if __name__ == "__main__":
    main()
