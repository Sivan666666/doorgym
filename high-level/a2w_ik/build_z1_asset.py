#!/usr/bin/env python3
"""Build the persistent Z1-only URDF asset used by a2wpush."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HIGH_LEVEL_ROOT = SCRIPT_DIR.parents[0]
REPO_ROOT = HIGH_LEVEL_ROOT.parents[0]
DEFAULT_Z1_ROOT = REPO_ROOT / "low-level" / "resources" / "robots" / "b1z1"
DEFAULT_Z1_FILE = "urdf/b1z1_basearn.urdf"
DEFAULT_OUT_ROOT = HIGH_LEVEL_ROOT / "data" / "asset" / "z1"
DEFAULT_Z1_ARM_FILE = "urdf/z1_arm.urdf"
Z1_ARM_LINKS = {
    "base",
    "link00",
    "link01",
    "link02",
    "link03",
    "link04",
    "link05",
    "link06",
    "gripperStator",
    "ee_gripper_link",
    "gripperMover",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-level/data/asset/z1 from the float_ik Z1 source URDF.")
    parser.add_argument("--z1_root", type=str, default=str(DEFAULT_Z1_ROOT))
    parser.add_argument("--z1_file", type=str, default=DEFAULT_Z1_FILE)
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--z1_arm_file", type=str, default=DEFAULT_Z1_ARM_FILE)
    return parser.parse_args()


def find_child(node: ET.Element, tag: str) -> ET.Element | None:
    for child in node:
        if child.tag == tag:
            return child
    return None


def referenced_mesh_names(root: ET.Element) -> set[str]:
    names = set()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            names.add(Path(filename).name)
            mesh.set("filename", f"../meshes/{Path(filename).name}")
    return names


def copy_referenced_meshes(src_mesh_dir: Path, dst_mesh_dir: Path, mesh_names: set[str]) -> None:
    if not src_mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {src_mesh_dir}")
    dst_mesh_dir.mkdir(parents=True, exist_ok=True)
    for mesh_name in sorted(mesh_names):
        src = src_mesh_dir / mesh_name
        if not src.exists():
            raise FileNotFoundError(f"Referenced Z1 mesh not found: {src}")
        shutil.copy2(src, dst_mesh_dir / mesh_name)


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


def write_filtered_urdf(source_urdf: Path, output_urdf: Path, keep_links: set[str]) -> set[str]:
    source_root = ET.parse(source_urdf).getroot()
    output_root = ET.Element(source_root.tag, {"name": "z1"})
    for elem in list(source_root):
        if elem.tag == "material":
            output_root.append(elem)
        elif elem.tag == "link" and elem.get("name") in keep_links:
            output_root.append(elem)
        elif elem.tag == "joint":
            parent = find_child(elem, "parent")
            child = find_child(elem, "child")
            if parent is None or child is None:
                continue
            if parent.get("link") in keep_links and child.get("link") in keep_links:
                output_root.append(elem)
    mesh_names = referenced_mesh_names(output_root)
    indent(output_root)
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output_urdf, encoding="utf-8", xml_declaration=True)
    return mesh_names


def build_asset(args: argparse.Namespace) -> Path:
    z1_root = Path(args.z1_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    z1_urdf = z1_root / args.z1_file
    z1_arm_file = out_root / args.z1_arm_file
    if not z1_urdf.exists():
        raise FileNotFoundError(f"Z1 URDF not found: {z1_urdf}")
    mesh_dir = out_root / "meshes"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_names = write_filtered_urdf(z1_urdf, z1_arm_file, Z1_ARM_LINKS)
    copy_referenced_meshes(z1_root / "meshes", mesh_dir, mesh_names)
    return z1_arm_file


def main() -> None:
    z1_arm_file = build_asset(parse_args())
    print(f"Wrote {z1_arm_file}")


if __name__ == "__main__":
    main()
