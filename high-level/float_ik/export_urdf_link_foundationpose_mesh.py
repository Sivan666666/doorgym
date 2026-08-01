#!/usr/bin/env python3
"""Export one URDF link's visual geometry in its rigid-body frame.

This helper is intended for FoundationPose registration. It supports the
primitive visual geometry used by the generated door assets and keeps the
resulting mesh origin aligned with the URDF link frame.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def parse_vector(value: str, expected: int = 3) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if values.shape != (expected,):
        raise ValueError(f"Expected {expected} values, got {value!r}")
    return values


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    xyz = parse_vector(origin.attrib.get("xyz", "0 0 0"))
    roll, pitch, yaw = parse_vector(origin.attrib.get("rpy", "0 0 0"))
    transform = trimesh.transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def visual_mesh(visual: ET.Element, urdf_dir: Path) -> trimesh.Trimesh:
    geometry = visual.find("geometry")
    if geometry is None:
        raise ValueError("URDF visual has no geometry")

    box = geometry.find("box")
    cylinder = geometry.find("cylinder")
    sphere = geometry.find("sphere")
    mesh_node = geometry.find("mesh")
    if box is not None:
        mesh = trimesh.creation.box(extents=parse_vector(box.attrib["size"]))
    elif cylinder is not None:
        mesh = trimesh.creation.cylinder(
            radius=float(cylinder.attrib["radius"]),
            height=float(cylinder.attrib["length"]),
            sections=64,
        )
    elif sphere is not None:
        mesh = trimesh.creation.icosphere(
            subdivisions=3,
            radius=float(sphere.attrib["radius"]),
        )
    elif mesh_node is not None:
        filename = mesh_node.attrib["filename"]
        if filename.startswith("package://"):
            raise ValueError(f"package:// URI is not supported: {filename}")
        loaded = trimesh.load((urdf_dir / filename).resolve(), force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected a triangle mesh for {filename}, got {type(loaded)}")
        mesh = loaded.copy()
        if "scale" in mesh_node.attrib:
            mesh.apply_scale(parse_vector(mesh_node.attrib["scale"]))
    else:
        raise NotImplementedError(f"Unsupported visual geometry: {ET.tostring(geometry)}")

    mesh.apply_transform(origin_matrix(visual.find("origin")))
    return mesh


def export_link(urdf: Path, link_name: str, output: Path) -> None:
    root = ET.parse(urdf).getroot()
    link = next((node for node in root.findall("link") if node.attrib.get("name") == link_name), None)
    if link is None:
        raise ValueError(f"Link {link_name!r} not found in {urdf}")

    meshes = [visual_mesh(visual, urdf.parent) for visual in link.findall("visual")]
    if not meshes:
        raise RuntimeError(f"No visual geometry found for link {link_name!r}")
    combined = trimesh.util.concatenate(meshes)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output)
    print(
        f"Exported {link_name} mesh to {output}\n"
        f"vertices={len(combined.vertices)} faces={len(combined.faces)} "
        f"bounds={combined.bounds.tolist()} extents={combined.extents.tolist()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--link", default="lever_handle")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_link(args.urdf.resolve(), args.link, args.output.resolve())


if __name__ == "__main__":
    main()
