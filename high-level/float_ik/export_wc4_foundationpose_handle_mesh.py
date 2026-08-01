#!/usr/bin/env python3
"""Export the WC4 lever_handle visuals as a standalone mesh.

The source WC4 URDF uses primitive boxes instead of a separate handle mesh.
This script recreates only the ``lever_handle`` link in that link's body frame,
which makes FoundationPose's output directly comparable with Isaac Gym's
lever_handle rigid-body pose.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_vector(value: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value.split())
    if len(values) != 3:
        raise ValueError(f"Expected xyz triplet, got {value!r}")
    return values


def transform_point(point, origin):
    xyz = parse_vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0)
    roll, pitch, yaw = (
        parse_vector(origin.attrib.get("rpy", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0)
    )
    x, y, z = point
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    x1, y1, z1 = x, cr * y - sr * z, sr * y + cr * z
    x2, y2, z2 = cp * x1 + sp * z1, y1, -sp * x1 + cp * z1
    x3, y3, z3 = cy * x2 - sy * y2, sy * x2 + cy * y2, z2
    return (x3 + xyz[0], y3 + xyz[1], z3 + xyz[2])


def export_link(urdf: Path, link_name: str, output: Path) -> None:
    root = ET.parse(urdf).getroot()
    link = next((node for node in root.findall("link") if node.attrib.get("name") == link_name), None)
    if link is None:
        raise ValueError(f"Link {link_name!r} not found in {urdf}")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    cube_faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        box = geometry.find("box")
        if box is None:
            raise NotImplementedError(f"Unsupported visual geometry in link {link_name}: {ET.tostring(geometry)}")
        sx, sy, sz = parse_vector(box.attrib["size"])
        local = (
            (-sx / 2, -sy / 2, -sz / 2), (sx / 2, -sy / 2, -sz / 2),
            (sx / 2, sy / 2, -sz / 2), (-sx / 2, sy / 2, -sz / 2),
            (-sx / 2, -sy / 2, sz / 2), (sx / 2, -sy / 2, sz / 2),
            (sx / 2, sy / 2, sz / 2), (-sx / 2, sy / 2, sz / 2),
        )
        offset = len(vertices)
        vertices.extend(transform_point(point, visual.find("origin")) for point in local)
        faces.extend(tuple(offset + index + 1 for index in face) for face in cube_faces)

    if not vertices:
        raise RuntimeError(f"No visual geometry found for link {link_name!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# WC4 lever_handle visual geometry in rigid-body frame", "o lever_handle"]
    lines.extend(f"v {x:.10f} {y:.10f} {z:.10f}" for x, y, z in vertices)
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mins = [min(point[axis] for point in vertices) for axis in range(3)]
    maxs = [max(point[axis] for point in vertices) for axis in range(3)]
    extents = [maxs[axis] - mins[axis] for axis in range(3)]
    print(
        f"Exported {link_name} mesh: {output}\n"
        f"vertices={len(vertices)} faces={len(faces)} bounds={[mins, maxs]} extents={extents}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/asset/door_set/wc4/model.urdf",
    )
    parser.add_argument("--link", default="lever_handle")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/asset/door_set/wc4/assets/meshes/lever_handle_foundationpose.obj",
    )
    args = parser.parse_args()
    export_link(args.urdf.resolve(), args.link, args.output.resolve())


if __name__ == "__main__":
    main()
