"""Shared DoorTwin candidate scoring and regression-safe selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _reports(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(summary.get("reports", []) or [])


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _grasp_z_abs(candidate: Any | None) -> float | None:
    if candidate is None:
        return None
    program = getattr(candidate, "skill_program", None)
    move = None if program is None else program.primitive("MoveEEToHandle")
    if move is None:
        return None
    pregrasp = move.params.get("pregrasp_offset")
    grasp = move.params.get("grasp_offset")
    if not isinstance(pregrasp, (list, tuple)) or not isinstance(grasp, (list, tuple)):
        return None
    if len(pregrasp) != 3 or len(grasp) != 3:
        return None
    try:
        # Static validation enforces equality. max() remains conservative for
        # legacy candidates that predate that invariant.
        value = max(abs(float(pregrasp[2])), abs(float(grasp[2])))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def candidate_score(
    summary: dict[str, Any],
    *,
    pass_open_angle_deg: float = 80.0,
    candidate: Any | None = None,
) -> dict[str, Any]:
    reports = _reports(summary)
    success = sum(bool(report.get("success", False)) for report in reports)
    traverse = sum(bool(report.get("body_passed", False)) for report in reports)
    opened = sum(_finite_float(report.get("door_open_deg")) >= pass_open_angle_deg for report in reports)
    unlock = sum(bool(report.get("handle_unlocked", False)) for report in reports)
    grasp = 0
    for report in reports:
        distance = report.get("metrics", {}).get("min_grasp_ee_handle_dist", report.get("ee_handle_dist"))
        if distance is not None and _finite_float(distance, math.inf) <= 0.09:
            grasp += 1
    collision = sum(bool(report.get("base_collision", False)) for report in reports)
    safe = len(reports) - collision
    mean_door = sum(_finite_float(report.get("door_open_deg")) for report in reports) / max(1, len(reports))
    mean_handle = sum(_finite_float(report.get("handle_rotation_deg")) for report in reports) / max(1, len(reports))
    tracking_values = [
        _finite_float(report.get("ee_tracking_error"))
        for report in reports
        if report.get("ee_tracking_error") is not None
        and math.isfinite(_finite_float(report.get("ee_tracking_error"), math.inf))
    ]
    mean_tracking = sum(tracking_values) / max(1, len(tracking_values))
    process_failures = len(summary.get("process_failures", []) or [])
    grasp_z_abs = _grasp_z_abs(candidate)
    grasp_z_preference = -grasp_z_abs if grasp_z_abs is not None else float("-inf")
    rank = (
        success,
        traverse,
        opened,
        unlock,
        grasp,
        safe,
        round(grasp_z_preference, 6),
        round(mean_door, 6),
        round(mean_handle, 6),
        round(-mean_tracking, 6),
        -process_failures,
    )
    return {
        "score_schema_version": 2,
        "rank": list(rank),
        "success_count": success,
        "traverse_count": traverse,
        "open_count": opened,
        "unlock_count": unlock,
        "grasp_count": grasp,
        "safe_count": safe,
        "collision_count": collision,
        "grasp_z_abs_m": grasp_z_abs,
        "mean_door_open_deg": mean_door,
        "mean_handle_rotation_deg": mean_handle,
        "mean_ee_tracking_error_m": mean_tracking,
        "process_failure_count": process_failures,
    }


def score_rank(score: dict[str, Any]) -> tuple[float, ...]:
    grasp_z_abs = score.get("grasp_z_abs_m")
    grasp_z_preference = (
        -float(grasp_z_abs)
        if grasp_z_abs is not None and math.isfinite(_finite_float(grasp_z_abs, math.inf))
        else float("-inf")
    )
    # Build from named fields so candidate graphs written before schema v2 are
    # still comparable after adding the real-robot grasp-height preference.
    return (
        float(score.get("success_count", 0)),
        float(score.get("traverse_count", 0)),
        float(score.get("open_count", 0)),
        float(score.get("unlock_count", 0)),
        float(score.get("grasp_count", 0)),
        float(score.get("safe_count", 0)),
        grasp_z_preference,
        float(score.get("mean_door_open_deg", 0.0)),
        float(score.get("mean_handle_rotation_deg", 0.0)),
        -float(score.get("mean_ee_tracking_error_m", 0.0)),
        -float(score.get("process_failure_count", 0)),
    )


@dataclass
class CandidateNode:
    candidate_hash: str
    parent_hash: str
    patch: dict[str, Any] = field(default_factory=dict)
    diagnostics: str = ""
    status: str = "pending"
    score: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "parent_hash": self.parent_hash,
            "patch": self.patch,
            "diagnostics": self.diagnostics,
            "status": self.status,
            "score": self.score,
            "artifacts": self.artifacts,
        }


class CandidateGraph:
    def __init__(self, initial_hash: str):
        self.initial_hash = str(initial_hash)
        self.best_hash = ""
        self.best_score: dict[str, Any] | None = None
        self.nodes: dict[str, CandidateNode] = {
            self.initial_hash: CandidateNode(self.initial_hash, "", status="pending")
        }

    def add(self, node: CandidateNode) -> None:
        if node.candidate_hash not in self.nodes:
            self.nodes[node.candidate_hash] = node

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateGraph":
        graph = cls(str(value["initial_hash"]))
        graph.best_hash = str(value.get("best_hash", ""))
        graph.best_score = value.get("best_score")
        graph.nodes = {}
        for raw in value.get("nodes", []) or []:
            node = CandidateNode(
                candidate_hash=str(raw["candidate_hash"]),
                parent_hash=str(raw.get("parent_hash", "")),
                patch=dict(raw.get("patch", {}) or {}),
                diagnostics=str(raw.get("diagnostics", "")),
                status=str(raw.get("status", "pending")),
                score=raw.get("score"),
                artifacts=dict(raw.get("artifacts", {}) or {}),
            )
            graph.nodes[node.candidate_hash] = node
        if graph.initial_hash not in graph.nodes:
            graph.nodes[graph.initial_hash] = CandidateNode(graph.initial_hash, "")
        return graph

    def evaluate(
        self,
        candidate_hash: str,
        summary: dict[str, Any],
        *,
        candidate: Any | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        node = self.nodes[str(candidate_hash)]
        score = candidate_score(summary, candidate=candidate)
        node.score = score
        node.artifacts.update(artifacts or {})
        if self.best_score is None or score_rank(score) > score_rank(self.best_score):
            if self.best_hash and self.best_hash in self.nodes and self.nodes[self.best_hash].status == "best":
                self.nodes[self.best_hash].status = "accepted"
            self.best_hash = node.candidate_hash
            self.best_score = score
            node.status = "best"
            return "new_best"
        if score_rank(score) < score_rank(self.best_score):
            node.status = "rejected_regression"
            return "rejected_regression"
        node.status = "tied_best_not_promoted"
        return "tied_best_not_promoted"

    def rejected_patches(self) -> list[dict[str, Any]]:
        return [
            {"candidate_hash": node.candidate_hash, "patch": node.patch, "diagnostics": node.diagnostics, "score": node.score}
            for node in self.nodes.values()
            if node.status in {"rejected_regression", "invalid"}
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "door_twin_candidate_graph_v1",
            "initial_hash": self.initial_hash,
            "best_hash": self.best_hash,
            "best_score": self.best_score,
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }
