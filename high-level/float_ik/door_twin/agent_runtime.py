"""Tool-driven DoorTwin Agent runtime with a Codex file-exchange backend."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .benchmark.evaluation import evaluate_asset, stage_metrics
from .benchmark.observer import compact_rollout_summary, select_phase_montages, structured_visual_diagnostics
from .benchmark.patches import CombinedRepairPatch
from .benchmark.run_agent_ablation import run_asset_probe, run_rollout, run_seed_set
from .benchmark.schema import BenchmarkManifest, Candidate, DoorCase, sha256_json
from .candidate_selection import CandidateGraph, CandidateNode
from .experience import DoorSignature, ExperienceCatalog, ExperienceStore
from .validation import summarize_physics_probes, validate_candidate_static, write_json


AGENT_SCHEMA_VERSION = "door_twin_agent_session_v1"
ACTION_SCHEMA_VERSION = "door_twin_agent_action_v1"


TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_guidance": "Read the canonical DoorTwin README or one allowlisted debug document.",
    "find_experiences": "Retrieve top-k structurally/textually similar validated door experiences.",
    "read_experience": "Read one retrieved experience's skill, debug document, and structured metadata.",
    "inspect_candidate": "Inspect the current candidate DoorSpec, URDF structure, signature, and skill.",
    "run_static_validation": "Run Level-1 static validation. Blocking failures gate simulation.",
    "run_physics_probe": "Run Level-2 load, hinge, handle, and grasp reachability probes.",
    "run_rollout": "Run Level-3 full development rollouts on the fixed seeds and score the candidate.",
    "inspect_rollout": "Read the latest compact rollout evidence and dominant failure.",
    "inspect_images": "Return deterministic phase montages plus structured visual diagnostics and patch hints.",
    "apply_candidate_patch": "Apply one bounded asset/skill JSON patch to the historical best candidate.",
    "compare_candidates": "Inspect the candidate graph, scores, accepted and rejected patches.",
    "select_best": "Rollback the current pointer to the historical best candidate.",
    "finish": "Finish exploration and evaluate the historical best candidate on held-out seeds.",
}


@dataclass(frozen=True)
class AgentAction:
    action: str
    arguments: dict[str, Any]
    diagnostics: str = ""

    @classmethod
    def from_obj(cls, value: Any) -> "AgentAction":
        if not isinstance(value, dict):
            raise ValueError("Agent action must be a JSON object")
        action = str(value.get("action", ""))
        if action not in TOOL_DESCRIPTIONS:
            raise ValueError(f"Unknown DoorTwin Agent tool: {action!r}")
        arguments = value.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            raise ValueError("Agent action.arguments must be an object")
        return cls(action, dict(arguments), str(value.get("diagnostics", "")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": ACTION_SCHEMA_VERSION, "action": self.action, "arguments": self.arguments, "diagnostics": self.diagnostics}


class ManualCodexBackend:
    """Atomic request/response exchange for the current Codex session."""

    def __init__(self, exchange_dir: str | Path, *, poll_s: float = 0.5, timeout_s: float = 86400.0):
        self.exchange_dir = Path(exchange_dir).expanduser().resolve()
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self.poll_s = float(poll_s)
        self.timeout_s = float(timeout_s)

    def next_action(self, request_index: int, context: dict[str, Any]) -> AgentAction:
        request = {
            "schema_version": "door_twin_codex_tool_request_v1",
            "request_index": int(request_index),
            "instructions": (
                "Act as DoorTwinAgent. Return exactly one JSON tool action. Use retrieved examples as reusable prior, "
                "never modify Python/mesh/URDF, never repeat rejected patches, and finish only with the historical best candidate. "
                "Keep MoveEEToHandle pregrasp_offset.z equal to grasp_offset.z. Subject to unchanged task success, "
                "prefer the smallest absolute shared z offset (closest to zero) for centered, real-robot-stable grasping."
            ),
            "action_schema": {"action": "one tool name", "arguments": {}, "diagnostics": "evidence-based reason"},
            "tools": TOOL_DESCRIPTIONS,
            "context": context,
        }
        request["request_fingerprint"] = sha256_json(request)
        request_path = self.exchange_dir / f"request_{request_index:04d}.json"
        response_path = self.exchange_dir / f"response_{request_index:04d}.json"
        if request_path.is_file():
            existing = json.loads(request_path.read_text(encoding="utf-8"))
            if existing.get("request_fingerprint") != request["request_fingerprint"]:
                raise RuntimeError(f"Existing manual request has different fingerprint: {request_path}")
        else:
            _atomic_json(request_path, request)
        print(f"DOORTWIN_CODEX_REQUEST request={request_path} response={response_path}", flush=True)
        deadline = time.monotonic() + self.timeout_s
        last_error = ""
        while True:
            if time.monotonic() >= deadline:
                detail = f"; last invalid response: {last_error}" if last_error else ""
                raise TimeoutError(f"Timed out waiting for Codex action: {response_path}{detail}")
            if response_path.is_file():
                try:
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                    response_fingerprint = str(response.get("request_fingerprint", "")) if isinstance(response, dict) else ""
                    if response_fingerprint and response_fingerprint != request["request_fingerprint"]:
                        raise ValueError("response request_fingerprint does not match the pending request")
                    return AgentAction.from_obj(response)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message != last_error:
                        print(f"DOORTWIN_CODEX_INVALID_RESPONSE path={response_path} error={message}", flush=True)
                        last_error = message
            time.sleep(self.poll_s)


class ScriptedBackend:
    """Deterministic backend used by tests and non-interactive smoke runs."""

    def __init__(self, actions: list[dict[str, Any]]):
        self.actions = [AgentAction.from_obj(action) for action in actions]

    def next_action(self, request_index: int, _context: dict[str, Any]) -> AgentAction:
        if request_index >= len(self.actions):
            return AgentAction("finish", {"evaluate_hidden": False}, "script exhausted")
        return self.actions[request_index]


class DoorTwinAgentSession:
    def __init__(
        self,
        *,
        run_root: str | Path,
        manifest: BenchmarkManifest,
        door: DoorCase,
        initial_candidate: Candidate,
        catalog: ExperienceCatalog,
        backend: Any,
        visual_feedback: bool = True,
        benchmark_mode: bool = False,
        max_tool_turns: int = 30,
        max_repairs: int = 5,
        force: bool = False,
        stream_output: bool = False,
    ):
        self.root = Path(run_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.door = door
        self.catalog = catalog
        self.backend = backend
        self.visual_feedback = bool(visual_feedback)
        self.benchmark_mode = bool(benchmark_mode)
        self.max_tool_turns = int(max_tool_turns)
        self.max_repairs = int(max_repairs)
        self.force = bool(force)
        self.stream_output = bool(stream_output)
        self.options = SimpleNamespace(dry_run=False, stream_output=self.stream_output, force=self.force)
        self.store = ExperienceStore(self.root)
        self.learned_store = ExperienceStore(catalog.path.parent)
        target_hashes = initial_candidate.metadata.get("excluded_prior_hashes", []) if benchmark_mode else []
        self.prior_snapshot = catalog.freeze(
            exclude_door_names=[door.door_name] if benchmark_mode else [],
            exclude_hashes=target_hashes,
        )
        self.candidates: dict[str, Candidate] = {initial_candidate.fingerprint: initial_candidate}
        self.current_hash = initial_candidate.fingerprint
        self.graph = CandidateGraph(initial_candidate.fingerprint)
        self.last_observation: dict[str, Any] = {"event": "session_started"}
        self.last_rollout: dict[str, Any] | None = None
        self.last_rollout_dir: Path | None = None
        self.last_rollout_hash: str | None = None
        self.static_reports: dict[str, dict[str, Any]] = {}
        self.probe_reports: dict[str, dict[str, Any]] = {}
        self.tool_turn = 0
        self.repair_count = 0
        self.finished = False
        self.retrieved_ids: set[str] = set()
        if (self.root / "session.json").is_file():
            self._resume(initial_candidate)
        else:
            self._save_candidate(initial_candidate)
            _atomic_json(self.root / "prior_snapshot.json", self.prior_snapshot)
            self._persist()

    @property
    def current(self) -> Candidate:
        return self.candidates[self.current_hash]

    @property
    def best(self) -> Candidate:
        return self.candidates[self.graph.best_hash or self.current_hash]

    def context(self) -> dict[str, Any]:
        return {
            "session": {
                "run_root": str(self.root),
                "door_name": self.door.door_name,
                "benchmark_mode": self.benchmark_mode,
                "visual_feedback": self.visual_feedback,
                "tool_turn": self.tool_turn,
                "remaining_tool_turns": self.max_tool_turns - self.tool_turn,
                "repair_count": self.repair_count,
                "remaining_repairs": self.max_repairs - self.repair_count,
            },
            "current_candidate_hash": self.current_hash,
            "best_candidate_hash": self.graph.best_hash,
            "best_score": self.graph.best_score,
            "optimization_preferences": {
                "hard_constraint": "MoveEEToHandle.pregrasp_offset.z == MoveEEToHandle.grasp_offset.z",
                "lexicographic_rule": "Never trade task success/stage completion/safety for grasp height.",
                "tie_break_preference": "Minimize abs(shared grasp z offset), i.e. prefer values closer to 0.",
                "motivation": "Avoid unnecessarily low simulated grasps that are more likely to loosen on the real robot.",
            },
            "prior_snapshot_fingerprint": self.prior_snapshot["fingerprint"],
            "retrieved_experience_ids": sorted(self.retrieved_ids),
            "rejected_patches": self.graph.rejected_patches(),
            "last_observation": self.last_observation,
        }

    def run(self) -> dict[str, Any]:
        while not self.finished and self.tool_turn < self.max_tool_turns:
            action = self.backend.next_action(self.tool_turn, self.context())
            observation = self.execute(action)
            self._append_trace(action, observation)
            self.last_observation = observation
            self.tool_turn += 1
            self._persist()
        if not self.finished:
            self.last_observation = self._finish({"evaluate_hidden": False}, reason="tool_budget_exhausted")
            self._persist()
        return self.final_report()

    def execute(self, action: AgentAction) -> dict[str, Any]:
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "read_guidance": self._read_guidance,
            "find_experiences": self._find_experiences,
            "read_experience": self._read_experience,
            "inspect_candidate": self._inspect_candidate,
            "run_static_validation": self._run_static,
            "run_physics_probe": self._run_physics,
            "run_rollout": self._run_full_rollout,
            "inspect_rollout": self._inspect_rollout,
            "inspect_images": self._inspect_images,
            "apply_candidate_patch": lambda args: self._apply_patch(args, action.diagnostics),
            "compare_candidates": self._compare_candidates,
            "select_best": self._select_best,
            "finish": self._finish,
        }
        try:
            result = handlers[action.action](action.arguments)
            return {"ok": True, "tool": action.action, "result": result}
        except Exception as exc:
            return {"ok": False, "tool": action.action, "error": f"{type(exc).__name__}: {exc}"}

    def _read_guidance(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = str(arguments.get("name", "agent_door_twin"))
        paths = {"agent_door_twin": Path(__file__).with_name("agent_door_twin.md")}
        for record in self.catalog.records:
            paths[record.experience_id] = record.debug_doc_path
        if name not in paths:
            raise ValueError(f"Guidance name is not allowlisted: {name}")
        text = paths[name].read_text(encoding="utf-8")
        return {"name": name, "path": str(paths[name]), "content": text[:50000], "truncated": len(text) > 50000}

    def _find_experiences(self, arguments: dict[str, Any]) -> dict[str, Any]:
        signature = DoorSignature.from_candidate(self.current)
        matches = self.catalog.search(
            signature,
            query=str(arguments.get("query", "")),
            failure_stage=str(arguments.get("failure_stage", "")),
            limit=min(5, max(1, int(arguments.get("limit", 3)))),
            exclude_door_names=[self.door.door_name] if self.benchmark_mode else [],
            allowed_snapshot=self.prior_snapshot,
        )
        self.retrieved_ids.update(match.record.experience_id for match in matches)
        return {"query_signature": signature.to_dict(), "matches": [match.to_dict() for match in matches]}

    def _read_experience(self, arguments: dict[str, Any]) -> dict[str, Any]:
        experience_id = str(arguments["experience_id"])
        if experience_id not in self.retrieved_ids:
            raise ValueError("Experience must be returned by find_experiences before it can be read")
        record = self.catalog.get(experience_id)
        return {
            **record.summary(),
            "skill": json.loads(record.skill_path.read_text(encoding="utf-8")),
            "debug_document": record.debug_doc_path.read_text(encoding="utf-8")[:50000],
        }

    def _inspect_candidate(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_hash": self.current_hash,
            "door_signature": DoorSignature.from_candidate(self.current).to_dict(),
            "public_context": self.current.public_context(),
            "runtime_spec": self.current.runtime_spec,
            "handle_bounding": self.current.handle_bounding,
            "skill_program": self.current.skill_program.to_dict(),
        }

    def _run_static(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        report = validate_candidate_static(self.current).to_dict()
        self.static_reports[self.current_hash] = report
        write_json(self.root / "validation" / self.current_hash / "static_validation.json", report)
        return report

    def _run_physics(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        static = self.static_reports.get(self.current_hash)
        if not static or not static.get("passed", False):
            raise RuntimeError("Current candidate must pass run_static_validation before physics probes")
        out = self.root / "probes" / self.current_hash
        paths = self.current.write(out / "candidate")
        base_seed = int(self.manifest.asset_probe_seed)
        load, _, _ = run_rollout(
            self.manifest, self.current, paths, base_seed, out / "load_stability", capture_images=False,
            dry_run=False, stream_output=self.stream_output, force=self.force, steps_override=100,
            extra_args=["--door_auto_open_force", "0"],
        )
        hinge, _ = run_asset_probe(self.manifest, self.current, paths, out / "hinge_motion", self.options)
        handle, _, _ = run_rollout(
            self.manifest, self.current, paths, base_seed + 1, out / "handle_motion", capture_images=False,
            dry_run=False, stream_output=self.stream_output, force=self.force, steps_override=min(self.manifest.steps, 750),
            extra_args=["--door_auto_open_force", "0"],
        )
        grasp, _, _ = run_rollout(
            self.manifest, self.current, paths, base_seed + 2, out / "grasp_reachability", capture_images=False,
            dry_run=False, stream_output=self.stream_output, force=self.force, steps_override=min(self.manifest.steps, 550),
            extra_args=["--door_auto_open_force", "0"],
        )
        report = summarize_physics_probes(
            self.current_hash,
            load_summary=load,
            hinge_summary=hinge,
            handle_summary=handle,
            grasp_summary=grasp,
            has_handle_dof=bool(str(self.current.runtime_spec.get("handle_dof_name", ""))),
        )
        report["artifacts"] = {"root": str(out), "load": str(out / "load_stability"), "hinge": str(out / "hinge_motion"), "handle": str(out / "handle_motion"), "grasp": str(out / "grasp_reachability")}
        self.probe_reports[self.current_hash] = report
        write_json(out / "probe_summary.json", report)
        return report

    def _run_full_rollout(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        static = self.static_reports.get(self.current_hash)
        probe = self.probe_reports.get(self.current_hash)
        if not static or not static.get("passed", False):
            raise RuntimeError("Current candidate has not passed static validation")
        if not probe or not probe.get("passed", False):
            raise RuntimeError("Current candidate has not passed all physics probes")
        out = self.root / "rollouts" / self.current_hash
        paths = self.current.write(out / "candidate")
        summary, elapsed = run_seed_set(
            self.manifest, self.current, paths, self.manifest.development_seeds, out / "development",
            capture_images=self.visual_feedback, args=self.options,
        )
        selection = self.graph.evaluate(
            self.current_hash,
            summary,
            candidate=self.current,
            artifacts={"rollout_root": str(out), "elapsed_s": elapsed},
        )
        node = self.graph.nodes[self.current_hash]
        self.last_rollout, self.last_rollout_dir = summary, out
        self.last_rollout_hash = self.current_hash
        self.store.append_repair(
            {
                "door_signature": DoorSignature.from_candidate(self.current).to_dict(),
                "candidate_before": node.parent_hash,
                "candidate_after": self.current_hash,
                "patch": node.patch,
                "diagnostics": node.diagnostics,
                "failure_evidence": compact_rollout_summary(summary),
                "status": selection,
                "before_score": None if not node.parent_hash else self.graph.nodes[node.parent_hash].score,
                "after_score": node.score,
                "seed_set": self.manifest.development_seeds,
                "artifacts": node.artifacts,
            }
        )
        if selection == "rejected_regression":
            self.current_hash = self.graph.best_hash
        self._persist_graph()
        result = {
            "candidate_selection": selection,
            "candidate_score": node.score,
            "current_hash_after_selection": self.current_hash,
            "best_hash": self.graph.best_hash,
            "summary": compact_rollout_summary(summary),
        }
        if int(summary.get("success_count", 0)) == len(self.manifest.development_seeds):
            result["stop_condition"] = "all_development_rollouts_succeeded"
            result["final"] = self._finish(
                {"evaluate_hidden": True},
                reason="all_development_rollouts_succeeded",
            )
        return result

    def _inspect_rollout(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        if self.last_rollout is None:
            raise RuntimeError("No development rollout has been run")
        counts = dict(self.last_rollout.get("failure_counts", {}) or {})
        failures = sorted(((name, int(count)) for name, count in counts.items() if name != "success"), key=lambda item: (-item[1], item[0]))
        return {"dominant_failure": failures[0][0] if failures else "", "summary": compact_rollout_summary(self.last_rollout)}

    def _inspect_images(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.visual_feedback:
            raise RuntimeError("Visual feedback is disabled for this session")
        if self.last_rollout is None:
            raise RuntimeError("No development rollout has been run")
        image_paths = select_phase_montages(self.last_rollout, limit=5)
        candidate = self.candidates.get(self.last_rollout_hash or self.current_hash, self.current)
        return {
            "image_paths": [str(path) for path in image_paths],
            "structured_visual_diagnostics": structured_visual_diagnostics(
                self.last_rollout,
                image_paths=image_paths,
                candidate=candidate,
            ),
        }

    def _apply_patch(self, arguments: dict[str, Any], diagnostics: str) -> dict[str, Any]:
        if self.repair_count >= self.max_repairs:
            raise RuntimeError("Repair budget exhausted")
        base = self.best
        base_experience_id = str(arguments.get("base_experience_id", "")).strip()
        patch_obj = dict(arguments.get("patch", arguments))
        patch_obj.pop("base_experience_id", None)
        patch_obj.setdefault("diagnostics", diagnostics)
        template_info: dict[str, Any] = {}
        if base_experience_id:
            if base_experience_id not in self.retrieved_ids:
                raise ValueError("base_experience_id must first be returned by find_experiences")
            experience = self.catalog.get(base_experience_id)
            base = base.clone()
            base.skill_program = experience_skill = self._load_experience_skill(base_experience_id)
            base.metadata["initial_skill_experience_id"] = base_experience_id
            template_info = {
                "base_experience_id": base_experience_id,
                "skill_sha256": experience.skill_sha256,
                "template_skill": experience_skill.to_dict(),
            }
        repair = CombinedRepairPatch.from_obj(patch_obj)
        candidate, info = repair.apply(base)
        parent = self.best
        if candidate.fingerprint == parent.fingerprint:
            self.store.append_repair(
                {
                    "door_signature": DoorSignature.from_candidate(parent).to_dict(),
                    "candidate_before": parent.fingerprint,
                    "candidate_after": parent.fingerprint,
                    "patch": patch_obj,
                    "diagnostics": diagnostics,
                    "status": "invalid",
                    "reason": "patch_did_not_change_candidate",
                    "seed_set": self.manifest.development_seeds,
                }
            )
            raise ValueError(f"Patch did not produce a valid candidate change: {info}")
        if candidate.fingerprint in self.graph.nodes:
            self.store.append_repair(
                {
                    "door_signature": DoorSignature.from_candidate(parent).to_dict(),
                    "candidate_before": parent.fingerprint,
                    "candidate_after": candidate.fingerprint,
                    "patch": patch_obj,
                    "diagnostics": diagnostics,
                    "status": "invalid",
                    "reason": "duplicate_candidate",
                    "seed_set": self.manifest.development_seeds,
                }
            )
            raise ValueError("Patch repeats a previously evaluated/pending candidate")
        self.repair_count += 1
        self.candidates[candidate.fingerprint] = candidate
        recorded_patch = {**patch_obj, **template_info}
        self.graph.add(CandidateNode(candidate.fingerprint, parent.fingerprint, patch=recorded_patch, diagnostics=diagnostics))
        self.current_hash = candidate.fingerprint
        self._save_candidate(candidate)
        self._persist_graph()
        return {"candidate_hash": candidate.fingerprint, "parent_hash": parent.fingerprint, "template": template_info, "patch_info": info, "repair_count": self.repair_count}

    def _compare_candidates(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        return self.graph.to_dict()

    def _select_best(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.graph.best_hash:
            raise RuntimeError("No candidate has a development score yet")
        self.current_hash = self.graph.best_hash
        return {"selected_hash": self.current_hash, "score": self.graph.best_score}

    def _finish(self, arguments: dict[str, Any], reason: str = "agent_finish") -> dict[str, Any]:
        evaluate_hidden = bool(arguments.get("evaluate_hidden", True))
        if evaluate_hidden and not self.graph.best_hash:
            raise RuntimeError("Hidden evaluation requires at least one development-scored candidate")
        if self.graph.best_hash:
            self.current_hash = self.graph.best_hash
        else:
            self.current_hash = self.graph.initial_hash
        best = self.current
        final_root = self.root / "best_candidate"
        paths = best.write(final_root)
        result: dict[str, Any] = {
            "reason": reason,
            "best_candidate_hash": best.fingerprint,
            "best_score": self.graph.best_score,
            "development_best_available": bool(self.graph.best_hash),
        }
        if evaluate_hidden:
            probe_summary, _ = run_asset_probe(self.manifest, best, paths, self.root / "hidden" / "asset_probe", self.options)
            heldout, _ = run_seed_set(
                self.manifest, best, paths, self.manifest.heldout_seeds, self.root / "hidden" / "rollouts",
                capture_images=False, args=self.options,
            )
            asset = evaluate_asset(best, self.door, probe_summary, motion_fallback_summary=heldout)
            stages = stage_metrics(heldout)
            eligible = bool(asset["structure_success"] and stages["success_count"] >= self.manifest.generation_success_min)
            result.update({"asset": asset, "heldout": stages, "experience_eligible": eligible})
            if eligible and not self.benchmark_mode:
                experience_id, path = self.learned_store.promote(
                    {
                        "door_name": self.door.door_name,
                        "door_signature": DoorSignature.from_candidate(best).to_dict(),
                        "candidate_hash": best.fingerprint,
                        "skill_program": best.skill_program.to_dict(),
                        "runtime_spec": best.runtime_spec,
                        "heldout": stages,
                        "asset": asset,
                        "prior_snapshot_fingerprint": self.prior_snapshot["fingerprint"],
                        "candidate_graph": self.graph.to_dict(),
                    }
                )
                result["promoted_experience"] = {"experience_id": experience_id, "path": str(path)}
            elif eligible:
                result["memory_writeback"] = "deferred_benchmark_pending"
        self.finished = True
        self._write_final_markdown(result)
        return result

    def final_report(self) -> dict[str, Any]:
        path = self.root / "final_result.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        result = {"schema_version": "door_twin_agent_result_v1", "finished": self.finished, "tool_turns": self.tool_turn, "repairs": self.repair_count, "current_hash": self.current_hash, "best_hash": self.graph.best_hash, "best_score": self.graph.best_score, "last_observation": self.last_observation}
        _atomic_json(path, result)
        return result

    def _save_candidate(self, candidate: Candidate) -> None:
        candidate.write(self.root / "candidates" / candidate.fingerprint)

    def _persist_graph(self) -> None:
        _atomic_json(self.root / "candidate_graph.json", self.graph.to_dict())

    def _persist(self) -> None:
        self._persist_graph()
        _atomic_json(
            self.root / "session.json",
            {
                "schema_version": AGENT_SCHEMA_VERSION,
                "door_name": self.door.door_name,
                "current_hash": self.current_hash,
                "best_hash": self.graph.best_hash,
                "tool_turn": self.tool_turn,
                "repair_count": self.repair_count,
                "finished": self.finished,
                "visual_feedback": self.visual_feedback,
                "benchmark_mode": self.benchmark_mode,
                "prior_snapshot": self.prior_snapshot,
                "retrieved_experience_ids": sorted(self.retrieved_ids),
                "last_observation": self.last_observation,
            },
        )

    def _load_experience_skill(self, experience_id: str):
        from .skill import SkillProgram

        record = self.catalog.get(experience_id)
        return SkillProgram.from_obj(json.loads(record.skill_path.read_text(encoding="utf-8")))

    def _resume(self, initial_candidate: Candidate) -> None:
        state = json.loads((self.root / "session.json").read_text(encoding="utf-8"))
        if state.get("schema_version") != AGENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported DoorTwin session schema in {self.root / 'session.json'}")
        if str(state.get("door_name")) != self.door.door_name:
            raise ValueError("Cannot resume a DoorTwin session with a different door")
        if bool(state.get("benchmark_mode", False)) != self.benchmark_mode:
            raise ValueError("Cannot change benchmark_mode while resuming a session")
        stored_prior = dict(state.get("prior_snapshot", {}) or {})
        if stored_prior.get("fingerprint") != self.prior_snapshot.get("fingerprint"):
            raise ValueError("Experience prior changed since the session started; refusing an unsafe resume")
        graph_path = self.root / "candidate_graph.json"
        self.graph = CandidateGraph.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
        self.candidates = {}
        for path in sorted((self.root / "candidates").glob("*/candidate.json")):
            candidate = Candidate.read(path, initial_candidate.source_config)
            self.candidates[candidate.fingerprint] = candidate
        if self.graph.initial_hash not in self.candidates:
            if initial_candidate.fingerprint != self.graph.initial_hash:
                raise ValueError("Initial candidate differs from the persisted session")
            self.candidates[initial_candidate.fingerprint] = initial_candidate
            self._save_candidate(initial_candidate)
        missing = sorted(set(self.graph.nodes) - set(self.candidates))
        if missing:
            raise FileNotFoundError(f"Persisted candidate artifacts are missing: {missing}")
        self.current_hash = str(state.get("current_hash") or self.graph.best_hash or self.graph.initial_hash)
        self.tool_turn = int(state.get("tool_turn", 0))
        self.repair_count = int(state.get("repair_count", 0))
        self.finished = bool(state.get("finished", False))
        self.retrieved_ids = {str(value) for value in state.get("retrieved_experience_ids", [])}
        self.last_observation = dict(state.get("last_observation", {}) or {})
        self.static_reports = {}
        for path in (self.root / "validation").glob("*/static_validation.json"):
            self.static_reports[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        self.probe_reports = {}
        for path in (self.root / "probes").glob("*/probe_summary.json"):
            self.probe_reports[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        rollout_path = self.root / "rollouts" / self.current_hash / "development" / "aggregate_summary.json"
        if not rollout_path.is_file() and self.graph.best_hash:
            rollout_path = self.root / "rollouts" / self.graph.best_hash / "development" / "aggregate_summary.json"
        if rollout_path.is_file():
            self.last_rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            self.last_rollout_dir = rollout_path.parents[1]
        _atomic_json(self.root / "prior_snapshot.json", self.prior_snapshot)

    def _append_trace(self, action: AgentAction, observation: dict[str, Any]) -> None:
        path = self.root / "tool_trace.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"turn": self.tool_turn, "action": action.to_dict(), "observation": observation}, ensure_ascii=False, allow_nan=False) + "\n")

    def _write_final_markdown(self, result: dict[str, Any]) -> None:
        lines = [
            "# DoorTwin Agent Final Report",
            "",
            f"- Door: `{self.door.door_name}`",
            f"- Best candidate: `{result.get('best_candidate_hash', '')}`",
            f"- Tool turns: {self.tool_turn}",
            f"- Repair attempts: {self.repair_count}",
            f"- Prior snapshot: `{self.prior_snapshot['fingerprint']}`",
            "",
            "## Final result",
            "",
            "```json",
            json.dumps(result, indent=2, ensure_ascii=False),
            "```",
        ]
        (self.root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _atomic_json(self.root / "final_result.json", {"schema_version": "door_twin_agent_result_v1", **result})


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
