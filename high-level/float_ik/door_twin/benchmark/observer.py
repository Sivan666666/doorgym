"""Fixed OpenAI Responses API observer for bounded DoorTwin JSON patches."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..skill import PATCH_BOUNDS
from .patches import ASSET_ALLOWED_FIELDS, ASSET_NUMERIC_BOUNDS, CombinedRepairPatch
from .schema import Candidate, sha256_json


OBSERVER_SCHEMA = {
    "failure_stage": "one concise failure category",
    "diagnostics": "short evidence-based diagnosis",
    "asset_patch": {"one_allowlisted_asset_field": "new value"},
    "skill_patch": {"one_allowlisted_skill_path": "new value"},
}


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("VLM output must decode to a JSON object")
    return data


def _response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    texts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    if not texts:
        raise ValueError("OpenAI response did not contain output text")
    return "\n".join(texts)


def select_phase_montages(summary: dict[str, Any], *, limit: int = 5) -> list[Path]:
    desired = ("grasp", "close_gripper", "rotate_handle", "push_door", "hold_home")
    reports = list(summary.get("reports", []) or [])
    reports.sort(key=lambda report: (bool(report.get("success", False)), int(report.get("env_id", 0))))
    by_phase: dict[str, Path] = {}
    fallback: list[Path] = []
    for report in reports:
        for record in (report.get("artifacts", {}) or {}).get("keyframes", []) or []:
            path = Path(str(record.get("path", ""))).expanduser()
            if not path.is_file() or not bool(record.get("valid", True)):
                continue
            if str(record.get("image", "")) == "multiview_montage":
                phase = str(record.get("phase", ""))
                by_phase.setdefault(phase, path)
                fallback.append(path)
    selected = [by_phase[phase] for phase in desired if phase in by_phase]
    for path in fallback:
        if path not in selected:
            selected.append(path)
    return selected[:limit]


def compact_rollout_summary(summary: dict[str, Any]) -> dict[str, Any]:
    reports = []
    for report in summary.get("reports", []) or []:
        metrics = report.get("metrics", {}) or {}
        trace = list(metrics.get("trace", []) or [])
        phase_evidence = dict(metrics.get("phase_metrics", {}) or {})
        if not phase_evidence:
            for phase in ("grasp", "close_gripper", "rotate_handle", "push_door", "traverse_door"):
                records = [record for record in trace if str(record.get("phase", "")) == phase]
                if not records:
                    continue
                phase_evidence[phase] = {
                    "start_step": int(records[0].get("step", -1)),
                    "end_step": int(records[-1].get("step", -1)),
                    "max_handle_rotation_deg": max(float(record.get("handle_rotation_deg", 0.0)) for record in records),
                    "max_door_open_deg": max(float(record.get("door_open_deg", 0.0)) for record in records),
                    "max_ee_tracking_error": max(float(record.get("ee_tracking_error", 0.0)) for record in records),
                }
        first_unlock = metrics.get("first_handle_unlock")
        if first_unlock is None:
            first_unlock_record = next(
                (record for record in trace if float(record.get("handle_rotation_deg", 0.0)) >= 40.0),
                None,
            )
            first_unlock = (
                None
                if first_unlock_record is None
                else {
                    "step": int(first_unlock_record.get("step", -1)),
                    "phase": str(first_unlock_record.get("phase", "")),
                }
            )
        reports.append(
            {
                "seed": report.get("benchmark_seed"),
                "success": report.get("success"),
                "failure_stage": report.get("failure_stage"),
                "door_open_deg": report.get("door_open_deg"),
                "handle_rotation_deg": report.get("handle_rotation_deg"),
                "ee_handle_dist": report.get("ee_handle_dist"),
                "ee_tracking_error": report.get("ee_tracking_error"),
                "base_collision": report.get("base_collision"),
                "body_passed": report.get("body_passed"),
                "handle_unlocked": report.get("handle_unlocked"),
                "final_phase": report.get("final_phase"),
                "secondary_failures": metrics.get("secondary_failures", []),
                "door_motion_sign": metrics.get("door_motion_sign"),
                "raw_hinge_deg_range": metrics.get("raw_hinge_deg_range"),
                "door_motion_relative_to_robot": metrics.get("door_motion_relative_to_robot"),
                "handle_delta_from_initial_base_at_max_open_m": metrics.get(
                    "handle_delta_from_initial_base_at_max_open_m"
                ),
                "first_door_open": metrics.get("first_door_open"),
                "first_handle_40deg": first_unlock,
                "first_base_collision": metrics.get("first_base_collision"),
                "phase_evidence": phase_evidence,
                "trace": trace,
            }
        )
    return {
        "success_count": summary.get("success_count", 0),
        "num_envs": summary.get("num_envs", len(reports)),
        "success_rate": summary.get("success_rate", 0.0),
        "failure_counts": summary.get("failure_counts", {}),
        "reports": reports,
        "process_failures": summary.get("process_failures", []),
    }


def structured_visual_diagnostics(
    summary: dict[str, Any],
    *,
    image_paths: list[Path] | None = None,
    candidate: Candidate | None = None,
) -> dict[str, Any]:
    """Convert Full-mode visual rollout evidence into bounded repair signals.

    The current implementation uses the fixed phase montages as the visual
    evidence carrier and pairs them with simulator trace geometry from the same
    keyframes. This gives the Agent explicit visual-style facts without asking it
    to infer geometry from path strings alone.
    """

    compact = compact_rollout_summary(summary)
    reports = list(compact.get("reports", []) or [])
    grasp_samples: list[dict[str, Any]] = []
    base_samples: list[dict[str, Any]] = []
    for report in reports:
        trace = list(report.get("trace", []) or [])
        grasp_records = [
            record
            for record in trace
            if str(record.get("phase", "")) in ("grasp", "close_gripper", "rotate_handle")
            and isinstance(record.get("ee_pos"), list)
            and isinstance(record.get("handle_goal"), list)
        ]
        if grasp_records:
            record = min(grasp_records, key=lambda item: float(item.get("ee_handle_dist", 1.0e9)))
            offset = _vector_delta(record.get("ee_pos"), record.get("handle_goal"))
            if offset is not None:
                grasp_samples.append(
                    {
                        "seed": report.get("seed"),
                        "step": record.get("step"),
                        "phase": record.get("phase"),
                        "ee_minus_handle_m": offset,
                        "ee_handle_dist_m": float(record.get("ee_handle_dist", 0.0)),
                    }
                )
        collision = report.get("first_base_collision")
        if collision:
            base_samples.append(
                {
                    "seed": report.get("seed"),
                    "collision": collision,
                    "first_handle_40deg": report.get("first_handle_40deg"),
                    "first_door_open": report.get("first_door_open"),
                    "door_open_deg": report.get("door_open_deg"),
                    "body_passed": report.get("body_passed"),
                    "success": report.get("success"),
                }
            )

    mean_offset = _mean_vector([sample["ee_minus_handle_m"] for sample in grasp_samples])
    vertical = _vertical_relation(mean_offset[2] if mean_offset else None)
    lateral = _lateral_relation(mean_offset if mean_offset else None)
    collision = _base_collision_diagnosis(base_samples)
    patch_hints = _visual_patch_hints(
        vertical=vertical,
        lateral=lateral,
        collision=collision,
        candidate=candidate,
    )
    return {
        "schema_version": "door_twin_structured_visual_diagnostics_v1",
        "source": "phase_montage_paths_plus_rollout_trace",
        "image_paths": [str(path) for path in (image_paths or [])],
        "grasp_alignment": {
            "samples": grasp_samples[:4],
            "mean_ee_minus_handle_m": mean_offset,
            "vertical_relation": vertical,
            "lateral_relation": lateral,
        },
        "base_collision": collision,
        "patch_hints": patch_hints,
        "notes": [
            "Use patch_hints as suggestions, not automatic edits.",
            "Keep MoveEEToHandle.pregrasp_offset.z equal to grasp_offset.z when applying any grasp-height fix.",
        ],
    }


def _vector_delta(lhs: Any, rhs: Any) -> list[float] | None:
    if not isinstance(lhs, list) or not isinstance(rhs, list) or len(lhs) < 3 or len(rhs) < 3:
        return None
    try:
        return [float(lhs[i]) - float(rhs[i]) for i in range(3)]
    except (TypeError, ValueError):
        return None


def _mean_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(3)]


def _vertical_relation(dz: float | None) -> dict[str, Any]:
    if dz is None:
        return {"label": "unknown", "dz_m": None}
    if dz > 0.025:
        label = "gripper_above_handle"
    elif dz < -0.025:
        label = "gripper_below_handle"
    else:
        label = "gripper_centered_vertically"
    return {"label": label, "dz_m": float(dz)}


def _lateral_relation(offset: list[float] | None) -> dict[str, Any]:
    if offset is None:
        return {"label": "unknown", "dominant_axis": None, "offset_m": None}
    xy = {"x": float(offset[0]), "y": float(offset[1])}
    axis = max(xy, key=lambda key: abs(xy[key]))
    value = xy[axis]
    if abs(value) <= 0.025:
        return {"label": "gripper_centered_laterally", "dominant_axis": axis, "offset_m": value}
    sign = "positive" if value > 0.0 else "negative"
    return {"label": f"gripper_{sign}_{axis}_side_of_handle", "dominant_axis": axis, "offset_m": value}


def _step(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    try:
        return int(value.get("step"))
    except (TypeError, ValueError):
        return None


def _base_collision_diagnosis(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"label": "no_base_collision_observed", "count": 0, "samples": []}
    preunlock = 0
    postunlock = 0
    opened_and_passed = 0
    blocking = 0
    for sample in samples:
        collision_step = _step(sample.get("collision"))
        unlock_step = _step(sample.get("first_handle_40deg"))
        open_step = _step(sample.get("first_door_open"))
        if collision_step is not None and unlock_step is not None and collision_step < unlock_step:
            preunlock += 1
        elif collision_step is not None and unlock_step is not None:
            postunlock += 1
        if bool(sample.get("body_passed")) and float(sample.get("door_open_deg") or 0.0) >= 80.0:
            opened_and_passed += 1
        else:
            blocking += 1
        if collision_step is not None and open_step is not None and collision_step < open_step:
            sample["collision_before_open"] = True
    if blocking:
        label = "blocking_base_collision"
    elif preunlock:
        label = "preunlock_collision_but_task_completes"
    elif postunlock:
        label = "minor_postunlock_collision"
    else:
        label = "base_collision_timing_unknown"
    return {
        "label": label,
        "count": len(samples),
        "preunlock_count": preunlock,
        "postunlock_count": postunlock,
        "opened_and_passed_count": opened_and_passed,
        "blocking_count": blocking,
        "samples": samples[:4],
    }


def _visual_patch_hints(
    *,
    vertical: dict[str, Any],
    lateral: dict[str, Any],
    collision: dict[str, Any],
    candidate: Candidate | None,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    move = candidate.skill_program.primitive("MoveEEToHandle") if candidate is not None else None
    current_grasp = list(move.params.get("grasp_offset", [0.0, 0.0, -0.01])) if move is not None else None
    if current_grasp and len(current_grasp) == 3:
        if vertical.get("label") == "gripper_above_handle":
            new_z = max(-0.20, float(current_grasp[2]) - 0.01)
            hints.append(
                {
                    "reason": "gripper is visually/trace above the handle",
                    "skill_patch": {
                        "MoveEEToHandle.pregrasp_offset": [0.15, 0.0, new_z],
                        "MoveEEToHandle.grasp_offset": [0.0, 0.0, new_z],
                    },
                }
            )
        elif vertical.get("label") == "gripper_below_handle":
            new_z = min(0.20, float(current_grasp[2]) + 0.01)
            hints.append(
                {
                    "reason": "gripper is visually/trace below the handle; prefer a centered grasp if success is preserved",
                    "skill_patch": {
                        "MoveEEToHandle.pregrasp_offset": [0.15, 0.0, new_z],
                        "MoveEEToHandle.grasp_offset": [0.0, 0.0, new_z],
                    },
                }
            )
    if lateral.get("label") not in ("unknown", "gripper_centered_laterally"):
        axis = lateral.get("dominant_axis")
        offset = float(lateral.get("offset_m") or 0.0)
        bias = [0.0, 0.0, 0.0]
        if axis == "x":
            bias[0] = max(-0.08, min(0.08, -0.5 * offset))
        elif axis == "y":
            bias[1] = max(-0.08, min(0.08, -0.5 * offset))
        if bias != [0.0, 0.0, 0.0]:
            hints.append(
                {
                    "reason": "gripper is laterally off the handle center",
                    "skill_patch": {"MoveEEToHandle.handle_goal_bias_world": bias},
                }
            )
    if collision.get("label") == "preunlock_collision_but_task_completes":
        hints.append(
            {
                "reason": "base contacts the door/frame before unlock/open while the task later completes",
                "skill_patch": {
                    "MoveTo:push.vx": 0.06,
                    "MoveTo:push.distance": 0.35,
                },
            }
        )
        hints.append(
            {
                "reason": "unlock happens too late relative to base motion",
                "skill_patch": {
                    "RotateHandle.duration_steps": 120,
                    "RotateHandle.local_delta": [0.0, -0.04, -0.04],
                },
            }
        )
    elif collision.get("label") == "blocking_base_collision":
        hints.append(
            {
                "reason": "base collision appears to block task completion",
                "skill_patch": {
                    "ApproachDoor.base_offset": 0.24,
                    "MoveTo:push.vx": 0.04,
                    "MoveTo:push.distance": 0.24,
                },
            }
        )
    return hints


@dataclass
class ObserverResult:
    patch: CombinedRepairPatch
    raw_response: dict[str, Any]
    output_text: str
    usage: dict[str, Any]
    request_fingerprint: str
    image_paths: list[str]
    persisted_request: dict[str, Any]
    elapsed_s: float


class OpenAIResponsesObserver:
    def __init__(
        self,
        *,
        model: str = "gpt-5",
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 180.0,
        max_retries: int = 3,
        mock_response: str | Path | None = None,
        manual_exchange_dir: str | Path | None = None,
        manual_wait_timeout_s: float = 86400.0,
    ):
        self.model = str(model)
        self.temperature = float(temperature)
        self.base_url = str(base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")).rstrip("/")
        self.api_key_env = str(api_key_env)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.mock_response = None if mock_response is None else Path(mock_response).expanduser().resolve()
        self.manual_exchange_dir = (
            None if manual_exchange_dir is None else Path(manual_exchange_dir).expanduser().resolve()
        )
        self.manual_wait_timeout_s = float(manual_wait_timeout_s)
        self._manual_call_index = 0

    def _system_prompt(self, *, diagnose_only: bool) -> str:
        mode = (
            "Diagnose only. Return empty asset_patch and skill_patch objects; do not propose edits."
            if diagnose_only
            else "Propose at most one bounded repair for this round. Never rewrite code, mesh, or URDF."
        )
        return (
            "You are the reproducible DoorTwin observer for a hinged push-door benchmark. "
            "Use only the supplied asset metadata, frozen retrieved priors, and rollout evidence. "
            f"{mode} Return JSON only with this shape: {json.dumps(OBSERVER_SCHEMA)}. "
            f"Allowed asset fields: {sorted(ASSET_ALLOWED_FIELDS)}. "
            f"Allowed skill paths: {sorted(PATCH_BOUNDS)}. "
            "MoveEEToHandle.pregrasp_offset.z and grasp_offset.z must be equal. "
            "Subject to unchanged rollout success, prefer the shared z offset with the smallest absolute value "
            "so the gripper remains centered on the handle for real deployment. "
            "Do not infer hidden annotations and do not add unlisted fields."
        )

    def _manual_call(self, payload: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
        """Exchange one observer request through files for a human/Codex-in-the-loop run."""

        assert self.manual_exchange_dir is not None
        self.manual_exchange_dir.mkdir(parents=True, exist_ok=True)
        request_payload = copy_without_image_data(payload)
        requested_images = [str(path) for path in image_paths]
        while True:
            index = self._manual_call_index
            self._manual_call_index += 1
            request_path = self.manual_exchange_dir / f"request_{index:04d}.json"
            response_path = self.manual_exchange_dir / f"response_{index:04d}.json"
            if request_path.is_file():
                try:
                    existing = json.loads(request_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if existing.get("request") == request_payload and existing.get("image_paths", []) == requested_images:
                    # Reuse an identical pending/completed exchange after a benchmark
                    # process restart instead of silently allocating a new request id.
                    break
            if not request_path.exists() and not response_path.exists():
                break
        request_record = {
            "schema_version": "door_twin_manual_observer_request_v1",
            "request_index": index,
            "request": request_payload,
            "image_paths": requested_images,
            "response_path": str(response_path),
            "response_schema": OBSERVER_SCHEMA,
        }
        if not request_path.is_file():
            temp_path = request_path.with_suffix(".json.tmp")
            temp_path.write_text(json.dumps(request_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temp_path.replace(request_path)
        print(
            f"MANUAL_VLM_REQUEST index={index} request={request_path} response={response_path}",
            flush=True,
        )
        deadline = time.monotonic() + self.manual_wait_timeout_s
        while not response_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for manual observer response: {response_path}")
            time.sleep(0.5)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise ValueError(f"Manual observer response must be a JSON object: {response_path}")
        # A responder may write either a native Responses API object or the bounded patch directly.
        if "output_text" not in response and "output" not in response:
            response = {
                "output_text": json.dumps(response, ensure_ascii=False),
                "usage": {"manual_observer_calls": 1},
                "manual_exchange": {"request": str(request_path), "response": str(response_path)},
            }
        return response

    def _call(self, payload: dict[str, Any], *, image_paths: list[Path] | None = None) -> dict[str, Any]:
        if self.mock_response is not None:
            return json.loads(self.mock_response.read_text(encoding="utf-8"))
        if self.manual_exchange_dir is not None:
            return self._manual_call(payload, image_paths or [])
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for automatic VLM generation/repair")
        endpoint = f"{self.base_url}/responses" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/responses"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"VLM request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _image_content(path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}

    def observe(
        self,
        candidate: Candidate,
        *,
        summary: dict[str, Any] | None,
        include_images: bool,
        initial_generation: bool = False,
        diagnose_only: bool = False,
        candidate_nonce: int = 0,
        repair_history: list[dict[str, Any]] | None = None,
        prior_context: dict[str, Any] | None = None,
    ) -> ObserverResult:
        context: dict[str, Any] = {
            "task": "initial_generation" if initial_generation else "rollout_repair",
            "candidate_nonce": int(candidate_nonce),
            "asset": candidate.public_context(),
            "current_candidate_hash": candidate.fingerprint,
        }
        if initial_generation:
            context["initial_generation_requirements"] = {
                "replicate": f"independent deterministic candidate replicate {candidate_nonce}; do not read or copy other replicates",
                "starting_point": (
                    "The current candidate is already the deterministic Rule-based result derived from the public "
                    "URDF and handle bounding box. Treat it as the trusted baseline."
                ),
                "patch_mode": (
                    "Return only a bounded residual patch justified by a retrieved experience. Preserve every "
                    "Rule-based field whose transfer is uncertain; do not regenerate the complete candidate."
                ),
                "structure_policy": (
                    "Only change body/DOF names or handle_goal_pos when public URDF/bbox evidence shows the "
                    "Rule-based value is wrong."
                ),
                "skill_policy": (
                    "Only change skill parameters that have a clear analogous successful prior; numeric offsets "
                    "must not be copied blindly across doors."
                ),
                "do_not_use": "hidden ground truth, target-door history, arbitrary code or URDF edits",
            }
        if prior_context:
            context["retrieved_door_twin_priors"] = prior_context
            context["prior_instruction"] = (
                "Use these frozen successful skills and accepted/rejected repair records as explicit engineering prior. "
                "Adapt them to the current public geometry; do not assume their numeric offsets transfer unchanged."
            )
        if summary is not None:
            context["rollout"] = compact_rollout_summary(summary)
        if repair_history:
            context["repair_history"] = repair_history
            context["repair_history_instruction"] = (
                "Candidates marked rejected_regression were rolled back. Do not repeat their patch; "
                "propose a different bounded change from the current best candidate."
            )
        text = json.dumps(context, indent=2, ensure_ascii=False)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        image_paths = select_phase_montages(summary or {}) if include_images else []
        content.extend(self._image_content(path) for path in image_paths)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "instructions": self._system_prompt(diagnose_only=diagnose_only),
            "input": [{"role": "user", "content": content}],
        }
        persisted_request = copy_without_image_data(payload)
        started = time.time()
        response = self._call(payload, image_paths=image_paths)
        elapsed_s = time.time() - started
        output_text = _response_text(response)
        patch = CombinedRepairPatch.from_obj(_json_from_text(output_text))
        return ObserverResult(
            patch=patch,
            raw_response=response,
            output_text=output_text,
            usage=dict(response.get("usage", {}) or {}),
            request_fingerprint=sha256_json(persisted_request),
            image_paths=[str(path) for path in image_paths],
            persisted_request=persisted_request,
            elapsed_s=elapsed_s,
        )


def copy_without_image_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a reproducible request description without duplicating large base64 images."""

    clean = json.loads(json.dumps(payload))
    for message in clean.get("input", []) or []:
        for item in message.get("content", []) or []:
            if item.get("type") == "input_image":
                data = str(item.get("image_url", ""))
                item["image_url"] = f"<embedded image: {len(data)} chars>"
    return clean


def write_observer_result(path: str | Path, result: ObserverResult) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_output": result.output_text,
        "usage": result.usage,
        "request_fingerprint": result.request_fingerprint,
        "image_paths": result.image_paths,
        "request": result.persisted_request,
        "elapsed_s": result.elapsed_s,
        "raw_response": result.raw_response,
    }
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
