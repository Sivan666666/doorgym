from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path


FLOAT_IK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = FLOAT_IK_ROOT.parents[1]
if str(FLOAT_IK_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOAT_IK_ROOT))

from door_twin.agent_runtime import AgentAction, DoorTwinAgentSession, ManualCodexBackend, ScriptedBackend  # noqa: E402
from door_twin.benchmark.evaluation import evaluate_asset  # noqa: E402
from door_twin.benchmark.schema import BenchmarkManifest, Candidate, DoorCase, rule_based_candidate  # noqa: E402
from door_twin.candidate_selection import CandidateGraph, CandidateNode  # noqa: E402
from door_twin.experience import DoorSignature, ExperienceCatalog  # noqa: E402
from door_twin.skill import MIN_DOOR_TWIN_FORWARD_DISTANCE_M, ProgramPatch  # noqa: E402
from door_twin.validation import summarize_physics_probes, validate_candidate_static  # noqa: E402


CFG = PROJECT_ROOT / "high-level/data/cfg/b1z1_opendoor.yaml"
RUNNER = PROJECT_ROOT / "high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py"
CATALOG = PROJECT_ROOT / "high-level/float_ik/door_twin/experience/catalog.yaml"


def make_fixture() -> tuple[BenchmarkManifest, DoorCase, Candidate]:
    door = DoorCase(id="99650069960003", door_name="99650069960003")
    manifest = BenchmarkManifest(
        path=CFG,
        name="tool_agent_test",
        base_door_cfg=CFG,
        runner_script=RUNNER,
        output_root=Path("/tmp"),
        doors=[door],
        development_seeds=[41001, 41002, 41003, 41004],
        heldout_seeds=list(range(42001, 42017)),
    )
    return manifest, door, Candidate.from_base_config(manifest, door, 0)


def summary(successes: int, *, door_deg: float = 90.0) -> dict:
    reports = []
    for index in range(4):
        reports.append(
            {
                "success": index < successes,
                "body_passed": index < successes,
                "door_open_deg": door_deg if index < successes else 0.0,
                "handle_unlocked": index < successes,
                "base_collision": False,
                "ee_handle_dist": 0.02,
                "ee_tracking_error": None,
                "metrics": {"min_grasp_ee_handle_dist": 0.02},
            }
        )
    return {"reports": reports, "process_failures": []}


def test_experience_retrieval_is_deterministic_and_excludes_target() -> None:
    catalog = ExperienceCatalog.load(CATALOG)
    query = DoorSignature(handle_mobility="fixed", handle_orientation="vertical", push_or_pull="push")
    first = catalog.search(query, query="fixed vertical handle", limit=3)
    second = catalog.search(query, query="fixed vertical handle", limit=3)
    assert [match.record.experience_id for match in first] == [match.record.experience_id for match in second]
    assert first[0].record.experience_id == "glass_door_reference"
    snapshot = catalog.freeze(exclude_door_names=["glass_door"])
    excluded = catalog.search(query, limit=5, exclude_door_names=["glass_door"], allowed_snapshot=snapshot)
    assert "glass_door_reference" not in {match.record.experience_id for match in excluded}
    hash_snapshot = catalog.freeze(exclude_hashes=[catalog.get("wc4_reference").skill_sha256])
    assert "wc4_reference" not in {item["experience_id"] for item in hash_snapshot["records"]}


def test_rule_based_candidate_uses_public_urdf_without_retrieval() -> None:
    manifest, door, _candidate = make_fixture()
    candidate = rule_based_candidate(manifest, door, 0)
    structure = candidate.urdf_structure()
    movable = {joint["name"] for joint in structure["joints"] if joint["type"] in {"revolute", "continuous", "prismatic"}}
    assert candidate.runtime_spec["door_dof_name"] in movable
    assert candidate.runtime_spec["door_dof_name"] != candidate.runtime_spec["handle_dof_name"]
    assert candidate.runtime_spec["door_body_name"] != candidate.runtime_spec["handle_body_name"]
    assert candidate.metadata["initial_generation"]["mode"] == "rule_based_public_geometry_v1"
    assert candidate.metadata["initial_generation"]["retrieval"] == {}
    moves = {
        str(primitive.params.get("stage")): float(primitive.params["distance"])
        for primitive in candidate.skill_program.primitives_named("MoveTo")
        if primitive.params.get("distance") is not None
    }
    assert moves["push"] + moves["traverse"] == MIN_DOOR_TWIN_FORWARD_DISTANCE_M


def test_generated_handle_goal_reference_is_diagnostic_not_structure_ground_truth() -> None:
    _manifest, _door, candidate = make_fixture()
    reference = [float(value) + 0.20 for value in candidate.handle_bounding["goal_pos"]]
    probe = {
        "reports": [
            {
                "steps": 100,
                "failure_stage": "success",
                "door_open_deg": 35.0,
                "door_spec": dict(candidate.runtime_spec),
            }
        ]
    }
    generated_reference = DoorCase(
        id=candidate.door_id,
        door_name=candidate.door_name,
        hidden_ground_truth={
            "handle_goal_pos": reference,
            "handle_goal_reference_type": "generated_initial_candidate",
            "handle_goal_verified": False,
        },
    )
    result = evaluate_asset(candidate, generated_reference, probe)
    assert result["frozen_reference_handle_goal_delta_m"] > 0.05
    assert result["handle_goal_check_required"] is False
    assert result["structure_success"] is True

    verified = DoorCase(
        id=candidate.door_id,
        door_name=candidate.door_name,
        hidden_ground_truth={"handle_goal_pos": reference, "handle_goal_verified": True},
    )
    verified_result = evaluate_asset(candidate, verified, probe)
    assert verified_result["handle_goal_check_passed"] is False
    assert verified_result["structure_success"] is False


def test_candidate_graph_rejects_regression_and_keeps_best() -> None:
    graph = CandidateGraph("initial")
    assert graph.evaluate("initial", summary(2)) == "new_best"
    graph.add(CandidateNode("better", "initial", patch={"x": 1}))
    assert graph.evaluate("better", summary(3)) == "new_best"
    graph.add(CandidateNode("worse", "better", patch={"x": 2}))
    assert graph.evaluate("worse", summary(1)) == "rejected_regression"
    assert graph.best_hash == "better"
    restored = CandidateGraph.from_dict(graph.to_dict())
    assert restored.best_hash == "better"
    assert restored.nodes["worse"].status == "rejected_regression"


def test_candidate_graph_prefers_grasp_z_closer_to_zero_only_when_success_ties() -> None:
    _manifest, _door, low_candidate = make_fixture()
    low_candidate.skill_program = ProgramPatch(
        {"MoveEEToHandle.grasp_offset": [0.0, 0.0, -0.04]}
    ).apply(low_candidate.skill_program)
    centered_candidate = low_candidate.clone()
    centered_candidate.skill_program = ProgramPatch(
        {"MoveEEToHandle.grasp_offset": [0.0, 0.0, -0.01]}
    ).apply(centered_candidate.skill_program)

    graph = CandidateGraph("low")
    assert graph.evaluate("low", summary(4), candidate=low_candidate) == "new_best"
    graph.add(CandidateNode("centered", "low"))
    assert graph.evaluate("centered", summary(4), candidate=centered_candidate) == "new_best"
    assert graph.best_score["grasp_z_abs_m"] == 0.01

    graph.add(CandidateNode("lower_success", "centered"))
    assert graph.evaluate("lower_success", summary(3), candidate=centered_candidate) == "rejected_regression"
    assert graph.best_hash == "centered"


def test_static_validation_rejects_mismatched_pregrasp_and_grasp_z() -> None:
    _manifest, _door, candidate = make_fixture()
    move = candidate.skill_program.primitive("MoveEEToHandle")
    assert move is not None
    move.params["pregrasp_offset"] = [0.15, 0.0, -0.04]
    move.params["grasp_offset"] = [0.0, 0.0, -0.01]
    report = validate_candidate_static(candidate)
    assert report.checks["pregrasp_grasp_z_match"] is False
    assert "HANDLE_APPROACH_Z_MISMATCH" in {issue.code for issue in report.issues}


def test_static_validation_rejects_short_push_and_traverse() -> None:
    _manifest, _door, candidate = make_fixture()
    for move in candidate.skill_program.primitives_named("MoveTo"):
        if move.params.get("stage") == "push":
            move.params["distance"] = 0.35
        elif move.params.get("stage") == "traverse":
            move.params["distance"] = 0.80
    report = validate_candidate_static(candidate)
    assert report.checks["minimum_forward_distance"] is False
    assert "FORWARD_DISTANCE_INSUFFICIENT" in {issue.code for issue in report.issues}


def test_static_validation_and_empty_probe_summary_are_json_safe() -> None:
    _manifest, _door, candidate = make_fixture()
    report = validate_candidate_static(candidate).to_dict()
    json.dumps(report, allow_nan=False)
    probes = summarize_physics_probes(
        candidate.fingerprint,
        load_summary={},
        hinge_summary={},
        handle_summary={},
        grasp_summary={},
        has_handle_dof=True,
    )
    assert probes["passed"] is False
    assert probes["measurements"]["max_pre_push_ee_tracking_error_m"] is None
    json.dumps(probes, allow_nan=False)


def test_grasp_probe_uses_grasp_phase_and_ignores_later_push_collision() -> None:
    report = {
        "steps": 550,
        "door_open_deg": 40.0,
        "handle_rotation_deg": 45.0,
        "handle_unlocked": True,
        "base_collision": True,
        "ee_handle_dist": 0.04,
        "metrics": {
            "max_pre_push_ee_tracking_error": 0.18,
            "min_grasp_ee_handle_dist": 0.04,
            "pre_push_joint_limit_hit": False,
            "phase_metrics": {
                "grasp": {"max_ee_tracking_error": 0.05, "base_collision": False},
                "push_door": {"max_ee_tracking_error": 0.10, "base_collision": True},
            },
        },
    }
    probes = summarize_physics_probes(
        "candidate",
        load_summary={"reports": [{"steps": 100, "failure_stage": ""}]},
        hinge_summary={"reports": [{"door_open_deg": 35.0}]},
        handle_summary={"reports": [report]},
        grasp_summary={"reports": [report]},
        has_handle_dof=True,
    )
    assert probes["passed"] is True
    assert abs(probes["measurements"]["max_pre_push_ee_tracking_error_m"] - 0.05) < 1.0e-9
    assert probes["measurements"]["base_collision"] is False


def test_hinge_probe_uses_physical_interaction_fallback_when_effort_probe_sleeps() -> None:
    report = {
        "steps": 750,
        "door_open_deg": 40.0,
        "handle_rotation_deg": 45.0,
        "handle_unlocked": True,
        "ee_handle_dist": 0.03,
        "metrics": {
            "min_grasp_ee_handle_dist": 0.03,
            "pre_push_joint_limit_hit": False,
            "phase_metrics": {"grasp": {"max_ee_tracking_error": 0.04, "base_collision": False}},
        },
    }
    probes = summarize_physics_probes(
        "candidate",
        load_summary={"reports": [{"steps": 100, "failure_stage": ""}]},
        hinge_summary={"reports": [{"door_open_deg": 0.0}]},
        handle_summary={"reports": [report]},
        grasp_summary={"reports": [report]},
        has_handle_dof=True,
    )
    assert probes["passed"] is True
    assert probes["measurements"]["hinge_motion_source"] == "interaction_probe"
    assert probes["measurements"]["hinge_motion_deg"] == 40.0


def test_wc4_static_validation_regression_passes() -> None:
    door = DoorCase(id="wc4", door_name="wc4")
    manifest = BenchmarkManifest(
        path=CFG,
        name="wc4_static_test",
        base_door_cfg=CFG,
        runner_script=RUNNER,
        output_root=Path("/tmp"),
        doors=[door],
    )
    candidate = Candidate.from_base_config(manifest, door, 0, sanitize_handle_goal=False)
    report = validate_candidate_static(candidate)
    assert report.passed, [issue.code for issue in report.issues]


def test_manual_codex_backend_uses_atomic_request_response(tmp_path: Path) -> None:
    backend = ManualCodexBackend(tmp_path, poll_s=0.01, timeout_s=2.0)

    def respond() -> None:
        request_path = tmp_path / "request_0000.json"
        for _ in range(200):
            if request_path.is_file():
                break
            time.sleep(0.01)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["request_fingerprint"]
        (tmp_path / "response_0000.json").write_text(
            json.dumps({"action": "inspect_candidate", "arguments": {}, "diagnostics": "test"}),
            encoding="utf-8",
        )

    thread = threading.Thread(target=respond)
    thread.start()
    action = backend.next_action(0, {"candidate": "abc"})
    thread.join(timeout=2)
    assert action.action == "inspect_candidate"


def test_session_resumes_and_can_initialize_skill_from_retrieved_experience(tmp_path: Path) -> None:
    manifest, door, candidate = make_fixture()
    catalog = ExperienceCatalog.load(CATALOG)
    backend = ScriptedBackend([])
    session = DoorTwinAgentSession(
        run_root=tmp_path,
        manifest=manifest,
        door=door,
        initial_candidate=candidate,
        catalog=catalog,
        backend=backend,
        max_tool_turns=30,
    )
    found = session.execute(AgentAction("find_experiences", {"query": "push traverse", "limit": 3}))
    assert found["ok"] is True
    experience_id = found["result"]["matches"][0]["experience_id"]
    patched = session.execute(
        AgentAction(
            "apply_candidate_patch",
            {"base_experience_id": experience_id, "patch": {"skill_patch": {"MoveEEToHandle.grasp_offset": [0, 0, -0.04]}}},
            "reuse a validated skill, then adapt grasp height",
        )
    )
    assert patched["ok"] is True
    pending_hash = patched["result"]["candidate_hash"]
    session.tool_turn = 2
    session.last_observation = patched
    session._persist()

    resumed = DoorTwinAgentSession(
        run_root=tmp_path,
        manifest=manifest,
        door=door,
        initial_candidate=candidate,
        catalog=catalog,
        backend=backend,
        max_tool_turns=30,
    )
    assert resumed.tool_turn == 2
    assert resumed.current_hash == pending_hash
    assert experience_id in resumed.retrieved_ids
    assert (tmp_path / "prior_snapshot.json").is_file()
