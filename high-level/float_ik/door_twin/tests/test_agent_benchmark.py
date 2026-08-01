from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml


FLOAT_IK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = FLOAT_IK_ROOT.parents[1]
if str(FLOAT_IK_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOAT_IK_ROOT))

from door_twin.benchmark.observer import OpenAIResponsesObserver, structured_visual_diagnostics  # noqa: E402
from door_twin.benchmark.patches import AssetConfigPatch, CombinedRepairPatch  # noqa: E402
from door_twin.benchmark.schema import (  # noqa: E402
    OURS_INITIALIZATION_PROTOCOL,
    BenchmarkManifest,
    Candidate,
    DoorCase,
    rule_based_candidate,
)
from door_twin.benchmark import run_agent_ablation as benchmark_runner  # noqa: E402
from door_twin.analyzer import RolloutTracker  # noqa: E402


CFG = PROJECT_ROOT / "high-level/data/cfg/b1z1_opendoor.yaml"
RUNNER = PROJECT_ROOT / "high-level/float_ik/isaacgym_float_ik_a2w_basearn_push_door_parallel.py"


def make_candidate() -> Candidate:
    door = DoorCase(id="99650069960003", door_name="99650069960003")
    manifest = BenchmarkManifest(
        path=CFG,
        name="test",
        base_door_cfg=CFG,
        runner_script=RUNNER,
        output_root=Path("/tmp"),
        doors=[door],
    )
    return Candidate.from_base_config(manifest, door, 0)


def test_public_context_hides_goal_annotation() -> None:
    candidate = make_candidate()
    context = candidate.public_context()
    assert "goal_pos" not in context["handle_bounding_without_goal"]
    source_cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    entry = next(
        value
        for values in source_cfg["env"]["asset"]["trainAssets"].values()
        for value in values.values()
        if value.get("name") == candidate.door_name
    )
    root = CFG.parents[2] / source_cfg["env"]["asset"]["assetRoot"] / source_cfg["env"]["asset"]["assetFileDoor"]
    hidden = json.loads((root / entry["handle_bounding"]).read_text(encoding="utf-8"))["goal_pos"]
    assert candidate.handle_bounding["goal_pos"] != hidden


def test_asset_patch_is_bounded_and_validates_urdf_names() -> None:
    candidate = make_candidate()
    patch = AssetConfigPatch(
        {
            "actor_scale": 99.0,
            "door_body_name": "link_1",
            "door_dof_name": "not_a_joint",
            "unknown": 1,
            "handle_goal_pos": [9.0, -9.0, 9.0],
        }
    )
    updated, info = patch.apply(candidate)
    assert updated.runtime_spec["actor_scale"] == 3.0
    assert updated.runtime_spec["door_body_name"] == "link_1"
    assert "door_dof_name" not in updated.runtime_spec
    assert sorted(info["rejected_fields"]) == ["door_dof_name", "unknown"]
    assert updated.handle_bounding["goal_pos"] == [2.0, -2.0, 2.5]


def test_combined_patch_changes_only_allowlisted_skill_values() -> None:
    candidate = make_candidate()
    patch = CombinedRepairPatch.from_obj(
        {
            "failure_stage": "grasp_miss",
            "diagnostics": "lower the grasp",
            "asset_patch": {},
            "skill_patch": {
                "MoveEEToHandle.grasp_offset": [0.0, 0.0, -0.05],
                "ArbitraryPython.code": "rm -rf /",
            },
        }
    )
    updated, info = patch.apply(candidate)
    move = updated.skill_program.primitive("MoveEEToHandle")
    assert move is not None
    assert move.params["grasp_offset"] == [0.0, 0.0, -0.05]
    assert move.params["pregrasp_offset"][2] == -0.05
    assert set(info["accepted_skill"]) == {"MoveEEToHandle.grasp_offset"}


def test_candidate_round_trip_preserves_fingerprint(tmp_path: Path) -> None:
    candidate = make_candidate()
    paths = candidate.write(tmp_path / "candidate")
    loaded = Candidate.read(paths["candidate"], CFG)
    assert loaded.fingerprint == candidate.fingerprint
    cfg = yaml.safe_load(paths["door_cfg"].read_text(encoding="utf-8"))
    block = cfg["env"]["asset"]["load_block"]
    assert len(cfg["env"]["asset"]["trainAssets"][block]) == 1


def test_mock_observer_returns_json_patch_without_network(tmp_path: Path) -> None:
    response = {
        "output_text": json.dumps(
            {
                "failure_stage": "grasp_miss",
                "diagnostics": "grasp is high",
                "asset_patch": {},
                "skill_patch": {"MoveEEToHandle.grasp_offset": [0.0, 0.0, -0.04]},
            }
        ),
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    response_path = tmp_path / "mock.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    observer = OpenAIResponsesObserver(mock_response=response_path)
    result = observer.observe(make_candidate(), summary=None, include_images=False, initial_generation=True)
    assert result.patch.failure_stage == "grasp_miss"
    assert result.usage["input_tokens"] == 10
    assert result.image_paths == []


def test_ours_initial_candidate_is_one_residual_patch_on_rule_based_base(tmp_path: Path) -> None:
    door = DoorCase(id="99650069960003", door_name="99650069960003")
    manifest = BenchmarkManifest(
        path=CFG,
        name="shared_initial_test",
        base_door_cfg=CFG,
        runner_script=RUNNER,
        output_root=tmp_path,
        doors=[door],
    )
    response = {
        "output_text": json.dumps(
            {
                "failure_stage": "initial_generation",
                "diagnostics": "small prior-supported residual",
                "asset_patch": {},
                "skill_patch": {"MoveTo:push.vx": 0.05},
            }
        ),
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    response_path = tmp_path / "mock_initial.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    observer = OpenAIResponsesObserver(mock_response=response_path)
    args = SimpleNamespace(initial_generation="vlm", force=False)

    candidate, generation = benchmark_runner.prepare_initial_candidate(
        manifest, door, 0, observer, args, catalog=None
    )
    rule_base = rule_based_candidate(manifest, door, 0)
    assert generation["protocol"] == OURS_INITIALIZATION_PROTOCOL
    assert generation["base_candidate_hash"] == rule_base.fingerprint
    assert generation["patch_semantics"] == "bounded_residual_on_rule_based_candidate"
    assert candidate.fingerprint != rule_base.fingerprint
    assert candidate.skill_program.primitives_named("MoveTo")[1].params["vx"] == 0.05

    saved_base = Candidate.read(
        tmp_path
        / "shared_initial_test/initial_candidates/99650069960003/candidate_00/rule_based_base/candidate.json",
        CFG,
    )
    assert saved_base.fingerprint == rule_base.fingerprint

    request_context = json.loads(
        observer.observe(
            rule_base,
            summary=None,
            include_images=False,
            initial_generation=True,
        ).persisted_request["input"][0]["content"][0]["text"]
    )
    requirements = request_context["initial_generation_requirements"]
    assert "trusted baseline" in requirements["starting_point"]
    assert "residual patch" in requirements["patch_mode"]


def test_manual_observer_file_exchange_needs_no_api_key(tmp_path: Path, monkeypatch) -> None:
    exchange = tmp_path / "exchange"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def respond() -> None:
        request = exchange / "request_0000.json"
        for _ in range(200):
            if request.is_file():
                break
            time.sleep(0.01)
        assert request.is_file()
        record = json.loads(request.read_text(encoding="utf-8"))
        assert record["image_paths"] == []
        Path(record["response_path"]).write_text(
            json.dumps(
                {
                    "failure_stage": "grasp_miss",
                    "diagnostics": "manual bridge test",
                    "asset_patch": {},
                    "skill_patch": {"MoveEEToHandle.grasp_offset": [0.0, 0.0, -0.02]},
                }
            ),
            encoding="utf-8",
        )

    thread = threading.Thread(target=respond)
    thread.start()
    observer = OpenAIResponsesObserver(manual_exchange_dir=exchange, manual_wait_timeout_s=5)
    result = observer.observe(make_candidate(), summary=None, include_images=False, initial_generation=True)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result.patch.failure_stage == "grasp_miss"
    assert result.usage["manual_observer_calls"] == 1


def test_rollout_tracker_exposes_push_pull_and_phase_diagnostics() -> None:
    tracker = RolloutTracker(
        0,
        "door",
        door_spec={"handle_dof_name": "joint_2"},
        door_motion_sign=-1.0,
        handle_unlock_threshold=0.5,
        save_trace=True,
        base_start=[1.0, 0.0],
        base_push=[0.0, 0.0],
    )
    tracker.update(
        step=0,
        phase="grasp",
        door_pos=[0.0, 0.0],
        handle_goal=[0.0, 0.0, 1.0],
        target_pos=[0.0, 0.0, 1.0],
        ee_pos=[0.0, 0.0, 1.0],
        ee_tracking_error=0.0,
        base_xy=[1.0, 0.0],
        base_collision=False,
        camera_available=False,
    )
    tracker.update(
        step=25,
        phase="push_door",
        door_pos=[-0.6, 0.6],
        handle_goal=[-0.2, 0.0, 1.0],
        target_pos=[-0.2, 0.0, 1.0],
        ee_pos=[-0.2, 0.0, 1.0],
        ee_tracking_error=0.01,
        base_xy=[0.8, 0.0],
        base_collision=True,
        camera_available=False,
    )
    report = tracker.finalize()
    metrics = report.metrics
    assert metrics["door_motion_sign"] == -1.0
    assert metrics["raw_hinge_deg_range"][0] < -34.0
    assert metrics["first_handle_unlock"] == {"step": 25, "phase": "push_door"}
    assert metrics["first_base_collision"] == {"step": 25, "phase": "push_door"}
    assert metrics["door_motion_relative_to_robot"] == "push_away_from_robot"
    assert metrics["phase_metrics"]["push_door"]["base_collision"] is True
    assert metrics["trace"][-1]["door_hinge_raw_deg"] < -34.0


def test_rollout_tracker_body_passed_uses_door_plane_and_robot_rear() -> None:
    tracker = RolloutTracker(
        0,
        "door",
        door_spec={"handle_dof_name": ""},
        require_traverse=True,
        base_start=[1.0, 0.0],
        base_push=[0.8, 0.0],
        pass_plane_point=[0.0, 0.0],
        pass_direction=[-1.0, 0.0],
        robot_rear_offset=0.65,
    )
    common = {
        "phase": "traverse_door",
        "door_pos": [-math.radians(90.0)],
        "handle_goal": [0.0, 0.0, 1.0],
        "target_pos": [0.0, 0.0, 1.0],
        "ee_pos": [0.0, 0.0, 1.0],
        "ee_tracking_error": 0.0,
        "base_collision": False,
        "camera_available": False,
    }
    tracker.update(step=0, base_xy=[0.8, 0.0], **common)
    assert tracker.body_passed is False
    tracker.update(step=1, base_xy=[-0.64, 0.0], **common)
    assert tracker.body_passed is False
    tracker.update(step=2, base_xy=[-0.65, 0.0], **common)
    report = tracker.finalize()
    assert report.body_passed is True
    assert report.metrics["body_passed_source"] == "door_plane_rear_extent"
    assert abs(report.metrics["pass_plane_tail_margin_m"]) < 1.0e-6


def _summary(*, successes: int, collisions: int, opened: int, unlocked: int, passed: int, trials: int = 2) -> dict:
    reports = []
    for index in range(trials):
        reports.append(
            {
                "success": index < successes,
                "failure_stage": "" if index < successes else "base_collision",
                "base_collision": index < collisions,
                "door_open_deg": 90.0 if index < opened else 0.0,
                "handle_unlocked": index < unlocked,
                "body_passed": index < passed,
                "ee_handle_dist": 0.02,
                "metrics": {"min_grasp_ee_handle_dist": 0.02},
            }
        )
    return {
        "reports": reports,
        "success_count": successes,
        "failure_counts": {"success": successes, "base_collision": trials - successes},
        "process_failures": [],
    }


def test_structured_visual_diagnostics_classifies_grasp_height_and_hints_equal_z() -> None:
    candidate = make_candidate()
    move = candidate.skill_program.primitive("MoveEEToHandle")
    assert move is not None
    move.params["pregrasp_offset"] = [0.15, 0.0, -0.01]
    move.params["grasp_offset"] = [0.0, 0.0, -0.01]
    summary = {
        "reports": [
            {
                "success": False,
                "failure_stage": "grasp_miss",
                "door_open_deg": 0.0,
                "body_passed": False,
                "metrics": {
                    "trace": [
                        {
                            "step": 10,
                            "phase": "close_gripper",
                            "ee_handle_dist": 0.04,
                            "ee_pos": [0.0, 0.0, 1.05],
                            "handle_goal": [0.0, 0.0, 1.0],
                        }
                    ]
                },
            }
        ],
        "success_count": 0,
        "failure_counts": {"grasp_miss": 1},
    }
    diagnostics = structured_visual_diagnostics(summary, candidate=candidate)
    assert diagnostics["grasp_alignment"]["vertical_relation"]["label"] == "gripper_above_handle"
    grasp_hints = [hint for hint in diagnostics["patch_hints"] if "MoveEEToHandle.grasp_offset" in hint["skill_patch"]]
    assert grasp_hints
    assert grasp_hints[0]["skill_patch"]["MoveEEToHandle.pregrasp_offset"][2] == -0.02
    assert grasp_hints[0]["skill_patch"]["MoveEEToHandle.grasp_offset"][2] == -0.02


def test_structured_visual_diagnostics_flags_preunlock_collision_with_task_complete() -> None:
    candidate = make_candidate()
    summary = {
        "reports": [
            {
                "success": False,
                "failure_stage": "base_collision",
                "door_open_deg": 90.0,
                "body_passed": True,
                "metrics": {
                    "first_base_collision": {"step": 100, "phase": "push_door"},
                    "first_handle_unlock": {"step": 150, "phase": "push_door"},
                    "first_door_open": {"step": 155, "phase": "push_door"},
                    "trace": [
                        {
                            "step": 50,
                            "phase": "grasp",
                            "ee_handle_dist": 0.01,
                            "ee_pos": [0.0, 0.0, 1.0],
                            "handle_goal": [0.0, 0.0, 1.0],
                        }
                    ],
                },
            }
        ],
        "success_count": 0,
        "failure_counts": {"base_collision": 1},
    }
    diagnostics = structured_visual_diagnostics(summary, candidate=candidate)
    assert diagnostics["base_collision"]["label"] == "preunlock_collision_but_task_completes"
    assert any("MoveTo:push.vx" in hint["skill_patch"] for hint in diagnostics["patch_hints"])


def test_development_score_prioritizes_success_then_safety_and_stages() -> None:
    baseline = benchmark_runner.development_score(
        _summary(successes=0, collisions=2, opened=2, unlocked=2, passed=2)
    )
    one_success = benchmark_runner.development_score(
        _summary(successes=1, collisions=1, opened=1, unlocked=1, passed=1)
    )
    safer_tie = benchmark_runner.development_score(
        _summary(successes=1, collisions=0, opened=1, unlocked=1, passed=1)
    )
    assert benchmark_runner._score_rank(one_success) > benchmark_runner._score_rank(baseline)
    assert benchmark_runner._score_rank(safer_tie) > benchmark_runner._score_rank(one_success)


def test_run_method_rolls_back_regression_and_evaluates_best_candidate(tmp_path: Path, monkeypatch) -> None:
    initial = make_candidate()
    manifest = BenchmarkManifest(
        path=CFG,
        name="best_candidate_test",
        base_door_cfg=CFG,
        runner_script=RUNNER,
        output_root=tmp_path,
        doors=[DoorCase(id=initial.door_id, door_name=initial.door_name)],
        candidate_repeats=1,
        max_repair_rounds=2,
        development_seeds=[1, 2],
        heldout_seeds=[3, 4],
    )
    development = iter(
        [
            _summary(successes=0, collisions=2, opened=1, unlocked=1, passed=2),
            _summary(successes=1, collisions=1, opened=2, unlocked=2, passed=2),
            _summary(successes=0, collisions=2, opened=2, unlocked=2, passed=2),
        ]
    )
    heldout = _summary(successes=1, collisions=1, opened=2, unlocked=2, passed=2)

    def fake_seed_set(_manifest, _candidate, _paths, seeds, _out, **_kwargs):
        return (next(development), 0.0) if seeds == manifest.development_seeds else (heldout, 0.0)

    monkeypatch.setattr(benchmark_runner, "run_seed_set", fake_seed_set)
    monkeypatch.setattr(benchmark_runner, "run_asset_probe", lambda *_args, **_kwargs: (heldout, 0.0))
    monkeypatch.setattr(
        benchmark_runner,
        "evaluate_asset",
        lambda *_args, **_kwargs: {"load_success": True, "structure_success": True},
    )
    monkeypatch.setattr(benchmark_runner, "write_observer_result", lambda *_args, **_kwargs: None)

    patches = iter(
        [
            CombinedRepairPatch.from_obj(
                {"failure_stage": "grasp", "skill_patch": {"MoveEEToHandle.grasp_offset": [0, 0, -0.02]}}
            ),
            CombinedRepairPatch.from_obj(
                {"failure_stage": "grasp", "skill_patch": {"MoveEEToHandle.grasp_offset": [0, 0, -0.01]}}
            ),
        ]
    )

    class FakeObserver:
        def observe(self, *_args, **_kwargs):
            return SimpleNamespace(
                patch=next(patches),
                usage={},
                elapsed_s=0.0,
                request_fingerprint="fake",
                image_paths=[],
                persisted_request={},
                output_text="{}",
                raw_response={},
            )

    expected_best, _ = CombinedRepairPatch.from_obj(
        {"skill_patch": {"MoveEEToHandle.grasp_offset": [0, 0, -0.02]}}
    ).apply(initial)
    result = benchmark_runner.run_method(
        manifest,
        manifest.doors[0],
        initial,
        "ours",
        FakeObserver(),
        heldout,
        0.0,
        {"load_success": True, "structure_success": True},
        SimpleNamespace(),
    )
    assert result["final_hash"] == expected_best.fingerprint
    assert result["best_development_round"] == 1
    assert result["repair_rounds_executed"] == 2
    assert result["development_evaluations"] == 3
    assert [record["candidate_selection"] for record in result["rounds"]] == [
        "new_best",
        "new_best",
        "rejected_regression",
    ]
    assert result["rounds"][-1]["rollback_to_hash"] == expected_best.fingerprint
