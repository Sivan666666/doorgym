import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


FLOAT_IK_ROOT = Path(__file__).resolve().parents[1]
if str(FLOAT_IK_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOAT_IK_ROOT))

from door_twin import (
    DoorTwinSpec,
    ProgramPatch,
    RolloutTracker,
    SkillProgram,
    apply_program_to_args,
    compute_skill_waypoints,
    default_skill_program_from_args,
    load_specs_from_config,
    profile_from_program,
    write_rollout_reports,
)


REPO_ROOT = FLOAT_IK_ROOT.parents[1]


def test_door_twin_spec_loads_wc4_from_cfg():
    specs = load_specs_from_config(REPO_ROOT / "high-level/data/cfg/b1z1_opendoor.yaml", door_name="wc4")
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "wc4"
    assert spec.supported
    assert len(spec.handle_goal_pos) == 3


def test_door_twin_spec_loads_button_door_from_cfg():
    specs = load_specs_from_config(REPO_ROOT / "high-level/data/cfg/b1z1_opendoor.yaml", door_name="button_door")
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "button_door"
    assert spec.supported
    assert spec.door_motion_sign_multiplier == pytest.approx(-1.0)
    assert spec.door_dof_name == "door_hinge_joint"
    assert spec.handle_dof_name == "handle_actuation_joint"
    assert spec.handle_goal_pos == pytest.approx([0.0845, 0.068, 0.0])


def test_door_twin_spec_loads_aigc_reference_export():
    export_dir = Path("/home/sivan/whole_body/door_aigc/reference_export")
    if not export_dir.is_dir():
        pytest.skip("AIGC reference export is not present on this machine")
    spec = DoorTwinSpec.from_aigc_export_dir(export_dir)
    assert spec.name == "reference_export"
    assert spec.supported
    assert spec.door_dof_name == "joint_1"
    assert spec.handle_dof_name == "joint_2"


def test_door_twin_spec_loads_record_materialization_cfg():
    cfg = REPO_ROOT / "high-level/experiments/isaacgym/b1z1_opendoor_record_materialization.yaml"
    if not cfg.is_file():
        pytest.skip("record_materialization config is not present on this machine")
    specs = load_specs_from_config(cfg)
    assert specs
    assert specs[0].supported
    assert "record_materialization" in specs[0].asset_path


def test_skill_program_roundtrip_and_local_waypoints(tmp_path):
    path = tmp_path / "skill.json"
    data = {
        "version": "door_skill_program_v1",
        "skill_program": [
            {"name": "MoveEEToHandle", "pregrasp_offset": [0.2, 0.01, 0.03], "grasp_offset": [0.0, 0.0, 0.01]},
            {"name": "RotateHandle", "local_delta": [0.0, 0.03, -0.02]},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    program = SkillProgram.from_obj(json.loads(path.read_text(encoding="utf-8")))
    profile = profile_from_program(program, SimpleNamespace(pregrasp_offset=0.15, grasp_offset=0.0, grasp_z_offset=-0.03))
    points = compute_skill_waypoints(np.array([1.0, 2.0, 0.5]), np.array([1.0, 0.0, 0.0]), profile)
    assert np.allclose(points.pregrasp, [1.2, 2.01, 0.53])
    assert np.allclose(points.grasp, [1.0, 2.0, 0.51])
    assert np.allclose(points.rotate, [1.0, 2.03, 0.49])


def test_program_patch_allowlist_and_clamp():
    program = SkillProgram.from_obj(
        {
            "skill_program": [
                {"name": "PushDoor", "distance": 1.0},
                {"name": "RotateHandle", "angle": 0.5},
            ]
        }
    )
    patch = ProgramPatch.from_obj(
        {
            "patch": {
                "PushDoor.distance": 5.0,
                "PushDoor.base_distance": 0.10,
                "RotateHandle.angle": -3.0,
                "UnsafePython.code": "rm -rf",
            }
        }
    )
    patched = patch.apply(program)
    assert patched.primitive("PushDoor").params["distance"] == 2.0
    assert "base_distance" not in patched.primitive("PushDoor").params
    assert patched.primitive("RotateHandle").params["angle"] == 0.05
    assert patched.metadata["execution_mode"] == "skill_interpreter"
    assert "PushDoor.base_distance" not in patch.clamped_patch()
    assert "UnsafePython.code" not in patch.clamped_patch()


def test_auto_program_starts_in_legacy_replay_mode():
    program = default_skill_program_from_args(SimpleNamespace())
    assert program.metadata["execution_mode"] == "legacy_replay"
    assert "base_distance" not in program.primitive("PushDoor").params
    assert {move.params["stage"] for move in program.primitives_named("MoveTo")} == {
        "approach",
        "push",
    }


def test_move_to_base_skill_applies_velocity_and_stage_patch():
    program = SkillProgram.from_obj(
        {
            "skill_program": [
                {"name": "MoveTo", "stage": "approach", "vx": 0.25, "vyaw": 0.10, "stop_distance": 0.22},
                {
                    "name": "MoveTo",
                    "stage": "push",
                    "vx": 0.15,
                    "vyaw": -0.20,
                    "distance": 0.45,
                    "duration_steps": 100,
                },
            ]
        }
    )
    args = SimpleNamespace(
        sim_dt=0.02,
        stop_distance=0.15,
        walk_min_speed=0.20,
        walk_steps=260,
        no_dynamic_walk_steps=False,
        door_push_steps=300,
        push_base_distance=0.35,
        push_base_yaw_delta=0.0,
        pass_through_door=False,
        return_home_steps=150,
    )
    apply_program_to_args(args, program)
    profile = profile_from_program(program, args)
    assert args.walk_min_speed == pytest.approx(0.25)
    assert args.stop_distance == pytest.approx(0.22)
    assert args.move_to_approach_vyaw == pytest.approx(0.10)
    assert args.push_base_distance == pytest.approx(0.45)
    assert args.push_base_yaw_delta == pytest.approx(-0.40)
    assert args.move_to_push_steps == 100
    assert set(profile.base_moves) == {"approach", "push"}

    patch = ProgramPatch({"MoveTo:push.vx": 5.0, "MoveTo:approach.vyaw": -3.0})
    patched = patch.apply(program)
    push = next(
        primitive
        for primitive in patched.primitives_named("MoveTo")
        if primitive.params.get("stage") == "push"
    )
    approach = next(
        primitive
        for primitive in patched.primitives_named("MoveTo")
        if primitive.params.get("stage") == "approach"
    )
    assert push.params["vx"] == pytest.approx(0.60)
    assert approach.params["vyaw"] == pytest.approx(-1.20)


def test_skill_program_pass_through_only_when_traverse_requested():
    args = SimpleNamespace(
        sim_dt=0.02,
        stop_distance=0.15,
        walk_min_speed=0.20,
        walk_steps=260,
        no_dynamic_walk_steps=False,
        door_push_steps=300,
        push_base_distance=0.35,
        push_base_yaw_delta=0.0,
        pass_through_door=True,
        pass_open_angle_deg=80.0,
        return_home_steps=150,
    )
    program = SkillProgram.from_obj({"skill_program": [{"name": "MoveTo", "stage": "push", "distance": 0.24}]})
    apply_program_to_args(args, program)
    assert args.pass_through_door is False

    traverse_args = SimpleNamespace(**vars(args))
    traverse_args.pass_through_door = False
    traverse_program = SkillProgram.from_obj(
        {"skill_program": [{"name": "TraverseDoor", "base_v": 0.18, "door_angle_target": 65.0}]}
    )
    apply_program_to_args(traverse_args, traverse_program)
    assert traverse_args.pass_through_door is True
    assert traverse_args.pass_open_angle_deg == pytest.approx(65.0)

    legacy_args = SimpleNamespace(**vars(args))
    legacy_args.pass_through_door = True
    legacy_program = default_skill_program_from_args(legacy_args)
    apply_program_to_args(legacy_args, legacy_program)
    assert legacy_args.pass_through_door is True


def test_rollout_summary_indexes_expert_trajectory(tmp_path):
    tracker = RolloutTracker(
        env_id=0,
        door_name="wc4",
        pass_open_angle_deg=80.0,
        door_motion_sign=-1.0,
        handle_unlock_threshold=0.1,
    )
    tracker.update(
        step=10,
        phase="hold_home",
        door_pos=[-np.pi / 2.0, 0.5],
        handle_goal=None,
        target_pos=None,
        ee_pos=None,
        ee_tracking_error=0.0,
        base_xy=[0.0, 0.0],
        base_collision=False,
        camera_available=True,
        base_vx=0.0,
        base_vyaw=0.0,
    )
    tracker.add_artifact(
        "expert_trajectory",
        {"path": "expert_raw/episode_000000.npz", "format": "door_dp_raw_npz_v1"},
    )
    summary = write_rollout_reports([tracker], tmp_path)
    assert summary["success_count"] == 1
    assert summary["expert_trajectories"] == [
        {
            "env_id": 0,
            "path": "expert_raw/episode_000000.npz",
            "format": "door_dp_raw_npz_v1",
        }
    ]


def test_failure_classification_keeps_late_collision_secondary():
    tracker = RolloutTracker(
        env_id=0,
        door_name="generated",
        door_motion_sign=-1.0,
        handle_unlock_threshold=0.1,
    )
    tracker.update(
        step=10,
        phase="grasp",
        door_pos=[0.0, 0.0],
        handle_goal=[1.0, 0.0, 1.0],
        target_pos=[1.0, 0.0, 1.0],
        ee_pos=[1.0, 0.0, 1.0],
        ee_tracking_error=0.0,
        base_xy=[0.0, 0.0],
        base_collision=False,
        camera_available=True,
    )
    tracker.update(
        step=20,
        phase="push_door",
        door_pos=[0.0, 0.0],
        handle_goal=[1.0, 0.0, 1.0],
        target_pos=[1.0, 0.0, 1.0],
        ee_pos=[1.0, 0.0, 1.0],
        ee_tracking_error=0.0,
        base_xy=[0.1, 0.0],
        base_collision=True,
        camera_available=True,
    )
    report = tracker.finalize()
    assert report.failure_stage == "base_collision"
    assert "handle_not_unlocked" in report.metrics["secondary_failures"]
