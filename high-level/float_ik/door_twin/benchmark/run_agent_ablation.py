#!/usr/bin/env python3
"""Run the paired five-way DoorTwin agent ablation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from door_twin.benchmark.evaluation import (  # noqa: E402
    aggregate_summaries,
    evaluate_asset,
    load_batch_summary,
    load_seed_summary,
    stage_metrics,
    synthetic_failed_summary,
)
from door_twin.benchmark.observer import OpenAIResponsesObserver, write_observer_result  # noqa: E402
from door_twin.benchmark.patches import CombinedRepairPatch  # noqa: E402
from door_twin.benchmark.schema import (  # noqa: E402
    ABLATION_METHODS,
    BenchmarkManifest,
    Candidate,
    DoorCase,
    OURS_INITIALIZATION_PROTOCOL,
    RULE_BASED_INITIALIZATION_PROTOCOL,
    load_benchmark_manifest,
    rule_based_candidate,
)
from door_twin.candidate_selection import candidate_score, score_rank  # noqa: E402
from door_twin.experience import DoorSignature, ExperienceCatalog  # noqa: E402
from door_twin.skill import ProgramPatch, heuristic_patches_for_failure  # noqa: E402


VISUAL_METHODS = {"ours", "validation_only"}
VLM_REPAIR_METHODS = {"ours", "without_visual_feedback"}
VALIDATION_ONLY_METHODS = {"validation_only_log", "validation_only"}
NO_DEVELOPMENT_METHODS = {"without_simulation_rollout", "rule_based"}


def _write_json(path: str | Path, value: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(out)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sum_usage(records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                totals[str(key)] = totals.get(str(key), 0.0) + float(value)
    return totals


def development_score(summary: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper around the shared Agent candidate score."""

    return candidate_score(summary)


def _score_rank(score: dict[str, Any]) -> tuple[float, ...]:
    return score_rank(score)


def _observer_repair_history(rounds: list[dict[str, Any]], current: dict[str, Any]) -> list[dict[str, Any]]:
    history = [*rounds, current]
    compact = []
    for record in history:
        item = {
            "evaluation": record.get("round"),
            "candidate_hash": record.get("candidate_hash_before"),
            "development_score": record.get("development_score"),
            "candidate_selection": record.get("candidate_selection"),
        }
        patch_info = record.get("patch_info")
        if patch_info:
            item["patch_applied_after_evaluation"] = {
                "asset": patch_info.get("asset", {}).get("accepted", {}),
                "skill": patch_info.get("accepted_skill", {}),
                "diagnostics": patch_info.get("diagnostics", ""),
            }
        compact.append(item)
    return compact


def observer_from_manifest(manifest: BenchmarkManifest, args: argparse.Namespace) -> OpenAIResponsesObserver:
    cfg = manifest.vlm
    return OpenAIResponsesObserver(
        model=str(args.vlm_model or cfg.get("model", "gpt-5")),
        temperature=float(cfg.get("temperature", 0.0)),
        base_url=cfg.get("base_url"),
        api_key_env=str(cfg.get("api_key_env", "OPENAI_API_KEY")),
        timeout_s=float(cfg.get("timeout_s", 180.0)),
        max_retries=int(cfg.get("max_retries", 3)),
        mock_response=args.mock_vlm_response or None,
        manual_exchange_dir=args.manual_vlm_dir or None,
        manual_wait_timeout_s=args.manual_vlm_timeout_s,
    )


def _remove_conflicting_args(args: list[str], names: set[str]) -> list[str]:
    result: list[str] = []
    skip_value = False
    for item in args:
        if skip_value:
            skip_value = False
            continue
        name = item.split("=", 1)[0]
        if name in names:
            if "=" not in item:
                skip_value = True
            continue
        result.append(item)
    return result


def rollout_command(
    manifest: BenchmarkManifest,
    candidate_paths: dict[str, Path],
    candidate: Candidate,
    seed: int,
    log_dir: Path,
    *,
    capture_images: bool,
    steps_override: int | None = None,
    extra_args: list[str] | None = None,
    num_envs: int = 1,
) -> list[str]:
    managed = {
        "--num_envs",
        "--steps",
        "--seed",
        "--door_cfg",
        "--door_name",
        "--skill_program_json",
        "--door_twin_log_dir",
        "--door_twin_camera_views",
    }
    common = _remove_conflicting_args(list(manifest.common_runner_args), managed)
    command = [
        manifest.python_executable,
        str(manifest.runner_script),
        "--num_envs",
        str(int(num_envs)),
        "--steps",
        str(manifest.steps if steps_override is None else steps_override),
        "--seed",
        str(seed),
        "--door_cfg",
        str(candidate_paths["door_cfg"]),
        "--door_name",
        candidate.door_name,
        "--skill_program_json",
        str(candidate_paths["skill_program"]),
        "--door_twin_log_dir",
        str(log_dir.resolve()),
        "--save_failed_rollouts",
        "--pass_open_angle_deg",
        "80",
        "--no_preview_trajectory_at_spawn",
        *common,
        *(extra_args or []),
    ]
    if capture_images:
        command.extend(
            [
                "--dump_keyframe_images",
                "--door_twin_camera_views",
                "front,wrist,handle_closeup,observer_left",
                "--enable_front_camera",
                "--enable_wrist_camera",
                "--camera_depth",
                "--no_show_camera_images",
            ]
        )
    return command


def run_rollout(
    manifest: BenchmarkManifest,
    candidate: Candidate,
    candidate_paths: dict[str, Path],
    seed: int,
    out_dir: Path,
    *,
    capture_images: bool,
    dry_run: bool,
    stream_output: bool,
    force: bool,
    steps_override: int | None = None,
    extra_args: list[str] | None = None,
    benchmark_seeds: list[int] | None = None,
) -> tuple[dict[str, Any], int, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "rollout_summary.json"
    process_path = out_dir / "process.json"
    seed_labels = [int(value) for value in (benchmark_seeds or [seed])]
    num_envs = len(seed_labels)
    if summary_path.is_file() and process_path.is_file() and not force:
        process = json.loads(process_path.read_text(encoding="utf-8"))
        if (
            process.get("candidate_hash") == candidate.fingerprint
            and list(process.get("benchmark_seeds", [seed])) == seed_labels
        ):
            try:
                summary = (
                    load_seed_summary(summary_path, seed)
                    if num_envs == 1
                    else load_batch_summary(summary_path, seed_labels)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                # A previous process may have exited during Isaac Gym teardown and
                # left behind an incomplete/synthetic summary. Re-run in that case.
                pass
            else:
                return summary, int(process.get("returncode", 0)), float(process.get("elapsed_s", 0.0))

    command = rollout_command(
        manifest,
        candidate_paths,
        candidate,
        seed,
        out_dir,
        capture_images=capture_images,
        steps_override=steps_override,
        extra_args=extra_args,
        num_envs=num_envs,
    )
    process_record = {
        "seed": int(seed),
        "candidate_hash": candidate.fingerprint,
        "benchmark_seeds": seed_labels,
        "num_envs": num_envs,
        "command": command,
        "command_shell": shlex.join(command),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        process_record.update({"returncode": 0, "elapsed_s": 0.0})
        _write_json(process_path, process_record)
        return synthetic_failed_summary(seed, 0, "dry-run: simulator was not executed"), 0, 0.0

    project_root = manifest.runner_script.parents[2]
    stdout_path = out_dir / "stdout.log"
    started = time.time()
    tail: list[str] = []
    with stdout_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
            tail = tail[-80:]
            if stream_output:
                print(f"[{candidate.door_id} c{candidate.candidate_index:02d} s{seed}] {line}", end="", flush=True)
        returncode = process.wait()
    elapsed = time.time() - started
    process_record.update({"returncode": int(returncode), "elapsed_s": elapsed, "stdout": str(stdout_path.resolve())})
    _write_json(process_path, process_record)
    if summary_path.is_file():
        try:
            summary = (
                load_seed_summary(summary_path, seed)
                if num_envs == 1
                else load_batch_summary(summary_path, seed_labels)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            summary = None
        if summary is not None:
            if returncode != 0:
                # Isaac Gym can segfault while destroying the simulator after it
                # has already completed every step and atomically written all
                # per-env reports. Keep those valid rollout results; the non-zero
                # teardown status remains recorded in process.json for auditing.
                process_record["summary_salvaged_after_nonzero_exit"] = True
                process_record["teardown_returncode"] = int(returncode)
                _write_json(process_path, process_record)
                warnings = list(summary.get("process_warnings", []) or [])
                warnings.append(
                    {
                        "type": "nonzero_exit_after_complete_summary",
                        "returncode": int(returncode),
                        "seed": int(seed),
                    }
                )
                summary["process_warnings"] = warnings
                _write_json(summary_path, _json_safe(summary))
            return summary, returncode, elapsed
    summary = synthetic_failed_summary(seed, returncode, "\n".join(tail))
    _write_json(summary_path, _json_safe(summary))
    return summary, returncode, elapsed


def run_seed_set(
    manifest: BenchmarkManifest,
    candidate: Candidate,
    candidate_paths: dict[str, Path],
    seeds: list[int],
    out_dir: Path,
    *,
    capture_images: bool,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float]:
    if not seeds:
        raise ValueError("run_seed_set requires at least one seed")
    # Start Isaac Gym once and execute the complete seed set as parallel envs.
    # seeds[0] is the process RNG seed; reports retain stable per-env labels.
    summary, _returncode, elapsed = run_rollout(
        manifest,
        candidate,
        candidate_paths,
        seeds[0],
        out_dir / f"batch_seed_{seeds[0]}_n{len(seeds)}",
        capture_images=capture_images,
        dry_run=args.dry_run,
        stream_output=args.stream_output,
        force=args.force,
        benchmark_seeds=seeds,
    )
    aggregate = aggregate_summaries([summary])
    _write_json(out_dir / "aggregate_summary.json", _json_safe(aggregate))
    return aggregate, elapsed


def run_asset_probe(
    manifest: BenchmarkManifest,
    candidate: Candidate,
    candidate_paths: dict[str, Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float]:
    summary, _returncode, elapsed = run_rollout(
        manifest,
        candidate,
        candidate_paths,
        manifest.asset_probe_seed,
        out_dir,
        capture_images=False,
        dry_run=args.dry_run,
        stream_output=args.stream_output,
        force=args.force,
        steps_override=manifest.asset_probe_steps,
        extra_args=[
            "--door_twin_asset_probe",
            "--door_auto_open_force",
            str(manifest.asset_probe_force),
            "--door_open_resistance",
            "0",
            "--door_open_damping",
            "0",
        ],
    )
    return summary, elapsed


def dominant_failure(summary: dict[str, Any]) -> str:
    failures = [
        (str(name), int(count))
        for name, count in (summary.get("failure_counts", {}) or {}).items()
        if str(name) != "success" and int(count) > 0
    ]
    failures.sort(key=lambda item: (-item[1], item[0]))
    return failures[0][0] if failures else ""


def heuristic_repair(summary: dict[str, Any]) -> CombinedRepairPatch:
    proposals = heuristic_patches_for_failure(dominant_failure(summary))
    patch = proposals[0] if proposals else ProgramPatch({}, "no deterministic patch for this failure")
    return CombinedRepairPatch(skill_patch=patch, diagnostics=patch.diagnostics, failure_stage=dominant_failure(summary))


def prepare_initial_candidate(
    manifest: BenchmarkManifest,
    door: DoorCase,
    candidate_index: int,
    observer: OpenAIResponsesObserver,
    args: argparse.Namespace,
    catalog: ExperienceCatalog | None = None,
) -> tuple[Candidate, dict[str, Any]]:
    root = manifest.output_root / manifest.name / "initial_candidates" / door.id / f"candidate_{candidate_index:02d}"
    candidate_path = root / "candidate.json"
    if candidate_path.is_file() and not args.force:
        resumed = Candidate.read(candidate_path, manifest.base_door_cfg)
        generation = dict(resumed.metadata.get("initial_generation", {}) or {})
        if (
            generation.get("protocol") == OURS_INITIALIZATION_PROTOCOL
            and generation.get("mode") == args.initial_generation
        ):
            return resumed, {"resumed": True, **generation}

    # Every Ours branch starts from the exact deterministic Rule-based
    # candidate. The initial VLM call may only contribute a bounded residual
    # patch on top of this base; it does not independently regenerate the
    # complete asset/skill candidate.
    candidate = rule_based_candidate(manifest, door, candidate_index)
    rule_based_hash = candidate.fingerprint
    candidate.write(root / "rule_based_base")
    generation_info: dict[str, Any] = {
        "mode": args.initial_generation,
        "protocol": OURS_INITIALIZATION_PROTOCOL,
        "base_protocol": RULE_BASED_INITIALIZATION_PROTOCOL,
        "base_candidate_hash": rule_based_hash,
        "patch_semantics": "bounded_residual_on_rule_based_candidate",
    }
    prior_context: dict[str, Any] = {}
    if catalog is not None:
        snapshot = catalog.freeze(exclude_door_names=[door.door_name])
        matches = catalog.search(
            DoorSignature.from_candidate(candidate),
            query="generate push door skill",
            limit=3,
            exclude_door_names=[door.door_name],
            allowed_snapshot=snapshot,
        )
        prior_context = {
            "snapshot_fingerprint": snapshot["fingerprint"],
            "matches": [
                {
                    **match.to_dict(),
                    "skill": json.loads(match.record.skill_path.read_text(encoding="utf-8")),
                    "debug_excerpt": match.record.debug_doc_path.read_text(encoding="utf-8")[:12000],
                }
                for match in matches
            ],
        }
        _write_json(root / "prior_snapshot.json", snapshot)
    if args.initial_generation == "vlm":
        result = observer.observe(
            candidate,
            summary=None,
            include_images=False,
            initial_generation=True,
            candidate_nonce=candidate_index,
            prior_context=prior_context,
        )
        write_observer_result(root / "initial_vlm_response.json", result)
        candidate, patch_info = result.patch.apply(candidate)
        generation_info.update(
            {
                "request_fingerprint": result.request_fingerprint,
                "usage": result.usage,
                "elapsed_s": result.elapsed_s,
                "patch": patch_info,
            }
        )
    generation_info["residual_candidate_hash"] = candidate.fingerprint
    generation_info["retrieval"] = prior_context
    candidate.metadata["initial_generation"] = generation_info
    candidate.write(root)
    return candidate, generation_info


def run_method(
    manifest: BenchmarkManifest,
    door: DoorCase,
    initial: Candidate,
    method: str,
    observer: OpenAIResponsesObserver,
    initial_heldout: dict[str, Any],
    initial_eval_elapsed: float,
    initial_asset: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    branch = manifest.output_root / manifest.name / "branches" / door.id / f"candidate_{initial.candidate_index:02d}" / method
    result_path = branch / "branch_result.json"
    initialization_protocol = (
        RULE_BASED_INITIALIZATION_PROTOCOL if method == "rule_based" else OURS_INITIALIZATION_PROTOCOL
    )
    if result_path.is_file() and not args.force:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            cached.get("initial_hash") == initial.fingerprint
            and cached.get("initialization_protocol") == initialization_protocol
            and list(cached.get("heldout_seeds", [])) == manifest.heldout_seeds
            and (
                method in NO_DEVELOPMENT_METHODS
                or list(cached.get("development_seeds", [])) == manifest.development_seeds
            )
        ):
            return cached
    branch.mkdir(parents=True, exist_ok=True)
    current = initial.clone()
    best_candidate = initial.clone()
    best_summary: dict[str, Any] | None = None
    best_score: dict[str, Any] | None = None
    best_round: int | None = None
    rejected_candidate_hashes: set[str] = set()
    initial_hash = current.fingerprint
    rounds: list[dict[str, Any]] = []
    total_elapsed = float(initial_eval_elapsed) + float(initial.metadata.get("initial_generation", {}).get("elapsed_s", 0.0))
    initial_usage = dict(initial.metadata.get("initial_generation", {}).get("usage", {}) or {})
    initial_vlm_calls = int(initial.metadata.get("initial_generation", {}).get("mode") == "vlm")
    repair_vlm_calls = 0
    usage_records = [initial_usage] if initial_usage else []
    invalid_patch_count = 0
    repair_patch_attempts = 0

    if method not in NO_DEVELOPMENT_METHODS:
        # Evaluation 0 scores the initial candidate. Each accepted patch gets a
        # subsequent evaluation, so no unvalidated final patch can leak into
        # held-out scoring.
        for round_index in range(manifest.max_repair_rounds + 1):
            round_dir = branch / "development" / f"round_{round_index:02d}"
            candidate_paths = current.write(round_dir / "candidate")
            summary, elapsed = run_seed_set(
                manifest,
                current,
                candidate_paths,
                manifest.development_seeds,
                round_dir / "rollouts",
                capture_images=method in VISUAL_METHODS,
                args=args,
            )
            total_elapsed += elapsed
            score = development_score(summary)
            round_record: dict[str, Any] = {
                "round": round_index,
                "candidate_hash_before": current.fingerprint,
                "summary": stage_metrics(summary),
                "failure_counts": summary.get("failure_counts", {}),
                "development_score": score,
            }
            if best_score is None or _score_rank(score) > _score_rank(best_score):
                best_candidate = current.clone()
                best_summary = summary
                best_score = score
                best_round = round_index
                round_record["candidate_selection"] = "new_best"
            elif _score_rank(score) < _score_rank(best_score):
                rejected_hash = current.fingerprint
                rejected_candidate_hashes.add(rejected_hash)
                round_record.update(
                    {
                        "candidate_selection": "rejected_regression",
                        "rollback_from_hash": rejected_hash,
                        "rollback_to_hash": best_candidate.fingerprint,
                    }
                )
                current = best_candidate.clone()
                summary = best_summary if best_summary is not None else summary
            else:
                round_record["candidate_selection"] = "tied_best_kept_for_exploration"
            round_record.update(
                {
                    "best_hash_after_evaluation": best_candidate.fingerprint,
                    "best_round_after_evaluation": best_round,
                    "best_development_score_after_evaluation": best_score,
                }
            )
            if int(summary.get("success_count", 0)) == len(manifest.development_seeds):
                round_record["stop_reason"] = "all_development_rollouts_succeeded"
                rounds.append(round_record)
                break
            if method in VALIDATION_ONLY_METHODS:
                diagnosis = observer.observe(
                    current,
                    summary=summary,
                    include_images=method == "validation_only",
                    diagnose_only=True,
                    prior_context=dict(initial.metadata.get("initial_generation", {}).get("retrieval", {}) or {}),
                )
                repair_vlm_calls += 1
                usage_records.append(diagnosis.usage)
                total_elapsed += diagnosis.elapsed_s
                write_observer_result(round_dir / "diagnosis_vlm_response.json", diagnosis)
                round_record.update(
                    {
                        "diagnosis": diagnosis.patch.diagnostics,
                        "diagnosed_failure_stage": diagnosis.patch.failure_stage,
                        "candidate_hash_after": current.fingerprint,
                        "diagnosis_input": "logs_and_images" if method == "validation_only" else "logs_only",
                        "stop_reason": "validation_only_no_repair",
                    }
                )
                rounds.append(round_record)
                break
            if round_index >= manifest.max_repair_rounds:
                round_record["stop_reason"] = "repair_budget_exhausted_after_validated_evaluation"
                rounds.append(round_record)
                break
            if method == "without_vlm_fix":
                repair = heuristic_repair(summary)
                source = "deterministic_heuristic"
            else:
                repair_result = observer.observe(
                    current,
                    summary=summary,
                    include_images=method == "ours",
                    initial_generation=False,
                    candidate_nonce=round_index,
                    repair_history=_observer_repair_history(rounds, round_record),
                    prior_context=dict(initial.metadata.get("initial_generation", {}).get("retrieval", {}) or {}),
                )
                repair_vlm_calls += 1
                usage_records.append(repair_result.usage)
                total_elapsed += repair_result.elapsed_s
                write_observer_result(round_dir / "repair_vlm_response.json", repair_result)
                repair = repair_result.patch
                source = "vlm_images" if method == "ours" else "vlm_logs_only"
            next_candidate, patch_info = repair.apply(current)
            repair_patch_attempts += 1
            patch_invalid = bool(
                patch_info.get("asset", {}).get("rejected_fields", [])
                or patch_info.get("rejected_skill_fields", [])
            )
            if patch_invalid:
                invalid_patch_count += 1
            round_record.update({"repair_source": source, "patch_info": patch_info, "candidate_hash_after": next_candidate.fingerprint})
            if next_candidate.fingerprint in rejected_candidate_hashes:
                invalid_patch_count += 1
                round_record["stop_reason"] = "repeated_previously_rejected_candidate"
                rounds.append(round_record)
                break
            if next_candidate.fingerprint == current.fingerprint:
                if not patch_invalid:
                    invalid_patch_count += 1
                round_record["stop_reason"] = "no_valid_candidate_change"
                rounds.append(round_record)
                break
            rounds.append(round_record)
            current = next_candidate

    # Held-out evaluation always uses the best candidate observed on the fixed
    # development seeds, never merely the final attempted patch.
    current = best_candidate.clone()
    final_dir = branch / "final_candidate"
    final_paths = current.write(final_dir)
    if method in NO_DEVELOPMENT_METHODS:
        heldout = initial_heldout
        asset = initial_asset
    else:
        asset_probe, asset_probe_elapsed = run_asset_probe(manifest, current, final_paths, branch / "asset_probe", args)
        total_elapsed += asset_probe_elapsed
        heldout, elapsed = run_seed_set(
            manifest,
            current,
            final_paths,
            manifest.heldout_seeds,
            branch / "heldout",
            capture_images=False,
            args=args,
        )
        total_elapsed += elapsed
        asset = evaluate_asset(current, door, asset_probe, motion_fallback_summary=heldout)
    heldout_stage = stage_metrics(heldout)
    result = {
        "schema_version": "door_twin_agent_branch_result_v1",
        "benchmark": manifest.name,
        "door_id": door.id,
        "door_name": door.door_name,
        "candidate_index": initial.candidate_index,
        "method": method,
        "initialization_protocol": initialization_protocol,
        "rule_based_base_hash": (
            initial.fingerprint
            if method == "rule_based"
            else initial.metadata.get("initial_generation", {}).get("base_candidate_hash")
        ),
        "initial_hash": initial_hash,
        "final_hash": current.fingerprint,
        "rounds": rounds,
        "repair_rounds_executed": repair_patch_attempts,
        "development_evaluations": len(rounds),
        "best_development_round": best_round,
        "best_development_score": best_score,
        "best_development_hash": best_candidate.fingerprint,
        "candidate_selection_policy": "lexicographic_best_with_regression_rollback",
        "vlm_initial_calls": initial_vlm_calls,
        "vlm_repair_calls": repair_vlm_calls,
        "vlm_calls": initial_vlm_calls + repair_vlm_calls,
        "vlm_usage": _sum_usage(usage_records),
        "invalid_patch_count": invalid_patch_count,
        "repair_patch_attempts": repair_patch_attempts,
        "invalid_patch_ratio": invalid_patch_count / max(1, repair_patch_attempts),
        "elapsed_s": total_elapsed,
        "asset": asset,
        "initial_asset": initial_asset,
        "initial_heldout": stage_metrics(initial_heldout),
        "heldout": heldout_stage,
        "generation_success": bool(asset["structure_success"] and heldout_stage["success_count"] >= manifest.generation_success_min),
        "heldout_seeds": manifest.heldout_seeds,
        "development_seeds": [] if method in NO_DEVELOPMENT_METHODS else manifest.development_seeds,
    }
    _write_json(result_path, _json_safe(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--methods", default="", help="Comma-separated methods; defaults to the manifest method list")
    parser.add_argument("--doors", default="", help="Optional comma-separated DoorCase ids")
    parser.add_argument("--candidate_indices", default="", help="Optional comma-separated candidate indices")
    parser.add_argument("--initial_generation", choices=("vlm", "metadata"), default="vlm")
    parser.add_argument("--vlm_model", default="")
    parser.add_argument("--mock_vlm_response", default="")
    parser.add_argument(
        "--manual_vlm_dir",
        default="",
        help="File-exchange directory for a human/Codex observer; no API key is required",
    )
    parser.add_argument("--manual_vlm_timeout_s", type=float, default=86400.0)
    parser.add_argument(
        "--experience_catalog",
        default=str(Path(__file__).resolve().parents[1] / "experience" / "catalog.yaml"),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stream_output", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow_nonformal_manifest", action="store_true")
    parser.add_argument("--shard_id", default="", help="Unique label for a parallel door/candidate shard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock_vlm_response and args.manual_vlm_dir:
        raise ValueError("Use only one of --mock_vlm_response and --manual_vlm_dir")
    manifest = load_benchmark_manifest(args.manifest, formal=not args.allow_nonformal_manifest)
    selected_methods = [name for name in args.methods.split(",") if name] if args.methods else list(manifest.methods)
    unknown = sorted(set(selected_methods) - set(manifest.methods))
    if unknown:
        raise ValueError(f"Methods not enabled by manifest: {unknown}")
    selected_doors = set(filter(None, args.doors.split(",")))
    candidate_indices = (
        [int(value) for value in args.candidate_indices.split(",") if value]
        if args.candidate_indices
        else list(range(manifest.candidate_repeats))
    )
    observer = observer_from_manifest(manifest, args)
    catalog = ExperienceCatalog.load(args.experience_catalog)
    all_results = []
    for door in manifest.doors:
        if selected_doors and door.id not in selected_doors:
            continue
        for candidate_index in candidate_indices:
            initial, _generation = prepare_initial_candidate(manifest, door, candidate_index, observer, args, catalog)
            initial_eval_dir = (
                manifest.output_root
                / manifest.name
                / "initial_candidates"
                / door.id
                / f"candidate_{candidate_index:02d}"
                / "heldout_scoring_only"
            )
            initial_paths = initial.write(initial_eval_dir / "candidate")
            initial_heldout, initial_eval_elapsed = run_seed_set(
                manifest,
                initial,
                initial_paths,
                manifest.heldout_seeds,
                initial_eval_dir / "rollouts",
                capture_images=False,
                args=args,
            )
            initial_asset_probe, initial_asset_elapsed = run_asset_probe(
                manifest,
                initial,
                initial_paths,
                initial_eval_dir / "asset_probe",
                args,
            )
            initial_eval_elapsed += initial_asset_elapsed
            initial_asset = evaluate_asset(initial, door, initial_asset_probe, motion_fallback_summary=initial_heldout)
            for method in selected_methods:
                print(f"DoorTwin benchmark: door={door.id} candidate={candidate_index} method={method}", flush=True)
                try:
                    method_initial = initial
                    method_heldout = initial_heldout
                    method_elapsed = initial_eval_elapsed
                    method_asset = initial_asset
                    if method == "rule_based":
                        method_initial = rule_based_candidate(manifest, door, candidate_index)
                        rule_dir = (
                            manifest.output_root
                            / manifest.name
                            / "rule_based_candidates"
                            / door.id
                            / f"candidate_{candidate_index:02d}"
                        )
                        rule_paths = method_initial.write(rule_dir / "candidate")
                        method_heldout, method_elapsed = run_seed_set(
                            manifest,
                            method_initial,
                            rule_paths,
                            manifest.heldout_seeds,
                            rule_dir / "heldout_scoring_only",
                            capture_images=False,
                            args=args,
                        )
                        rule_probe, probe_elapsed = run_asset_probe(
                            manifest, method_initial, rule_paths, rule_dir / "asset_probe", args
                        )
                        method_elapsed += probe_elapsed
                        method_asset = evaluate_asset(
                            method_initial,
                            door,
                            rule_probe,
                            motion_fallback_summary=method_heldout,
                        )
                    all_results.append(
                        run_method(
                            manifest,
                            door,
                            method_initial,
                            method,
                            observer,
                            method_heldout,
                            method_elapsed,
                            method_asset,
                            args,
                        )
                    )
                except Exception as exc:
                    failure = {
                        "door_id": door.id,
                        "candidate_index": candidate_index,
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(
                        manifest.output_root
                        / manifest.name
                        / "branches"
                        / door.id
                        / f"candidate_{candidate_index:02d}"
                        / method
                        / "branch_error.json",
                        failure,
                    )
                    print(f"ERROR {failure}", file=sys.stderr, flush=True)
    shard_id = args.shard_id or str(os.getpid())
    _write_json(manifest.output_root / manifest.name / f"run_index_{shard_id}.json", {"results": all_results})
    print(f"Completed {len(all_results)} branch(es). Run the report script on {manifest.output_root / manifest.name}")


if __name__ == "__main__":
    main()
