#!/usr/bin/env python3
"""Build CSV/JSON/Markdown statistics for a DoorTwin agent benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from door_twin.benchmark.schema import (  # noqa: E402
    METHOD_LABELS,
    OURS_INITIALIZATION_PROTOCOL,
    RULE_BASED_INITIALIZATION_PROTOCOL,
    load_benchmark_manifest,
)

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - optional in lightweight report environments
    wilcoxon = None


def load_results(run_root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((run_root / "branches").glob("*/candidate_*/*/branch_result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["result_path"] = str(path.resolve())
        results.append(result)
    return results


def audit_invariants(run_root: Path, results: list[dict[str, Any]], manifest: Any) -> list[str]:
    violations: list[str] = []
    hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    rule_hashes: dict[tuple[str, int], str] = {}
    for item in results:
        key = (str(item["door_id"]), int(item["candidate_index"]))
        method = str(item["method"])
        if method != "rule_based":
            hashes[key].add(str(item.get("initial_hash", "")))
            if item.get("initialization_protocol") != OURS_INITIALIZATION_PROTOCOL:
                violations.append(f"{key}/{method}: Ours branch did not use the shared Rule-based + residual protocol")
        else:
            rule_hashes[key] = str(item.get("initial_hash", ""))
            if item.get("initialization_protocol") != RULE_BASED_INITIALIZATION_PROTOCOL:
                violations.append(f"{key}/{method}: Rule-based branch used the wrong initialization protocol")
        if list(item.get("heldout_seeds", [])) != list(manifest.heldout_seeds):
            violations.append(f"{key}/{method}: held-out seeds differ from manifest")
        expected_dev = [] if method in ("without_simulation_rollout", "rule_based") else list(manifest.development_seeds)
        if list(item.get("development_seeds", [])) != expected_dev:
            violations.append(f"{key}/{method}: development seeds differ from protocol")
        if method in ("validation_only_log", "validation_only") and item.get("initial_hash") != item.get("final_hash"):
            violations.append(f"{key}/{method}: candidate changed in diagnosis-only branch")
        if method in ("without_simulation_rollout", "rule_based") and item.get("rounds"):
            violations.append(f"{key}/{method}: development rollout records exist")
        if method == "rule_based" and int(item.get("vlm_calls", 0)) != 0:
            violations.append(f"{key}/{method}: rule-based branch called the VLM")
        if method == "without_vlm_fix" and int(item.get("vlm_repair_calls", 0)) != 0:
            violations.append(f"{key}/{method}: repair-time VLM was called")
        selected_best = str(item.get("best_development_hash", ""))
        if selected_best and str(item.get("final_hash", "")) != selected_best:
            violations.append(f"{key}/{method}: final candidate is not the selected development-best candidate")
        branch = Path(item["result_path"]).parent
        if method in ("without_visual_feedback", "validation_only_log"):
            response_name = (
                "repair_vlm_response.json"
                if method == "without_visual_feedback"
                else "diagnosis_vlm_response.json"
            )
            for response_path in branch.glob(f"development/round_*/{response_name}"):
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("image_paths"):
                    violations.append(f"{key}/{method}: log-only request included image paths")
        if method in ("ours", "validation_only"):
            pattern = "repair_vlm_response.json" if method == "ours" else "diagnosis_vlm_response.json"
            for response_path in branch.glob(f"development/round_*/{pattern}"):
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if not response.get("image_paths"):
                    violations.append(f"{key}/{method}: visual observer call had no valid montage")
    for key, values in hashes.items():
        if len(values) != 1:
            violations.append(f"{key}: branches do not share one initial candidate hash")
    for item in results:
        method = str(item["method"])
        if method == "rule_based":
            continue
        key = (str(item["door_id"]), int(item["candidate_index"]))
        expected = rule_hashes.get(key)
        if expected and str(item.get("rule_based_base_hash", "")) != expected:
            violations.append(f"{key}/{method}: residual candidate was not derived from the paired Rule-based hash")
    return sorted(set(violations))


def bootstrap_ci(values: list[float], *, samples: int = 10000, seed: int = 615455575) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict[str, Any]:
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    discordant = b01 + b10
    if discordant == 0:
        p = 1.0
    else:
        lower = min(b01, b10)
        p = min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant))
    return {"ours_only": b10, "other_only": b01, "discordant": discordant, "p_value": p}


def paired_map(results: list[dict[str, Any]], field: str) -> dict[str, dict[tuple[str, int], Any]]:
    mapped: dict[str, dict[tuple[str, int], Any]] = defaultdict(dict)
    for result in results:
        mapped[result["method"]][(result["door_id"], int(result["candidate_index"]))] = result.get(field)
    return mapped


def method_summary(items: list[dict[str, Any]], generation_min: int) -> dict[str, Any]:
    generation = [float(bool(item.get("generation_success", False))) for item in items]
    asset_load = [float(bool(item.get("asset", {}).get("load_success", False))) for item in items]
    asset_structure = [float(bool(item.get("asset", {}).get("structure_success", False))) for item in items]
    initial_asset_structure = [float(bool(item.get("initial_asset", {}).get("structure_success", False))) for item in items]
    heldout_trials = sum(int(item.get("heldout", {}).get("trials", 0)) for item in items)
    heldout_success = sum(int(item.get("heldout", {}).get("success_count", 0)) for item in items)
    valid_items = [item for item in items if item.get("asset", {}).get("structure_success", False)]
    conditional_trials = sum(int(item.get("heldout", {}).get("trials", 0)) for item in valid_items)
    conditional_success = sum(int(item.get("heldout", {}).get("success_count", 0)) for item in valid_items)
    first_success = [int(item.get("initial_heldout", {}).get("success_count", 0)) for item in items]
    first_trials = [int(item.get("initial_heldout", {}).get("trials", 0)) for item in items]
    ci_low, ci_high = bootstrap_ci(generation)
    usage: dict[str, float] = defaultdict(float)
    for item in items:
        for key, value in (item.get("vlm_usage", {}) or {}).items():
            if isinstance(value, (int, float)):
                usage[str(key)] += float(value)
    return {
        "candidates": len(items),
        "asset_load_success_count": int(sum(asset_load)),
        "asset_load_rate": mean(asset_load) if asset_load else 0.0,
        "asset_structure_success_count": int(sum(asset_structure)),
        "asset_structure_rate": mean(asset_structure) if asset_structure else 0.0,
        "initial_asset_structure_success_count": int(sum(initial_asset_structure)),
        "initial_asset_structure_rate": mean(initial_asset_structure) if initial_asset_structure else 0.0,
        "first_rollout_success_count": sum(first_success),
        "first_rollout_trials": sum(first_trials),
        "first_rollout_success_rate": sum(first_success) / max(1, sum(first_trials)),
        "generation_success_count": int(sum(generation)),
        "generation_success_min_heldout": generation_min,
        "generation_success_rate": mean(generation) if generation else 0.0,
        "generation_success_ci95": [ci_low, ci_high],
        "heldout_success_count": heldout_success,
        "heldout_trials": heldout_trials,
        "heldout_rollout_success_rate": heldout_success / max(1, heldout_trials),
        "conditional_trajectory_success_count": conditional_success,
        "conditional_trajectory_trials": conditional_trials,
        "conditional_trajectory_success_rate": conditional_success / max(1, conditional_trials),
        "mean_repair_rounds": mean(float(item.get("repair_rounds_executed", 0)) for item in items) if items else 0.0,
        "mean_elapsed_s": mean(float(item.get("elapsed_s", 0.0)) for item in items) if items else 0.0,
        "vlm_calls": sum(int(item.get("vlm_calls", 0)) for item in items),
        "vlm_initial_calls": sum(int(item.get("vlm_initial_calls", 0)) for item in items),
        "vlm_repair_calls": sum(int(item.get("vlm_repair_calls", 0)) for item in items),
        "vlm_usage": dict(usage),
        "invalid_patch_count": sum(int(item.get("invalid_patch_count", 0)) for item in items),
        "repair_patch_attempts": sum(int(item.get("repair_patch_attempts", 0)) for item in items),
        "invalid_patch_ratio": (
            sum(int(item.get("invalid_patch_count", 0)) for item in items)
            / max(1, sum(int(item.get("repair_patch_attempts", 0)) for item in items))
        ),
        "grasp_rate": sum(int(item.get("heldout", {}).get("grasp_count", 0)) for item in items) / max(1, heldout_trials),
        "unlock_rate": sum(int(item.get("heldout", {}).get("unlock_count", 0)) for item in items) / max(1, heldout_trials),
        "open_rate": sum(int(item.get("heldout", {}).get("open_count", 0)) for item in items) / max(1, heldout_trials),
        "traverse_rate": sum(int(item.get("heldout", {}).get("traverse_count", 0)) for item in items) / max(1, heldout_trials),
    }


def write_candidate_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "door_id",
        "candidate_index",
        "method",
        "initial_hash",
        "final_hash",
        "asset_load_success",
        "initial_asset_structure_success",
        "asset_structure_success",
        "initial_heldout_success_count",
        "heldout_success_count",
        "heldout_trials",
        "generation_success",
        "repair_rounds_executed",
        "development_evaluations",
        "best_development_round",
        "best_development_hash",
        "vlm_calls",
        "invalid_patch_count",
        "elapsed_s",
        "result_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "door_id": item["door_id"],
                    "candidate_index": item["candidate_index"],
                    "method": item["method"],
                    "initial_hash": item["initial_hash"],
                    "final_hash": item["final_hash"],
                    "asset_load_success": item.get("asset", {}).get("load_success", False),
                    "initial_asset_structure_success": item.get("initial_asset", {}).get("structure_success", False),
                    "asset_structure_success": item.get("asset", {}).get("structure_success", False),
                    "initial_heldout_success_count": item.get("initial_heldout", {}).get("success_count", 0),
                    "heldout_success_count": item.get("heldout", {}).get("success_count", 0),
                    "heldout_trials": item.get("heldout", {}).get("trials", 0),
                    "generation_success": item.get("generation_success", False),
                    "repair_rounds_executed": item.get("repair_rounds_executed", 0),
                    "development_evaluations": item.get("development_evaluations", 0),
                    "best_development_round": item.get("best_development_round"),
                    "best_development_hash": item.get("best_development_hash", ""),
                    "vlm_calls": item.get("vlm_calls", 0),
                    "invalid_patch_count": item.get("invalid_patch_count", 0),
                    "elapsed_s": item.get("elapsed_s", 0.0),
                    "result_path": item["result_path"],
                }
            )


def write_diagnosis_template(path: Path, results: list[dict[str, Any]]) -> None:
    fields = ["door_id", "candidate_index", "method", "round", "predicted_failure", "reviewer_1", "reviewer_2", "adjudicated"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            if item["method"] not in (
                "ours",
                "without_visual_feedback",
                "validation_only_log",
                "validation_only",
            ):
                continue
            for round_record in item.get("rounds", []) or []:
                predicted = round_record.get("diagnosed_failure_stage") or round_record.get("patch_info", {}).get("failure_stage", "")
                writer.writerow(
                    {
                        "door_id": item["door_id"],
                        "candidate_index": item["candidate_index"],
                        "method": item["method"],
                        "round": round_record.get("round", 0),
                        "predicted_failure": predicted,
                        "reviewer_1": "",
                        "reviewer_2": "",
                        "adjudicated": "",
                    }
                )


def diagnosis_accuracy(path: Path) -> dict[str, Any]:
    by_method: dict[str, list[bool]] = defaultdict(list)
    reviewer_agreement: list[bool] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r1, r2 = str(row.get("reviewer_1", "")).strip(), str(row.get("reviewer_2", "")).strip()
            adjudicated = str(row.get("adjudicated", "")).strip()
            if r1 and r2:
                reviewer_agreement.append(r1 == r2)
            target = adjudicated or (r1 if r1 and r1 == r2 else "")
            if target:
                by_method[str(row.get("method", ""))].append(str(row.get("predicted_failure", "")).strip() == target)
    return {
        "reviewer_raw_agreement": mean(reviewer_agreement) if reviewer_agreement else None,
        "labeled_rows": sum(len(values) for values in by_method.values()),
        "methods": {
            method: {"correct": sum(values), "count": len(values), "accuracy": mean(values)}
            for method, values in sorted(by_method.items())
        },
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# DoorTwin Agent Ablation Benchmark",
        "",
        "## Protocol",
        "",
        f"- Doors: {summary['protocol']['doors']}",
        f"- Initial candidates per door: {summary['protocol']['candidate_repeats']}",
        f"- Development seeds: `{summary['protocol']['development_seeds']}`",
        f"- Held-out seeds: `{summary['protocol']['heldout_seeds']}`",
        f"- Maximum repair rounds: {summary['protocol']['max_repair_rounds']}",
        f"- Asset probe: seed {summary['protocol']['asset_probe_seed']}, {summary['protocol']['asset_probe_steps']} steps",
        f"- Generation success: at least {summary['protocol']['generation_success_min']}/{len(summary['protocol']['heldout_seeds'])} held-out successes and valid asset structure.",
        f"- Completed branches: {summary['completeness']['completed_branches']}/{summary['completeness']['expected_branches']}",
        "",
        "## Main results",
        "",
        "| Method | Initial asset | Final asset | Asset load | First rollout | Generation success (95% CI) | Held-out rollout | Conditional trajectory | Mean rounds | VLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        ci = values["generation_success_ci95"]
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} "
            f"| {values['initial_asset_structure_success_count']}/{values['candidates']} ({100*values['initial_asset_structure_rate']:.1f}%) "
            f"| {values['asset_structure_success_count']}/{values['candidates']} ({100*values['asset_structure_rate']:.1f}%) "
            f"| {values['asset_load_success_count']}/{values['candidates']} ({100*values['asset_load_rate']:.1f}%) "
            f"| {values['first_rollout_success_count']}/{values['first_rollout_trials']} ({100*values['first_rollout_success_rate']:.1f}%) "
            f"| {values['generation_success_count']}/{values['candidates']} ({100*values['generation_success_rate']:.1f}%, {100*ci[0]:.1f}–{100*ci[1]:.1f}) "
            f"| {values['heldout_success_count']}/{values['heldout_trials']} ({100*values['heldout_rollout_success_rate']:.1f}%) "
            f"| {values['conditional_trajectory_success_count']}/{values['conditional_trajectory_trials']} ({100*values['conditional_trajectory_success_rate']:.1f}%) "
            f"| {values['mean_repair_rounds']:.2f} | {values['vlm_calls']} |"
        )
    lines.extend(["", "## Stage success", "", "| Method | Grasp | Unlock | Open ≥80° | Traverse |", "|---|---:|---:|---:|---:|"])
    for method, values in summary["methods"].items():
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {100*values['grasp_rate']:.1f}% | {100*values['unlock_rate']:.1f}% "
            f"| {100*values['open_rate']:.1f}% | {100*values['traverse_rate']:.1f}% |"
        )
    lines.extend(["", "## Paired comparison against Ours", "", "```json", json.dumps(summary["paired_tests"], indent=2), "```", ""])
    if summary.get("diagnosis_accuracy"):
        lines.extend(["", "## Failure diagnosis", "", "```json", json.dumps(summary["diagnosis_accuracy"], indent=2), "```"])
    else:
        lines.append("Diagnostic accuracy is intentionally omitted until two blinded reviewer labels and adjudication are filled in `diagnosis_labels_template.csv`.")
    if summary.get("invariant_violations"):
        lines.extend(["", "## Protocol invariant violations", ""])
        lines.extend(f"- {value}" for value in summary["invariant_violations"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_root", default="", help="Defaults to manifest.output_root/manifest.name")
    parser.add_argument("--diagnosis_labels", default="", help="Completed blinded-review CSV generated from diagnosis_labels_template.csv")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when completeness or protocol invariants fail")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_benchmark_manifest(args.manifest, formal=False)
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else manifest.output_root / manifest.name
    results = load_results(run_root)
    if not results:
        raise RuntimeError(f"No branch_result.json files found under {run_root}")
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_method[result["method"]].append(result)
    methods = {method: method_summary(items, manifest.generation_success_min) for method, items in sorted(by_method.items())}

    paired_generation = paired_map(results, "generation_success")
    paired_rounds = paired_map(results, "repair_rounds_executed")
    paired_tests = {}
    ours = paired_generation.get("ours", {})
    for method, values in paired_generation.items():
        if method == "ours":
            continue
        keys = sorted(set(ours) & set(values))
        record = {"pairs": len(keys), "mcnemar": mcnemar_exact([bool(ours[k]) for k in keys], [bool(values[k]) for k in keys])}
        if wilcoxon is not None and keys:
            x = [float(paired_rounds.get("ours", {}).get(k, 0)) for k in keys]
            y = [float(paired_rounds.get(method, {}).get(k, 0)) for k in keys]
            if any(a != b for a, b in zip(x, y)):
                test = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                record["repair_rounds_wilcoxon"] = {"statistic": float(test.statistic), "p_value": float(test.pvalue)}
            else:
                record["repair_rounds_wilcoxon"] = {"statistic": 0.0, "p_value": 1.0}
        paired_tests[method] = record

    expected = len(manifest.doors) * manifest.candidate_repeats * len(manifest.methods)
    violations = audit_invariants(run_root, results, manifest)
    summary = {
        "schema_version": "door_twin_agent_benchmark_report_v1",
        "protocol": {
            "doors": len(manifest.doors),
            "candidate_repeats": manifest.candidate_repeats,
            "development_seeds": manifest.development_seeds,
            "heldout_seeds": manifest.heldout_seeds,
            "max_repair_rounds": manifest.max_repair_rounds,
            "generation_success_min": manifest.generation_success_min,
            "asset_probe_seed": manifest.asset_probe_seed,
            "asset_probe_steps": manifest.asset_probe_steps,
        },
        "completeness": {"expected_branches": expected, "completed_branches": len(results)},
        "invariant_violations": violations,
        "methods": methods,
        "paired_tests": paired_tests,
    }
    if args.diagnosis_labels:
        summary["diagnosis_accuracy"] = diagnosis_accuracy(Path(args.diagnosis_labels).expanduser().resolve())
    report_dir = run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_candidate_csv(report_dir / "candidate_results.csv", results)
    write_diagnosis_template(report_dir / "diagnosis_labels_template.csv", results)
    (report_dir / "REPORT.md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"Wrote DoorTwin benchmark report to {report_dir}")
    if args.strict and (len(results) != expected or violations):
        raise SystemExit(f"Strict benchmark audit failed: completed={len(results)}/{expected}, violations={len(violations)}")


if __name__ == "__main__":
    main()
