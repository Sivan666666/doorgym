"""Local orchestration helpers for the Door Digital Twin V1 loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .optimizer import propose_patches_from_summary
from .skill import SkillProgram


def load_rollout_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Rollout summary must be a JSON object: {path}")
    return data


def propose_next_program(program: SkillProgram, summary: dict[str, Any]) -> SkillProgram:
    patches = propose_patches_from_summary(summary)
    next_program = program
    for patch in patches:
        next_program = patch.apply(next_program)
    next_program.metadata["door_twin_source_summary"] = str(summary.get("summary_path", ""))
    next_program.metadata["door_twin_success_rate"] = float(summary.get("success_rate", 0.0))
    return next_program


def write_next_program(program: SkillProgram, summary_path: str | Path, output_path: str | Path) -> SkillProgram:
    summary = load_rollout_summary(summary_path)
    next_program = propose_next_program(program, summary)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(next_program.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return next_program
