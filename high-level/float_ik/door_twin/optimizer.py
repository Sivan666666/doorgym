"""Tiny local optimizer utilities for Door Digital Twin V1.

This module intentionally does not call a VLM. A VLM observer can write the
same ProgramPatch JSON that these helpers consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .skill import ProgramPatch, SkillProgram, heuristic_patches_for_failure, load_skill_program


def load_program_patch(path: str | Path) -> ProgramPatch:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return ProgramPatch.from_obj(json.load(f))


def apply_patch_file(program_path: str | Path, patch_path: str | Path, output_path: str | Path) -> SkillProgram:
    program = load_skill_program(program_path)
    patched = load_program_patch(patch_path).apply(program)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(patched.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return patched


def propose_patches_from_summary(summary: dict[str, Any]) -> list[ProgramPatch]:
    counts = dict(summary.get("failure_counts", {}) or {})
    failures = [(name, count) for name, count in counts.items() if name != "success" and int(count) > 0]
    failures.sort(key=lambda item: item[1], reverse=True)
    if not failures:
        return []
    return heuristic_patches_for_failure(failures[0][0])

