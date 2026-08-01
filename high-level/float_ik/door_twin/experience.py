"""Structured DoorTwin experience catalog and deterministic retrieval.

The catalog is deliberately read-only during a benchmark run.  Normal agent
sessions may append validated learned records through :class:`ExperienceStore`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = "door_twin_experience_catalog_v1"
LEARNED_SCHEMA_VERSION = "door_twin_learned_experience_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _resolved(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", value.lower()) if len(token) >= 2}


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _extent(bounds: dict[str, Any], lo_key: str, hi_key: str) -> tuple[float, float, float] | None:
    lo, hi = _vector3(bounds.get(lo_key)), _vector3(bounds.get(hi_key))
    if lo is None or hi is None:
        return None
    return tuple(abs(b - a) for a, b in zip(lo, hi))


def infer_handle_orientation(handle_bounds: dict[str, Any]) -> str:
    extent = _extent(handle_bounds, "handle_min", "handle_max")
    if extent is None:
        return "unknown"
    axis = max(range(3), key=lambda index: extent[index])
    return ("horizontal_x", "horizontal_y", "vertical")[axis]


@dataclass(frozen=True)
class DoorSignature:
    mechanism_type: str = "hinged_lever_door"
    push_or_pull: str = "push"
    hinge_side: str = "unknown"
    hinge_axis: str = "unknown"
    handle_mobility: str = "movable"
    handle_orientation: str = "unknown"
    door_dimensions: tuple[float, float, float] | None = None
    handle_dimensions: tuple[float, float, float] | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DoorSignature":
        def dimensions(name: str) -> tuple[float, float, float] | None:
            vector = _vector3(value.get(name))
            return vector

        return cls(
            mechanism_type=str(value.get("mechanism_type", "hinged_lever_door")),
            push_or_pull=str(value.get("push_or_pull", "push")),
            hinge_side=str(value.get("hinge_side", "unknown")),
            hinge_axis=str(value.get("hinge_axis", "unknown")),
            handle_mobility=str(value.get("handle_mobility", "movable")),
            handle_orientation=str(value.get("handle_orientation", "unknown")),
            door_dimensions=dimensions("door_dimensions"),
            handle_dimensions=dimensions("handle_dimensions"),
            tags=tuple(str(tag) for tag in value.get("tags", []) or []),
        )

    @classmethod
    def from_candidate(cls, candidate: Any) -> "DoorSignature":
        runtime = dict(candidate.runtime_spec)
        structure = candidate.urdf_structure()
        requested = str(runtime.get("door_dof_name", ""))
        door_joint = next(
            (joint for joint in structure.get("joints", []) if joint.get("name") == requested),
            None,
        )
        axis = "unknown" if door_joint is None else str(door_joint.get("axis") or "unknown")
        return cls(
            mechanism_type="hinged_lever_door",
            push_or_pull="push",
            hinge_side=str(runtime.get("hinge_side", runtime.get("handle_side", "unknown"))),
            hinge_axis=axis,
            handle_mobility="movable" if str(runtime.get("handle_dof_name", "")) else "fixed",
            handle_orientation=infer_handle_orientation(candidate.handle_bounding),
            door_dimensions=_extent(candidate.bounding_box, "min", "max"),
            handle_dimensions=_extent(candidate.handle_bounding, "handle_min", "handle_max"),
            tags=tuple(str(tag) for tag in runtime.get("experience_tags", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("door_dimensions", "handle_dimensions", "tags"):
            if result[key] is not None:
                result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    door_name: str
    signature: DoorSignature
    skill_path: Path
    debug_doc_path: Path
    skill_sha256: str
    debug_doc_sha256: str
    initial_failures: tuple[str, ...] = ()
    accepted_repairs: tuple[str, ...] = ()
    rejected_repairs: tuple[str, ...] = ()
    final_validation: dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        skill = json.loads(self.skill_path.read_text(encoding="utf-8"))
        return " ".join(
            [
                self.experience_id,
                self.door_name,
                canonical_json(skill.get("metadata", {})),
                self.debug_doc_path.read_text(encoding="utf-8"),
                " ".join(self.signature.tags),
                " ".join(self.initial_failures),
                " ".join(self.accepted_repairs),
                " ".join(self.rejected_repairs),
            ]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "door_name": self.door_name,
            "door_signature": self.signature.to_dict(),
            "skill_path": str(self.skill_path),
            "skill_sha256": self.skill_sha256,
            "debug_doc_path": str(self.debug_doc_path),
            "debug_doc_sha256": self.debug_doc_sha256,
            "initial_failures": list(self.initial_failures),
            "accepted_repairs": list(self.accepted_repairs),
            "rejected_repairs": list(self.rejected_repairs),
            "final_validation": self.final_validation,
        }


@dataclass(frozen=True)
class RetrievalMatch:
    record: ExperienceRecord
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**self.record.summary(), "retrieval_score": self.score, "retrieval_reasons": list(self.reasons)}


class ExperienceCatalog:
    def __init__(self, path: str | Path, records: Iterable[ExperienceRecord]):
        self.path = Path(path).resolve()
        self.records = tuple(records)
        self._by_id = {record.experience_id: record for record in self.records}
        if len(self._by_id) != len(self.records):
            raise ValueError("Experience ids must be unique")

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceCatalog":
        catalog_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}: {catalog_path}")
        records = []
        for item in raw.get("experiences", []) or []:
            skill = _resolved(item["skill_path"], catalog_path.parent)
            doc = _resolved(item["debug_doc_path"], catalog_path.parent)
            if not skill.is_file() or not doc.is_file():
                raise FileNotFoundError(skill if not skill.is_file() else doc)
            expected_skill = str(item.get("skill_sha256", ""))
            expected_doc = str(item.get("debug_doc_sha256", ""))
            skill_hash, doc_hash = sha256_file(skill), sha256_file(doc)
            if expected_skill and expected_skill != skill_hash:
                raise ValueError(f"Skill hash mismatch for {item['experience_id']}")
            if expected_doc and expected_doc != doc_hash:
                raise ValueError(f"Debug-doc hash mismatch for {item['experience_id']}")
            records.append(
                ExperienceRecord(
                    experience_id=str(item["experience_id"]),
                    door_name=str(item.get("door_name", item["experience_id"])),
                    signature=DoorSignature.from_mapping(dict(item.get("door_signature", {}) or {})),
                    skill_path=skill,
                    debug_doc_path=doc,
                    skill_sha256=skill_hash,
                    debug_doc_sha256=doc_hash,
                    initial_failures=tuple(str(v) for v in item.get("initial_failures", []) or []),
                    accepted_repairs=tuple(str(v) for v in item.get("accepted_repairs", []) or []),
                    rejected_repairs=tuple(str(v) for v in item.get("rejected_repairs", []) or []),
                    final_validation=dict(item.get("final_validation", {}) or {}),
                )
            )
        learned_root = catalog_path.parent / "learned"
        for learned_path in sorted(learned_root.glob("*/experience.json")):
            item = json.loads(learned_path.read_text(encoding="utf-8"))
            skill = _resolved(item["skill_path"], learned_path.parent)
            doc = _resolved(item["debug_doc_path"], learned_path.parent)
            if not skill.is_file() or not doc.is_file():
                continue
            records.append(
                ExperienceRecord(
                    experience_id=str(item["experience_id"]),
                    door_name=str(item["door_name"]),
                    signature=DoorSignature.from_mapping(dict(item.get("door_signature", {}) or {})),
                    skill_path=skill,
                    debug_doc_path=doc,
                    skill_sha256=sha256_file(skill),
                    debug_doc_sha256=sha256_file(doc),
                    initial_failures=tuple(str(v) for v in item.get("initial_failures", []) or []),
                    accepted_repairs=tuple(str(v) for v in item.get("accepted_repairs", []) or []),
                    rejected_repairs=tuple(str(v) for v in item.get("rejected_repairs", []) or []),
                    final_validation=dict(item.get("final_validation", {}) or {}),
                )
            )
        return cls(catalog_path, records)

    def get(self, experience_id: str) -> ExperienceRecord:
        try:
            return self._by_id[str(experience_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown DoorTwin experience: {experience_id}") from exc

    def freeze(
        self,
        *,
        exclude_door_names: Iterable[str] = (),
        exclude_hashes: Iterable[str] = (),
    ) -> dict[str, Any]:
        excluded = {str(value) for value in exclude_door_names}
        hashes = {str(value) for value in exclude_hashes if str(value)}
        records = [
            record.summary()
            for record in self.records
            if record.door_name not in excluded
            and record.skill_sha256 not in hashes
            and record.debug_doc_sha256 not in hashes
        ]
        payload = {
            "schema_version": "door_twin_prior_snapshot_v1",
            "catalog": str(self.path),
            "excluded_door_names": sorted(excluded),
            "excluded_hashes": sorted(hashes),
            "records": records,
        }
        payload["fingerprint"] = sha256_json(payload)
        return payload

    def search(
        self,
        signature: DoorSignature,
        *,
        query: str = "",
        failure_stage: str = "",
        limit: int = 3,
        exclude_door_names: Iterable[str] = (),
        allowed_snapshot: dict[str, Any] | None = None,
    ) -> list[RetrievalMatch]:
        excluded = {str(value) for value in exclude_door_names}
        allowed_ids = None
        if allowed_snapshot is not None:
            allowed_ids = {str(item["experience_id"]) for item in allowed_snapshot.get("records", [])}
        query_tokens = _tokens(" ".join([query, failure_stage]))
        matches = []
        for record in self.records:
            if record.door_name in excluded or (allowed_ids is not None and record.experience_id not in allowed_ids):
                continue
            score, reasons = _signature_score(signature, record.signature)
            overlap = query_tokens & _tokens(record.searchable_text())
            if overlap:
                text_score = min(3.0, 0.35 * len(overlap))
                score += text_score
                reasons.append(f"text overlap: {', '.join(sorted(overlap)[:8])}")
            matches.append(RetrievalMatch(record, round(score, 6), tuple(reasons)))
        matches.sort(key=lambda match: (-match.score, match.record.experience_id))
        return matches[: max(1, int(limit))]


def _signature_score(query: DoorSignature, candidate: DoorSignature) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    weights = {
        "mechanism_type": 4.0,
        "push_or_pull": 4.0,
        "handle_mobility": 3.0,
        "handle_orientation": 3.0,
        "hinge_axis": 2.0,
        "hinge_side": 1.5,
    }
    for field_name, weight in weights.items():
        left, right = getattr(query, field_name), getattr(candidate, field_name)
        if left == right and left not in ("", "unknown"):
            score += weight
            reasons.append(f"{field_name}={left}")
    for field_name in ("door_dimensions", "handle_dimensions"):
        left, right = getattr(query, field_name), getattr(candidate, field_name)
        if left is None or right is None:
            continue
        relative = sum(abs(a - b) / max(abs(a), abs(b), 1.0e-3) for a, b in zip(left, right)) / 3.0
        similarity = max(0.0, 1.0 - relative)
        score += similarity
        if similarity >= 0.5:
            reasons.append(f"{field_name} similarity={similarity:.2f}")
    tag_overlap = set(query.tags) & set(candidate.tags)
    if tag_overlap:
        score += min(2.0, 0.5 * len(tag_overlap))
        reasons.append(f"tags={','.join(sorted(tag_overlap))}")
    return score, reasons


class ExperienceStore:
    """Append-only, content-addressed session/learned experience storage."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def append_repair(self, record: dict[str, Any]) -> tuple[str, Path]:
        payload = {"schema_version": "door_twin_repair_record_v1", **record}
        record_id = sha256_json(payload)
        payload["record_id"] = record_id
        path = self.root / "repairs" / f"{record_id}.json"
        existed = path.is_file()
        _atomic_json(path, payload)
        if not existed:
            with (self.root / "repair_records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        return record_id, path

    def promote(self, record: dict[str, Any]) -> tuple[str, Path]:
        payload = {"schema_version": LEARNED_SCHEMA_VERSION, **record}
        experience_id = sha256_json(payload)
        payload["experience_id"] = experience_id
        folder = self.root / "learned" / experience_id
        skill_path = folder / "skill_program.json"
        debug_path = folder / "debug.md"
        _atomic_json(skill_path, dict(payload["skill_program"]))
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            "# Learned DoorTwin experience\n\n"
            f"- door: `{payload.get('door_name', '')}`\n"
            f"- candidate: `{payload.get('candidate_hash', '')}`\n"
            f"- prior snapshot: `{payload.get('prior_snapshot_fingerprint', '')}`\n\n"
            "The candidate passed static/physics validation and the configured hidden rollout threshold.\n",
            encoding="utf-8",
        )
        payload["skill_path"] = "skill_program.json"
        payload["debug_doc_path"] = "debug.md"
        payload["final_validation"] = {"asset": payload.get("asset", {}), "heldout": payload.get("heldout", {})}
        path = folder / "experience.json"
        _atomic_json(path, payload)
        return experience_id, path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
