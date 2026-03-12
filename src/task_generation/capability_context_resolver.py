"""Helpers for resolving capabilities and chapter context into generation units."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
from src.schemas.io_utils import load_capabilities


logger = logging.getLogger(__name__)


@dataclass
class GenerationUnit:
    """Unit of generation with persisted capability lineage and chapter context."""

    capability: Capability
    context_text: str
    chapter_id: str
    chapter_relpath: str
    book_name: str


def get_book_name_from_relpath(chapter_relpath: str) -> str:
    """Return the top-level book directory name from a chapter relative path."""
    parts = chapter_relpath.split("/")
    return parts[0] if parts else "unknown_book"


def _stable_slug(value: str) -> str:
    """Create a short stable slug suitable for schema ids."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "unknown"


def _stable_short_hash(value: str, length: int = 8) -> str:
    """Return a short stable hash for compact unique ids."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def build_placeholder_capability(
    *,
    chapter_id: str,
    chapter_index: int,
    chapter_relpath: str,
    domain_name: str,
    stage2_capabilities_tag: Optional[str] = None,
    stage2_loaded_area_ids: Optional[List[str]] = None,
    stage2_loaded_capability_ids: Optional[List[str]] = None,
) -> Capability:
    """Schema-valid chapter-derived placeholder capability with stable unique ids."""
    slug = _stable_slug(chapter_id)
    short_hash = _stable_short_hash(chapter_relpath)
    domain = Domain(
        domain_name=domain_name or "unknown_domain",
        domain_id="domain_000",
        domain_description="Chapter-derived placeholder domain for task-generation runner",
    )
    area = Area(
        area_name=f"placeholder_area_{chapter_id}",
        area_id=f"area_ch_{slug}_{short_hash}",
        domain=domain,
        area_description="Chapter-derived placeholder area for task-generation runner",
    )
    return Capability(
        capability_name=f"placeholder_capability_{chapter_id}",
        capability_id=f"cap_ch_{slug}_{short_hash}",
        area=area,
        capability_description="Chapter-derived placeholder capability for task-generation runner",
        generation_metadata={
            "lineage_mode": "chapter_derived_placeholder",
            "source_chapter_id": chapter_id,
            "source_chapter_relpath": chapter_relpath,
            "source_chapter_index": chapter_index,
            "stage2_capabilities_tag": stage2_capabilities_tag,
            "stage2_loaded_area_ids": stage2_loaded_area_ids or [],
            "stage2_loaded_capability_ids": stage2_loaded_capability_ids or [],
        },
    )


def load_stage2_capability_artifacts(
    *,
    output_base_dir: Path,
    experiment_id: str,
    capabilities_tag: Optional[str],
) -> Tuple[List[Capability], List[str], List[str]]:
    """Load Stage-2 capability artifacts for lineage/traceability if available."""
    if not capabilities_tag or capabilities_tag == "placeholder_capabilities_tag":
        return [], [], []

    capabilities_base_dir = (
        output_base_dir / experiment_id / "capabilities" / capabilities_tag
    )
    if not capabilities_base_dir.exists():
        logger.warning(
            "Stage-2 capability directory not found for capabilities_tag=%s at %s",
            capabilities_tag,
            capabilities_base_dir,
        )
        return [], [], []

    loaded_capabilities: List[Capability] = []
    loaded_area_ids: List[str] = []
    loaded_capability_ids: List[str] = []

    for area_dir in sorted(d for d in capabilities_base_dir.iterdir() if d.is_dir()):
        capabilities_path = area_dir / "capabilities.json"
        if not capabilities_path.exists():
            continue
        capabilities, _ = load_capabilities(capabilities_path)
        loaded_capabilities.extend(capabilities)
        loaded_area_ids.append(area_dir.name)
        loaded_capability_ids.extend(cap.capability_id for cap in capabilities)

    return loaded_capabilities, loaded_area_ids, loaded_capability_ids


def load_capability_chapter_mapping(
    mapping_path: Optional[Path],
) -> Dict[str, List[str]]:
    """Load optional capability-to-chapter mapping config."""
    if mapping_path is None:
        return {}
    if not mapping_path.exists():
        raise FileNotFoundError(f"Capability chapter mapping not found: {mapping_path}")

    raw: Any = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Capability chapter mapping must be a JSON object: {mapping_path}"
        )

    normalized: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            normalized[key] = [value]
        elif isinstance(value, list):
            normalized[key] = [str(v) for v in value if str(v).strip()]
    return normalized


def resolve_chapter_paths_for_capability(
    capability: Capability,
    chapter_by_relpath: Dict[str, Path],
    capability_chapter_mapping: Dict[str, List[str]],
) -> List[Path]:
    """Resolve chapter files for a Stage-2 capability with mapping/fallback behavior."""
    keys = [
        capability.capability_id,
        f"{capability.area.area_id}/{capability.capability_id}",
        capability.capability_name,
    ]
    for key in keys:
        relpaths = capability_chapter_mapping.get(key)
        if not relpaths:
            continue
        resolved: List[Path] = []
        missing: List[str] = []
        for relpath in relpaths:
            p = chapter_by_relpath.get(relpath)
            if p is None:
                missing.append(relpath)
            else:
                resolved.append(p)
        if missing:
            raise FileNotFoundError(
                "Capability chapter mapping references missing chapter file(s) for "
                f"{capability.capability_id}: {missing}"
            )
        if resolved:
            return resolved

    chapter_paths = sorted(chapter_by_relpath.values())
    if len(chapter_paths) == 1:
        return chapter_paths
    return chapter_paths


def build_context_bundle_from_chapters(
    chapter_paths: List[Path],
    *,
    chapter_root_dir: Path,
) -> Tuple[str, str, str, str]:
    """Combine chapter texts into a single context bundle for generation."""
    if not chapter_paths:
        raise ValueError("At least one chapter path is required to build context.")

    relpaths = [p.relative_to(chapter_root_dir).as_posix() for p in chapter_paths]
    chapter_ids = [p.stem for p in chapter_paths]
    texts = [p.read_text(encoding="utf-8") for p in chapter_paths]

    if len(chapter_paths) == 1:
        return (
            texts[0],
            chapter_ids[0],
            relpaths[0],
            get_book_name_from_relpath(relpaths[0]),
        )

    sections = [
        f"[Source Chapter: {relpath}]\n{text}" for relpath, text in zip(relpaths, texts)
    ]
    context_text = "\n\n".join(sections)
    chapter_id = "__".join(chapter_ids)
    chapter_relpath = "__".join(relpaths)
    book_names = sorted({get_book_name_from_relpath(relpath) for relpath in relpaths})
    book_name = book_names[0] if len(book_names) == 1 else "multiple_books"
    return context_text, chapter_id, chapter_relpath, book_name


def prepare_generation_units(
    *,
    chapter_files: List[Path],
    chapter_root_dir: Path,
    stage2_capabilities: List[Capability],
    blueprint_domain: str,
    capabilities_input_tag: str,
    stage2_area_ids: List[str],
    stage2_capability_ids: List[str],
    capability_chapter_mapping: Dict[str, List[str]],
) -> List[GenerationUnit]:
    """Create generation units from either Stage-2 capabilities or placeholders."""
    chapter_by_relpath = {
        path.relative_to(chapter_root_dir).as_posix(): path for path in chapter_files
    }
    generation_units: List[GenerationUnit] = []

    if stage2_capabilities:
        for capability in stage2_capabilities:
            matched_chapter_paths = resolve_chapter_paths_for_capability(
                capability,
                chapter_by_relpath,
                capability_chapter_mapping,
            )
            (
                context_text,
                chapter_id,
                chapter_relpath,
                book_name,
            ) = build_context_bundle_from_chapters(
                matched_chapter_paths,
                chapter_root_dir=chapter_root_dir,
            )
            generation_units.append(
                GenerationUnit(
                    capability=capability,
                    context_text=context_text,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    book_name=book_name,
                )
            )
        return generation_units

    for chapter_idx, chapter_path in enumerate(chapter_files):
        chapter_relpath = chapter_path.relative_to(chapter_root_dir).as_posix()
        chapter_id = chapter_path.stem
        generation_units.append(
            GenerationUnit(
                capability=build_placeholder_capability(
                    chapter_id=chapter_id,
                    chapter_index=chapter_idx,
                    chapter_relpath=chapter_relpath,
                    domain_name=blueprint_domain,
                    stage2_capabilities_tag=capabilities_input_tag,
                    stage2_loaded_area_ids=stage2_area_ids,
                    stage2_loaded_capability_ids=stage2_capability_ids,
                ),
                context_text=chapter_path.read_text(encoding="utf-8"),
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                book_name=get_book_name_from_relpath(chapter_relpath),
            )
        )

    return generation_units
