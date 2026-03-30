"""Output and persistence helpers for task-generation Stage 3."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.schemas.capability_schemas import Capability
from src.schemas.io_utils import save_tasks
from src.schemas.metadata_schemas import PipelineMetadata
from src.schemas.task_schemas import Task


def build_task_output_path(
    *,
    output_base_dir: Path,
    experiment_id: str,
    out_tag: str,
    capability: Capability,
) -> Path:
    """Build the standard Stage-3 tasks.json output path."""
    return (
        output_base_dir
        / experiment_id
        / "tasks"
        / out_tag
        / capability.area.area_id
        / capability.capability_id
        / "tasks.json"
    )


def build_checkpoint_path(
    *,
    chapter_out_path: Path,
    checkpoint_enabled: bool,
    checkpoint_dir_name: str,
    checkpoint_file_name: str,
) -> Optional[Path]:
    """Build and create the deterministic checkpoint path for a generation unit."""
    if not checkpoint_enabled:
        return None
    checkpoint_dir = chapter_out_path.parent / checkpoint_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / checkpoint_file_name


def build_pipeline_metadata(
    *,
    experiment_id: str,
    output_base_dir: Path,
    input_stage_tag: str,
    output_stage_tag: str,
    resume: bool,
    timestamp: Optional[datetime] = None,
) -> PipelineMetadata:
    """Create standard pipeline metadata for Stage-3 outputs/checkpoints."""
    ts = timestamp or datetime.now(timezone.utc)
    return PipelineMetadata(
        experiment_id=experiment_id,
        output_base_dir=str(output_base_dir),
        timestamp=ts.isoformat().replace("+00:00", "Z"),
        input_stage_tag=input_stage_tag,
        output_stage_tag=output_stage_tag,
        resume=resume,
    )


def write_json_artifact(
    output_path: Path,
    payload: Dict[str, Any],
) -> None:
    """Write a JSON artifact with standard UTF-8 formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_dedup_report(
    *,
    chapter_out_path: Path,
    report: Dict[str, Any],
) -> Path:
    """Write chapter-level dedup report and return its path."""
    report_path = chapter_out_path.parent / "dedup_report.json"
    write_json_artifact(report_path, report)
    return report_path


def write_verification_stats(
    *,
    chapter_out_path: Path,
    chapter_id: str,
    chapter_relpath: str,
    book_name: str,
    capability_id: str,
    area_id: str,
    verification_logs: List[Dict[str, Any]],
) -> Path:
    """Write verification stats and return the saved path."""
    stats_path = chapter_out_path.parent / "verification_stats.json"
    payload = {
        "chapter_id": chapter_id,
        "chapter_relpath": chapter_relpath,
        "book_name": book_name,
        "capability_id": capability_id,
        "area_id": area_id,
        "num_verifier_calls": len(verification_logs),
        "verification_logs": verification_logs,
    }
    write_json_artifact(stats_path, payload)
    return stats_path


def write_token_stats(
    *,
    chapter_out_path: Path,
    chapter_id: str,
    chapter_relpath: str,
    book_name: str,
    capability_id: str,
    area_id: str,
    token_usage_logs: List[Dict[str, Any]],
) -> Path:
    """Write token accounting stats and return the saved path."""
    stats_path = chapter_out_path.parent / "token_stats.json"

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    usage_available_calls = 0
    usage_missing_calls = 0
    by_stage: Dict[str, Dict[str, int]] = {}
    by_model_role: Dict[str, Dict[str, int]] = {}

    for entry in token_usage_logs:
        input_tokens = entry.get("input_tokens")
        output_tokens = entry.get("output_tokens")
        record_total = entry.get("total_tokens")
        usage_available = bool(entry.get("usage_available", False))

        if usage_available:
            safe_input = int(input_tokens or 0)
            safe_output = int(output_tokens or 0)
            safe_total = int(record_total or (safe_input + safe_output))
            usage_available_calls += 1
            total_input_tokens += safe_input
            total_output_tokens += safe_output
            total_tokens += safe_total
        else:
            safe_input = 0
            safe_output = 0
            safe_total = 0
            usage_missing_calls += 1

        stage = str(entry.get("stage") or "unknown")
        stage_bucket = by_stage.setdefault(
            stage,
            {
                "calls": 0,
                "usage_available_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        stage_bucket["calls"] += 1
        if usage_available:
            stage_bucket["usage_available_calls"] += 1
            stage_bucket["input_tokens"] += safe_input
            stage_bucket["output_tokens"] += safe_output
            stage_bucket["total_tokens"] += safe_total

        model_role = str(entry.get("model_role") or "unknown")
        role_bucket = by_model_role.setdefault(
            model_role,
            {
                "calls": 0,
                "usage_available_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        role_bucket["calls"] += 1
        if usage_available:
            role_bucket["usage_available_calls"] += 1
            role_bucket["input_tokens"] += safe_input
            role_bucket["output_tokens"] += safe_output
            role_bucket["total_tokens"] += safe_total

    payload = {
        "chapter_id": chapter_id,
        "chapter_relpath": chapter_relpath,
        "book_name": book_name,
        "capability_id": capability_id,
        "area_id": area_id,
        "num_model_calls": len(token_usage_logs),
        "token_usage_summary": {
            "usage_available_calls": usage_available_calls,
            "usage_missing_calls": usage_missing_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "by_stage": by_stage,
            "by_model_role": by_model_role,
        },
        "token_usage_logs": token_usage_logs,
    }
    write_json_artifact(stats_path, payload)
    return stats_path


def save_task_outputs(
    *,
    tasks: List[Task],
    discarded_tasks: Optional[List[Task]],
    metadata: PipelineMetadata,
    chapter_out_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """Save primary task outputs and optional discarded-task outputs."""
    save_tasks(tasks, metadata, chapter_out_path)
    discarded_path: Optional[Path] = None
    if discarded_tasks:
        discarded_path = chapter_out_path.parent / "discarded_tasks.json"
        save_tasks(discarded_tasks, metadata, discarded_path)
    return chapter_out_path, discarded_path
