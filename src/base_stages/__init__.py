"""Base (non-agentic) pipeline stage entrypoints.

Imports are intentionally lazy to avoid importing optional Stage-3 agentic
dependencies during unrelated module imports (for example doctest collection).
"""

from typing import Optional

from omegaconf import DictConfig


def run_stage0(cfg: DictConfig) -> None:
    """Run Stage 0 (setup)."""
    from src.base_stages.stage0_setup import run_stage0 as _run_stage0

    return _run_stage0(cfg)


def run_stage1(cfg: DictConfig) -> str:
    """Run Stage 1 (areas)."""
    from src.base_stages.stage1_areas import run_stage1 as _run_stage1

    return _run_stage1(cfg)


def run_stage2(
    cfg: DictConfig, areas_tag: str, capabilities_tag: Optional[str] = None
) -> str:
    """Run Stage 2 (capabilities)."""
    from src.base_stages.stage2_capabilities import run_stage2 as _run_stage2

    return _run_stage2(cfg, areas_tag, capabilities_tag)


def run_stage3(
    cfg: DictConfig, capabilities_tag: str, tasks_tag: Optional[str] = None
) -> str:
    """Run Stage 3 (tasks)."""
    from src.base_stages.stage3_tasks import run_stage3 as _run_stage3

    return _run_stage3(cfg, capabilities_tag, tasks_tag)


def run_stage4(
    cfg: DictConfig, tasks_tag: str, solution_tag: Optional[str] = None
) -> str:
    """Run Stage 4 (solutions)."""
    from src.base_stages.stage4_solutions import run_stage4 as _run_stage4

    return _run_stage4(cfg, tasks_tag, solution_tag)


def run_stage5(
    cfg: DictConfig, solution_tag: str, validation_tag: Optional[str] = None
) -> str:
    """Run Stage 5 (validation)."""
    from src.base_stages.stage5_validation import run_stage5 as _run_stage5

    return _run_stage5(cfg, solution_tag, validation_tag)


__all__ = [
    "run_stage0",
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "run_stage4",
    "run_stage5",
]
