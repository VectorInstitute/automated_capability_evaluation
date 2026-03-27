"""Task solving module with debate-based approach."""

from __future__ import annotations

from typing import Any


def _missing_solve_tasks(*_args: Any, **_kwargs: Any) -> Any:
    """Raise a clear error when optional generator dependencies are missing."""
    raise ImportError(
        "src.task_solver.generator requires optional dependencies that are not "
        "installed in this environment."
    )


solve_tasks = _missing_solve_tasks

try:
    from .generator import solve_tasks as _solve_tasks
except Exception:
    pass
else:
    solve_tasks = _solve_tasks

__all__ = ["solve_tasks"]
