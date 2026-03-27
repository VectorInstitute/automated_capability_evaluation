"""Task solving module with debate-based approach."""

try:
    from .generator import solve_tasks
except (ImportError, Exception):
    solve_tasks = None  # autogen_core/datasets not installed; self_contrast still works

__all__ = ["solve_tasks"]
