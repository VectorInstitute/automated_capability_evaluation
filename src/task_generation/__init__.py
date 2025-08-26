"""Task generation package for multi-agent debate-based task generation."""

from .generator import generate_tasks
from .moderator import TaskModerator
from .scientist import TaskScientist


__all__ = [
    "generate_tasks",
    "TaskModerator",
    "TaskScientist",
]
