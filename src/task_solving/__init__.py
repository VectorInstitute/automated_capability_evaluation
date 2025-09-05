"""Task solving module with debate-based approach."""

from .generator import solve_tasks_with_debate, load_tasks_from_file
from .messages import Task, TaskSolutionRequest, AgentSolution, FinalSolution
from .moderator import TaskSolvingModerator
from .scientist import TaskSolvingScientist

__all__ = [
    "solve_tasks_with_debate",
    "load_tasks_from_file",
    "Task",
    "TaskSolutionRequest", 
    "AgentSolution",
    "FinalSolution",
    "TaskSolvingModerator",
    "TaskSolvingScientist",
] 