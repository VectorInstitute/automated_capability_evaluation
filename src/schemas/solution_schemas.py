"""Schemas for solution generation stage (Stage 4).

Defines TaskSolution dataclass for task solution, including solution text,
reasoning, and optional numerical answer.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.task_schemas import Task


@dataclass
class TaskSolution:
    """Dataclass for task solution."""

    task: Task
    solution: str
    reasoning: str
    numerical_answer: Optional[str] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        """Get task_id from the task object for convenience."""
        return self.task.task_id

    @property
    def task_statement(self) -> str:
        """Get task statement from the task object for convenience."""
        return self.task.task_statement

    def to_dict(self):
        """Convert to dictionary.

        Flattens the task object fields into the result for JSON serialization.
        """
        result = self.task.to_dict()
        result["solution"] = self.solution
        result["reasoning"] = self.reasoning
        if self.numerical_answer is not None:
            result["numerical_answer"] = self.numerical_answer
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        task = Task.from_dict(data)
        return cls(
            task=task,
            solution=data["solution"],
            reasoning=data["reasoning"],
            numerical_answer=data.get("numerical_answer"),
            generation_metadata=data.get("generation_metadata", {}),
        )
