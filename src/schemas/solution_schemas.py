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

    task_id: str
    task: str
    solution: str
    reasoning: str
    task_obj: Task
    numerical_answer: Optional[str] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = self.task_obj.to_dict()
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
        task_obj = Task.from_dict(data)
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            solution=data["solution"],
            reasoning=data["reasoning"],
            task_obj=task_obj,
            numerical_answer=data.get("numerical_answer"),
            generation_metadata=data.get("generation_metadata", {}),
        )
