"""Schemas for validation stage (Stage 5).

Defines ValidationResult dataclass for validation result, including
verification status, feedback, and optional score.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.solution_schemas import TaskSolution


@dataclass
class ValidationResult:
    """Dataclass for validation result."""

    task_solution: TaskSolution
    verification: bool
    feedback: str
    score: Optional[float] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        """Get task_id from the task_solution for convenience."""
        return self.task_solution.task_id

    @property
    def task_text(self) -> str:
        """Get task text from the task_solution for convenience."""
        return self.task_solution.task_text

    def to_dict(self):
        """Convert to dictionary.

        Flattens the task_solution fields into the result for JSON serialization.
        """
        result = self.task_solution.to_dict()
        result["verification"] = self.verification
        result["feedback"] = self.feedback
        if self.score is not None:
            result["score"] = self.score
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        task_solution = TaskSolution.from_dict(data)
        return cls(
            task_solution=task_solution,
            verification=data["verification"],
            feedback=data["feedback"],
            score=data.get("score"),
            generation_metadata=data.get("generation_metadata", {}),
        )
