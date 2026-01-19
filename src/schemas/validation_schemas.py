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

    task_id: str
    task: str
    task_solution: TaskSolution
    verification: bool
    feedback: str
    score: Optional[float] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
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
            task_id=data["task_id"],
            task=data["task"],
            task_solution=task_solution,
            verification=data["verification"],
            feedback=data["feedback"],
            score=data.get("score"),
            generation_metadata=data.get("generation_metadata", {}),
        )
