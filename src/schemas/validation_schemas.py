"""Schemas for validation stage (Stage 5).

Defines ValidationResult dataclass for validation result, including
verification status, feedback, and optional score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.schemas.solution_schemas import TaskSolution


@dataclass
class ValidationResult:
    """Dataclass for validation result."""

    task_solution: TaskSolution
    verification: bool
    feedback: str
    score: Optional[float] = None
    generation_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        """Get task_id from the task_solution for convenience."""
        return self.task_solution.task_id

    @property
    def task_statement(self) -> str:
        """Get task statement from the task_solution for convenience."""
        return self.task_solution.task_statement

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Flattens the task_solution fields into the result for JSON serialization.
        """
        result: Dict[str, Any] = self.task_solution.to_dict()
        result["verification"] = self.verification
        result["feedback"] = self.feedback
        if self.score is not None:
            result["score"] = self.score
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationResult:
        """Create from dictionary."""
        task_solution = TaskSolution.from_dict(data)
        return cls(
            task_solution=task_solution,
            verification=data["verification"],
            feedback=data["feedback"],
            score=data.get("score"),
            generation_metadata=data.get("generation_metadata", {}),
        )
