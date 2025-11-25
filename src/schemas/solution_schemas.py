"""Schemas for solution generation stage."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TaskSolution:
    """Represents the complete solution for a task."""

    task_id: str
    task: str
    capability: str
    capability_id: str
    area: str
    area_id: str
    domain: str
    domain_id: str
    solution: str
    reasoning: str
    numerical_answer: Optional[str] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "capability": self.capability,
            "capability_id": self.capability_id,
            "area": self.area,
            "area_id": self.area_id,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "solution": self.solution,
            "reasoning": self.reasoning,
        }
        if self.numerical_answer is not None:
            result["numerical_answer"] = self.numerical_answer
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            capability=data["capability"],
            capability_id=data["capability_id"],
            area=data["area"],
            area_id=data["area_id"],
            domain=data["domain"],
            domain_id=data["domain_id"],
            solution=data["solution"],
            reasoning=data["reasoning"],
            numerical_answer=data.get("numerical_answer"),
            generation_metadata=data.get("generation_metadata", {}),
        )
