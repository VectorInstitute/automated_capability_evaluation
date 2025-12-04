"""Schemas for solution generation stage (Stage 4).

Defines TaskSolution dataclass for task solution, including solution text,
reasoning, and optional numerical answer.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
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
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "solution": self.solution,
            "reasoning": self.reasoning,
            "capability_id": self.task_obj.capability.capability_id,
            "capability": self.task_obj.capability.name,
            "capability_description": self.task_obj.capability.description,
            "area": self.task_obj.capability.area.name,
            "area_id": self.task_obj.capability.area.area_id,
            "area_description": self.task_obj.capability.area.description,
            "domain": self.task_obj.capability.area.domain.name,
            "domain_id": self.task_obj.capability.area.domain.domain_id,
        }
        if self.numerical_answer is not None:
            result["numerical_answer"] = self.numerical_answer
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        domain = Domain(
            name=data["domain"],
            domain_id=data["domain_id"],
            description=data.get("domain_description"),
        )
        area = Area(
            name=data["area"],
            area_id=data["area_id"],
            domain=domain,
            description=data["area_description"],
        )
        capability = Capability(
            name=data["capability"],
            capability_id=data["capability_id"],
            area=area,
            description=data["capability_description"],
        )
        task_obj = Task(
            task_id=data["task_id"],
            task=data["task"],
            capability=capability,
        )
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            solution=data["solution"],
            reasoning=data["reasoning"],
            task_obj=task_obj,
            numerical_answer=data.get("numerical_answer"),
            generation_metadata=data.get("generation_metadata", {}),
        )
