"""Schemas for validation stage (Stage 5).

Defines ValidationResult dataclass for validation result, including
verification status, feedback, and optional score.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
from src.schemas.task_schemas import Task


@dataclass
class ValidationResult:
    """Dataclass for validation result."""

    task_id: str
    task: str
    verification: bool
    feedback: str
    task_obj: Task
    score: Optional[float] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "verification": self.verification,
            "feedback": self.feedback,
            "capability_id": self.task_obj.capability.capability_id,
            "capability": self.task_obj.capability.name,
            "capability_description": self.task_obj.capability.description,
            "area": self.task_obj.capability.area.name,
            "area_id": self.task_obj.capability.area.area_id,
            "area_description": self.task_obj.capability.area.description,
            "domain": self.task_obj.capability.area.domain.name,
            "domain_id": self.task_obj.capability.area.domain.domain_id,
        }
        if self.score is not None:
            result["score"] = self.score
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
            verification=data["verification"],
            feedback=data["feedback"],
            task_obj=task_obj,
            score=data.get("score"),
            generation_metadata=data.get("generation_metadata", {}),
        )
