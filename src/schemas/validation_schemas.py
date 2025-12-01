"""Schemas for validation stage (Stage 5).

Defines ValidationResult dataclass representing the validation outcome for a task,
including verification status, feedback, and optional score.
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
    score: Optional[float] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)
    task_obj: Optional[Task] = None  # Full task object with hierarchy

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "verification": self.verification,
            "feedback": self.feedback,
        }
        if self.task_obj is not None and self.task_obj.capability is not None:
            result["capability_id"] = self.task_obj.capability.capability_id
            result["capability"] = self.task_obj.capability.name
            if self.task_obj.capability.area is not None:
                result["area"] = self.task_obj.capability.area.name
                result["area_id"] = self.task_obj.capability.area.area_id
                if self.task_obj.capability.area.domain is not None:
                    result["domain"] = self.task_obj.capability.area.domain.name
                    result["domain_id"] = self.task_obj.capability.area.domain.domain_id
        if self.score is not None:
            result["score"] = self.score
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        task_obj = None
        if "capability" in data and "capability_id" in data:
            area = None
            if "area" in data and "area_id" in data:
                domain = None
                if "domain" in data and "domain_id" in data:
                    domain = Domain(
                        name=data["domain"],
                        domain_id=data["domain_id"],
                        description=None,
                    )
                area = Area(
                    name=data["area"],
                    area_id=data["area_id"],
                    description=None,
                    domain=domain,
                )
            capability = Capability(
                name=data["capability"],
                capability_id=data["capability_id"],
                description=None,
                area=area,
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
            score=data.get("score"),
            generation_metadata=data.get("generation_metadata", {}),
            task_obj=task_obj,
        )
