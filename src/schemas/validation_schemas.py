"""Schemas for validation stage."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ValidationResult:
    """Validation result for a single task."""

    task_id: str
    task: str
    capability: str
    capability_id: str
    area: str
    area_id: str
    domain: str
    domain_id: str
    verification: bool
    feedback: str
    score: Optional[float] = None
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
            "verification": self.verification,
            "feedback": self.feedback,
        }
        if self.score is not None:
            result["score"] = self.score
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
            verification=data["verification"],
            feedback=data["feedback"],
            score=data.get("score"),
            generation_metadata=data.get("generation_metadata", {}),
        )
