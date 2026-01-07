"""Schemas for task generation stage (Stage 3).

Defines Task dataclass for task. Tasks are concrete evaluation items
that test a capability.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.capability_schemas import Capability


@dataclass
class Task:
    """Dataclass for task."""

    task_id: str
    task: str
    capability: Capability
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "capability_id": self.capability.capability_id,
            "capability": self.capability.name,
            "capability_description": self.capability.description,
            "area": self.capability.area.name,
            "area_id": self.capability.area.area_id,
            "area_description": self.capability.area.description,
            "domain": self.capability.area.domain.name,
            "domain_id": self.capability.area.domain.domain_id,
        }
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        capability = Capability.from_dict(
            {
                "name": data["capability"],
                "capability_id": data["capability_id"],
                "description": data["capability_description"],
                "area": data["area"],
                "area_id": data["area_id"],
                "area_description": data["area_description"],
                "domain": data["domain"],
                "domain_id": data["domain_id"],
                "domain_description": data.get("domain_description"),
            }
        )
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            capability=capability,
            generation_metadata=data.get("generation_metadata", {}),
        )
