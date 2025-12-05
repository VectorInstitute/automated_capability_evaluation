"""Schemas for task generation stage (Stage 3).

Defines Task dataclass for task. Tasks are concrete evaluation items
that test a capability.
"""

from dataclasses import dataclass

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain


@dataclass
class Task:
    """Dataclass for task."""

    task_id: str
    task: str
    capability: Capability

    def to_dict(self):
        """Convert to dictionary."""
        return {
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
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            capability=capability,
        )
