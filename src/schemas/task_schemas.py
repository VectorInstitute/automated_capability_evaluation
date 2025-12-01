"""Schemas for task generation stage (Stage 3).

Defines Task dataclass representing a specific task for a capability. Tasks are
concrete evaluation items that test a capability (e.g., "Create a monthly budget").
"""

from dataclasses import dataclass
from typing import Optional

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain


@dataclass
class Task:
    """Dataclass for task."""

    task_id: str
    task: str
    capability: Optional[Capability] = None

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "task": self.task,
        }
        if self.capability is not None:
            result["capability_id"] = self.capability.capability_id
            result["capability"] = self.capability.name
            if self.capability.area is not None:
                result["area"] = self.capability.area.name
                result["area_id"] = self.capability.area.area_id
                if self.capability.area.domain is not None:
                    result["domain"] = self.capability.area.domain.name
                    result["domain_id"] = self.capability.area.domain.domain_id
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        capability = None
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
        return cls(
            task_id=data["task_id"],
            task=data["task"],
            capability=capability,
        )
