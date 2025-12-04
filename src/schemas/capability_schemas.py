"""Schemas for capability generation stage (Stage 2).

Defines Capability dataclass for capability within an area. Capabilities
are specific skills or abilities.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.area_schemas import Area
from src.schemas.domain_schemas import Domain


@dataclass
class Capability:
    """Dataclass for capability."""

    name: str
    capability_id: str
    area: Area
    description: str
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "capability_id": self.capability_id,
            "area": self.area.name,
            "area_id": self.area.area_id,
            "area_description": self.area.description,
            "domain": self.area.domain.name,
            "domain_id": self.area.domain.domain_id,
            "description": self.description,
        }
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
        return cls(
            name=data["name"],
            capability_id=data["capability_id"],
            area=area,
            description=data["description"],
            generation_metadata=data.get("generation_metadata", {}),
        )
