"""Schemas for capability generation stage (Stage 2).

Defines Capability dataclass for capability within an area. Capabilities
are specific skills or abilities (e.g., "Budget Creation" within "Budgeting" area).
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
    description: Optional[str] = None
    area: Optional[Area] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "capability_id": self.capability_id,
        }
        if self.area is not None:
            result["area"] = self.area.name
            result["area_id"] = self.area.area_id
            if self.area.domain is not None:
                result["domain"] = self.area.domain.name
                result["domain_id"] = self.area.domain.domain_id
        if self.description is not None:
            result["description"] = self.description
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
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
        return cls(
            name=data["name"],
            capability_id=data["capability_id"],
            description=data.get("description"),
            area=area,
            generation_metadata=data.get("generation_metadata", {}),
        )
