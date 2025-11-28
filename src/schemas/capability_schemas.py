"""Schemas for capability generation stage."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Capability:
    """Represents a capability within an area."""

    name: str
    capability_id: str
    description: Optional[str] = None
    area: str = ""
    area_id: str = ""
    domain: str = ""
    domain_id: str = ""
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "capability_id": self.capability_id,
            "area": self.area,
            "area_id": self.area_id,
            "domain": self.domain,
            "domain_id": self.domain_id,
        }
        if self.description is not None:
            result["description"] = self.description
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            name=data["name"],
            capability_id=data["capability_id"],
            description=data.get("description"),
            area=data.get("area", ""),
            area_id=data.get("area_id", ""),
            domain=data.get("domain", ""),
            domain_id=data.get("domain_id", ""),
            generation_metadata=data.get("generation_metadata", {}),
        )
