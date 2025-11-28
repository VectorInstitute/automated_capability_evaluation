"""Schemas for area generation stage."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Area:
    """Represents a domain area."""

    name: str
    area_id: str
    description: Optional[str] = None
    domain: str = ""
    domain_id: str = ""
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
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
            area_id=data["area_id"],
            description=data.get("description"),
            domain=data.get("domain", ""),
            domain_id=data.get("domain_id", ""),
            generation_metadata=data.get("generation_metadata", {}),
        )
