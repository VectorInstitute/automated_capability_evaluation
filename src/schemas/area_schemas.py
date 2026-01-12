"""Schemas for area generation stage (Stage 1).

Defines Area dataclass for domain area. Areas are high-level categories
within a domain.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.domain_schemas import Domain


@dataclass
class Area:
    """Dataclass for domain area."""

    name: str
    area_id: str
    domain: Domain
    description: str
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "area_id": self.area_id,
            "domain": self.domain.name,
            "domain_id": self.domain.domain_id,
            "description": self.description,
        }
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        domain = Domain.from_dict(
            {
                "name": data["domain"],
                "domain_id": data["domain_id"],
                "description": data.get("domain_description"),
            }
        )
        return cls(
            name=data["name"],
            area_id=data["area_id"],
            domain=domain,
            description=data["description"],
            generation_metadata=data.get("generation_metadata", {}),
        )
