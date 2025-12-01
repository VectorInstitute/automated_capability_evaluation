"""Schemas for area generation stage (Stage 1).

Defines Area dataclass representing a domain area. Areas are high-level categories
within a domain (e.g., "Budgeting" within "Personal Finance").
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.schemas.domain_schemas import Domain


@dataclass
class Area:
    """Dataclass for domain area."""

    name: str
    area_id: str
    description: Optional[str] = None
    domain: Optional[Domain] = None
    generation_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "area_id": self.area_id,
        }
        if self.domain is not None:
            result["domain"] = self.domain.name
            result["domain_id"] = self.domain.domain_id
        if self.description is not None:
            result["description"] = self.description
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        domain = None
        if "domain" in data and "domain_id" in data:
            domain = Domain(
                name=data["domain"],
                domain_id=data["domain_id"],
                description=None,
            )
        return cls(
            name=data["name"],
            area_id=data["area_id"],
            description=data.get("description"),
            domain=domain,
            generation_metadata=data.get("generation_metadata", {}),
        )
