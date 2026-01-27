"""Schemas for area generation stage (Stage 1).

Defines Area dataclass for domain area. Areas are high-level categories
within a domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.schemas.domain_schemas import Domain


@dataclass
class Area:
    """Dataclass for domain area."""

    area_name: str
    area_id: str
    domain: Domain
    area_description: str
    generation_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "area_name": self.area_name,
            "area_id": self.area_id,
            "area_description": self.area_description,
            **self.domain.to_dict(),
        }
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Area:
        """Create from dictionary."""
        domain = Domain.from_dict(data)
        return cls(
            area_name=data["area_name"],
            area_id=data["area_id"],
            domain=domain,
            area_description=data["area_description"],
            generation_metadata=data.get("generation_metadata", {}),
        )
