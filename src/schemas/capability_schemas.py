"""Schemas for capability generation stage (Stage 2).

Defines Capability dataclass for capability within an area. Capabilities
are specific skills or abilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.schemas.area_schemas import Area


@dataclass
class Capability:
    """Dataclass for capability."""

    capability_name: str
    capability_id: str
    area: Area
    capability_description: str
    generation_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "capability_name": self.capability_name,
            "capability_id": self.capability_id,
            "capability_description": self.capability_description,
            **self.area.to_dict(),
        }
        if self.generation_metadata:
            result["generation_metadata"] = self.generation_metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capability":
        """Create from dictionary."""
        area = Area.from_dict(data)
        return cls(
            capability_name=data["capability_name"],
            capability_id=data["capability_id"],
            area=area,
            capability_description=data["capability_description"],
            generation_metadata=data.get("generation_metadata", {}),
        )
