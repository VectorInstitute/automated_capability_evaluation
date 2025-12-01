"""Schemas for domain (Stage 0).

Defines Domain dataclass representing the domain being evaluated in the experiment.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Domain:
    """Represents a domain."""

    name: str
    domain_id: str
    description: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "domain_id": self.domain_id,
        }
        if self.description is not None:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            name=data["name"],
            domain_id=data["domain_id"],
            description=data.get("description"),
        )
