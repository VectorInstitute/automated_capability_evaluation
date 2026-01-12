"""Schemas for domain (Stage 0).

Defines Domain dataclass for domain.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Domain:
    """Dataclass for domain."""

    domain_name: str
    domain_id: str
    domain_description: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "domain_name": self.domain_name,
            "domain_id": self.domain_id,
        }
        if self.domain_description is not None:
            result["domain_description"] = self.domain_description
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            domain_name=data["domain_name"],
            domain_id=data["domain_id"],
            domain_description=data.get("domain_description"),
        )
