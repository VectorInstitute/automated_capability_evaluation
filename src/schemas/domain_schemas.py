"""Schemas for domain (Stage 0).

Defines Domain dataclass for domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Domain:
    """Dataclass for domain."""

    domain_name: str
    domain_id: str
    domain_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "domain_name": self.domain_name,
            "domain_id": self.domain_id,
        }
        if self.domain_description is not None:
            result["domain_description"] = self.domain_description
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Domain:
        """Create from dictionary."""
        return cls(
            domain_name=data["domain_name"],
            domain_id=data["domain_id"],
            domain_description=data.get("domain_description"),
        )
