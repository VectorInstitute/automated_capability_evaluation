"""Schemas for experiment setup stage (Stage 0)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Experiment:
    """Represents experiment metadata and configuration."""

    experiment_id: str
    domain: str
    domain_id: str
    pipeline_type: Optional[str] = None
    configuration: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize configuration if not provided."""
        if self.configuration is None:
            self.configuration = {}

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "configuration": self.configuration,
        }
        if self.pipeline_type is not None:
            result["pipeline_type"] = self.pipeline_type
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            domain=data["domain"],
            domain_id=data["domain_id"],
            pipeline_type=data.get("pipeline_type"),
            configuration=data.get("configuration", {}),
        )


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
