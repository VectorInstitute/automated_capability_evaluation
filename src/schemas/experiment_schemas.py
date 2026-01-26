"""Schemas for experiment setup stage (Stage 0).

Defines Experiment dataclass containing experiment configuration and metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Experiment:
    """Dataclass for experiment metadata and configuration."""

    experiment_id: str
    domain: str
    domain_id: str
    pipeline_type: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "domain_id": self.domain_id,
            "configuration": self.configuration,
        }
        if self.pipeline_type is not None:
            result["pipeline_type"] = self.pipeline_type
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            domain=data["domain"],
            domain_id=data["domain_id"],
            pipeline_type=data.get("pipeline_type"),
            configuration=data.get("configuration", {}),
        )
