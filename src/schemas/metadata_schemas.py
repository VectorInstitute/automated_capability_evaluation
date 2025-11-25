"""Metadata schemas for pipeline stages."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PipelineMetadata:
    """Standard metadata for all pipeline stage outputs."""

    experiment_id: str
    output_base_dir: str
    timestamp: str
    input_stage_tag: Optional[str] = None
    output_stage_tag: Optional[str] = None
    resume: bool = False

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self):
        """Convert to dictionary."""
        result = {
            "experiment_id": self.experiment_id,
            "output_base_dir": self.output_base_dir,
            "timestamp": self.timestamp,
            "resume": self.resume,
        }
        if self.input_stage_tag is not None:
            result["input_stage_tag"] = self.input_stage_tag
        if self.output_stage_tag is not None:
            result["output_stage_tag"] = self.output_stage_tag
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            output_base_dir=data["output_base_dir"],
            timestamp=data["timestamp"],
            input_stage_tag=data.get("input_stage_tag"),
            output_stage_tag=data.get("output_stage_tag"),
            resume=data.get("resume", False),
        )
