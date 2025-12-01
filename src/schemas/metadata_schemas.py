"""Metadata schemas for pipeline stages.

This module defines PipelineMetadata, which provides execution context and traceability
for all pipeline stage outputs. It tracks experiment ID, timestamps, input/output
version tags, and resume state. Used by all save/load functions and serialized in
JSON output files.

Note: PipelineMetadata tracks execution context, not content (content identifiers are
in the data objects themselves).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PipelineMetadata:
    """Standard metadata for all pipeline stage outputs.

    Provides execution context, traceability, and resumability for pipeline stages.
    Included with every stage output to track which experiment produced it, when it was
    generated, which input version was used, and whether the run was resumed.

    Attributes
    ----------
        experiment_id: Unique identifier for the experiment.
        output_base_dir: Base directory path where all pipeline outputs are stored.
        timestamp: ISO 8601 formatted timestamp (e.g., "2025-11-06T12:00:00Z").
            Auto-generated if not provided.
        input_stage_tag: Optional tag for the input version from previous stage
            (e.g., "_20251009_122040"). None for Stage 0.
        output_stage_tag: Optional tag for this output version
            (e.g., "_20251009_131252"). None for Stage 0.
        resume: Boolean indicating if this run was resumed from a checkpoint.
    """

    experiment_id: str
    output_base_dir: str
    timestamp: str
    input_stage_tag: Optional[str] = None
    output_stage_tag: Optional[str] = None
    resume: bool = False

    def __post_init__(self):
        """Set default timestamp if not provided.

        Automatically generates a UTC timestamp in ISO 8601 format if not set.
        """
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self):
        """Convert metadata to dictionary for JSON serialization."""
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
        """Create PipelineMetadata from dictionary (e.g., loaded from JSON)."""
        return cls(
            experiment_id=data["experiment_id"],
            output_base_dir=data["output_base_dir"],
            timestamp=data["timestamp"],
            input_stage_tag=data.get("input_stage_tag"),
            output_stage_tag=data.get("output_stage_tag"),
            resume=data.get("resume", False),
        )
