"""Schemas for task generation stage."""

from dataclasses import dataclass


@dataclass
class Task:
    """Represents a task for a capability."""

    task_id: str
    task: str
    capability_id: str
    capability: str
    area: str
    area_id: str
    domain: str
    domain_id: str

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task": self.task,
            "capability_id": self.capability_id,
            "capability": self.capability,
            "area": self.area,
            "area_id": self.area_id,
            "domain": self.domain,
            "domain_id": self.domain_id,
        }
