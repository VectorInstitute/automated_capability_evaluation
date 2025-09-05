"""Message types and data classes for task generation."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Capability:
    """A capability with name, description, domain, and area."""

    name: str
    description: str
    domain: str
    area: str


@dataclass
class ProblemProposalRequest:
    """Request for problem proposals from scientists."""

    capability_name: str
    capability_description: str
    capability_domain: str
    capability_area: str
    num_problems: int
    sample_tasks: List[str]
    iteration: int = 1


@dataclass
class ScientistProblemProposal:
    """Problem proposal from a scientist."""

    scientist_id: str
    capability_name: str
    problems: Dict[str, str]  # task_id -> task_text
    iteration: int
