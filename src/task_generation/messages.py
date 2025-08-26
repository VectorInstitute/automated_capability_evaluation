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


@dataclass
class ScientistProblemProposal:
    """Problem proposal from a scientist."""

    scientist_id: str
    capability_name: str
    problems: Dict[str, str]  # task_id -> task_text
    iteration: int


@dataclass
class ModeratorProblemReview:
    """Moderator's review and filtering of problems."""

    capability_name: str
    final_problems: Dict[str, str]  # task_id -> task_text
    rejected_problems: Dict[str, str]  # task_id -> rejection_reason
    iteration: int


@dataclass
class SolutionRequest:
    """Request for scientists to solve problems."""

    capability_name: str
    capability_description: str
    capability_domain: str
    capability_area: str
    problems: Dict[str, str]  # task_id -> task_text


@dataclass
class ScientistSolutionProposal:
    """Solution proposal from a scientist."""

    scientist_id: str
    capability_name: str
    solutions: Dict[str, str]  # task_id -> solution


@dataclass
class FinalTaskSet:
    """Final task set with problems and solutions."""

    capability_name: str
    tasks: Dict[str, Dict[str, str]]  # task_id -> {problem, answer}
