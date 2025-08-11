"""Dataclasses for the area generation system."""

from dataclasses import dataclass


@dataclass
class Domain:
    """A domain of capability areas."""

    name: str


@dataclass
class AreaProposalRequest:
    """Initial request for area proposals from scientists."""

    domain: str
    num_areas: int


@dataclass
class ScientistAreaProposal:
    """Area proposal from a scientist."""

    scientist_id: str
    proposal: str
    round: int


@dataclass
class ModeratorMergeRequest:
    """Request for moderator to merge scientist proposals."""

    domain: str
    num_final_areas: int
    scientist_a_proposal: str
    scientist_b_proposal: str
    round: int


@dataclass
class ModeratorMergedProposal:
    """Merged proposal from moderator."""

    merged_proposal: str
    round: int
    is_finalized: bool


@dataclass
class ScientistRevisionRequest:
    """Request for scientist to review and revise moderator's proposal."""

    scientist_id: str
    moderator_proposal: str
    round: int


@dataclass
class FinalAreasResponse:
    """Final areas response."""

    areas: str
