"""Message types for the capability generation system."""

from dataclasses import dataclass


@dataclass
class Area:
    """A capability area with name and description."""

    name: str
    description: str


@dataclass
class CapabilityProposalRequest:
    """Initial request for capability proposals from scientists."""

    area_name: str
    area_description: str
    num_capabilities: int


@dataclass
class ScientistCapabilityProposal:
    """Capability proposal from a scientist."""

    scientist_id: str
    proposal: str
    area_name: str
    round: int


@dataclass
class CapabilityRevisionRequest:
    """Request for scientist to review and revise moderator's proposal."""

    scientist_id: str
    moderator_proposal: str
    area_name: str
    round: int 