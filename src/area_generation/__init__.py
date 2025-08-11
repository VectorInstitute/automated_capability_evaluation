"""Area generation package for multi-agent debate-based capability area generation."""

from .messages import (
    AreaProposalRequest,
    Domain,
    FinalAreasResponse,
    ModeratorMergedProposal,
    ModeratorMergeRequest,
    ScientistAreaProposal,
    ScientistRevisionRequest,
)
from .generator import generate_areas
from .moderator import AreaModerator
from .scientist import AreaScientist


__all__ = [
    "Domain",
    "AreaProposalRequest",
    "ScientistAreaProposal",
    "ModeratorMergeRequest",
    "ModeratorMergedProposal",
    "ScientistRevisionRequest",
    "FinalAreasResponse",
    "AreaScientist",
    "AreaModerator",
    "generate_areas",
]
