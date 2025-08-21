"""Capability generation package for multi-agent debate-based capability generation."""

from .generator import generate_capabilities, generate_capabilities_for_area
from .messages import (
    Area,
    CapabilityProposalRequest,
    CapabilityRevisionRequest,
    ScientistCapabilityProposal,
)
from .moderator import CapabilityModerator
from .scientist import CapabilityScientist


__all__ = [
    "Area",
    "CapabilityProposalRequest",
    "ScientistCapabilityProposal",
    "CapabilityRevisionRequest",
    "CapabilityScientist",
    "CapabilityModerator",
    "generate_capabilities",
    "generate_capabilities_for_area",
]
