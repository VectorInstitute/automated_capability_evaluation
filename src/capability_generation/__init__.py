"""Capability generation package for multi-agent debate-based capability generation."""

from .messages import (
    Area,
    CapabilityProposalRequest,
    ScientistCapabilityProposal,
    CapabilityRevisionRequest,
)
from .generator import generate_capabilities, generate_capabilities_for_area
from .moderator import CapabilityModerator, normalize_capabilities
from .scientist import CapabilityScientist

__all__ = [
    "Area",
    "CapabilityProposalRequest",
    "ScientistCapabilityProposal",
    "CapabilityRevisionRequest",
    "CapabilityScientist",
    "CapabilityModerator",
    "normalize_capabilities",
    "generate_capabilities",
    "generate_capabilities_for_area",
] 