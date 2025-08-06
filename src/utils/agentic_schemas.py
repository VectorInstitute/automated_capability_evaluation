"""Pydantic schemas for structured debate outputs."""

from typing import List

from pydantic import BaseModel


class Area(BaseModel):
    """Represents a capability area within a domain."""

    id: int
    name: str
    description: str = ""


class AreaBundle(BaseModel):
    """Bundle of capability areas with finalization status."""

    areas: List[Area]
    finalized: bool


class CapabilityJSON(BaseModel):
    """Represents a capability within an area."""

    id: int
    name: str
    description: str = ""


class CapabilityBundle(BaseModel):
    """Bundle of capabilities with finalization status."""

    capabilities: List[CapabilityJSON]
    finalized: bool


class TaskJSON(BaseModel):
    """Represents a task for evaluating a capability."""

    id: str
    problem: str
    answer: str
    difficulty: str = "medium"


class TaskBundle(BaseModel):
    """Bundle of tasks with finalization status."""

    tasks: List[TaskJSON]
    finalized: bool
