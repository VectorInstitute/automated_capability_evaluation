"""Message types for task solving debate system."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Task:
    """Task to be solved."""

    task_id: str
    problem: str
    capability_name: str
    area_name: str
    task_type: str = "unknown"


@dataclass
class TaskSolutionRequest:
    """Request to solve a task."""

    task_id: str
    problem: str
    capability_name: str
    area_name: str
    task_type: str = "unknown"
    round_number: int = 1
    moderator_guidance: str = ""


@dataclass
class AgentSolution:
    """Solution proposed by an agent."""

    agent_id: str
    task_id: str
    thought: str
    final_answer: str
    answer: str
    numerical_answer: str
    round_number: int
    capability_name: str
    area_name: str
    task_type: str = "unknown"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "thought": self.thought,
            "final_answer": self.final_answer,
            "answer": self.answer,
            "numerical_answer": self.numerical_answer,
            "round_number": str(self.round_number),
            "capability_name": self.capability_name,
            "area_name": self.area_name,
            "task_type": self.task_type,
        }


@dataclass
class AgentRevisionRequest:
    """Request for agent to revise solution based on other agents' solutions."""

    task_id: str
    problem: str
    capability_name: str
    area_name: str
    other_solutions: List[Dict[str, str]]
    round_number: int
    task_type: str = "unknown"


@dataclass
class ConsensusCheck:
    """Check if consensus has been reached."""

    task_id: str
    solutions: List[Dict[str, str]]
    round_number: int
    task_type: str = "unknown"


@dataclass
class FinalSolution:
    """Final solution for a task."""

    task_id: str
    capability_name: str
    area_name: str
    problem: str
    solution: str
    answer: str
    numerical_answer: str
    reasoning: str
    consensus_reached: bool
    total_rounds: int
    all_solutions: List[Dict[str, str]]
    task_type: str = "unknown"
