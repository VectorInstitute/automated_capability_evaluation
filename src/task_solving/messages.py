"""Message types for task solving debate system."""

from dataclasses import dataclass
from typing import Any, Dict, List

from autogen_core import BaseMessage


@dataclass
class Task(BaseMessage):
    """Task to be solved."""
    
    task_id: str
    task_content: Dict[str, Any]
    capability_id: str


@dataclass
class TaskSolutionRequest(BaseMessage):
    """Request to solve a task."""
    
    task: Task
    round_number: int = 1


@dataclass
class AgentSolution(BaseMessage):
    """Solution proposed by an agent."""
    
    agent_id: str
    task_id: str
    thought: str
    final_answer: str
    round_number: int


@dataclass
class AgentRevisionRequest(BaseMessage):
    """Request for agent to revise solution based on other agents' solutions."""
    
    task: Task
    other_solutions: List[AgentSolution]
    round_number: int


@dataclass
class ConsensusCheck(BaseMessage):
    """Check if consensus has been reached."""
    
    task_id: str
    solutions: List[AgentSolution]
    round_number: int


@dataclass
class FinalSolution(BaseMessage):
    """Final solution for a task."""
    
    task_id: str
    solution: str
    reasoning: str
    consensus_reached: bool
    total_rounds: int
    all_solutions: List[AgentSolution] 