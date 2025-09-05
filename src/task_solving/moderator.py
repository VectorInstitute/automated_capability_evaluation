"""Task solving moderator agent for managing the debate process."""

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, List

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from langfuse import Langfuse

from src.task_solving.messages import (
    AgentRevisionRequest,
    AgentSolution,
    ConsensusCheck,
    FinalSolution,
    Task,
    TaskSolutionRequest,
)
from src.utils.agentic_prompts import (
    TASK_MODERATOR_CONSENSUS_PROMPT,
    TASK_MODERATOR_SYSTEM_MESSAGE,
)


log = logging.getLogger("task_solving.moderator")


@default_subscription
class TaskSolvingModerator(RoutedAgent):
    """Moderator that manages task solving debate and checks for consensus."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        num_solvers: int,
        max_rounds: int,
        output_dir: Path,
        langfuse_client: Langfuse = None,
    ) -> None:
        super().__init__("Task Solving Moderator")
        self._model_client = model_client
        self._num_solvers = num_solvers
        self._max_rounds = max_rounds
        self._output_dir = output_dir
        self._langfuse_client = langfuse_client
        
        # Track solutions by task_id and round
        self._solutions_buffer: Dict[str, Dict[int, List[AgentSolution]]] = {}
        self._current_round: Dict[str, int] = {}
        self._final_solutions: Dict[str, FinalSolution] = {}

    def _extract_consensus_components(self, response: str) -> tuple[bool, str, str]:
        """Extract consensus decision, solution, and reasoning from response."""
        consensus_match = re.search(r"CONSENSUS_REACHED:\s*(true|false)", response, re.IGNORECASE)
        solution_match = re.search(r"FINAL_SOLUTION:\s*(.*?)(?=REASONING:|$)", response, re.DOTALL | re.IGNORECASE)
        reasoning_match = re.search(r"REASONING:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        
        consensus_reached = consensus_match.group(1).lower() == "true" if consensus_match else False
        final_solution = solution_match.group(1).strip() if solution_match else "NONE"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return consensus_reached, final_solution, reasoning

    def _check_simple_consensus(self, solutions: List[AgentSolution]) -> tuple[bool, str]:
        """Simple consensus check - if all agents have the same final answer."""
        if not solutions:
            return False, ""
        
        # Extract final answers and normalize them
        answers = [sol.final_answer.strip().lower() for sol in solutions]
        
        # Check if all answers are the same
        if len(set(answers)) == 1:
            return True, solutions[0].final_answer
        
        return False, ""

    @message_handler
    async def handle_task(self, message: Task, ctx: MessageContext) -> None:
        """Handle a task and initiate the solving process."""
        with self._langfuse_client.start_as_current_span(
            name=f"moderator_handle_task_{message.task_id}"
        ) as span:
            try:
                msg = f"Moderator received task: {message.task_id}"
                log.info(msg)
                span.update(
                    metadata={
                        "task_received": msg,
                        "task_id": message.task_id,
                        "capability_id": message.capability_id,
                    }
                )

                # Initialize tracking for this task
                self._solutions_buffer[message.task_id] = {}
                self._current_round[message.task_id] = 1

                # Send initial solution request to all solvers
                await self.publish_message(
                    TaskSolutionRequest(task=message, round_number=1),
                    topic_id=DefaultTopicId(),
                )

                span.update(
                    metadata={"solution_request_sent": f"Round 1 solution request sent for task {message.task_id}"}
                )

            except Exception as e:
                error_msg = f"Error handling task {message.task_id}: {str(e)}"
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg})

    @message_handler
    async def handle_agent_solution(self, message: AgentSolution, ctx: MessageContext) -> None:
        """Handle solution from an agent."""
        with self._langfuse_client.start_as_current_span(
            name=f"moderator_handle_solution_{message.task_id}_round_{message.round_number}"
        ) as span:
            try:
                task_id = message.task_id
                round_num = message.round_number
                
                msg = f"Moderator received solution from agent {message.agent_id} for task {task_id}, round {round_num}"
                log.info(msg)
                span.update(
                    metadata={
                        "solution_received": msg,
                        "task_id": task_id,
                        "agent_id": message.agent_id,
                        "round": round_num,
                    }
                )

                # Initialize round buffer if needed
                if round_num not in self._solutions_buffer[task_id]:
                    self._solutions_buffer[task_id][round_num] = []

                # Add solution to buffer
                self._solutions_buffer[task_id][round_num].append(message)

                # Check if we have all solutions for this round
                if len(self._solutions_buffer[task_id][round_num]) == self._num_solvers:
                    await self._check_consensus_and_proceed(task_id, round_num, ctx)

                span.update(
                    metadata={
                        "solutions_collected": f"{len(self._solutions_buffer[task_id][round_num])}/{self._num_solvers} for round {round_num}"
                    }
                )

            except Exception as e:
                error_msg = f"Error handling solution from agent {message.agent_id}: {str(e)}"
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg})

    async def _check_consensus_and_proceed(self, task_id: str, round_num: int, ctx: MessageContext) -> None:
        """Check for consensus and either finalize or start next round."""
        with self._langfuse_client.start_as_current_span(
            name=f"moderator_consensus_check_{task_id}_round_{round_num}"
        ) as span:
            try:
                solutions = self._solutions_buffer[task_id][round_num]
                
                # First try simple consensus check
                simple_consensus, simple_solution = self._check_simple_consensus(solutions)
                
                if simple_consensus:
                    # Simple consensus reached
                    final_solution = FinalSolution(
                        task_id=task_id,
                        solution=simple_solution,
                        reasoning="All agents provided the same answer",
                        consensus_reached=True,
                        total_rounds=round_num,
                        all_solutions=self._get_all_solutions_for_task(task_id),
                    )
                    
                    self._final_solutions[task_id] = final_solution
                    await self._save_final_solution(final_solution)
                    
                    span.update(
                        metadata={
                            "consensus_reached": True,
                            "method": "simple",
                            "final_solution": simple_solution[:100],
                        }
                    )
                    return

                # If no simple consensus and we haven't reached max rounds, use LLM to check
                if round_num < self._max_rounds:
                    # Use LLM moderator to check for consensus
                    task_content = ""  # We need to get the original task content
                    # For now, let's get it from the first solution's context or we need to store it
                    
                    # Format solutions for LLM
                    all_solutions_text = "\n\n".join([
                        f"Agent {sol.agent_id}:\nReasoning: {sol.thought}\nFinal Answer: {sol.final_answer}"
                        for sol in solutions
                    ])
                    
                    prompt = TASK_MODERATOR_CONSENSUS_PROMPT.format(
                        problem_text=task_content,  # We need to store this from the original task
                        all_solutions=all_solutions_text
                    )
                    
                    system_message = SystemMessage(content=TASK_MODERATOR_SYSTEM_MESSAGE)
                    user_message = UserMessage(content=prompt, source="user")

                    response = await self._model_client.create(
                        messages=[system_message, user_message],
                        cancellation_token=ctx.cancellation_token,
                    )

                    consensus_reached, final_solution_text, reasoning = self._extract_consensus_components(response.content)
                    
                    if consensus_reached:
                        # LLM found consensus
                        final_solution = FinalSolution(
                            task_id=task_id,
                            solution=final_solution_text,
                            reasoning=reasoning,
                            consensus_reached=True,
                            total_rounds=round_num,
                            all_solutions=self._get_all_solutions_for_task(task_id),
                        )
                        
                        self._final_solutions[task_id] = final_solution
                        await self._save_final_solution(final_solution)
                        
                        span.update(
                            metadata={
                                "consensus_reached": True,
                                "method": "llm_moderator",
                                "final_solution": final_solution_text[:100],
                            }
                        )
                        return
                    else:
                        # No consensus, start next round
                        next_round = round_num + 1
                        self._current_round[task_id] = next_round
                        
                        # We need the original task to send revision requests
                        # For now, create a placeholder task
                        task = Task(task_id=task_id, task_content={"task": task_content}, capability_id="")
                        
                        await self.publish_message(
                            AgentRevisionRequest(
                                task=task,
                                other_solutions=solutions,
                                round_number=next_round,
                            ),
                            topic_id=DefaultTopicId(),
                        )
                        
                        span.update(
                            metadata={
                                "consensus_reached": False,
                                "next_round_started": next_round,
                            }
                        )
                else:
                    # Max rounds reached, no consensus
                    final_solution = FinalSolution(
                        task_id=task_id,
                        solution="No consensus reached",
                        reasoning=f"Maximum rounds ({self._max_rounds}) reached without consensus",
                        consensus_reached=False,
                        total_rounds=round_num,
                        all_solutions=self._get_all_solutions_for_task(task_id),
                    )
                    
                    self._final_solutions[task_id] = final_solution
                    await self._save_final_solution(final_solution)
                    
                    span.update(
                        metadata={
                            "consensus_reached": False,
                            "max_rounds_reached": True,
                        }
                    )

            except Exception as e:
                error_msg = f"Error checking consensus for task {task_id}: {str(e)}"
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg})

    def _get_all_solutions_for_task(self, task_id: str) -> List[AgentSolution]:
        """Get all solutions for a task across all rounds."""
        all_solutions = []
        for round_solutions in self._solutions_buffer[task_id].values():
            all_solutions.extend(round_solutions)
        return all_solutions

    async def _save_final_solution(self, final_solution: FinalSolution) -> None:
        """Save the final solution to a file."""
        try:
            output_file = self._output_dir / f"task_{final_solution.task_id}_solution.json"
            
            solution_data = {
                "task_id": final_solution.task_id,
                "solution": final_solution.solution,
                "reasoning": final_solution.reasoning,
                "consensus_reached": final_solution.consensus_reached,
                "total_rounds": final_solution.total_rounds,
                "all_solutions": [
                    {
                        "agent_id": sol.agent_id,
                        "thought": sol.thought,
                        "final_answer": sol.final_answer,
                        "round_number": sol.round_number,
                    }
                    for sol in final_solution.all_solutions
                ],
            }
            
            with open(output_file, "w") as f:
                json.dump(solution_data, f, indent=2)
            
            log.info(f"Saved final solution for task {final_solution.task_id} to {output_file}")
            
        except Exception as e:
            log.error(f"Error saving final solution for task {final_solution.task_id}: {str(e)}")
            log.error(traceback.format_exc()) 