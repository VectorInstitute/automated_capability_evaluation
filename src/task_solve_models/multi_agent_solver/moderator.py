"""Task solver moderator agent for managing the debate process."""

import json
import logging
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

from src.task_solve_models.multi_agent_solver.messages import (
    AgentRevisionRequest,
    AgentSolution,
    FinalSolution,
    Task,
    TaskSolutionRequest,
)
from src.task_solve_models.multi_agent_solver.prompts import (
    TASK_MODERATOR_CONSENSUS_PROMPT,
    TASK_MODERATOR_SYSTEM_MESSAGE,
    TASK_MODERATOR_BREAKDOWN_PROMPT,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("task_solver.moderator")


@default_subscription
class TaskSolverModerator(RoutedAgent):
    """Moderator that manages task solver debate and checks for consensus.

    Attributes
    ----------
    _model_client : ChatCompletionClient
        ChatCompletionClient for LLM interactions.
    _num_solvers : int
        Number of solver agents participating in the debate.
    _max_rounds : int
        Maximum number of debate rounds allowed before forcing a conclusion.
    _output_dir : Path
        Directory path where final solutions are saved.
    _langfuse_client : Langfuse
        Langfuse client for tracing and logging debate activity.
    _solutions_buffer : Dict[int, List[AgentSolution]]
        Buffer storing solutions from all agents, keyed by task_id and
        organized by round number.
    _current_round : int
        Counter tracking the current debate round (0-indexed).
    _final_solutions : FinalSolution
        Storage for the final consensus solution once reached.
    _tasks : Task
        Original task data used for consensus checking and validation.
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        num_solvers: int,
        max_rounds: int,
        output_dir: Path,
        langfuse_client: Langfuse,
    ) -> None:
        super().__init__("Task Solver Moderator")
        self._model_client = model_client
        self._num_solvers = num_solvers
        self._max_rounds = max_rounds
        self._output_dir = output_dir
        self._langfuse_client = langfuse_client

        # Track solutions by task_id and round
        self._solutions_buffer: Dict[int, List[AgentSolution]]
        self._current_round = 0
        self._final_solutions: FinalSolution
        self._tasks: Task  # Store original tasks for consensus checking

    def _extract_consensus_components(
        self, response: str
    ) -> tuple[bool, str, str, str, str]:
        """Extract consensus, solution, answer, reasoning, and numerical answer from JSON."""
        try:
            parsed = parse_llm_json_response(response)
            consensus_reached = parsed.get("consensus_reached", False)
            final_solution = parsed.get("final_solution", "NONE")
            answer = parsed.get("answer", "NONE")
            reasoning = parsed.get("reasoning", "No reasoning provided")
            numerical_answer = parsed.get("numerical_answer")

            # Robustness: If the model nested the answer inside final_solution (common issue)
            if (answer == "NONE" or answer == "null") and isinstance(final_solution, str):
                try:
                    # Check if final_solution itself is a JSON string
                    if final_solution.strip().startswith("{"):
                        inner_parsed = parse_llm_json_response(final_solution)
                        if "answer" in inner_parsed and answer == "NONE":
                            answer = str(inner_parsed["answer"])
                        if "numerical_answer" in inner_parsed and numerical_answer is None:
                            numerical_answer = inner_parsed["numerical_answer"]
                        if "reasoning" in inner_parsed and (reasoning == "No reasoning provided" or not reasoning):
                            reasoning = inner_parsed["reasoning"]
                except Exception:
                    pass

            # Ensure final_solution is a string for storage
            if isinstance(final_solution, (dict, list)):
                final_solution = json.dumps(final_solution, ensure_ascii=False)
            else:
                final_solution = str(final_solution)

            # Convert numerical_answer to string representation
            if numerical_answer is not None:
                numerical_answer = str(numerical_answer)
            else:
                numerical_answer = "null"

            return consensus_reached, final_solution, answer, reasoning, numerical_answer

        except Exception as e:
            msg = f"Error extracting consensus components: {e}"
            log.error(msg)
            log.error(traceback.format_exc())
            raise

    def _check_simple_consensus(
        self, solutions: List[AgentSolution]
    ) -> tuple[bool, str, str, str]:
        """Check consensus; if all agents have the same final answer."""
        
        # FILTER OUT ERROR RESPONSES
        valid_solutions = []
        for sol in solutions:
            # Check for common error patterns in answer or final_answer
            is_error = False
            if "error" in sol.answer.lower() or "failed" in sol.answer.lower():
                is_error = True
            if "error" in sol.final_answer.lower() and len(sol.final_answer) < 100: # Short error message
                is_error = True
            
            if not is_error:
                valid_solutions.append(sol)
        
        # If we filtered out solutions, log it
        if len(valid_solutions) < len(solutions):
             log.warning(f"Filtered out {len(solutions) - len(valid_solutions)} error/failed solutions.")

        # If NO valid solutions, return failure
        if not valid_solutions:
            return False, "", "null", "null"
            
        def clean_answer(a: str) -> str:
            if not a: return ""
            a = a.strip().lower()
            # Remove trailing periods, stars, etc.
            import re
            a = re.sub(r'[^a-z0-9]$', '', a)
            return a

        # Check for strict consensus on the structured 'answer' field
        answers = [clean_answer(sol.answer) for sol in valid_solutions if sol.answer and clean_answer(sol.answer) != "no answer found"]
        
        if len(answers) == len(valid_solutions) and len(set(answers)) == 1:
             return True, valid_solutions[0].final_answer, valid_solutions[0].answer, valid_solutions[0].numerical_answer

        # First check numerical answers if they exist
        def clean_num(n: str) -> str:
            if n == "null": return "null"
            # Remove unit symbols and extra formatting
            import re
            n = re.sub(r'[^\d\.-]', '', n)
            return n

        numerical_answers = [
            clean_num(sol.numerical_answer) for sol in valid_solutions if sol.numerical_answer != "null"
        ]
        if (
            len(numerical_answers) == len(valid_solutions)
            and len(set(numerical_answers)) == 1
        ):
            return True, valid_solutions[0].final_answer, valid_solutions[0].answer, valid_solutions[0].numerical_answer

        return False, "", "null", "null"

    @message_handler
    async def handle_task(self, message: Task, ctx: MessageContext) -> None:
        """Handle a task and initiate the solver process."""
        with self._langfuse_client.start_as_current_span(
            name=f"moderator_handle_task_{message.task_id}"
        ) as span:
            try:
                msg = f"Moderator received task: {message.task_id}, {message.capability_name} round {self._current_round}"
                log.info(msg)
                span.update(
                    metadata={
                        "task_received": msg,
                        "task_id": message.task_id,
                        "capability_name": message.capability_name,
                        "area_name": message.area_name,
                    }
                )

                # Initialize tracking for this task
                self._solutions_buffer = {}
                self._tasks = message

                # Step 1: Moderator Guidance (Planning)
                guidance = ""
                try:
                    plan_prompt = TASK_MODERATOR_BREAKDOWN_PROMPT.format(problem_text=message.problem)
                    plan_response = await self._model_client.create(
                        messages=[UserMessage(content=plan_prompt, source="user")],
                        cancellation_token=ctx.cancellation_token
                    )
                    guidance = str(plan_response.content).strip()
                    log.info(f"Moderator generated guidance for task {message.task_id}")
                except Exception as e:
                    log.error(f"Moderator failed to generate guidance: {e}")
                    guidance = "No specific guidance provided. Solve the problem directly."

                # Send initial solution request to all solvers WITH GUIDANCE
                await self.publish_message(
                    TaskSolutionRequest(
                        task_id=message.task_id,
                        problem=message.problem,
                        capability_name=message.capability_name,
                        area_name=message.area_name,
                        task_type=message.task_type,
                        round_number=self._current_round,
                        moderator_guidance=guidance,
                    ),
                    topic_id=DefaultTopicId(),
                )

                span.update(
                    metadata={
                        "solution_request_sent": f"Round {self._current_round} solution request sent for task {message.task_id}",
                        "guidance": guidance[:100] + "..."
                    }
                )

            except Exception as e:
                error_msg = f"Error handling task {message.task_id}: {str(e)}"
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg})

    @message_handler
    async def handle_agent_solution(
        self, message: AgentSolution, ctx: MessageContext
    ) -> None:
        """Handle solution from an agent."""
        with self._langfuse_client.start_as_current_span(
            name=f"moderator_handle_solution_{message.task_id}_round_{message.round_number}"
        ) as span:
            try:
                task_id = message.task_id
                round_num = message.round_number

                msg = f"Moderator received solution from agent {message.agent_id} for task {task_id}, {message.capability_name}, {message.area_name} round {round_num}"
                log.info(msg)
                span.update(
                    metadata={
                        "solution_received": msg,
                        "task_id": task_id,
                        "agent_id": message.agent_id,
                        "round": round_num,
                    }
                )

                if round_num != self._current_round:
                    msg = f"Moderator received solution from agent {message.agent_id} for task {task_id}, {message.capability_name}, {message.area_name} round {round_num} but current round is {self._current_round}"
                    log.error(msg)
                    span.update(metadata={"error": msg})
                    raise Exception(msg)

                # Initialize round buffer if needed
                if self._current_round not in self._solutions_buffer:
                    self._solutions_buffer[self._current_round] = []

                # Add solution to buffer
                self._solutions_buffer[self._current_round].append(message)

                msg = f"{len(self._solutions_buffer[self._current_round])}/{self._num_solvers} solutions collected for round {self._current_round}"
                log.info(msg)
                span.update(metadata={"solutions_collected": msg})

                if (
                    len(self._solutions_buffer[self._current_round])
                    == self._num_solvers
                ):
                    await self._check_consensus_and_proceed(task_id, ctx)

            except Exception as e:
                error_msg = (
                    f"Error handling solution from agent {message.agent_id}: {str(e)}"
                )
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg})

    def _enforce_format(self, answer: str, task_type: str) -> str:
        """Enforce answer format based on task type."""
        answer = str(answer).strip()
        
        if task_type == "bool":
            if answer.lower() in ["true", "yes", "correct", "1", "1.0"]:
                return "True"
            if answer.lower() in ["false", "no", "incorrect", "0", "0.0"]:
                return "False"
            # Heuristic for sentence answers
            if "true" in answer.lower() and "false" not in answer.lower():
                return "True"
            if "false" in answer.lower() and "true" not in answer.lower():
                return "False"
                
        elif task_type == "mcq":
            # Extract first single letter if possible
            if len(answer) == 1 and answer.upper() in ["A", "B", "C", "D", "E"]:
                return answer.upper()
            # If answer is like "A.", "Choice A", etc.
            import re
            match = re.search(r'\b([A-E])\b', answer.upper())
            if match:
                return match.group(1)
        
        return answer

    async def _check_consensus_and_proceed(
        self, task_id: str, ctx: MessageContext
    ) -> None:
        """Check for consensus and either finalize or start next round."""
        with self._langfuse_client.start_as_current_span(
            name=f"moderator_consensus_check_{task_id}_round_{self._current_round}"
        ) as span:
            try:
                solutions = self._solutions_buffer[self._current_round]

                # First try simple consensus check
                simple_consensus, simple_solution, simple_answer, simple_numerical = (
                    self._check_simple_consensus(solutions)
                )

                if simple_consensus:
                    # Enforce format before saving
                    final_answer = self._enforce_format(simple_answer, self._tasks.task_type)
                    
                    final_solution = FinalSolution(
                        task_id=task_id,
                        capability_name=self._tasks.capability_name,
                        area_name=self._tasks.area_name,
                        problem=self._tasks.problem,
                        solution=simple_solution,
                        answer=final_answer,
                        numerical_answer=simple_numerical,
                        reasoning="All agents provided the same answer",
                        consensus_reached=True,
                        total_rounds=self._current_round,
                        all_solutions=self._get_all_solutions(),
                        task_type=self._tasks.task_type,
                    )

                    self._final_solutions = final_solution
                    await self._save_final_solution(final_solution)

                    span.update(
                        metadata={
                            "consensus_reached": True,
                            "method": "simple",
                            "final_solution": simple_solution[:100],
                        }
                    )
                    return

                # If simple consensus failed, run LLM Moderator (Judge/Consensus Check)
                stored_task = self._tasks

                # Format solutions for LLM
                all_solutions_text = "\n\n".join(
                    [
                        f"Agent {sol.agent_id}:\nReasoning: {sol.thought}\nFinal Answer: {sol.final_answer}\nStructured Answer: {sol.answer}"
                        for sol in solutions
                    ]
                )

                prompt = TASK_MODERATOR_CONSENSUS_PROMPT.format(
                    problem_text=stored_task.problem,
                    all_solutions=all_solutions_text,
                    current_round=self._current_round,
                    max_rounds=self._max_rounds,
                )

                system_message = SystemMessage(
                    content=TASK_MODERATOR_SYSTEM_MESSAGE
                )
                user_message = UserMessage(content=prompt, source="user")

                # Retry Logic for Moderator Consensus
                MAX_MODERATOR_ATTEMPTS = 3
                consensus_reached = False
                final_solution_text = ""
                answer = "null"
                reasoning = ""
                numerical_answer = "null"

                for attempt in range(MAX_MODERATOR_ATTEMPTS):
                    try:
                        response = await self._model_client.create(
                            messages=[system_message, user_message],
                            cancellation_token=ctx.cancellation_token,
                            extra_create_args={"response_format": {"type": "json_object"}},
                        )
                        
                        content = str(response.content).strip()
                        if not content:
                            raise ValueError("Empty response from moderator model")

                        (
                            consensus_reached,
                            final_solution_text,
                            answer,
                            reasoning,
                            numerical_answer,
                        ) = self._extract_consensus_components(content)
                        
                        # If we parsed successfully, break the loop
                        break
                    
                    except Exception as e:
                        log.warning(f"Moderator consensus check failed (Attempt {attempt+1}/{MAX_MODERATOR_ATTEMPTS}): {e}")
                        if attempt < MAX_MODERATOR_ATTEMPTS - 1:
                            import asyncio
                            await asyncio.sleep(2 ** attempt)
                        else:
                            log.error("Moderator failed all attempts to check consensus.")
                            # Fall through to logic that handles 'no consensus' or retry next round
                            # But if we failed to get JSON, we likely treat it as no consensus for now
                            consensus_reached = False

                if consensus_reached:
                    # LLM found consensus (or acted as Judge)
                    # Enforce format before saving
                    final_answer = self._enforce_format(answer, self._tasks.task_type)

                    final_solution = FinalSolution(
                        task_id=task_id,
                        capability_name=self._tasks.capability_name,
                        area_name=self._tasks.area_name,
                        problem=self._tasks.problem,
                        solution=final_solution_text,
                        answer=final_answer,
                        numerical_answer=numerical_answer,
                        reasoning=reasoning,
                        consensus_reached=True,
                        total_rounds=self._current_round,
                        all_solutions=self._get_all_solutions(),
                        task_type=self._tasks.task_type,
                    )

                    self._final_solutions = final_solution
                    await self._save_final_solution(final_solution)

                    span.update(
                        metadata={
                            "consensus_reached": True,
                            "method": "llm_moderator",
                            "final_solution": final_solution_text[:100],
                        }
                    )
                    return

                # If consensus was NOT reached (LLM asked for more rounds), check if we can continue
                if self._current_round < self._max_rounds:
                    # Start next round
                    self._current_round += 1

                    # Send revision request with flattened task data
                    stored_task = self._tasks  # Get the original task

                    await self.publish_message(
                        AgentRevisionRequest(
                            task_id=stored_task.task_id,
                            problem=stored_task.problem,
                            capability_name=stored_task.capability_name,
                            area_name=stored_task.area_name,
                            task_type=stored_task.task_type,
                            other_solutions=[
                                {
                                    "agent_id": sol.agent_id,
                                    "task_id": sol.task_id,
                                    "thought": sol.thought,
                                    "final_answer": sol.final_answer,
                                    "answer": sol.answer,
                                    "numerical_answer": sol.numerical_answer,
                                    "round_number": str(sol.round_number),
                                }
                                for sol in solutions
                            ],
                            round_number=self._current_round,
                        ),
                        topic_id=DefaultTopicId(),
                    )

                    span.update(
                        metadata={
                            "consensus_reached": False,
                            "next_round_started": self._current_round,
                        }
                    )
                else:
                    # Max rounds reached, judge failed to reach consensus/pick a winner
                    final_solution = FinalSolution(
                        task_id=task_id,
                        capability_name=self._tasks.capability_name,
                        area_name=self._tasks.area_name,
                        problem=self._tasks.problem,
                        solution="No consensus reached",
                        answer="null",
                        numerical_answer="null",
                        reasoning=f"Maximum rounds ({self._max_rounds}) reached without consensus/judgement",
                        consensus_reached=False,
                        total_rounds=self._current_round,
                        all_solutions=self._get_all_solutions(),
                        task_type=self._tasks.task_type,
                    )

                    self._final_solutions = final_solution
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

    def _get_all_solutions(self) -> List[Dict[str, str]]:
        return [
            sol.to_dict() for sols in self._solutions_buffer.values() for sol in sols
        ]

    async def _save_final_solution(self, final_solution: FinalSolution) -> None:
        """Save the final solution to a file."""
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self._output_dir / f"{final_solution.task_id}_solution.json"

            solution_data = {
                "task_id": final_solution.task_id,
                "capability_name": final_solution.capability_name,
                "area_name": final_solution.area_name,
                "problem": final_solution.problem,
                "solution": final_solution.solution,
                "answer": final_solution.answer,
                "numerical_answer": final_solution.numerical_answer,
                "reasoning": final_solution.reasoning,
                "consensus_reached": final_solution.consensus_reached,
                "total_rounds": final_solution.total_rounds,
                "all_solutions": [
                    {
                        "agent_id": sol["agent_id"],
                        "task_id": sol["task_id"],
                        "thought": sol["thought"],
                        "final_answer": sol["final_answer"],
                        "answer": sol.get("answer", "null"),
                        "numerical_answer": sol["numerical_answer"],
                        "round_number": sol["round_number"],
                    }
                    for sol in final_solution.all_solutions
                ],
            }

            with open(output_file, "w") as f:
                json.dump(solution_data, f, indent=2)

            log.info(
                f"Saved final solution for task {final_solution.task_id} to {output_file}"
            )

        except Exception as e:
            log.error(
                f"Error saving final solution for task {final_solution.task_id}: {str(e)}"
            )
            log.error(traceback.format_exc())
