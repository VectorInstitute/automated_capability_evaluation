"""Task solver agent for solver tasks through debate."""

import logging
import traceback

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

from src.task_solver.messages import (
    AgentRevisionRequest,
    AgentSolution,
    TaskSolutionRequest,
)
from src.utils.agentic_prompts import (
    TASK_SOLVER_ROUND_1_PROMPT,
    TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT,
    TASK_SOLVER_SYSTEM_MESSAGE,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("task_solver.scientist")


@default_subscription
class TaskSolverScientist(RoutedAgent):
    """A scientist that solves tasks through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        langfuse_client: Langfuse,
    ) -> None:
        super().__init__(f"Task Solver Scientist {scientist_id}")
        self._model_client = model_client
        self._scientist_id = scientist_id
        self._langfuse_client = langfuse_client

    def _extract_solution_components(self, response: str) -> tuple[str, str, str]:
        """Extract thought, final answer, and numerical answer from JSON response."""
        try:
            parsed = parse_llm_json_response(response)
            thought = parsed.get("thought", response.strip())
            final_answer = parsed.get("final_answer", "No clear answer provided")
            numerical_answer = parsed.get("numerical_answer")

            # Convert numerical_answer to string representation
            if numerical_answer is not None:
                numerical_answer = str(numerical_answer)
            else:
                numerical_answer = "null"

            return thought, final_answer, numerical_answer

        except Exception as e:
            msg = f"Failed to parse JSON response: {e} \n Response: {response}"
            log.error(msg)
            log.error(traceback.format_exc())
            raise

    @message_handler
    async def handle_task_solution_request(
        self, message: TaskSolutionRequest, ctx: MessageContext
    ) -> None:
        """Handle initial task solution request."""
        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_initial_solution_request"
        ) as span:
            try:
                msg = f"Scientist {self._scientist_id} handling initial solution request for task: {message.task_id}, capability: {message.capability_name} round: {message.round_number}"
                log.info(msg)
                span.update(
                    metadata={
                        "solution_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task_id,
                        "capability": message.capability_name,
                        "round": message.round_number,
                    }
                )

                prompt = TASK_SOLVER_ROUND_1_PROMPT.format(problem_text=message.problem)

                system_message = SystemMessage(content=TASK_SOLVER_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                response = await self._model_client.create(
                    [system_message, user_message]
                )

                response_content = str(response.content)
                thought, final_answer, numerical_answer = (
                    self._extract_solution_components(response_content)
                )

                solution = AgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    numerical_answer=numerical_answer,
                    round_number=message.round_number,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "solution_generated": f"Scientist {self._scientist_id} generated solution for task {message.task_id}, capability: {message.capability_name} round: {message.round_number}",
                    }
                )

            except Exception as e:
                msg = f"Error in scientist {self._scientist_id} task solution request: {str(e)}"
                log.error(msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": msg})

    @message_handler
    async def handle_agent_revision_request(
        self, message: AgentRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request with other agents' solutions."""
        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_round_{message.round_number}"
        ) as span:
            try:
                msg = f"Scientist {self._scientist_id} handling revision request for task: {message.task_id}, capability: {message.capability_name} round: {message.round_number}"
                log.info(msg)
                span.update(
                    metadata={
                        "revision_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task_id,
                        "round": message.round_number,
                        "num_other_solutions": len(message.other_solutions),
                    }
                )

                # Format other scientists' solutions
                other_solutions_text = "\n\n".join(
                    [
                        f"Scientist {sol['agent_id']}: Reasoning: {sol['thought']}, Final solution: {sol['final_answer']}"
                        for sol in message.other_solutions
                        if sol["agent_id"]
                        != self._scientist_id  # Don't include its own solution
                    ]
                )

                prompt = TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT.format(
                    other_solutions=other_solutions_text, problem_text=message.problem
                )

                system_message = SystemMessage(content=TASK_SOLVER_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                response = await self._model_client.create(
                    [system_message, user_message]
                )

                response_content = str(response.content)
                thought, final_answer, numerical_answer = (
                    self._extract_solution_components(response_content)
                )

                solution = AgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    numerical_answer=numerical_answer,
                    round_number=message.round_number,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "revision_generated": f"Scientist {self._scientist_id} generated revision for task {message.task_id}, capability: {message.capability_name}, round: {message.round_number}",
                    }
                )

            except Exception as e:
                msg = f"Error in scientist {self._scientist_id} agent revision request: {str(e)}"
                log.error(msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": msg})
