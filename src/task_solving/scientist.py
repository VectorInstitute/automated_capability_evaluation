"""Task solver agent for solving tasks through debate."""

import logging
import re
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

from src.task_solving.messages import (
    AgentRevisionRequest,
    AgentSolution,
    TaskSolutionRequest,
)
from src.utils.agentic_prompts import (
    TASK_SOLVER_ROUND_1_PROMPT,
    TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT,
    TASK_SOLVER_SYSTEM_MESSAGE,
)


log = logging.getLogger("task_solving.solver")


@default_subscription
class TaskSolvingScientist(RoutedAgent):
    """A scientist that solves tasks through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        langfuse_client: Langfuse = None,
    ) -> None:
        super().__init__(f"Task Solving Scientist {scientist_id}")
        self._model_client = model_client
        self._scientist_id = scientist_id
        self._langfuse_client = langfuse_client

    def _extract_solution_components(self, response: str) -> tuple[str, str]:
        """Extract thought and final answer from the response."""
        thought_match = re.search(r"THOUGHT:\s*(.*?)(?=FINAL ANSWER:|$)", response, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"FINAL ANSWER:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        
        thought = thought_match.group(1).strip() if thought_match else response.strip()
        final_answer = answer_match.group(1).strip() if answer_match else "No clear answer provided"
        
        return thought, final_answer

    @message_handler
    async def handle_task_solution_request(
        self, message: TaskSolutionRequest, ctx: MessageContext
    ) -> None:
        """Handle initial task solution request (Round 1)."""
        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_round_1"
        ) as span:
            try:
                task_text = message.task.task_content.get("task", "")
                
                msg = f"Scientist {self._scientist_id} handling initial solution request for task: {message.task.task_id}"
                log.info(msg)
                span.update(
                    metadata={
                        "solution_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task.task_id,
                        "round": message.round_number,
                    }
                )

                prompt = TASK_SOLVER_ROUND_1_PROMPT.format(problem_text=task_text)
                
                system_message = SystemMessage(content=TASK_SOLVER_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                response = await self._model_client.create(
                    messages=[system_message, user_message],
                    cancellation_token=ctx.cancellation_token,
                )

                response_content = response.content
                thought, final_answer = self._extract_solution_components(response_content)

                solution = AgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    round_number=message.round_number,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "solution_generated": f"Scientist {self._scientist_id} generated solution for task {message.task.task_id}",
                        "final_answer": final_answer[:100],  # Truncate for logging
                    }
                )

            except Exception as e:
                error_msg = f"Error in scientist {self._scientist_id} round 1: {str(e)}"
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg})

    @message_handler
    async def handle_agent_revision_request(
        self, message: AgentRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request with other agents' solutions."""
        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_round_{message.round_number}"
        ) as span:
            try:
                task_text = message.task.task_content.get("task", "")
                
                msg = f"Scientist {self._scientist_id} handling revision request for task: {message.task.task_id}, round: {message.round_number}"
                log.info(msg)
                span.update(
                    metadata={
                        "revision_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task.task_id,
                        "round": message.round_number,
                        "num_other_solutions": len(message.other_solutions),
                    }
                )

                # Format other scientists' solutions
                other_solutions_text = "\n\n".join([
                    f"Scientist {sol.agent_id}: Reasoning: {sol.thought}, Final solution: {sol.final_answer}"
                    for sol in message.other_solutions
                    if sol.agent_id != self._scientist_id  # Don't include our own solution
                ])

                prompt = TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT.format(
                    other_solutions=other_solutions_text,
                    problem_text=task_text
                )
                
                system_message = SystemMessage(content=TASK_SOLVER_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                response = await self._model_client.create(
                    messages=[system_message, user_message],
                    cancellation_token=ctx.cancellation_token,
                )

                response_content = response.content
                thought, final_answer = self._extract_solution_components(response_content)

                solution = AgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    round_number=message.round_number,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "revision_generated": f"Scientist {self._scientist_id} generated revision for task {message.task.task_id}",
                        "final_answer": final_answer[:100],  # Truncate for logging
                    }
                )

            except Exception as e:
                error_msg = f"Error in scientist {self._scientist_id} round {message.round_number}: {str(e)}"
                log.error(error_msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": error_msg}) 