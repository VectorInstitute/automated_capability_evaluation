"""Task solver agent for solver tasks through debate."""

import json
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

from src.task_solve_models.multi_agent_solver.messages import (
    AgentRevisionRequest,
    AgentSolution,
    TaskSolutionRequest,
)
from src.task_solve_models.multi_agent_solver.prompts import (
    TASK_SOLVER_ROUND_1_PROMPT,
    TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT,
    TASK_SOLVER_SYSTEM_MESSAGE,
    TASK_SOLVER_FORMATTER_PROMPT,
    TASK_SOLVER_FORMATTER_SYSTEM_MESSAGE,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("task_solver.scientist")

MAX_MODEL_ATTEMPTS = 3


@default_subscription
class TaskSolverScientist(RoutedAgent):
    """A scientist that solves tasks through debate.

    Attributes
    ----------
    _model_client : ChatCompletionClient
        ChatCompletionClient for generating solutions via LLM.
    _scientist_id : str
        Unique identifier for this scientist agent in the debate.
    _langfuse_client : Langfuse
        Langfuse client for tracing and logging scientist activity.
    """

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

    def _extract_solution_components(self, response: str) -> tuple[str, str, str, str]:
        """Extract thought, final answer, answer, and numerical answer from JSON response."""
        try:
            # Local fix for common LLM JSON errors (like missing commas between fields)
            # Matches: "value" \n "key": -> "value", \n "key":
            import re
            response = re.sub(r'"\s*\n\s*"(\w+)":', r'",\n"\1":', response)

            parsed = parse_llm_json_response(response)
            thought_raw = parsed.get("thought", response.strip())
            final_answer_raw = parsed.get("final_answer", "No clear answer provided")
            answer_raw = parsed.get("answer", "No answer provided")
            numerical_answer = parsed.get("numerical_answer")

            thought = (
                json.dumps(thought_raw, ensure_ascii=False)
                if isinstance(thought_raw, (dict, list))
                else str(thought_raw).strip()
            )
            final_answer = (
                json.dumps(final_answer_raw, ensure_ascii=False, indent=2)
                if isinstance(final_answer_raw, (dict, list))
                else str(final_answer_raw).strip()
            )
            answer = (
                json.dumps(answer_raw, ensure_ascii=False, indent=2)
                if isinstance(answer_raw, (dict, list))
                else str(answer_raw).strip()
            )

            if numerical_answer is not None:
                numerical_answer = str(numerical_answer)
            else:
                numerical_answer = "null"

            return thought, final_answer, answer, numerical_answer

        except Exception as e:
            msg = f"Failed to parse JSON response: {e} \n Response: {response}"
            log.error(msg)
            log.error(traceback.format_exc())
            raise

    async def _generate_solution_payload(
        self, system_message: SystemMessage, user_message: UserMessage
    ) -> tuple[str, str, str, str]:
        """Call the model in two steps: Reasoning (Text) -> Formatting (JSON)."""
        
        last_error: Exception | None = None
        
        for attempt in range(MAX_MODEL_ATTEMPTS):
            try:
                # 1. Generate Reasoning (Allowing free text/markdown)
                # Note: We do NOT enforce JSON mode here to avoid the escaping/math complexity issues.
                reasoning_response = await self._model_client.create(
                    [system_message, user_message],
                )
                reasoning_content = str(getattr(reasoning_response, "content", "") or "").strip()
                
                if not reasoning_content:
                    raise ValueError("Empty reasoning response from model")

                # 2. Format into JSON (Strict Mode)
                # We create a new prompt context for the formatting step
                formatter_system = SystemMessage(content=TASK_SOLVER_FORMATTER_SYSTEM_MESSAGE)
                formatter_user = UserMessage(
                    content=TASK_SOLVER_FORMATTER_PROMPT.format(agent_response=reasoning_content), 
                    source="user"
                )

                # We DO enforce JSON mode here for guaranteed structure
                json_response = await self._model_client.create(
                    [formatter_system, formatter_user],
                    json_output=True,
                    extra_create_args={"response_format": {"type": "json_object"}},
                )
                
                json_content = str(getattr(json_response, "content", "") or "").strip()
                
                if not json_content:
                    raise ValueError("Empty JSON response from formatter")

                return self._extract_solution_components(json_content)

            except Exception as exc:
                last_error = exc
                log.warning(
                    "Scientist %s failed in generation attempt %d/%d: %s",
                    self._scientist_id,
                    attempt + 1,
                    MAX_MODEL_ATTEMPTS,
                    exc,
                )

        log.error(
            f"Scientist {self._scientist_id} could not obtain valid JSON "
            f"after {MAX_MODEL_ATTEMPTS} attempts. Returning error payload."
        )
        return (
            "Failed to generate valid JSON response.",
            f"Error: generation failed. Last error: {str(last_error)}",
            "ERROR",
            "null",
        )

    @message_handler
    async def handle_task_solution_request(
        self, message: TaskSolutionRequest, ctx: MessageContext
    ) -> None:
        """Handle initial task solution request."""
        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_initial_solution_request"
        ) as span:
            try:
                msg = (
                    f"Scientist {self._scientist_id} handling initial solution request "
                    f"for task: {message.task_id}, capability: {message.capability_name}, area: {message.area_name}"
                    f"round: {message.round_number}"
                )
                log.info(msg)
                span.update(
                    metadata={
                        "solution_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task_id,
                        "capability": message.capability_name,
                        "area": message.area_name,
                        "round": message.round_number,
                    }
                )

                prompt = TASK_SOLVER_ROUND_1_PROMPT.format(problem_text=message.problem)

                system_message = SystemMessage(content=TASK_SOLVER_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                (
                    thought,
                    final_answer,
                    answer,
                    numerical_answer,
                ) = await self._generate_solution_payload(system_message, user_message)

                solution = AgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    answer=answer,
                    numerical_answer=numerical_answer,
                    round_number=message.round_number,
                    capability_name=message.capability_name,
                    area_name=message.area_name,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "solution_generated": (
                            f"Scientist {self._scientist_id} generated solution for task "
                            f"{message.task_id}, capability: {message.capability_name}, area: {message.area_name}"
                            f"round: {message.round_number}"
                        ),
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
                msg = (
                    f"Scientist {self._scientist_id} handling revision request for task: "
                    f"{message.task_id}, capability: {message.capability_name}, area: {message.area_name}"
                    f"round: {message.round_number}"
                )
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

                other_solutions_text = "\n\n".join(
                    [
                        (
                            f"Scientist {sol['agent_id']}: Reasoning: {sol['thought']}, "
                            f"Final solution: {sol['final_answer']}"
                        )
                        for sol in message.other_solutions
                        if sol["agent_id"] != self._scientist_id
                    ]
                )

                prompt = TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT.format(
                    other_solutions=other_solutions_text,
                    problem_text=message.problem,
                )

                system_message = SystemMessage(content=TASK_SOLVER_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                (
                    thought,
                    final_answer,
                    answer,
                    numerical_answer,
                ) = await self._generate_solution_payload(system_message, user_message)

                solution = AgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    answer=answer,
                    numerical_answer=numerical_answer,
                    round_number=message.round_number,
                    capability_name=message.capability_name,
                    area_name=message.area_name,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "revision_generated": (
                            f"Scientist {self._scientist_id} generated revision for task "
                            f"{message.task_id}, capability: {message.capability_name}, area: {message.area_name}"
                            f"round: {message.round_number}"
                        ),
                    }
                )

            except Exception as e:
                msg = f"Error in scientist {self._scientist_id} agent revision request: {str(e)}"
                log.error(msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": msg})
