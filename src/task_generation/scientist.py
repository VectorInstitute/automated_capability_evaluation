"""Task scientist agent for generating problems and solutions."""

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

from src.task_generation.messages import (
    ProblemProposalRequest,
    ScientistProblemProposal,
    ScientistSolutionProposal,
    SolutionRequest,
)
from src.utils.agentic_prompts import (
    TASK_SCIENTIST_PROBLEM_SYSTEM_PROMPT,
    TASK_SCIENTIST_PROBLEM_USER_PROMPT,
    TASK_SCIENTIST_SOLUTION_SYSTEM_PROMPT,
    TASK_SCIENTIST_SOLUTION_USER_PROMPT,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("agentic_task_gen.scientist")


@default_subscription
class TaskScientist(RoutedAgent):
    """Scientist that generates problems and solutions."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        langfuse_client: Langfuse,
        domain: str = "",
    ) -> None:
        super().__init__(f"Task Scientist {scientist_id}")
        self._scientist_id = scientist_id
        self._model_client = model_client
        self._domain = domain
        self._langfuse_client = langfuse_client

    @message_handler
    async def handle_problem_proposal_request(
        self, message: ProblemProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle problem proposal request."""
        with self._langfuse_client.start_as_current_span(
            name=f"task_scientist_{self._scientist_id}_problem_proposal"
        ) as span:
            try:
                msg = f"Task Scientist {self._scientist_id} generating {message.num_problems} problems for capability: {message.capability_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "problem_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "capability_name": message.capability_name,
                        "capability_description": message.capability_description,
                        "num_problems": message.num_problems,
                    }
                )

                sample_tasks_text = ""
                if message.sample_tasks:
                    sample_tasks_text = "\n".join(
                        [f"- {task}" for task in message.sample_tasks]
                    )
                else:
                    sample_tasks_text = "(No sample tasks provided)"

                system_prompt = TASK_SCIENTIST_PROBLEM_SYSTEM_PROMPT.format(
                    scientist_id=self._scientist_id,
                )

                user_prompt = TASK_SCIENTIST_PROBLEM_USER_PROMPT.format(
                    num_problems=message.num_problems,
                    capability_name=message.capability_name,
                    capability_description=message.capability_description,
                    capability_domain=message.capability_domain,
                    sample_tasks_text=sample_tasks_text,
                )

                system_message = SystemMessage(content=system_prompt)
                user_message = UserMessage(content=user_prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = f"Task Scientist {self._scientist_id} is parsing LLM response"
                log.info(msg)
                span.update(
                    metadata={
                        "llm_response_received": msg,
                        "scientist_id": self._scientist_id,
                    }
                )

                parsed = parse_llm_json_response(model_result.content)
                problems = parsed.get("problems", {})

                msg = f"Task Scientist {self._scientist_id} proposing {len(problems)} problems for capability: {message.capability_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "problem_proposal_published": msg,
                        "scientist_id": self._scientist_id,
                        "capability_name": message.capability_name,
                        "num_problems_generated": len(problems),
                    }
                )

                await self.publish_message(
                    ScientistProblemProposal(
                        scientist_id=self._scientist_id,
                        capability_name=message.capability_name,
                        problems=problems,
                        iteration=0,
                    ),
                    topic_id=DefaultTopicId(),
                )

            except Exception as e:
                error_msg = f"Error in Task Scientist {self._scientist_id} handle_problem_proposal_request: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "problem_request_error": error_msg,
                        "scientist_id": self._scientist_id,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    @message_handler
    async def handle_solution_request(
        self, message: SolutionRequest, ctx: MessageContext
    ) -> None:
        """Handle solution request for problems."""
        with self._langfuse_client.start_as_current_span(
            name=f"task_scientist_{self._scientist_id}_solution_proposal"
        ) as span:
            try:
                msg = f"Task Scientist {self._scientist_id} solving {len(message.problems)} problems for capability: {message.capability_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "solution_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "capability_name": message.capability_name,
                        "num_problems": len(message.problems),
                    }
                )

                problems_json = json.dumps(message.problems, indent=2)

                system_prompt = TASK_SCIENTIST_SOLUTION_SYSTEM_PROMPT.format(
                    scientist_id=self._scientist_id,
                    capability_domain=message.capability_domain,
                    capability_name=message.capability_name,
                )

                user_prompt = TASK_SCIENTIST_SOLUTION_USER_PROMPT.format(
                    problems=problems_json,
                )

                system_message = SystemMessage(content=system_prompt)
                user_message = UserMessage(content=user_prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = f"Task Scientist {self._scientist_id} is parsing LLM response"
                log.info(msg)
                span.update(
                    metadata={
                        "llm_response_received": msg,
                        "scientist_id": self._scientist_id,
                    }
                )

                parsed = parse_llm_json_response(model_result.content)
                solutions = parsed.get("solutions", {})

                msg = f"Task Scientist {self._scientist_id} publishing solutions for capability: {message.capability_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "solution_proposal_published": msg,
                        "scientist_id": self._scientist_id,
                        "capability_name": message.capability_name,
                        "num_solutions_generated": len(solutions),
                    }
                )

                await self.publish_message(
                    ScientistSolutionProposal(
                        scientist_id=self._scientist_id,
                        capability_name=message.capability_name,
                        solutions=solutions,
                    ),
                    topic_id=DefaultTopicId(),
                )

            except Exception as e:
                error_msg = f"Error in Task Scientist {self._scientist_id} handle_solution_request: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "solution_request_error": error_msg,
                        "scientist_id": self._scientist_id,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise
