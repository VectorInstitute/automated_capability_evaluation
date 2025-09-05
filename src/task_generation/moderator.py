"""Task moderator agent for managing task generation workflow."""

import json
import logging
import math
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

from src.task_generation.messages import (
    Capability,
    ProblemProposalRequest,
    ScientistProblemProposal,
)
from src.utils.agentic_prompts import (
    TASK_MODERATOR_PROBLEM_SYSTEM_PROMPT,
    TASK_MODERATOR_PROBLEM_USER_PROMPT,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("agentic_task_gen.moderator")


@default_subscription
class TaskModerator(RoutedAgent):
    """Moderator that merges scientist task proposals and manages iteration."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        num_scientists: int,
        num_final_problems: int,
        buffer_param: int,
        output_dir: Path,
        domain: str,
        langfuse_client: Langfuse,
        max_round: int = 5,
    ) -> None:
        super().__init__("Task Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_final_problems = num_final_problems
        self._buffer_param = buffer_param
        self._output_dir = output_dir
        self._domain = domain
        self._langfuse_client = langfuse_client
        self._max_round = max_round

        self._num_remaining = self._num_final_problems
        self._final_problems: Dict[str, str] = {}  # {task_id: problem_text}
        self._capability: (
            Capability  # Store original capability info (set in first message)
        )
        self._current_round = 0

        # Problem design state
        self._problem_proposals: Dict[int, List[ScientistProblemProposal]] = {}

    @message_handler
    async def handle_capability(self, message: Capability, ctx: MessageContext) -> None:
        """Start problem design for a capability."""
        with self._langfuse_client.start_as_current_span(
            name="task_moderator_handle_capability"
        ) as span:
            try:
                capability_name = message.name
                msg = f"Task Moderator starting problem design for capability: {capability_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "capability_received": msg,
                        "capability_name": capability_name,
                        "capability_description": message.description,
                        "capability_area": message.area,
                    }
                )

                self._capability = message
                self._problem_proposals[self._current_round] = []

                await self._start_problem_iteration()

            except Exception as e:
                error_msg = f"Error in Task Moderator handle_capability: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "handle_capability_error": error_msg,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    async def _start_problem_iteration(self) -> None:
        """Start a problem generation iteration."""
        try:
            # Check if we've reached the maximum number of rounds
            if self._current_round >= self._max_round:
                log.info(
                    f"Maximum rounds ({self._max_round}) reached for capability: {self._capability.name}.\
                    Finalizing with {len(self._final_problems)} problems."
                )
                await self._finalize_tasks_without_solutions()
                return

            if self._num_remaining <= 0:
                log.info(
                    f"Problem design completed for capability: {self._capability.name}"
                )
                await self._finalize_tasks_without_solutions()
                return

            # Calculate problems per scientist: ceil(num_remaining / M) + B
            problems_per_scientist = (
                math.ceil(self._num_remaining / self._num_scientists)
                + self._buffer_param
            )

            log.info(
                f"Task Moderator requesting {problems_per_scientist} problems per scientist for capability: {self._capability.name} (remaining: {self._num_remaining}, round: {self._current_round}/{self._max_round})"
            )

            # Get sample tasks from existing final problems
            sample_tasks = list(self._final_problems.values())[
                :3
            ]  # Use up to 3 existing problems as samples

            # Send problem proposal requests to all scientists
            await self.publish_message(
                ProblemProposalRequest(
                    capability_name=self._capability.name,
                    capability_description=self._capability.description,
                    capability_domain=self._capability.domain,
                    capability_area=self._capability.area,
                    num_problems=problems_per_scientist,
                    sample_tasks=sample_tasks,
                    iteration=self._current_round,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(f"Error in Task Moderator _start_problem_iteration: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_scientist_problem_proposal(
        self, message: ScientistProblemProposal, ctx: MessageContext
    ) -> None:
        """Handle problem proposals from scientists."""
        try:
            log.info(
                f"Task Moderator received problem proposal from Scientist {message.scientist_id} for capability: {message.capability_name}"
            )

            self._problem_proposals[self._current_round].append(message)

            # Check if we have all proposals for this iteration
            current_proposals = self._problem_proposals[self._current_round]
            if len(current_proposals) == self._num_scientists:
                log.info(
                    f"Task Moderator received all problem proposals for capability: {self._capability.name}, proceeding to filter"
                )
                await self._filter_and_select_problems()

        except Exception as e:
            log.error(f"Error in Task Moderator handle_scientist_problem_proposal: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _filter_and_select_problems(self) -> None:
        """Filter and select problems using moderator LLM."""
        try:
            log.info(
                f"Task Moderator filtering problems for capability: {self._capability.name}"
            )

            # Collect all proposed problems
            current_proposals = self._problem_proposals[self._current_round]
            all_problems = {}
            scientist_attribution = {}

            for proposal in current_proposals:
                for task_id, problem_text in proposal.problems.items():
                    unique_id = f"{proposal.scientist_id}_{task_id}"
                    all_problems[unique_id] = problem_text
                    scientist_attribution[unique_id] = proposal.scientist_id

            if not all_problems:
                log.warning(
                    f"No problems received for capability: {self._capability.name}"
                )
                return

            # Format problems for moderator
            problems_text = ""
            for scientist_id in set(scientist_attribution.values()):
                problems_text += f"Scientist {scientist_id}:\n"
                for task_id, problem in all_problems.items():
                    if scientist_attribution[task_id] == scientist_id:
                        task_name = task_id.split("_", 1)[1]  # Remove scientist prefix
                        problems_text += f"- {task_name}: {problem}\n"
                problems_text += "\n"

            user_prompt = TASK_MODERATOR_PROBLEM_USER_PROMPT.format(
                capability_name=self._capability.name,
                capability_description=self._capability.description,
                capability_domain=self._capability.domain,
                problems_text=problems_text,
            )

            system_message = SystemMessage(content=TASK_MODERATOR_PROBLEM_SYSTEM_PROMPT)
            user_message = UserMessage(content=user_prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            raw_content = model_result.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            # Extract JSON from response using robust parser
            try:
                parsed = parse_llm_json_response(raw_content)
                final_tasks = parsed.get("final_tasks", {})
            except Exception as e:
                log.error(
                    f"Error parsing JSON from moderator: {e}\nOutput: {raw_content}"
                )
                final_tasks = {}

            num_selected = min(len(final_tasks), self._num_remaining)

            # Add selected problems to final set
            selected_count = 0
            for _, problem_text in final_tasks.items():
                if selected_count < num_selected:
                    final_task_id = f"task_{len(self._final_problems) + 1}"
                    self._final_problems[final_task_id] = problem_text
                    selected_count += 1

            # Update remaining count
            self._num_remaining = self._num_remaining - selected_count

            log.info(
                f"Task Moderator selected {selected_count} problems for {self._capability.name}, {self._num_remaining} remaining"
            )

            if self._num_remaining > 0:
                # Increment round counter before starting next iteration
                self._current_round += 1
                await self._start_problem_iteration()
            else:
                await self._finalize_tasks_without_solutions()

        except Exception as e:
            log.error(f"Error in Task Moderator _filter_and_select_problems: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _finalize_tasks_without_solutions(self) -> None:
        """Finalize tasks with problems only."""
        try:
            log.info(
                f"Task Moderator finalizing tasks for capability: {self._capability.name}"
            )

            if not self._final_problems:
                log.error(
                    f"No final problems available for capability: {self._capability.name}"
                )
                return

            # Create tasks with problems only
            final_tasks = {}
            for task_id, problem_text in self._final_problems.items():
                final_tasks[task_id] = {
                    "task": problem_text,
                    "capability_id": self._capability.name,
                }

            # Save final tasks
            await self._save_tasks_to_file(final_tasks)
            log.info(
                f"Task generation completed for capability: {self._capability.name} ({len(final_tasks)} tasks)"
            )

        except Exception as e:
            log.error(f"Error in Task Moderator _finalize_tasks_without_solutions: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _save_tasks_to_file(self, tasks: Dict[str, Dict[str, str]]) -> None:
        """Save final tasks to file."""
        try:
            # Create capability directory
            capability_dir = self._output_dir / self._capability.name
            capability_dir.mkdir(parents=True, exist_ok=True)

            # Save tasks
            tasks_file = capability_dir / "tasks.json"
            with open(tasks_file, "w", encoding="utf-8") as f:
                json.dump({"tasks": tasks}, f, indent=2, ensure_ascii=False)

            log.info(
                f"Saved {len(tasks)} tasks for capability '{self._capability.name}' to {tasks_file}"
            )
        except Exception as e:
            log.error(f"Error saving tasks for capability {self._capability.name}: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise
