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
    ) -> None:
        super().__init__("Task Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_final_problems = num_final_problems
        self._buffer_param = buffer_param
        self._output_dir = output_dir
        self._domain = domain
        self._langfuse_client = langfuse_client

        self._num_remaining: Dict[str, int] = {}
        self._final_problems: Dict[
            str, Dict[str, str]
        ] = {}  # capability -> {task_id: problem_text}
        self._capabilities: Dict[str, Capability] = {}  # Store original capability info

        # Problem design state
        self._problem_proposals: Dict[
            str, List[ScientistProblemProposal]
        ] = {}  # capability -> proposals

    @message_handler
    async def handle_capability(self, message: Capability, ctx: MessageContext) -> None:
        """Start problem design for a capability."""
        with self._langfuse_client.start_as_current_span(
            name="task_moderator_handle_capability"
        ) as span:
            try:
                msg = f"Task Moderator starting problem design for capability: {message.name}"
                log.info(msg)
                span.update(
                    metadata={
                        "capability_received": msg,
                        "capability_name": message.name,
                        "capability_description": message.description,
                        "capability_area": message.area,
                    }
                )

                self._num_remaining[message.name] = self._num_final_problems
                self._final_problems[message.name] = {}
                self._capabilities[message.name] = message

                await self._start_problem_iteration(message)

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

    async def _start_problem_iteration(self, capability: Capability) -> None:
        """Start a problem generation iteration."""
        try:
            num_remaining = self._num_remaining[capability.name]
            if num_remaining <= 0:
                log.info(f"Problem design completed for capability: {capability.name}")
                await self._finalize_tasks_without_solutions(capability.name)
                return

            # Calculate problems per scientist: ceil(num_remaining / M) + B
            problems_per_scientist = (
                math.ceil(num_remaining / self._num_scientists) + self._buffer_param
            )

            log.info(
                f"Task Moderator requesting {problems_per_scientist} problems per scientist for capability: {capability.name} (remaining: {num_remaining})"
            )

            # Get sample tasks from existing final problems
            sample_tasks = list(self._final_problems[capability.name].values())[
                :3
            ]  # Use up to 3 existing problems as samples

            # Send problem proposal requests to all scientists
            await self.publish_message(
                ProblemProposalRequest(
                    capability_name=capability.name,
                    capability_description=capability.description,
                    capability_domain=capability.domain,
                    capability_area=capability.area,
                    num_problems=problems_per_scientist,
                    sample_tasks=sample_tasks,
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

            capability_name = message.capability_name
            if capability_name not in self._problem_proposals:
                self._problem_proposals[capability_name] = []

            self._problem_proposals[capability_name].append(message)

            # Check if we have all proposals for this iteration
            current_proposals = [
                p
                for p in self._problem_proposals[capability_name]
                if p.iteration == message.iteration
            ]
            if len(current_proposals) == self._num_scientists:
                log.info(
                    f"Task Moderator received all problem proposals for capability: {capability_name}, proceeding to filter"
                )
                await self._filter_and_select_problems(
                    capability_name, message.iteration
                )

        except Exception as e:
            log.error(f"Error in Task Moderator handle_scientist_problem_proposal: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _filter_and_select_problems(
        self, capability_name: str, iteration: int
    ) -> None:
        """Filter and select problems using moderator LLM."""
        try:
            log.info(
                f"Task Moderator filtering problems for capability: {capability_name}"
            )

            # Collect all proposed problems
            current_proposals = [
                p
                for p in self._problem_proposals[capability_name]
                if p.iteration == iteration
            ]
            all_problems = {}
            scientist_attribution = {}

            for proposal in current_proposals:
                for task_id, problem_text in proposal.problems.items():
                    unique_id = f"{proposal.scientist_id}_{task_id}"
                    all_problems[unique_id] = problem_text
                    scientist_attribution[unique_id] = proposal.scientist_id

            if not all_problems:
                log.warning(f"No problems received for capability: {capability_name}")
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

            system_prompt = TASK_MODERATOR_PROBLEM_SYSTEM_PROMPT

            capability_info = self._capabilities[capability_name]
            user_prompt = TASK_MODERATOR_PROBLEM_USER_PROMPT.format(
                capability_name=capability_info.name,
                capability_description=capability_info.description,
                capability_domain=capability_info.domain,
                problems_text=problems_text,
            )

            system_message = SystemMessage(content=system_prompt)
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

            num_remaining = self._num_remaining[capability_name]
            num_selected = min(len(final_tasks), num_remaining)

            # Add selected problems to final set
            selected_count = 0
            for _, problem_text in final_tasks.items():
                if selected_count < num_selected:
                    final_task_id = (
                        f"task_{len(self._final_problems[capability_name]) + 1}"
                    )
                    self._final_problems[capability_name][final_task_id] = problem_text
                    selected_count += 1

            # Update remaining count
            self._num_remaining[capability_name] = num_remaining - selected_count

            log.info(
                f"Task Moderator selected {selected_count} problems for {capability_name}, {self._num_remaining[capability_name]} remaining"
            )

            if self._num_remaining[capability_name] > 0:
                capability = self._capabilities[capability_name]
                await self._start_problem_iteration(capability)
            else:
                await self._finalize_tasks_without_solutions(capability_name)

        except Exception as e:
            log.error(f"Error in Task Moderator _filter_and_select_problems: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _finalize_tasks_without_solutions(self, capability_name: str) -> None:
        """Finalize tasks with problems only."""
        try:
            log.info(
                f"Task Moderator finalizing tasks for capability: {capability_name}"
            )

            final_problems = self._final_problems[capability_name]
            if not final_problems:
                log.error(
                    f"No final problems available for capability: {capability_name}"
                )
                return

            # Create tasks with problems only
            final_tasks = {}
            for task_id, problem_text in final_problems.items():
                final_tasks[task_id] = {
                    "task": problem_text,
                    "capability_id": capability_name,
                }

            # Save final tasks
            await self._save_tasks_to_file(capability_name, final_tasks)
            log.info(
                f"Task generation completed for capability: {capability_name} ({len(final_tasks)} tasks)"
            )

        except Exception as e:
            log.error(f"Error in Task Moderator _finalize_tasks_without_solutions: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _save_tasks_to_file(
        self, capability_name: str, tasks: Dict[str, Dict[str, str]]
    ) -> None:
        """Save final tasks to file."""
        try:
            # Create capability directory
            capability_dir = self._output_dir / capability_name
            capability_dir.mkdir(parents=True, exist_ok=True)

            # Save tasks
            tasks_file = capability_dir / "tasks.json"
            with open(tasks_file, "w", encoding="utf-8") as f:
                json.dump({"tasks": tasks}, f, indent=2, ensure_ascii=False)

            log.info(
                f"Saved {len(tasks)} tasks for capability '{capability_name}' to {tasks_file}"
            )
        except Exception as e:
            log.error(f"Error saving tasks for capability {capability_name}: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise
