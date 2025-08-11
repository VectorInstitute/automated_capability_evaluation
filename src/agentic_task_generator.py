"""Multi-agent task generation system for generating tasks for each capability."""

import asyncio
import json
import logging
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger("agentic_task_gen")


@dataclass
class Capability:
    """A capability with name, description, domain, and area."""

    name: str
    description: str
    domain: str
    area: str


@dataclass
class ProblemProposalRequest:
    """Request for problem proposals from scientists."""

    capability_name: str
    capability_description: str
    capability_domain: str
    capability_area: str
    num_problems: int
    sample_tasks: Optional[List[str]] = None


@dataclass
class ScientistProblemProposal:
    """Problem proposal from a scientist."""

    scientist_id: str
    capability_name: str
    problems: Dict[str, str]  # task_id -> task_text
    iteration: int


@dataclass
class ModeratorProblemReview:
    """Moderator's review and filtering of problems."""

    capability_name: str
    final_problems: Dict[str, str]  # task_id -> task_text
    rejected_problems: Dict[str, str]  # task_id -> rejection_reason
    iteration: int


@dataclass
class SolutionRequest:
    """Request for scientists to solve problems."""

    capability_name: str
    capability_description: str
    capability_domain: str
    capability_area: str
    problems: Dict[str, str]  # task_id -> task_text


@dataclass
class ScientistSolutionProposal:
    """Solution proposal from a scientist."""

    scientist_id: str
    capability_name: str
    solutions: Dict[str, str]  # task_id -> solution


@dataclass
class FinalTaskSet:
    """Final task set with problems and solutions."""

    capability_name: str
    tasks: Dict[str, Dict[str, str]]  # task_id -> {problem, answer}


@default_subscription
class TaskScientist(RoutedAgent):
    """Scientist that generates problems and solutions."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        domain: str = "",
    ) -> None:
        super().__init__(f"Task Scientist {scientist_id}")
        self._scientist_id = scientist_id
        self._model_client = model_client
        self._domain = domain

    @message_handler
    async def handle_problem_proposal_request(
        self, message: ProblemProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle problem proposal request."""
        try:
            log.info(
                f"Task Scientist {self._scientist_id} generating {message.num_problems} problems for capability: {message.capability_name}"
            )

            sample_tasks_text = ""
            if message.sample_tasks:
                sample_tasks_text = "\n".join(
                    [f"- {task}" for task in message.sample_tasks]
                )
            else:
                sample_tasks_text = "(No sample tasks provided)"

            system_prompt = f"""You are Scientist {self._scientist_id}, an expert in designing tasks for evaluating a given capability. You will be shown the capability's name, description, domain, and a few sample tasks. Your goal is to propose novel, diverse, and non-trivial task problems that assess different aspects of this capability.

You will be particularly rewarded for:
- Ensuring clear alignment with the capability,
- Avoiding overlap or redundancy,
- Proposing tasks that vary in difficulty and structure.

Your response must follow this format exactly:
THOUGHT: <brief reasoning about the kind of tasks you're proposing>
RESPONSE JSON:
{{
  "task_1": "<TASK_TEXT_1>",
  "task_2": "<TASK_TEXT_2>",
  ...
}}

Make sure:
- All tasks are within the scope of the capability.
- Tasks are phrased as standalone problem descriptions, without any answers or solutions.
- LaTeX strings are properly escaped (e.g., \\\\[2x + 3 = 11\\\\]).
- Each task is distinct from the others and covers a different aspect or sub-skill."""

            user_prompt = f"""Design {message.num_problems} tasks for the following capability:

Name: {message.capability_name}
Description: {message.capability_description}
Domain: {message.capability_domain}
Sample tasks:
{sample_tasks_text}"""

            system_message = SystemMessage(content=system_prompt)
            user_message = UserMessage(content=user_prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            raw_content = model_result.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            # Extract JSON from response
            try:
                json_start = raw_content.find("{")
                if json_start != -1:
                    json_part = raw_content[json_start:]
                    problems = json.loads(json_part)
                else:
                    problems = {}
                    log.warning(
                        f"No JSON found in scientist {self._scientist_id} response"
                    )
            except Exception as e:
                log.error(
                    f"Error parsing JSON from scientist {self._scientist_id}: {e}"
                )
                problems = {}

            log.info(
                f"Task Scientist {self._scientist_id} proposing {len(problems)} problems for capability: {message.capability_name}"
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
            log.error(
                f"Error in Task Scientist {self._scientist_id} handle_problem_proposal_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_solution_request(
        self, message: SolutionRequest, ctx: MessageContext
    ) -> None:
        """Handle solution request for problems."""
        try:
            log.info(
                f"Task Scientist {self._scientist_id} solving {len(message.problems)} problems for capability: {message.capability_name}"
            )

            solutions = {}

            for task_id, problem_text in message.problems.items():
                system_prompt = f"""You are Scientist {self._scientist_id}, an expert in {message.capability_domain}. You are solving a task related to the capability: {message.capability_name}.

Provide a clear, accurate, and complete solution to the given problem. Your solution should be correct and well-reasoned."""

                user_prompt = f"""Solve the following problem:

{problem_text}

Provide your solution clearly and concisely."""

                system_message = SystemMessage(content=system_prompt)
                user_message = UserMessage(content=user_prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                raw_content = model_result.content
                if not isinstance(raw_content, str):
                    raw_content = str(raw_content)

                solutions[task_id] = raw_content

            log.info(
                f"Task Scientist {self._scientist_id} publishing solutions for capability: {message.capability_name}"
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
            log.error(
                f"Error in Task Scientist {self._scientist_id} handle_solution_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


@default_subscription
class TaskModerator(RoutedAgent):
    """Moderator that merges scientist task proposals and manages iteration."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        num_scientists: int,
        num_final_problems: int,
        buffer_param: int,
        agreement_threshold: float,
        output_dir: Path,
        domain: str,
    ) -> None:
        super().__init__("Task Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_final_problems = num_final_problems
        self._buffer_param = buffer_param
        self._agreement_threshold = agreement_threshold
        self._output_dir = output_dir
        self._domain = domain

        # Algorithm 1 state
        self._num_remaining: Dict[str, int] = {}
        self._final_problems: Dict[
            str, Dict[str, str]
        ] = {}  # capability -> {task_id: problem_text}
        self._capabilities: Dict[str, Capability] = {}  # Store original capability info

        # Problem design state
        self._problem_proposals: Dict[
            str, List[ScientistProblemProposal]
        ] = {}  # capability -> proposals

        # Solution design state
        self._solution_proposals: Dict[
            str, List[ScientistSolutionProposal]
        ] = {}  # capability -> solutions

    @message_handler
    async def handle_capability(self, message: Capability, ctx: MessageContext) -> None:
        """Handle capability and start Algorithm 1 for problem design."""
        try:
            log.info(
                f"Task Moderator starting problem design for capability: {message.name}"
            )

            # Initialize Algorithm 1 state
            self._num_remaining[message.name] = self._num_final_problems
            self._final_problems[message.name] = {}
            self._capabilities[message.name] = message  # Store original capability info

            await self._start_problem_iteration(message)

        except Exception as e:
            log.error(f"Error in Task Moderator handle_capability: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _start_problem_iteration(self, capability: Capability) -> None:
        """Start a problem generation iteration (Algorithm 1)."""
        try:
            num_remaining = self._num_remaining[capability.name]
            if num_remaining <= 0:
                log.info(
                    f"Problem design completed for capability: {capability.name}, starting solution design"
                )
                await self._start_solution_design(capability)
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
        """Filter and select problems using moderator LLM (Algorithm 1 steps 3-6)."""
        try:
            log.info(
                f"Task Moderator filtering problems for capability: {capability_name}"
            )

            # Get capability info
            # For now, we'll use the capability_name and
            # assume we can get info from context
            # In a real implementation, we might want to store capability objects

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

            system_prompt = """You are the Moderator overseeing capability-based task design. Your task is to review proposed tasks from multiple scientist agents and synthesize a final, high-quality task set for the capability.

Your responsibilities:
- Eliminate any task that is not clearly aligned with the capability.
- Merge or remove tasks that are redundant or overly similar.
- Ensure that the final set of tasks is diverse, non-trivial, and tests different facets of the capability.
- Include a brief justification for each rejected or significantly modified task.

Your response should follow this format exactly:

THOUGHT: <your summary of strengths and weaknesses of the proposed tasks and your curation plan>
RESPONSE JSON:
{
  "final_tasks": {
    "task_1": "<FINAL_TASK_1>",
    "task_2": "<FINAL_TASK_2>",
    ...
  },
  "rejected_tasks": {
    "task_from_scientist_A": "Reason for rejection or modification",
    "task_from_scientist_B": "Reason for rejection or modification",
    ...
  }
}"""

            capability_info = self._capabilities[capability_name]
            user_prompt = f"""Below is a capability and task proposals from multiple scientist agents. Curate the final task set by filtering, editing, or merging as needed.

Name: {capability_info.name}
Description: {capability_info.description}
Domain: {capability_info.domain}

Proposed Tasks:
{problems_text}"""

            system_message = SystemMessage(content=system_prompt)
            user_message = UserMessage(content=user_prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            raw_content = model_result.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            # Extract JSON from response
            try:
                json_start = raw_content.find("{")
                if json_start != -1:
                    json_part = raw_content[json_start:]
                    result = json.loads(json_part)
                    final_tasks = result.get("final_tasks", {})
                    rejected_tasks = result.get("rejected_tasks", {})
                else:
                    final_tasks = {}
                    rejected_tasks = {}
                    log.warning(
                        f"No JSON found in moderator response for {capability_name}"
                    )
            except Exception as e:
                log.error(f"Error parsing JSON from moderator: {e}")
                final_tasks = {}
                rejected_tasks = {}

            # Update Algorithm 1 state
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
            log.info(
                f"Rejected {len(rejected_tasks)} problems: {list(rejected_tasks.keys())}"
            )

            # Continue Algorithm 1 or move to solution design
            if self._num_remaining[capability_name] > 0:
                # Need more problems, start another iteration
                capability = self._capabilities[capability_name]
                await self._start_problem_iteration(capability)
            else:
                # Problem design complete, start solution design
                capability = self._capabilities[capability_name]
                await self._start_solution_design(capability)

        except Exception as e:
            log.error(f"Error in Task Moderator _filter_and_select_problems: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _start_solution_design(self, capability: Capability) -> None:
        """Start solution design phase."""
        try:
            log.info(
                f"Task Moderator starting solution design for capability: {capability.name}"
            )

            final_problems = self._final_problems[capability.name]
            if not final_problems:
                log.error(
                    f"No final problems available for capability: {capability.name}"
                )
                return

            # Send solution requests to all scientists
            await self.publish_message(
                SolutionRequest(
                    capability_name=capability.name,
                    capability_description=capability.description,
                    capability_domain=capability.domain,
                    capability_area=capability.area,
                    problems=final_problems,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(f"Error in Task Moderator _start_solution_design: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_scientist_solution_proposal(
        self, message: ScientistSolutionProposal, ctx: MessageContext
    ) -> None:
        """Handle solution proposals from scientists."""
        try:
            log.info(
                f"Task Moderator received solution proposal from Scientist {message.scientist_id} for capability: {message.capability_name}"
            )

            capability_name = message.capability_name
            if capability_name not in self._solution_proposals:
                self._solution_proposals[capability_name] = []

            self._solution_proposals[capability_name].append(message)

            # Check if we have all solutions
            if len(self._solution_proposals[capability_name]) == self._num_scientists:
                log.info(
                    f"Task Moderator received all solutions for capability: {capability_name}, determining consensus"
                )
                await self._determine_solution_consensus(capability_name)

        except Exception as e:
            log.error(
                f"Error in Task Moderator handle_scientist_solution_proposal: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _determine_solution_consensus(self, capability_name: str) -> None:
        """Determine solution consensus and finalize tasks."""
        try:
            log.info(
                f"Task Moderator determining solution consensus for capability: {capability_name}"
            )

            solutions_by_task: Dict[
                str, Dict[str, str]
            ] = {}  # task_id -> [scientist_id -> solution]

            for proposal in self._solution_proposals[capability_name]:
                for task_id, solution in proposal.solutions.items():
                    if task_id not in solutions_by_task:
                        solutions_by_task[task_id] = {}
                    solutions_by_task[task_id][proposal.scientist_id] = solution

            final_tasks = {}

            for task_id, problem_text in self._final_problems[capability_name].items():
                if task_id in solutions_by_task:
                    scientist_solutions = solutions_by_task[task_id]

                    # Simple consensus: find most common solution
                    solution_counts: Dict[str, int] = {}
                    for solution in scientist_solutions.values():
                        solution_counts[solution] = solution_counts.get(solution, 0) + 1

                    if solution_counts:
                        most_common_solution = max(
                            solution_counts.keys(), key=lambda x: solution_counts[x]
                        )
                        agreement_rate = solution_counts[most_common_solution] / len(
                            scientist_solutions
                        )

                        if agreement_rate >= self._agreement_threshold:
                            final_tasks[task_id] = {
                                "problem": problem_text,
                                "answer": most_common_solution,
                            }
                            log.info(
                                f"Task {task_id}: consensus achieved ({agreement_rate:.2f} agreement)"
                            )
                        else:
                            log.warning(
                                f"Task {task_id}: low agreement ({agreement_rate:.2f}), requires human review"
                            )
                            # For now, use most common solution but mark it
                            final_tasks[task_id] = {
                                "problem": problem_text,
                                "answer": most_common_solution,
                                "requires_human_review": "true",
                                "agreement_rate": str(agreement_rate),
                            }

            # Save final tasks
            await self._save_tasks_to_file(capability_name, final_tasks)
            log.info(f"Task generation completed for capability: {capability_name}")

        except Exception as e:
            log.error(f"Error in Task Moderator _determine_solution_consensus: {e}")
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


async def generate_tasks_for_capability(
    cfg: DictConfig, capability: Capability, output_dir: Path
) -> None:
    """Generate tasks for a single capability."""
    try:
        log.info(f"Generating tasks for capability: {capability.name}")

        runtime = SingleThreadedAgentRuntime()

        # Register scientists
        await TaskScientist.register(
            runtime,
            "TaskScientistA",
            lambda: TaskScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_a.name,
                    seed=cfg.agents.scientist_a.seed,
                ),
                scientist_id="A",
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        await TaskScientist.register(
            runtime,
            "TaskScientistB",
            lambda: TaskScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_b.name,
                    seed=cfg.agents.scientist_b.seed,
                ),
                scientist_id="B",
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        # Register moderator
        await TaskModerator.register(
            runtime,
            "TaskModerator",
            lambda: TaskModerator(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.moderator.name,
                    seed=cfg.agents.moderator.seed,
                ),
                num_scientists=cfg.task_cfg.num_scientists,
                num_final_problems=cfg.task_cfg.num_final_problems,
                buffer_param=cfg.task_cfg.buffer_param,
                agreement_threshold=cfg.task_cfg.agreement_threshold,
                output_dir=output_dir,
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        # Start runtime and process the capability
        runtime.start()
        await runtime.publish_message(capability, DefaultTopicId())
        log.info(f"Capability message published: {capability.name}")

        # Wait for the runtime to stop when idle
        try:
            await runtime.stop_when_idle()
            log.info(f"Completed generating tasks for capability: {capability.name}")
        except Exception as e:
            log.error(
                f"Error while generating tasks for capability {capability.name}: {e}"
            )
            raise

    except Exception as e:
        log.error(f"Error in generating tasks for {capability.name}: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


async def generate_tasks(cfg: DictConfig) -> None:
    """Generate tasks for all capabilities."""
    try:
        log.info("Starting task generation process")

        # Read capabilities from all areas
        capabilities_dir = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain
            / cfg.exp_cfg.exp_id
            / "capabilities"
        )

        if not capabilities_dir.exists():
            raise FileNotFoundError(
                f"Capabilities directory not found: {capabilities_dir}"
            )

        capabilities = []

        # Iterate through area directories
        for area_dir in capabilities_dir.iterdir():
            if area_dir.is_dir():
                capabilities_file = area_dir / "capabilities.json"
                if capabilities_file.exists():
                    with open(capabilities_file, "r", encoding="utf-8") as f:
                        capabilities_data = json.load(f)

                    if (
                        isinstance(capabilities_data, dict)
                        and "capabilities" in capabilities_data
                    ):
                        for cap_dict in capabilities_data["capabilities"]:
                            if (
                                isinstance(cap_dict, dict)
                                and "name" in cap_dict
                                and "description" in cap_dict
                            ):
                                capabilities.append(
                                    Capability(
                                        name=cap_dict["name"],
                                        description=cap_dict["description"],
                                        domain=cap_dict.get(
                                            "domain", cfg.capabilities_cfg.domain
                                        ),
                                        area=cap_dict.get("area", area_dir.name),
                                    )
                                )

        if not capabilities:
            raise ValueError(f"No valid capabilities found in {capabilities_dir}")

        log.info(f"Found {len(capabilities)} capabilities to process")

        output_dir = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain
            / cfg.exp_cfg.exp_id
            / "tasks"
        )
        log.info(f"Output directory: {output_dir}")

        # Process each capability individually
        for i, capability in enumerate(capabilities):
            log.info(
                f"Processing capability {i + 1}/{len(capabilities)}: {capability.name}"
            )

            await generate_tasks_for_capability(cfg, capability, output_dir)

            log.info(
                f"Completed capability {i + 1}/{len(capabilities)}: {capability.name}"
            )

            await asyncio.sleep(1)

    except Exception as e:
        log.error(f"Error in generate_tasks: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent task generation system."""
    log.info("Starting multi-agent task generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        asyncio.run(generate_tasks(cfg))
    except Exception as e:
        log.error(f"Task generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
