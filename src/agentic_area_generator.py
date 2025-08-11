"""Multi-agent debate system for generating capability areas."""

import asyncio
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


log = logging.getLogger("agentic_area_gen")

DEFAULT_AREAS_JSON = '{"areas": []}'
DEFAULT_NUM_SCIENTISTS = 2


@dataclass
class Domain:
    """A domain of capability areas."""

    name: str


@dataclass
class AreaProposalRequest:
    """Initial request for area proposals from scientists."""

    domain: str
    num_areas: int


@dataclass
class ScientistAreaProposal:
    """Area proposal from a scientist."""

    scientist_id: str
    proposal: str
    round: int


@dataclass
class ModeratorMergeRequest:
    """Request for moderator to merge scientist proposals."""

    domain: str
    num_final_areas: int
    scientist_a_proposal: str
    scientist_b_proposal: str
    round: int


@dataclass
class ModeratorMergedProposal:
    """Merged proposal from moderator."""

    merged_proposal: str
    round: int
    is_finalized: bool


@dataclass
class ScientistRevisionRequest:
    """Request for scientist to review and revise moderator's proposal."""

    scientist_id: str
    moderator_proposal: str
    round: int


@dataclass
class FinalAreasResponse:
    """Final areas response."""

    areas: str


def normalize_areas(s: str, expected: int) -> str:
    """Ensure payload has JSON with an 'areas' list of size <= expected."""
    try:
        data = json.loads(s)
        if isinstance(data, dict) and isinstance(data.get("areas"), list):
            data["areas"] = data["areas"][:expected]
            return json.dumps(data, indent=2)
    except Exception:
        pass
    return s


@default_subscription
class AreaScientist(RoutedAgent):
    """A scientist that generates capability areas through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        max_round: int,
    ) -> None:
        super().__init__(f"Area Scientist {scientist_id}")
        self._scientist_id = scientist_id
        self._model_client = model_client
        self._max_round = max_round
        self._round = 0

    @message_handler
    async def handle_area_proposal_request(
        self, message: AreaProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle initial area proposal request."""
        try:
            log.info(
                f"Scientist {self._scientist_id} handling area proposal request for domain: {message.domain}"
            )

            prompt = f"""You are Scientist {self._scientist_id}. You are an expert in evaluating large language models (LLMs) in the domain of {message.domain}. Your task is to independently propose a list of {message.num_areas} high-level, non-overlapping **capability areas** that collectively cover the space of skills relevant to this domain.

Each area should:
- Represent a broad but distinct dimension of LLM competence.
- Be clearly distinct from the other proposed areas (no overlap).
- Contain enough conceptual room to allow for multiple fine-grained capabilities in the next stage.

For each area, provide:
1. A short name (a few words).
2. A 2–3 sentence description that defines its boundaries and justifies its inclusion.

Please return your proposal in the following format:
RESPONSE JSON:
{{
  "area_0": {{
    "name": <STR>,
    "description": <STR>
  }},
  ...
}}"""

            system_message = SystemMessage(
                content="You are an expert capability researcher specializing in LLM evaluation."
            )
            user_message = UserMessage(content=prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            raw_content = model_result.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            log.info(f"Scientist {self._scientist_id} publishing area proposal")
            await self.publish_message(
                ScientistAreaProposal(
                    scientist_id=self._scientist_id,
                    proposal=raw_content,
                    round=self._round,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(
                f"Error in Scientist {self._scientist_id} handle_area_proposal_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_revision_request(
        self, message: ScientistRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request from moderator."""
        try:
            if message.scientist_id != self._scientist_id:
                return  # Not for this scientist

            log.info(
                f"Scientist {self._scientist_id} handling revision request for round {message.round}"
            )

            prompt = f"""You are Scientist {self._scientist_id}. You are reviewing the merged set of capability areas proposed by the Moderator.

Moderator's Proposal:
{message.moderator_proposal}

Please review the proposed areas carefully and suggest any of the following:
- Minor refinements or clarifications to area descriptions.
- Proposed merges/splits where you see overlap or conceptual drift.
- Additions of missing areas or removal of unneeded ones.

Keep your feedback constructive and focused on improving clarity, coverage, and non-overlap. Avoid unnecessary changes.

Return your revised proposal in the following format:
RESPONSE JSON:
{{
  "area_0": {{
    "name": <STR>,
    "description": <STR>
  }},
  ...
}}"""

            system_message = SystemMessage(
                content="You are an expert capability researcher specializing in LLM evaluation."
            )
            user_message = UserMessage(content=prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            raw_content = model_result.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            log.info(
                f"Scientist {self._scientist_id} publishing revised proposal for round {message.round}"
            )
            await self.publish_message(
                ScientistAreaProposal(
                    scientist_id=self._scientist_id,
                    proposal=raw_content,
                    round=message.round,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(
                f"Error in Scientist {self._scientist_id} handle_revision_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


@default_subscription
class AreaModerator(RoutedAgent):
    """Moderator that merges scientist proposals and manages iteration."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        num_scientists: int,
        num_final_areas: int,
        max_round: int,
        output_dir: Path,
    ) -> None:
        super().__init__("Area Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_final_areas = num_final_areas
        self._max_round = max_round
        self._output_dir = output_dir
        self._round = 0
        self._proposals_buffer: Dict[int, List[ScientistAreaProposal]] = {}
        self._domain = ""

    @message_handler
    async def handle_domain(self, message: Domain, ctx: MessageContext) -> None:
        """Handle the domain message and initiate area proposal process."""
        try:
            log.info(f"Moderator received domain: {message.name}")
            self._domain = message.name

            # Send initial proposal request to scientists
            await self.publish_message(
                AreaProposalRequest(
                    domain=message.name, num_areas=self._num_final_areas
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(f"Error in Moderator handle_domain: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_scientist_proposal(
        self, message: ScientistAreaProposal, ctx: MessageContext
    ) -> None:
        """Handle area proposals from scientists."""
        try:
            log.info(
                f"Moderator received proposal from Scientist {message.scientist_id} for round {message.round}"
            )

            self._proposals_buffer.setdefault(message.round, []).append(message)

            if len(self._proposals_buffer[message.round]) == self._num_scientists:
                log.info(
                    f"Moderator received all proposals for round {message.round}, proceeding to merge"
                )

                # Get proposals from both scientists
                proposals = self._proposals_buffer[message.round]
                scientist_a_proposal = next(
                    p.proposal for p in proposals if p.scientist_id == "A"
                )
                scientist_b_proposal = next(
                    p.proposal for p in proposals if p.scientist_id == "B"
                )

                await self._merge_proposals(
                    scientist_a_proposal, scientist_b_proposal, message.round
                )

        except Exception as e:
            log.error(f"Error in Moderator handle_scientist_proposal: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _merge_proposals(
        self, scientist_a_proposal: str, scientist_b_proposal: str, round_num: int
    ) -> None:
        """Merge scientist proposals using LLM."""
        try:
            log.info(f"Moderator merging proposals for round {round_num}")

            finalized_instruction = ""
            if round_num >= self._max_round - 1:
                finalized_instruction = """
If you judge the merged set to be clear, comprehensive, and non-overlapping, you may declare the area design finalized.
To finalize, add the field:
"finalized": true
at the end of the JSON response."""

            prompt = f"""You are the Moderator. Two scientist agents have independently proposed a list of high-level capability areas for evaluating large language models in the domain of {self._domain}.

Below are their proposals:

Scientist A Proposal:
{scientist_a_proposal}

Scientist B Proposal:
{scientist_b_proposal}

Your task is to merge their proposals into a unified set of {self._num_final_areas} areas. In doing so:
- Eliminate overlaps and redundant areas.
- Justify any removals, merges, or renamings.
- Ensure that the final set is mutually exclusive and collectively exhaustive for this domain.

You will then submit this merged proposal for review by the scientist agents. If either scientist provides substantive suggestions, you may revise the proposal and initiate another round of review.{finalized_instruction}

Present the merged areas in the following format:
{{
  "area_0": {{
    "name": <STR>,
    "description": <STR>
  }},
  ...{', "finalized": <true|false>' if finalized_instruction else ""}
}}

Be thoughtful and concise in your output."""

            system_message = SystemMessage(
                content="You are an expert moderator specializing in capability area design for LLM evaluation."
            )
            user_message = UserMessage(content=prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            raw_content = model_result.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            # Check if finalized
            is_finalized = False
            try:
                parsed = json.loads(raw_content)
                is_finalized = parsed.get("finalized", False)
            except Exception:
                pass

            if is_finalized or round_num >= self._max_round - 1:
                log.info(f"Moderator finalizing areas after round {round_num}")
                await self._finalize_areas(raw_content)
            else:
                log.info(
                    f"Moderator sending merged proposal for revision in round {round_num}"
                )
                self._round = round_num + 1

                # Send to scientists for revision
                await self.publish_message(
                    ScientistRevisionRequest(
                        scientist_id="A",
                        moderator_proposal=raw_content,
                        round=self._round,
                    ),
                    topic_id=DefaultTopicId(),
                )
                await self.publish_message(
                    ScientistRevisionRequest(
                        scientist_id="B",
                        moderator_proposal=raw_content,
                        round=self._round,
                    ),
                    topic_id=DefaultTopicId(),
                )

        except Exception as e:
            log.error(f"Error in Moderator _merge_proposals: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _finalize_areas(self, final_areas: str) -> None:
        """Save final areas to file."""
        try:
            log.info("Moderator finalizing and saving areas")

            # Convert to the expected format with "areas" list
            try:
                json_start = final_areas.find("{")
                if json_start != -1:
                    json_part = final_areas[json_start:]
                    parsed = json.loads(json_part)
                else:
                    parsed = json.loads(final_areas)

                if "finalized" in parsed:
                    del parsed["finalized"]  # Remove finalized flag

                # Convert area_0, area_1 format to areas list
                areas_list = []
                i = 0
                while f"area_{i}" in parsed:
                    areas_list.append(parsed[f"area_{i}"])
                    i += 1

                final_format = {"areas": areas_list}
                final_areas_json = json.dumps(final_format, indent=2)
            except Exception as e:
                log.warning(f"Could not parse final areas JSON: {e}")
                final_areas_json = final_areas

            self._save_areas_to_file(final_areas_json)
            log.info("Area generation completed successfully")

        except Exception as e:
            log.error(f"Error in Moderator _finalize_areas: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _save_areas_to_file(self, areas: str) -> None:
        """Save the generated areas to a file in the specified directory structure."""
        try:
            # Create the output directory if it doesn't exist
            self._output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Created output directory: {self._output_dir}")

            # Save as JSON file
            areas_file = self._output_dir / "areas.json"

            try:
                # Try to parse as JSON first, if it's already JSON format
                areas_data = json.loads(areas) if isinstance(areas, str) else areas
            except json.JSONDecodeError as e:
                log.warning(f"Areas string is not valid JSON, wrapping it: {e}")
                # If not valid JSON, wrap in a simple structure
                areas_data = {
                    "raw_areas": areas,
                    "error": "Original content was not valid JSON",
                }

            with open(areas_file, "w", encoding="utf-8") as f:
                json.dump(areas_data, f, indent=2, ensure_ascii=False)

            # Also save as text file for easy reading
            areas_text_file = self._output_dir / "areas.txt"
            with open(areas_text_file, "w", encoding="utf-8") as f:
                f.write(str(areas))

            log.info(f"Areas saved to {areas_file} and {areas_text_file}")

        except Exception as e:
            log.error(f"Failed to save areas to file: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


async def generate_areas(cfg: DictConfig) -> None:
    """Generate areas using multi-agent debate system."""
    try:
        log.info("Starting area generation process")

        max_round = cfg.debate_cfg.max_round
        runtime = SingleThreadedAgentRuntime()

        output_dir = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain
            / cfg.exp_cfg.exp_id
            / "areas"
        )
        log.info(f"Output directory: {output_dir}")

        await AreaScientist.register(
            runtime,
            "AreaScientistA",
            lambda: AreaScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_a.name,
                    seed=cfg.agents.scientist_a.seed,
                ),
                scientist_id="A",
                max_round=max_round,
            ),
        )

        await AreaScientist.register(
            runtime,
            "AreaScientistB",
            lambda: AreaScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_b.name,
                    seed=cfg.agents.scientist_b.seed,
                ),
                scientist_id="B",
                max_round=max_round,
            ),
        )

        await AreaModerator.register(
            runtime,
            "AreaModerator",
            lambda: AreaModerator(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.moderator.name,
                    seed=cfg.agents.moderator.seed,
                ),
                num_scientists=DEFAULT_NUM_SCIENTISTS,
                num_final_areas=cfg.capabilities_cfg.num_capability_areas,
                max_round=max_round,
                output_dir=output_dir,
            ),
        )

        # Use domain from config
        domain = Domain(name=cfg.capabilities_cfg.domain)
        runtime.start()
        await runtime.publish_message(domain, DefaultTopicId())
        log.info(f"Domain message published: {domain.name}")

        # Wait for the runtime to stop when idle.
        try:
            await runtime.stop_when_idle()
            log.info("Runtime stopped when idle")
        except Exception as e:
            log.error(f"Error while waiting for runtime to stop: {e}")
            raise

    except Exception as e:
        log.error(f"Error in generate_areas: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based area generation system."""
    log.info("Starting multi-agent debate-based area generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        asyncio.run(generate_areas(cfg))
    except Exception as e:
        log.error(f"Area generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
