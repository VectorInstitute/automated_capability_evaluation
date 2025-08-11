"""Multi-agent debate system for generating capabilities for each area."""

import asyncio
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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


log = logging.getLogger("agentic_cap_gen")

DEFAULT_CAPABILITIES_JSON = '{"capabilities": []}'


@dataclass
class Area:
    """A capability area with name and description."""

    name: str
    description: str


@dataclass
class CapabilityProposalRequest:
    """Initial request for capability proposals from scientists."""

    area_name: str
    area_description: str
    num_capabilities: int


@dataclass
class ScientistCapabilityProposal:
    """Capability proposal from a scientist."""

    scientist_id: str
    proposal: str
    area_name: str
    round: int


@dataclass
class CapabilityRevisionRequest:
    """Request for scientist to review and revise moderator's proposal."""

    scientist_id: str
    moderator_proposal: str
    area_name: str
    round: int


def normalize_capabilities(s: str, expected: int, domain: str = "") -> str:
    """Ensure payload has JSON with a 'capabilities' list of size <= expected."""
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            if "capabilities" in data and isinstance(data["capabilities"], list):
                data["capabilities"] = data["capabilities"][:expected]
                return json.dumps(data, indent=2)
            capabilities: List[Dict[str, Any]] = []
            i = 0
            while f"capability_{i}" in data and len(capabilities) < expected:
                cap = data[f"capability_{i}"]
                capabilities.append(cap)
                i += 1
            return json.dumps({"capabilities": capabilities}, indent=2)
    except Exception:
        pass
    return s


@default_subscription
class CapabilityScientist(RoutedAgent):
    """Generates capabilities through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        max_round: int,
        expected_capabilities: int,
        domain: str = "",
    ) -> None:
        super().__init__(f"Capability Scientist {scientist_id}")
        self._scientist_id = scientist_id
        self._model_client = model_client
        self._max_round = max_round
        self._expected_capabilities = expected_capabilities
        self._domain = domain
        self._round = 0

    @message_handler
    async def handle_capability_proposal_request(
        self, message: CapabilityProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle initial capability proposal request."""
        try:
            log.info(
                f"Capability Scientist {self._scientist_id} handling proposal request for area: {message.area_name}"
            )

            prompt = f"""You are Scientist {self._scientist_id}. You have been assigned the area: "{message.area_name}".

Your task is to propose {message.num_capabilities} specific, **non-overlapping capabilities** within this area that test different aspects of LLM performance.

Each capability should:
- Be clearly within the scope of the area.
- Be distinct from the others (no overlap).
- Be testable via concrete tasks in later stages.

Provide each capability with:
1. A concise name (lowercase_with_underscores).
2. A 2–3 sentence description justifying its purpose.

Output format:
RESPONSE JSON:
{{
  "capability_0": {{
    "name": <STR>,
    "description": <STR>,
    "area": "{message.area_name}"
  }},
  ...
}}

Area Description: {message.area_description}"""

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
                f"Capability Scientist {self._scientist_id} publishing capability proposal for area: {message.area_name}"
            )
            await self.publish_message(
                ScientistCapabilityProposal(
                    scientist_id=self._scientist_id,
                    proposal=raw_content,
                    area_name=message.area_name,
                    round=self._round,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(
                f"Error in Capability Scientist {self._scientist_id} handle_capability_proposal_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_revision_request(
        self, message: CapabilityRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request from moderator."""
        try:
            if message.scientist_id != self._scientist_id:
                return  # Not for this scientist

            log.info(
                f"Capability Scientist {self._scientist_id} handling revision request for area: {message.area_name}, round {message.round}"
            )

            prompt = f"""You are Scientist {self._scientist_id}. The Moderator has proposed a merged list of capabilities for the area "{message.area_name}".

Moderator's Proposal:
{message.moderator_proposal}

Please review and revise the merged capability list by:
- Clarifying or refining capability descriptions.
- Flagging capabilities that may be overlapping or vague.
- Proposing any additions or deletions if you believe something important is missing or redundant.

Be concise and constructive in your revisions.

Return the updated list in the following format:
RESPONSE JSON:
{{
  "capability_0": {{
    "name": <STR>,
    "description": <STR>,
    "area": "{message.area_name}"
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
                f"Capability Scientist {self._scientist_id} publishing revised proposal for area: {message.area_name}, round {message.round}"
            )
            await self.publish_message(
                ScientistCapabilityProposal(
                    scientist_id=self._scientist_id,
                    proposal=raw_content,
                    area_name=message.area_name,
                    round=message.round,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(
                f"Error in Capability Scientist {self._scientist_id} handle_revision_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


@default_subscription
class CapabilityModerator(RoutedAgent):
    """Moderator that merges scientist capability proposals and manages iteration."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        num_scientists: int,
        num_capabilities: int,
        max_round: int,
        output_dir: Path,
        domain: str,
    ) -> None:
        super().__init__("Capability Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_capabilities = num_capabilities
        self._max_round = max_round
        self._output_dir = output_dir
        self._domain = domain
        self._round = 0
        self._proposals_buffer: Dict[
            str, Dict[int, List[ScientistCapabilityProposal]]
        ] = {}

    @message_handler
    async def handle_area(self, message: Area, ctx: MessageContext) -> None:
        """Handle area messages and initiate capability proposal process."""
        try:
            log.info(f"Capability Moderator received area: {message.name}")

            # Send initial proposal request to scientists
            await self.publish_message(
                CapabilityProposalRequest(
                    area_name=message.name,
                    area_description=message.description,
                    num_capabilities=self._num_capabilities,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(f"Error in Capability Moderator handle_area: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_scientist_proposal(
        self, message: ScientistCapabilityProposal, ctx: MessageContext
    ) -> None:
        """Handle capability proposals from scientists."""
        try:
            log.info(
                f"Capability Moderator received proposal from Scientist {message.scientist_id} for area: {message.area_name}, round {message.round}"
            )

            area_key = message.area_name
            if area_key not in self._proposals_buffer:
                self._proposals_buffer[area_key] = {}

            self._proposals_buffer[area_key].setdefault(message.round, []).append(
                message
            )

            if (
                len(self._proposals_buffer[area_key][message.round])
                == self._num_scientists
            ):
                log.info(
                    f"Capability Moderator received all proposals for area: {message.area_name}, round {message.round}, proceeding to merge"
                )

                # Get proposals from both scientists
                proposals = self._proposals_buffer[area_key][message.round]
                scientist_a_proposal = next(
                    p.proposal for p in proposals if p.scientist_id == "A"
                )
                scientist_b_proposal = next(
                    p.proposal for p in proposals if p.scientist_id == "B"
                )

                await self._merge_proposals(
                    scientist_a_proposal,
                    scientist_b_proposal,
                    message.area_name,
                    message.round,
                )

        except Exception as e:
            log.error(f"Error in Capability Moderator handle_scientist_proposal: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _merge_proposals(
        self,
        scientist_a_proposal: str,
        scientist_b_proposal: str,
        area_name: str,
        round_num: int,
    ) -> None:
        """Merge scientist proposals using LLM."""
        try:
            log.info(
                f"Capability Moderator merging proposals for area: {area_name}, round {round_num}"
            )

            finalized_instruction = ""
            if round_num >= self._max_round - 1:
                finalized_instruction = """
If, after incorporating feedback or upon review, you judge the merged set to be clear, comprehensive, and non-overlapping within the area, you may declare the capability design finalized.
To finalize, add the field:
"finalized": true
at the end of your JSON response."""

            prompt = f"""You are the Moderator. Two scientist agents have independently proposed a list of capabilities within the capability area: "{area_name}".

Below are their proposals:

Scientist A Proposal:
{scientist_a_proposal}

Scientist B Proposal:
{scientist_b_proposal}

Your task is to merge these proposals into a unified set of capabilities for the area. In doing so:
- Eliminate redundancy and overlapping capabilities.
- Ensure all capabilities are clearly within the scope of the area.
- Ensure all capabilities are distinct from one another.
- Improve clarity and precision in naming and descriptions, where needed.

You will then submit this merged capability list for review by the scientist agents. If either scientist provides substantive suggestions, you may revise the list and initiate another round of review.{finalized_instruction}

Present the merged capabilities in the following format:
{{
  "capability_0": {{
    "name": "<STR>",
    "description": "<STR>",
    "area": "{area_name}"
  }},
  ...{', "finalized": <true|false>' if finalized_instruction else ""}
}}

Be thoughtful and concise in your output."""

            system_message = SystemMessage(
                content="You are an expert moderator specializing in capability design for LLM evaluation."
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
                json_start = raw_content.find("{")
                if json_start != -1:
                    json_part = raw_content[json_start:]
                    parsed = json.loads(json_part)
                    is_finalized = parsed.get("finalized", False)
            except Exception:
                pass

            if is_finalized or round_num >= self._max_round - 1:
                log.info(
                    f"Capability Moderator finalizing capabilities for area: {area_name} after round {round_num}"
                )
                await self._finalize_capabilities(raw_content, area_name)
            else:
                log.info(
                    f"Capability Moderator sending merged proposal for revision in area: {area_name}, round {round_num}"
                )
                next_round = round_num + 1

                # Send to scientists for revision
                await self.publish_message(
                    CapabilityRevisionRequest(
                        scientist_id="A",
                        moderator_proposal=raw_content,
                        area_name=area_name,
                        round=next_round,
                    ),
                    topic_id=DefaultTopicId(),
                )
                await self.publish_message(
                    CapabilityRevisionRequest(
                        scientist_id="B",
                        moderator_proposal=raw_content,
                        area_name=area_name,
                        round=next_round,
                    ),
                    topic_id=DefaultTopicId(),
                )

        except Exception as e:
            log.error(f"Error in Capability Moderator _merge_proposals: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _finalize_capabilities(
        self, final_capabilities: str, area_name: str
    ) -> None:
        """Save final capabilities to file."""
        try:
            log.info(
                f"Capability Moderator finalizing and saving capabilities for area: {area_name}"
            )

            # Convert to the expected format with "capabilities" list
            try:
                json_start = final_capabilities.find("{")
                if json_start != -1:
                    json_part = final_capabilities[json_start:]
                    parsed = json.loads(json_part)
                else:
                    parsed = json.loads(final_capabilities)

                if "finalized" in parsed:
                    del parsed["finalized"]  # Remove finalized flag

                # Convert capability_0, capability_1 format to capabilities list
                capabilities_list = []
                i = 0
                while f"capability_{i}" in parsed:
                    cap = parsed[f"capability_{i}"]
                    # Add domain and area fields manually
                    if isinstance(cap, dict):
                        if "domain" not in cap:
                            cap["domain"] = self._domain
                        if "area" not in cap:
                            cap["area"] = area_name
                    capabilities_list.append(cap)
                    i += 1

                final_format = {"capabilities": capabilities_list}
                final_capabilities_json = json.dumps(final_format, indent=2)
            except Exception as e:
                log.warning(f"Could not parse final capabilities JSON: {e}")
                final_capabilities_json = final_capabilities

            self._save_capabilities_to_file(final_capabilities_json, area_name)
            log.info(
                f"Capability generation for area '{area_name}' completed successfully"
            )

        except Exception as e:
            log.error(f"Error in Capability Moderator _finalize_capabilities: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _save_capabilities_to_file(self, capabilities: str, area_name: str) -> None:
        """Save the generated capabilities JSON payload to disk under the area."""
        try:
            # Ensure output dir exists
            area_dir = self._output_dir / area_name
            area_dir.mkdir(parents=True, exist_ok=True)

            try:
                parsed = json.loads(capabilities)
            except json.JSONDecodeError:
                # Wrap raw string if not valid JSON
                parsed = {"capabilities": capabilities}

            # Write to file
            out_path = area_dir / "capabilities.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            log.info(f"Saved capabilities JSON for area '{area_name}' to {out_path}")
        except Exception as e:
            log.error(f"Failed to save capabilities for area {area_name}: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


async def generate_capabilities_for_area(
    cfg: DictConfig, area: Area, output_dir: Path
) -> None:
    """Generate capabilities for a single area."""
    try:
        log.info(f"Generating capabilities for area: {area.name}")

        runtime = SingleThreadedAgentRuntime()

        await CapabilityScientist.register(
            runtime,
            "CapabilityScientistA",
            lambda: CapabilityScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_a.name,
                    seed=cfg.agents.scientist_a.seed,
                ),
                scientist_id="A",
                max_round=cfg.debate_cfg.max_round,
                expected_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        await CapabilityScientist.register(
            runtime,
            "CapabilityScientistB",
            lambda: CapabilityScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_b.name,
                    seed=cfg.agents.scientist_b.seed,
                ),
                scientist_id="B",
                max_round=cfg.debate_cfg.max_round,
                expected_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        await CapabilityModerator.register(
            runtime,
            "CapabilityModerator",
            lambda: CapabilityModerator(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.moderator.name,
                    seed=cfg.agents.moderator.seed,
                ),
                num_scientists=2,
                num_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                max_round=cfg.debate_cfg.max_round,
                output_dir=output_dir,
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        # Start runtime and process the area
        runtime.start()
        await runtime.publish_message(area, DefaultTopicId())
        log.info(f"Area message published: {area.name}")

        # Wait for the runtime to stop when idle
        try:
            await runtime.stop_when_idle()
            log.info(f"Completed generating capabilities for area: {area.name}")
        except Exception as e:
            log.error(f"Error while generating capabilities for area {area.name}: {e}")
            raise

    except Exception as e:
        log.error(f"Error in generating capabilities for {area.name}: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


async def generate_capabilities(cfg: DictConfig) -> None:
    """Generate capabilities using multi-agent debate system for each area."""
    try:
        log.info("Starting capability generation process")

        # Read areas from the areas.json file
        areas_file = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain
            / cfg.exp_cfg.exp_id
            / "areas"
            / "areas.json"
        )

        if not areas_file.exists():
            raise FileNotFoundError(f"Areas file not found: {areas_file}")

        with open(areas_file, "r", encoding="utf-8") as f:
            areas_data = json.load(f)

        # Parse areas from the JSON data
        areas = []
        if isinstance(areas_data, dict) and "areas" in areas_data:
            for area_dict in areas_data["areas"]:
                if (
                    isinstance(area_dict, dict)
                    and "name" in area_dict
                    and "description" in area_dict
                ):
                    areas.append(
                        Area(
                            name=area_dict["name"], description=area_dict["description"]
                        )
                    )

        if not areas:
            raise ValueError(f"No valid areas found in {areas_file}")

        log.info(
            f"Found {len(areas)} areas to process: {[area.name for area in areas]}"
        )

        output_dir = (
            Path.home()
            / cfg.debate_cfg.output_dir
            / cfg.capabilities_cfg.domain
            / cfg.exp_cfg.exp_id
            / "capabilities"
        )
        log.info(f"Output directory: {output_dir}")

        # Process each area individually with fresh agents
        for i, area in enumerate(areas):
            log.info(f"Processing area {i + 1}/{len(areas)}: {area.name}")

            await generate_capabilities_for_area(cfg, area, output_dir)

            log.info(f"Completed area {i + 1}/{len(areas)}: {area.name}")

            await asyncio.sleep(1)

    except Exception as e:
        log.error(f"Error in generate_capabilities: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")
        raise


@hydra.main(version_base=None, config_path="cfg", config_name="agentic_config")
def main(cfg: DictConfig) -> None:
    """Run the multi-agent debate-based capability generation system."""
    log.info("Starting multi-agent debate-based capability generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        asyncio.run(generate_capabilities(cfg))
    except Exception as e:
        log.error(f"Capability generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
