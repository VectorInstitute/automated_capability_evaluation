"""Multi-agent debate system for generating capabilities for each area."""

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
    TypeSubscription,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from omegaconf import DictConfig, OmegaConf

from src.utils.agentic_prompts import (
    CAPABILITY_GENERATION_SCIENTIST_PROMPT,
)


log = logging.getLogger("agentic_cap_gen")

DEFAULT_CAPABILITIES_JSON = '{"capabilities": []}'


@dataclass
class Area:
    """A capability area with name and description."""

    name: str
    description: str


@dataclass
class CapabilityRequest:
    """A request for capabilities within an area."""

    content: str
    area_name: str
    area_description: str


@dataclass
class IntermediateCapabilityResponse:
    """An intermediate response from a scientist."""

    capabilities: str
    round: int
    area_name: str


@dataclass
class FinalCapabilityResponse:
    """A final response from a scientist."""

    capabilities: str
    area_name: str


def normalize_capabilities(s: str, expected: int, domain: str = "") -> str:
    """Ensure payload has JSON with a 'capabilities' list of size <= expected."""
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            if "capabilities" in data and isinstance(data["capabilities"], list):
                data["capabilities"] = data["capabilities"][:expected]
                return json.dumps(data, indent=2)
            capabilities = []
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
        topic_type: str,
        num_neighbors: int,
        max_round: int,
        expected_capabilities: int,
        domain: str = "",
    ) -> None:
        super().__init__("A Capability scientist.")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._expected_capabilities = expected_capabilities
        self._domain = domain
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateCapabilityResponse]] = {}
        self._system_messages = [
            SystemMessage(content=CAPABILITY_GENERATION_SCIENTIST_PROMPT)
        ]
        self._round = 0
        self._max_round = max_round

    @message_handler
    async def handle_request(
        self, message: CapabilityRequest, ctx: MessageContext
    ) -> None:
        """Handle capability generation requests."""
        try:
            area_name = message.area_name
            log.info(
                f"Capability scientist {self.id} handling request for area: {area_name}, round {self._round}"
            )

            self._history.append(UserMessage(content=message.content, source="user"))

            model_result = await self._model_client.create(
                self._system_messages + self._history
            )

            # Process the model result content
            raw_content = model_result.content
            if not isinstance(raw_content, str):
                log.error(f"Model result content is not a string: {type(raw_content)}")
                raw_content = str(raw_content)

            # Normalize capabilities to ensure correct shape and count
            capabilities = normalize_capabilities(
                raw_content, self._expected_capabilities, self._domain
            )

            self._history.append(
                AssistantMessage(
                    content=capabilities,
                    source=self.metadata.get("type", "CapabilityScientist"),
                )
            )

            self._round += 1

            if self._round == self._max_round:
                log.info(
                    f"Capability scientist {self.id} publishing final response for area: {area_name}"
                )
                await self.publish_message(
                    FinalCapabilityResponse(
                        capabilities=capabilities, area_name=area_name
                    ),
                    topic_id=DefaultTopicId(),
                )
            else:
                log.info(
                    f"Capability scientist {self.id} publishing intermediate response for area: {area_name}"
                )
                await self.publish_message(
                    IntermediateCapabilityResponse(
                        capabilities=capabilities,
                        round=self._round,
                        area_name=area_name,
                    ),
                    topic_id=DefaultTopicId(type=self._topic_type),
                )

        except Exception as e:
            log.error(f"Error in CapabilityScientist {self.id} handle_request: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_response(
        self, message: IntermediateCapabilityResponse, ctx: MessageContext
    ) -> None:
        """Handle intermediate responses from peers."""
        try:
            area_name = message.area_name
            log.info(
                f"Capability scientist {self.id} handling response for area: {area_name} from round {message.round}"
            )

            self._buffer.setdefault(message.round, []).append(message)

            if len(self._buffer[message.round]) == self._num_neighbors:
                log.info(
                    f"Capability scientist {self.id} round {message.round} for area {area_name}: "
                    f"Received all responses from {self._num_neighbors} neighbors."
                )

                prompt = f"These are the capabilities generated by other scientists for the '{area_name}' area:\n"
                for resp in self._buffer[message.round]:
                    prompt += f"One scientist solution: {resp.capabilities}\n"
                prompt += (
                    f"Using the capabilities generated by other scientists as additional information, "
                    f"can you provide your answer to the capability generation problem for the '{area_name}' area? "
                    f"Return ONLY JSON with a 'capabilities' list."
                )

                await self.publish_message(
                    CapabilityRequest(
                        content=prompt, area_name=area_name, area_description=""
                    ),
                    topic_id=DefaultTopicId(type=self._topic_type),
                )

                self._buffer.pop(message.round)

        except Exception as e:
            log.error(f"Error in Capability scientist {self.id} handle_response: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


@default_subscription
class CapabilityAggregator(RoutedAgent):
    """Aggregates final responses and saves results."""

    def __init__(
        self, num_scientists: int, num_capabilities: int, output_dir: Path, domain: str
    ) -> None:
        super().__init__("Capability aggregator")
        self._num_scientists = num_scientists
        self._num_capabilities = num_capabilities
        self._buffer: List[FinalCapabilityResponse] = []
        self._output_dir = output_dir
        self._domain = domain

    @message_handler
    async def handle_area(self, message: Area, ctx: MessageContext) -> None:
        """Handle area messages."""
        try:
            log.info(f"Capability aggregator received area: {message.name}")

            prompt = (
                f"Generate {self._num_capabilities} diverse and novel capabilities for the '{message.name}' area. "
                f"Area description: {message.description}. "
                f"Each capability should be relevant to this area and designed to assess LLM abilities. "
                f"Return ONLY JSON with a 'capabilities' list with exactly {self._num_capabilities} items. "
                f"Each item must include: 'name', 'description', 'instructions', and 'tasks' (exactly 3, each with 'problem' and 'answer')."
            )

            await self.publish_message(
                CapabilityRequest(
                    content=prompt,
                    area_name=message.name,
                    area_description=message.description,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(f"Error in CapabilityAggregator handle_area: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_final_capability_response(
        self, message: FinalCapabilityResponse, ctx: MessageContext
    ) -> None:
        """Handle final capability responses from scientists."""
        try:
            area_name = message.area_name
            log.info(
                f"Capability aggregator received final response for area '{area_name}' from scientist {ctx.sender}"
            )

            self._buffer.append(message)

            if len(self._buffer) == self._num_scientists:
                capabilities_list = [resp.capabilities for resp in self._buffer]

                if not capabilities_list:
                    log.error(
                        f"No capabilities received from scientists for area: {area_name}"
                    )
                    majority_capabilities = DEFAULT_CAPABILITIES_JSON
                else:
                    # Get the most common response
                    majority_capabilities = max(
                        set(capabilities_list), key=capabilities_list.count
                    )
                    majority_capabilities = self._limit_capabilities_to_count(
                        majority_capabilities, self._num_capabilities
                    )

                log.info(
                    f"Final majority_capabilities for area '{area_name}': {majority_capabilities}"
                )

                # Save the results to file
                try:
                    self._save_capabilities_to_file(majority_capabilities, area_name)
                    log.info(
                        f"Capabilities for area '{area_name}' successfully saved to file"
                    )
                except Exception as save_error:
                    log.error(
                        f"Failed to save capabilities for area '{area_name}' to file: {save_error}"
                    )

                self._buffer.clear()

                log.info(
                    f"Capability generation for area '{area_name}' completed successfully"
                )

        except Exception as e:
            log.error(
                f"Error in CapabilityAggregator handle_final_capability_response: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _limit_capabilities_to_count(
        self, capabilities_str: str, max_count: int
    ) -> str:
        """Limit the number of capabilities in the JSON string to max_count."""
        try:
            capabilities_data = json.loads(capabilities_str)
            if (
                isinstance(capabilities_data, dict)
                and "capabilities" in capabilities_data
                and isinstance(capabilities_data["capabilities"], list)
            ):
                capabilities_data["capabilities"] = capabilities_data["capabilities"][
                    :max_count
                ]
                return json.dumps(capabilities_data, indent=2)
            return capabilities_str
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log.warning(
                f"Could not limit capabilities count due to JSON parsing error: {e}"
            )
            raise e

    def _save_capabilities_to_file(self, capabilities: str, area_name: str) -> None:
        """Save the generated capabilities JSON payload to disk under the area."""
        try:
            # Ensure output dir exists
            area_dir = self._output_dir / area_name
            area_dir.mkdir(parents=True, exist_ok=True)

            # Normalize capabilities JSON string and enforce max count again for safety
            normalized = self._limit_capabilities_to_count(
                capabilities, self._num_capabilities
            )
            try:
                parsed = json.loads(normalized)
            except json.JSONDecodeError:
                # Wrap raw string if not valid JSON
                parsed = {"capabilities": normalized}

            # Add domain field to each capability
            for cap in parsed.get("capabilities", []):
                if isinstance(cap, dict) and "domain" not in cap:
                    cap["domain"] = self._domain
                if isinstance(cap, dict) and "area" not in cap:
                    cap["area"] = area_name

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
                topic_type="CapabilityScientistA",
                num_neighbors=1,
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
                topic_type="CapabilityScientistB",
                num_neighbors=1,
                max_round=cfg.debate_cfg.max_round,
                expected_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        await CapabilityAggregator.register(
            runtime,
            "CapabilityAggregator",
            lambda: CapabilityAggregator(
                num_scientists=2,
                num_capabilities=cfg.capabilities_cfg.num_capabilities_per_area,
                output_dir=output_dir,
                domain=cfg.capabilities_cfg.domain,
            ),
        )

        await runtime.add_subscription(
            TypeSubscription("CapabilityScientistA", "CapabilityScientistB")
        )
        await runtime.add_subscription(
            TypeSubscription("CapabilityScientistB", "CapabilityScientistA")
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
