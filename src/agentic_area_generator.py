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


log = logging.getLogger("agentic_area_gen")

DEFAULT_AREAS_JSON = '{"areas": []}'
DEFAULT_NUM_SCIENTISTS = 2


@dataclass
class Domain:
    """A domain of capability areas."""

    name: str


@dataclass
class AreaRequest:
    """A request for capability areas."""

    content: str


@dataclass
class IntermediateAreaResponse:
    """An intermediate response from a scientist."""

    areas: str
    round: int


@dataclass
class FinalAreaResponse:
    """A final response from a scientist."""

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
        topic_type: str,
        num_neighbors: int,
        max_round: int,
        expected_areas: int,
    ) -> None:
        super().__init__("An area scientist.")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._expected_areas = expected_areas
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateAreaResponse]] = {}
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an expert capability researcher. "
                    "Given a domain, propose non-overlapping areas that comprehensively cover it. "
                    "Each area must include a name and description. Return ONLY JSON with an 'areas' list."
                )
            )
        ]
        self._round = 0
        self._max_round = max_round

    @message_handler
    async def handle_request(self, message: AreaRequest, ctx: MessageContext) -> None:
        """Handle area generation requests."""
        try:
            log.info(f"Area Scientist {self.id} handling request, round {self._round}")

            self._history.append(UserMessage(content=message.content, source="user"))

            model_result = await self._model_client.create(
                self._system_messages + self._history
            )

            # Process the model result content
            raw_content = model_result.content
            if not isinstance(raw_content, str):
                log.error(f"Model result content is not a string: {type(raw_content)}")
                raw_content = str(raw_content)

            # For areas, normalize to ensure correct shape and count
            areas = normalize_areas(raw_content, self._expected_areas)

            self._history.append(
                AssistantMessage(
                    content=areas, source=self.metadata.get("type", "AreaScientist")
                )
            )

            self._round += 1

            if self._round == self._max_round:
                log.info(f"Area Scientist {self.id} publishing final response")
                await self.publish_message(
                    FinalAreaResponse(areas=areas), topic_id=DefaultTopicId()
                )
            else:
                log.info(f"Area Scientist {self.id} publishing intermediate response")
                await self.publish_message(
                    IntermediateAreaResponse(
                        areas=areas,
                        round=self._round,
                    ),
                    topic_id=DefaultTopicId(type=self._topic_type),
                )

        except Exception as e:
            log.error(f"Error in AreaScientist {self.id} handle_request: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_response(
        self, message: IntermediateAreaResponse, ctx: MessageContext
    ) -> None:
        """Handle intermediate responses from peers."""
        try:
            log.info(
                f"Area Scientist {self.id} handling response from round {message.round}"
            )

            self._buffer.setdefault(message.round, []).append(message)

            if len(self._buffer[message.round]) == self._num_neighbors:
                log.info(
                    f"Area Scientist {self.id} round {message.round}:\nReceived all responses from {self._num_neighbors} neighbors."
                )

                prompt = "These are the areas generated by other scientists:\n"
                for resp in self._buffer[message.round]:
                    prompt += f"One scientist solution: {resp.areas}\n"
                prompt += (
                    "Using the areas generated by other scientists as additional information, "
                    "can you provide your answer to the area generation problem? "
                    "Return ONLY JSON with an 'areas' list."
                )

                await self.publish_message(
                    AreaRequest(content=prompt),
                    topic_id=DefaultTopicId(type=self._topic_type),
                )

                self._buffer.pop(message.round)

        except Exception as e:
            log.error(f"Error in Area Scientist {self.id} handle_response: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise


@default_subscription
class AreaAggregator(RoutedAgent):
    """Aggregates final responses and saves results."""

    def __init__(self, num_scientists: int, num_areas: int, output_dir: Path) -> None:
        super().__init__("Area Aggregator")
        self._num_scientists = num_scientists
        self._num_areas = num_areas
        self._buffer: List[FinalAreaResponse] = []
        self._output_dir = output_dir

    @message_handler
    async def handle_domain(self, message: Domain, ctx: MessageContext) -> None:
        """Handle the domain message from the moderator."""
        try:
            log.info(f"Area Aggregator received domain: {message.name}")

            prompt = (
                f"Generate {self._num_areas} non-overlapping capability areas for the following domain: {message.name}. "
                f"Return ONLY JSON with an 'areas' list with exactly {self._num_areas} items (each item: name, description)."
            )

            await self.publish_message(
                AreaRequest(content=prompt), topic_id=DefaultTopicId()
            )

        except Exception as e:
            log.error(f"Error in AreaAggregator handle_domain: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_final_area_response(
        self, message: FinalAreaResponse, ctx: MessageContext
    ) -> None:
        """Handle the final area response from a scientist."""
        try:
            log.info(
                f"AreaAggregator received final response from scientist {ctx.sender}"
            )
            self._buffer.append(message)

            if len(self._buffer) == self._num_scientists:
                areas = [resp.areas for resp in self._buffer]

                if not areas:
                    log.error("No areas received from scientists")
                    majority_areas = DEFAULT_AREAS_JSON
                else:
                    # Get the most common response
                    majority_areas = max(set(areas), key=areas.count)
                    majority_areas = self._limit_areas_to_count(
                        majority_areas, self._num_areas
                    )

                log.info(f"Final majority_areas: {majority_areas}")

                # Save the results to file
                try:
                    self._save_areas_to_file(majority_areas)
                    log.info("Areas successfully saved to file")
                except Exception as save_error:
                    log.error(f"Failed to save areas to file: {save_error}")

                log.info("Area generation completed successfully")

                self._buffer.clear()

        except Exception as e:
            log.error(f"Error in AreaAggregator handle_final_area_response: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _limit_areas_to_count(self, areas_str: str, max_count: int) -> str:
        """Limit the number of areas in the JSON string to max_count."""
        try:
            areas_data = json.loads(areas_str)
            if (
                isinstance(areas_data, dict)
                and "areas" in areas_data
                and isinstance(areas_data["areas"], list)
            ):
                areas_data["areas"] = areas_data["areas"][:max_count]
                return json.dumps(areas_data, indent=2)
            return areas_str
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log.warning(f"Could not limit areas count due to JSON parsing error: {e}")
            raise e

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
    """Generate areas using AutoGen multi-agent debate system."""
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
                    model=cfg.agents.scientist_a.name
                ),
                topic_type="AreaScientistA",
                num_neighbors=1,
                max_round=max_round,
                expected_areas=cfg.capabilities_cfg.num_capability_areas,
            ),
        )

        await AreaScientist.register(
            runtime,
            "AreaScientistB",
            lambda: AreaScientist(
                model_client=OpenAIChatCompletionClient(
                    model=cfg.agents.scientist_b.name
                ),
                topic_type="AreaScientistB",
                num_neighbors=1,
                max_round=max_round,
                expected_areas=cfg.capabilities_cfg.num_capability_areas,
            ),
        )

        await AreaAggregator.register(
            runtime,
            "AreaAggregator",
            lambda: AreaAggregator(
                num_scientists=DEFAULT_NUM_SCIENTISTS,
                num_areas=cfg.capabilities_cfg.num_capability_areas,
                output_dir=output_dir,
            ),
        )

        await runtime.add_subscription(
            TypeSubscription("AreaScientistA", "AreaScientistB")
        )
        await runtime.add_subscription(
            TypeSubscription("AreaScientistB", "AreaScientistA")
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
    """Run the AutoGen multi-agent debate-based area generation system."""
    log.info("Starting AutoGen multi-agent debate-based area generation")
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    try:
        asyncio.run(generate_areas(cfg))
    except Exception as e:
        log.error(f"Area generation failed: {e}")
        log.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
