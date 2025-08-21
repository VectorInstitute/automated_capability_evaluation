"""Area moderator agent for managing the debate process."""

import json
import logging
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

from ..utils.agentic_prompts import (
    AREA_MODERATOR_MERGE_PROMPT,
    AREA_MODERATOR_SYSTEM_MESSAGE,
)
from .messages import (
    AreaProposalRequest,
    Domain,
    ScientistAreaProposal,
    ScientistRevisionRequest,
)


log = logging.getLogger("agentic_area_gen.moderator")


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

            prompt = AREA_MODERATOR_MERGE_PROMPT.format(
                domain=self._domain,
                scientist_a_proposal=scientist_a_proposal,
                scientist_b_proposal=scientist_b_proposal,
                num_final_areas=self._num_final_areas,
            )

            system_message = SystemMessage(content=AREA_MODERATOR_SYSTEM_MESSAGE)
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
                revised_areas = parsed.get("areas", {})
                is_finalized = parsed.get("finalized", False)

            except Exception:
                log.error(f"Error parsing merged areas JSON: {raw_content}")
                raise

            if is_finalized or round_num >= self._max_round - 1:
                log.info(f"Moderator finalizing areas after round {round_num}")
                await self._finalize_areas(revised_areas)
            else:
                log.info(
                    f"Moderator sending merged proposal for revision in round {round_num}"
                )

                # Use the already parsed areas instead of parsing again
                revision_content = json.dumps(revised_areas)

                # Send to scientists for revision
                await self.publish_message(
                    ScientistRevisionRequest(
                        moderator_proposal=revision_content,
                        round=round_num + 1,
                    ),
                    topic_id=DefaultTopicId(),
                )
                await self.publish_message(
                    ScientistRevisionRequest(
                        moderator_proposal=revision_content,
                        round=round_num + 1,
                    ),
                    topic_id=DefaultTopicId(),
                )

        except Exception as e:
            log.error(f"Error in Moderator _merge_proposals: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _finalize_areas(self, final_areas: dict) -> None:
        """Save final areas to file."""
        try:
            log.info("Moderator finalizing and saving areas")

            areas_list = []
            i = 0
            while f"area_{i}" in final_areas:
                areas_list.append(final_areas[f"area_{i}"])
                i += 1

            final_format = {"areas": areas_list}
            final_areas_json = json.dumps(final_format, indent=2)

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

            log.info(f"Areas saved to {areas_file}.")

        except Exception as e:
            log.error(f"Failed to save areas to file: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise
