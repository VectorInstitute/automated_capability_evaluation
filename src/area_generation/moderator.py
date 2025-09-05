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
from langfuse import Langfuse

from src.area_generation.messages import (
    AreaProposalRequest,
    Domain,
    ScientistAreaProposal,
    ScientistRevisionRequest,
)
from src.utils.agentic_prompts import (
    AREA_MODERATOR_MERGE_PROMPT,
    AREA_MODERATOR_SYSTEM_MESSAGE,
)
from src.utils.json_utils import parse_llm_json_response


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
        langfuse_client: Langfuse = None,
    ) -> None:
        super().__init__("Area Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_final_areas = num_final_areas
        self._max_round = max_round
        self._output_dir = output_dir
        self._langfuse_client = langfuse_client
        self._round = 0
        self._proposals_buffer: Dict[int, List[ScientistAreaProposal]] = {}
        self._domain = ""

    @message_handler
    async def handle_domain(self, message: Domain, ctx: MessageContext) -> None:
        """Handle the domain message and initiate area proposal process."""
        with self._langfuse_client.start_as_current_span(
            name="moderator_handle_domain"
        ) as span:
            try:
                msg = f"Moderator received domain: {message.name}"
                log.info(msg)
                span.update(metadata={"domain_received": msg, "domain": message.name})

                self._domain = message.name

                await self.publish_message(
                    AreaProposalRequest(
                        domain=message.name, num_areas=self._num_final_areas
                    ),
                    topic_id=DefaultTopicId(),
                )

                span.update(
                    metadata={
                        "proposal_request_sent": f"Sent proposal request for {self._num_final_areas} areas",
                        "num_areas": self._num_final_areas,
                        "domain": message.name,
                    }
                )

            except Exception as e:
                error_msg = f"Error in Moderator handle_domain: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "handle_domain_error": error_msg,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    @message_handler
    async def handle_scientist_proposal(
        self, message: ScientistAreaProposal, ctx: MessageContext
    ) -> None:
        """Handle area proposals from scientists."""
        with self._langfuse_client.start_as_current_span(
            name="moderator_handle_proposal"
        ) as span:
            try:
                if self._round != message.round:
                    error_msg = f"Moderator received proposal for round {message.round} but current round is {self._round}"
                    log.error(error_msg)

                    span.update(
                        level="ERROR",
                        status_message=error_msg,
                        metadata={
                            "round_mismatch": error_msg,
                            "received_round": message.round,
                            "current_round": self._round,
                        },
                    )

                    raise Exception(error_msg)

                msg = f"Moderator received proposal from Scientist {message.scientist_id} for round {self._round}"
                log.info(msg)
                span.update(
                    metadata={
                        "proposal_received": msg,
                        "scientist_id": message.scientist_id,
                        "round": self._round,
                    }
                )

                self._proposals_buffer.setdefault(self._round, []).append(message)

                if len(self._proposals_buffer[self._round]) == self._num_scientists:
                    msg = f"Moderator received all proposals for round {self._round}, proceeding to merge"
                    log.info(msg)
                    span.update(
                        metadata={
                            "all_proposals_received": msg,
                            "round": self._round,
                            "num_proposals": len(self._proposals_buffer[self._round]),
                        }
                    )

                    proposals = self._proposals_buffer[self._round]
                    scientist_a_proposal = next(
                        p.proposal for p in proposals if p.scientist_id == "A"
                    )
                    scientist_b_proposal = next(
                        p.proposal for p in proposals if p.scientist_id == "B"
                    )

                    await self._merge_proposals(
                        scientist_a_proposal, scientist_b_proposal
                    )

            except Exception as e:
                error_msg = f"Error in Moderator handle_scientist_proposal: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "proposal_handling_error": error_msg,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    async def _merge_proposals(
        self, scientist_a_proposal: str, scientist_b_proposal: str
    ) -> None:
        """Merge scientist proposals using LLM."""
        with self._langfuse_client.start_as_current_span(
            name="moderator_merge_proposals"
        ) as span:
            try:
                msg = f"Moderator merging proposals for round {self._round}"
                log.info(msg)
                span.update(metadata={"merge_started": msg, "round": self._round})

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

                msg = "Moderator is parsing LLM response."
                log.info(msg)
                span.update(metadata={"llm_response_received": msg})

                parsed = parse_llm_json_response(model_result.content)
                revised_areas = parsed.get("areas", {})
                is_finalized = parsed.get("finalized", False)

                if is_finalized or self._round >= self._max_round - 1:
                    msg = f"Moderator finalizing areas after round {self._round}"
                    log.info(msg)
                    span.update(
                        metadata={
                            "decision_finalize": msg,
                            "round": self._round,
                            "is_finalized": is_finalized,
                            "reached_max_rounds": self._round >= self._max_round - 1,
                        }
                    )

                    await self._finalize_areas(revised_areas)
                else:
                    msg = f"Moderator sending merged proposal for revision in round {self._round}"
                    log.info(msg)
                    span.update(
                        metadata={
                            "decision_continue": msg,
                            "round": self._round,
                            "next_round": self._round + 1,
                        }
                    )

                    self._round += 1
                    revision_content = json.dumps(revised_areas)

                    await self.publish_message(
                        ScientistRevisionRequest(
                            scientist_id="A",
                            moderator_proposal=revision_content,
                            round=self._round,
                        ),
                        topic_id=DefaultTopicId(),
                    )
                    await self.publish_message(
                        ScientistRevisionRequest(
                            scientist_id="B",
                            moderator_proposal=revision_content,
                            round=self._round,
                        ),
                        topic_id=DefaultTopicId(),
                    )

                    span.update(
                        metadata={
                            "revision_requests_sent": f"Sent revision requests for round {self._round}",
                            "round": self._round,
                            "scientists": ["A", "B"],
                        }
                    )

            except Exception as e:
                error_msg = f"Error in Moderator _merge_proposals: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "merge_error": error_msg,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    async def _finalize_areas(self, final_areas: dict) -> None:
        """Save final areas to file."""
        with self._langfuse_client.start_as_current_span(
            name="moderator_finalize_areas"
        ) as span:
            try:
                msg = "Moderator finalizing and saving areas"
                log.info(msg)
                span.update(metadata={"finalization_started": msg})

                areas_list = []
                i = 0
                while f"area_{i}" in final_areas:
                    areas_list.append(final_areas[f"area_{i}"])
                    i += 1

                final_format = {"areas": areas_list}
                final_areas_json = json.dumps(final_format, indent=2)

                self._save_areas_to_file(final_areas_json)

                msg = "Area generation completed successfully"
                log.info(msg)
                span.update(
                    metadata={"areas_finalized": msg, "num_areas": len(areas_list)}
                )

            except Exception as e:
                error_msg = f"Error in Moderator _finalize_areas: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "finalize_error": error_msg,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    def _save_areas_to_file(self, areas: str) -> None:
        """Save the generated areas to a file in the specified directory structure."""
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)

            log.debug(f"Created output directory: {self._output_dir}")

            areas_file = self._output_dir / "areas.json"

            try:
                areas_data = json.loads(areas) if isinstance(areas, str) else areas
            except json.JSONDecodeError as e:
                warning_msg = f"Areas string is not valid JSON, wrapping it: {e}"
                log.warning(warning_msg)

                areas_data = {
                    "raw_areas": areas,
                    "error": "Original content was not valid JSON",
                }

            with open(areas_file, "w", encoding="utf-8") as f:
                json.dump(areas_data, f, indent=2, ensure_ascii=False)

            msg = f"Areas saved to {areas_file}"
            log.info(msg)

        except Exception as e:
            error_msg = f"Failed to save areas to file: {e}"
            traceback_msg = f"Traceback: {traceback.format_exc()}"

            log.error(error_msg)
            log.error(traceback_msg)
            raise
