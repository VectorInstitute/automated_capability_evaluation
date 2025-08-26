"""Capability moderator agent for managing the debate process."""

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

from src.capability_generation.messages import (
    Area,
    CapabilityProposalRequest,
    CapabilityRevisionRequest,
    ScientistCapabilityProposal,
)
from src.utils.agentic_prompts import (
    CAPABILITY_MODERATOR_MERGE_PROMPT,
    CAPABILITY_MODERATOR_SYSTEM_MESSAGE,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("agentic_cap_gen.moderator")


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
        langfuse_client: Langfuse = None,
    ) -> None:
        super().__init__("Capability Moderator")
        self._model_client = model_client
        self._num_scientists = num_scientists
        self._num_capabilities = num_capabilities
        self._max_round = max_round
        self._output_dir = output_dir
        self._domain = domain
        self._langfuse_client = langfuse_client
        self._round = 0
        self._proposals_buffer: Dict[
            str, Dict[int, List[ScientistCapabilityProposal]]
        ] = {}
        self._current_area = ""

    @message_handler
    async def handle_area(self, message: Area, ctx: MessageContext) -> None:
        """Handle the area message and initiate capability proposal process."""
        with self._langfuse_client.start_as_current_span(
            name="capability_moderator_handle_area"
        ) as span:
            try:
                msg = f"Capability Moderator received area: {message.name}"
                log.info(msg)
                span.update(
                    metadata={
                        "area_received": msg,
                        "area_name": message.name,
                        "area_description": message.description,
                    }
                )

                self._current_area = message.name
                self._round = 0
                self._proposals_buffer[message.name] = {}

                await self.publish_message(
                    CapabilityProposalRequest(
                        area_name=message.name,
                        area_description=message.description,
                        num_capabilities=self._num_capabilities,
                    ),
                    topic_id=DefaultTopicId(),
                )

                span.update(
                    metadata={
                        "proposal_request_sent": f"Sent capability proposal request for {self._num_capabilities} capabilities",
                        "num_capabilities": self._num_capabilities,
                        "area_name": message.name,
                    }
                )

            except Exception as e:
                error_msg = f"Error in Capability Moderator handle_area: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "handle_area_error": error_msg,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    @message_handler
    async def handle_scientist_proposal(
        self, message: ScientistCapabilityProposal, ctx: MessageContext
    ) -> None:
        """Handle capability proposals from scientists."""
        with self._langfuse_client.start_as_current_span(
            name="capability_moderator_handle_proposal"
        ) as span:
            try:
                msg = f"Capability Moderator received proposal from Scientist {message.scientist_id} for area: {message.area_name}, round {message.round}"
                log.info(msg)
                span.update(
                    metadata={
                        "proposal_received": msg,
                        "scientist_id": message.scientist_id,
                        "area_name": message.area_name,
                        "round": message.round,
                    }
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
                    msg = f"Capability Moderator received all proposals for area: {message.area_name}, round {message.round}, proceeding to merge"
                    log.info(msg)
                    span.update(
                        metadata={
                            "all_proposals_received": msg,
                            "area_name": message.area_name,
                            "round": message.round,
                            "num_proposals": len(
                                self._proposals_buffer[area_key][message.round]
                            ),
                        }
                    )

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
                error_msg = (
                    f"Error in Capability Moderator handle_scientist_proposal: {e}"
                )
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
        self,
        scientist_a_proposal: str,
        scientist_b_proposal: str,
        area_name: str,
        round_num: int,
    ) -> None:
        """Merge scientist capability proposals using LLM."""
        with self._langfuse_client.start_as_current_span(
            name="capability_moderator_merge_proposals"
        ) as span:
            try:
                msg = f"Capability Moderator merging proposals for area: {area_name}, round {round_num}"
                log.info(msg)
                span.update(
                    metadata={
                        "merge_started": msg,
                        "area_name": area_name,
                        "round": round_num,
                    }
                )

                prompt = CAPABILITY_MODERATOR_MERGE_PROMPT.format(
                    domain=self._domain,
                    area_name=area_name,
                    scientist_a_proposal=scientist_a_proposal,
                    scientist_b_proposal=scientist_b_proposal,
                    num_capabilities=self._num_capabilities,
                )

                system_message = SystemMessage(
                    content=CAPABILITY_MODERATOR_SYSTEM_MESSAGE
                )
                user_message = UserMessage(content=prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = "Capability Moderator is parsing LLM response"
                log.info(msg)
                span.update(metadata={"llm_response_received": msg})

                parsed = parse_llm_json_response(model_result.content)
                revised_capabilities = parsed.get("capabilities", {})
                is_finalized = parsed.get("finalized", False)

                if is_finalized or round_num >= self._max_round - 1:
                    msg = f"Capability Moderator finalizing capabilities for area: {area_name} after round {round_num}"
                    log.info(msg)
                    span.update(
                        metadata={
                            "decision_finalize": msg,
                            "area_name": area_name,
                            "round": round_num,
                            "is_finalized": is_finalized,
                            "reached_max_rounds": round_num >= self._max_round - 1,
                        }
                    )

                    await self._finalize_capabilities(revised_capabilities, area_name)
                else:
                    msg = f"Capability Moderator sending merged proposal for revision for area: {area_name} in round {round_num}"
                    log.info(msg)
                    span.update(
                        metadata={
                            "decision_continue": msg,
                            "area_name": area_name,
                            "round": round_num,
                            "next_round": round_num + 1,
                        }
                    )

                    next_round = round_num + 1
                    revision_content = json.dumps(revised_capabilities)

                    await self.publish_message(
                        CapabilityRevisionRequest(
                            scientist_id="A",
                            area_name=area_name,
                            moderator_proposal=revision_content,
                            round=next_round,
                        ),
                        topic_id=DefaultTopicId(),
                    )
                    await self.publish_message(
                        CapabilityRevisionRequest(
                            scientist_id="B",
                            area_name=area_name,
                            moderator_proposal=revision_content,
                            round=next_round,
                        ),
                        topic_id=DefaultTopicId(),
                    )

                    span.update(
                        metadata={
                            "revision_requests_sent": f"Sent revision requests for round {next_round}",
                            "round": next_round,
                            "scientists": ["A", "B"],
                        }
                    )

            except Exception as e:
                error_msg = f"Error in Capability Moderator _merge_proposals: {e}"
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

    async def _finalize_capabilities(
        self, final_capabilities: dict, area_name: str
    ) -> None:
        """Save final capabilities to file."""
        with self._langfuse_client.start_as_current_span(
            name="capability_moderator_finalize"
        ) as span:
            try:
                msg = f"Capability Moderator finalizing and saving capabilities for area: {area_name}"
                log.info(msg)
                span.update(
                    metadata={"finalization_started": msg, "area_name": area_name}
                )

                capabilities_list = []
                i = 0
                while f"capability_{i}" in final_capabilities:
                    capabilities_list.append(final_capabilities[f"capability_{i}"])
                    i += 1

                final_format = {"capabilities": capabilities_list}
                final_capabilities_json = json.dumps(final_format, indent=2)

                self._save_capabilities_to_file(final_capabilities_json, area_name)

                msg = f"Capability generation completed successfully for area: {area_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "capabilities_finalized": msg,
                        "area_name": area_name,
                        "num_capabilities": len(capabilities_list),
                    }
                )

            except Exception as e:
                error_msg = f"Error in Capability Moderator _finalize_capabilities: {e}"
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

    def _save_capabilities_to_file(self, capabilities: str, area_name: str) -> None:
        """Save the generated capabilities to a file."""
        try:
            area_dir = self._output_dir / area_name.replace(" ", "_")
            area_dir.mkdir(parents=True, exist_ok=True)

            out_path = area_dir / "capabilities.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(capabilities)

            msg = f"Saved capabilities JSON for area '{area_name}' to {out_path}"
            log.info(msg)
        except Exception as e:
            error_msg = f"Failed to save capabilities for area {area_name}: {e}"
            traceback_msg = f"Traceback: {traceback.format_exc()}"

            log.error(error_msg)
            log.error(traceback_msg)
            raise
