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

from ..utils.agentic_prompts import (
    CAPABILITY_MODERATOR_MERGE_PROMPT,
    CAPABILITY_MODERATOR_SYSTEM_MESSAGE,
)
from ..utils.json_utils import parse_llm_json_response
from .messages import (
    Area,
    CapabilityProposalRequest,
    CapabilityRevisionRequest,
    ScientistCapabilityProposal,
)


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

            prompt = CAPABILITY_MODERATOR_MERGE_PROMPT.format(
                area_name=area_name,
                scientist_a_proposal=scientist_a_proposal,
                scientist_b_proposal=scientist_b_proposal,
            )

            system_message = SystemMessage(content=CAPABILITY_MODERATOR_SYSTEM_MESSAGE)
            user_message = UserMessage(content=prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            # Check if finalized
            is_finalized = False
            try:
                parsed = parse_llm_json_response(model_result.content)
                revised_capabilities = parsed.get("capabilities", {})
                is_finalized = parsed.get("finalized", False)

            except Exception:
                log.error(
                    f"Error parsing merged capabilities JSON: {model_result.content}"
                )
                raise

            if is_finalized or round_num >= self._max_round - 1:
                log.info(
                    f"Capability Moderator finalizing capabilities for area: {area_name} after round {round_num}"
                )
                await self._finalize_capabilities(revised_capabilities, area_name)
            else:
                log.info(
                    f"Capability Moderator sending merged proposal for revision in area: {area_name}, round {round_num}"
                )

                round_num += 1
                revision_content = json.dumps(revised_capabilities)

                # Send to scientists for revision
                await self.publish_message(
                    CapabilityRevisionRequest(
                        scientist_id="A",
                        moderator_proposal=revision_content,
                        area_name=area_name,
                        round=round_num,
                    ),
                    topic_id=DefaultTopicId(),
                )
                await self.publish_message(
                    CapabilityRevisionRequest(
                        scientist_id="B",
                        moderator_proposal=revision_content,
                        area_name=area_name,
                        round=round_num,
                    ),
                    topic_id=DefaultTopicId(),
                )

        except Exception as e:
            log.error(f"Error in Capability Moderator _merge_proposals: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _finalize_capabilities(
        self, final_capabilities: dict, area_name: str
    ) -> None:
        """Save final capabilities to file."""
        try:
            log.info(
                f"Capability Moderator finalizing and saving capabilities for area: {area_name}"
            )

            if "finalized" in final_capabilities:
                del final_capabilities["finalized"]  # Remove finalized flag

            # Convert capability_0, capability_1 format to capabilities list
            capabilities_list = []
            i = 0
            while f"capability_{i}" in final_capabilities:
                cap = final_capabilities[f"capability_{i}"]
                # Add domain and area fields manually
                if isinstance(cap, dict):
                    if "domain" not in cap:
                        cap["domain"] = self._domain
                    if "area" not in cap:
                        cap["area"] = area_name
                capabilities_list.append(cap)
                i += 1

            final_format = {"capabilities": capabilities_list}
            self._save_capabilities_to_file(final_format, area_name)
            log.info(
                f"Capability generation for area '{area_name}' completed successfully"
            )

        except Exception as e:
            log.error(f"Error in Capability Moderator _finalize_capabilities: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _save_capabilities_to_file(self, capabilities: dict, area_name: str) -> None:
        """Save the generated capabilities JSON payload to disk under the area."""
        try:
            # Ensure output dir exists
            area_dir = self._output_dir / area_name
            area_dir.mkdir(parents=True, exist_ok=True)

            # Write to file
            out_path = area_dir / "capabilities.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(capabilities, f, indent=2, ensure_ascii=False)
            log.info(f"Saved capabilities JSON for area '{area_name}' to {out_path}")
        except Exception as e:
            log.error(f"Failed to save capabilities for area {area_name}: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            raise
