"""Capability scientist agent for generating capabilities within areas."""

import json
import logging
import traceback

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
    CAPABILITY_SCIENTIST_INITIAL_PROMPT,
    CAPABILITY_SCIENTIST_REVISION_PROMPT,
    CAPABILITY_SCIENTIST_SYSTEM_MESSAGE,
)
from ..utils.json_utils import parse_llm_json_response
from .messages import (
    CapabilityProposalRequest,
    CapabilityRevisionRequest,
    ScientistCapabilityProposal,
)


log = logging.getLogger("agentic_cap_gen.scientist")


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

            prompt = CAPABILITY_SCIENTIST_INITIAL_PROMPT.format(
                scientist_id=self._scientist_id,
                area_name=message.area_name,
                num_capabilities=message.num_capabilities,
                area_description=message.area_description,
            )

            system_message = SystemMessage(content=CAPABILITY_SCIENTIST_SYSTEM_MESSAGE)
            user_message = UserMessage(content=prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            try:
                parsed = parse_llm_json_response(model_result.content)
                proposal_content = json.dumps(parsed["capabilities"])
            except Exception as e:
                log.error(f"Could not parse scientist response as JSON: {e}")
                raise

            log.info(
                f"Capability Scientist {self._scientist_id} publishing capability proposal for area: {message.area_name}"
            )
            await self.publish_message(
                ScientistCapabilityProposal(
                    scientist_id=self._scientist_id,
                    proposal=proposal_content,
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

            prompt = CAPABILITY_SCIENTIST_REVISION_PROMPT.format(
                scientist_id=self._scientist_id,
                area_name=message.area_name,
                moderator_proposal=message.moderator_proposal,
            )

            system_message = SystemMessage(content=CAPABILITY_SCIENTIST_SYSTEM_MESSAGE)
            user_message = UserMessage(content=prompt, source="user")

            model_result = await self._model_client.create(
                [system_message, user_message]
            )

            try:
                parsed = parse_llm_json_response(model_result.content)
                proposal_content = json.dumps(parsed["capabilities"])
            except Exception as e:
                log.error(f"Could not parse scientist response as JSON: {e}")
                raise

            log.info(
                f"Capability Scientist {self._scientist_id} publishing revised proposal for area: {message.area_name}, round {message.round}"
            )
            await self.publish_message(
                ScientistCapabilityProposal(
                    scientist_id=self._scientist_id,
                    proposal=proposal_content,
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
