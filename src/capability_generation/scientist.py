"""Capability scientist agent for generating capabilities."""

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
from langfuse import Langfuse

from src.capability_generation.messages import (
    CapabilityProposalRequest,
    CapabilityRevisionRequest,
    ScientistCapabilityProposal,
)
from src.utils.agentic_prompts import (
    CAPABILITY_SCIENTIST_INITIAL_PROMPT,
    CAPABILITY_SCIENTIST_REVISION_PROMPT,
    CAPABILITY_SCIENTIST_SYSTEM_MESSAGE,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("agentic_cap_gen.scientist")


@default_subscription
class CapabilityScientist(RoutedAgent):
    """A scientist that generates capabilities through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        langfuse_client: Langfuse = None,
    ) -> None:
        super().__init__(f"Capability Scientist {scientist_id}")
        self._scientist_id = scientist_id
        self._model_client = model_client
        self._langfuse_client = langfuse_client

    @message_handler
    async def handle_capability_proposal_request(
        self, message: CapabilityProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle initial capability proposal request."""
        with self._langfuse_client.start_as_current_span(
            name=f"capability_scientist_{self._scientist_id}_initial_proposal"
        ) as span:
            try:
                msg = f"Capability Scientist {self._scientist_id} handling proposal request for area: {message.area_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "proposal_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "area_name": message.area_name,
                        "area_description": message.area_description,
                        "num_capabilities": message.num_capabilities,
                    }
                )

                prompt = CAPABILITY_SCIENTIST_INITIAL_PROMPT.format(
                    scientist_id=self._scientist_id,
                    area_name=message.area_name,
                    area_description=message.area_description,
                    num_capabilities=message.num_capabilities,
                )

                system_message = SystemMessage(
                    content=CAPABILITY_SCIENTIST_SYSTEM_MESSAGE
                )
                user_message = UserMessage(content=prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = (
                    f"Capability Scientist {self._scientist_id} is parsing LLM response"
                )
                log.info(msg)
                span.update(
                    metadata={
                        "llm_response_received": msg,
                        "scientist_id": self._scientist_id,
                    }
                )

                parsed = parse_llm_json_response(model_result.content)
                proposal_content = json.dumps(parsed["capabilities"])

                msg = f"Capability Scientist {self._scientist_id} publishing capability proposal for area: {message.area_name}"
                log.info(msg)
                span.update(
                    metadata={
                        "proposal_published": msg,
                        "scientist_id": self._scientist_id,
                        "area_name": message.area_name,
                        "round": 0,
                    }
                )

                await self.publish_message(
                    ScientistCapabilityProposal(
                        scientist_id=self._scientist_id,
                        proposal=proposal_content,
                        area_name=message.area_name,
                        round=0,
                    ),
                    topic_id=DefaultTopicId(),
                )

            except Exception as e:
                error_msg = f"Error in Capability Scientist {self._scientist_id} handle_capability_proposal_request: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "proposal_request_error": error_msg,
                        "scientist_id": self._scientist_id,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise

    @message_handler
    async def handle_revision_request(
        self, message: CapabilityRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request from moderator."""
        if message.scientist_id != self._scientist_id:
            return

        with self._langfuse_client.start_as_current_span(
            name=f"capability_scientist_{self._scientist_id}_revision"
        ) as span:
            try:
                msg = f"Capability Scientist {self._scientist_id} handling revision request for area: {message.area_name}, round {message.round}"
                log.info(msg)
                span.update(
                    metadata={
                        "revision_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "area_name": message.area_name,
                        "round": message.round,
                    }
                )

                prompt = CAPABILITY_SCIENTIST_REVISION_PROMPT.format(
                    scientist_id=self._scientist_id,
                    area_name=message.area_name,
                    moderator_proposal=message.moderator_proposal,
                )

                system_message = SystemMessage(
                    content=CAPABILITY_SCIENTIST_SYSTEM_MESSAGE
                )
                user_message = UserMessage(content=prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = (
                    f"Capability Scientist {self._scientist_id} is parsing LLM response"
                )
                log.info(msg)
                span.update(
                    metadata={
                        "llm_response_received": msg,
                        "scientist_id": self._scientist_id,
                        "round": message.round,
                    }
                )

                parsed = parse_llm_json_response(model_result.content)
                proposal_content = json.dumps(parsed["capabilities"])

                msg = f"Capability Scientist {self._scientist_id} publishing revised proposal for area: {message.area_name}, round {message.round}"
                log.info(msg)
                span.update(
                    metadata={
                        "revised_proposal_published": msg,
                        "scientist_id": self._scientist_id,
                        "area_name": message.area_name,
                        "round": message.round,
                    }
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
                error_msg = f"Error in Capability Scientist {self._scientist_id} handle_revision_request: {e}"
                traceback_msg = f"Traceback: {traceback.format_exc()}"

                log.error(error_msg)
                log.error(traceback_msg)

                span.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={
                        "revision_request_error": error_msg,
                        "scientist_id": self._scientist_id,
                        "error": str(e),
                        "traceback": traceback_msg,
                    },
                )
                raise
