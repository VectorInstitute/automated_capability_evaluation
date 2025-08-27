"""Area scientist agent for generating capability areas."""

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

from src.area_generation.messages import (
    AreaProposalRequest,
    ScientistAreaProposal,
    ScientistRevisionRequest,
)
from src.utils.agentic_prompts import (
    AREA_SCIENTIST_INITIAL_PROMPT,
    AREA_SCIENTIST_REVISION_PROMPT,
    AREA_SCIENTIST_SYSTEM_MESSAGE,
)
from src.utils.json_utils import parse_llm_json_response


log = logging.getLogger("agentic_area_gen.scientist")


@default_subscription
class AreaScientist(RoutedAgent):
    """A scientist that generates capability areas through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        langfuse_client: Langfuse = None,
    ) -> None:
        super().__init__(f"Area Scientist {scientist_id}")
        self._model_client = model_client
        self._scientist_id = scientist_id
        self._langfuse_client = langfuse_client

    @message_handler
    async def handle_area_proposal_request(
        self, message: AreaProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle initial area proposal request."""
        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_initial_proposal"
        ) as span:
            try:
                msg = f"Scientist {self._scientist_id} handling area proposal request for domain: {message.domain}"
                log.info(msg)
                span.update(
                    metadata={
                        "proposal_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "domain": message.domain,
                        "num_areas": message.num_areas,
                    }
                )

                prompt = AREA_SCIENTIST_INITIAL_PROMPT.format(
                    scientist_id=self._scientist_id,
                    domain=message.domain,
                    num_areas=message.num_areas,
                )

                system_message = SystemMessage(content=AREA_SCIENTIST_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = f"Scientist {self._scientist_id} is parsing LLM response"
                log.info(msg)
                span.update(
                    metadata={
                        "llm_response_received": msg,
                        "scientist_id": self._scientist_id,
                    }
                )

                parsed = parse_llm_json_response(model_result.content)
                proposal_content = json.dumps(parsed["areas"])

                msg = f"Scientist {self._scientist_id} publishing area proposal"
                log.info(msg)
                span.update(
                    metadata={
                        "proposal_published": msg,
                        "scientist_id": self._scientist_id,
                        "round": 0,
                    }
                )

                await self.publish_message(
                    ScientistAreaProposal(
                        scientist_id=self._scientist_id,
                        proposal=proposal_content,
                        round=0,
                    ),
                    topic_id=DefaultTopicId(),
                )

            except Exception as e:
                error_msg = f"Error in Scientist {self._scientist_id} handle_area_proposal_request: {e}"
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
        self, message: ScientistRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request from moderator."""
        if message.scientist_id != self._scientist_id:
            return

        with self._langfuse_client.start_as_current_span(
            name=f"scientist_{self._scientist_id}_revision"
        ) as span:
            try:
                round_num = message.round

                msg = f"Scientist {self._scientist_id} handling revision request for round {round_num}"
                log.info(msg)
                span.update(
                    metadata={
                        "revision_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "round": round_num,
                    }
                )

                prompt = AREA_SCIENTIST_REVISION_PROMPT.format(
                    scientist_id=self._scientist_id,
                    moderator_proposal=message.moderator_proposal,
                )

                system_message = SystemMessage(content=AREA_SCIENTIST_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                model_result = await self._model_client.create(
                    [system_message, user_message]
                )

                msg = f"Scientist {self._scientist_id} is parsing LLM response"
                log.info(msg)
                span.update(
                    metadata={
                        "llm_response_received": msg,
                        "scientist_id": self._scientist_id,
                        "round": round_num,
                    }
                )

                parsed = parse_llm_json_response(model_result.content)
                proposal_content = json.dumps(parsed["areas"])

                msg = f"Scientist {self._scientist_id} publishing revised proposal for round {round_num}"
                log.info(msg)
                span.update(
                    metadata={
                        "revised_proposal_published": msg,
                        "scientist_id": self._scientist_id,
                        "round": round_num,
                    }
                )

                await self.publish_message(
                    ScientistAreaProposal(
                        scientist_id=self._scientist_id,
                        proposal=proposal_content,
                        round=round_num,
                    ),
                    topic_id=DefaultTopicId(),
                )

            except Exception as e:
                error_msg = f"Error in Scientist {self._scientist_id} handle_revision_request: {e}"
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
