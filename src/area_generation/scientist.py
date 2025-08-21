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

from ..utils.agentic_prompts import (
    AREA_SCIENTIST_INITIAL_PROMPT,
    AREA_SCIENTIST_REVISION_PROMPT,
    AREA_SCIENTIST_SYSTEM_MESSAGE,
)
from ..utils.json_utils import parse_llm_json_response
from .messages import (
    AreaProposalRequest,
    ScientistAreaProposal,
    ScientistRevisionRequest,
)


log = logging.getLogger("agentic_area_gen.scientist")


@default_subscription
class AreaScientist(RoutedAgent):
    """A scientist that generates capability areas through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
    ) -> None:
        super().__init__(f"Area Scientist {scientist_id}")
        self._model_client = model_client
        self._scientist_id = scientist_id

    @message_handler
    async def handle_area_proposal_request(
        self, message: AreaProposalRequest, ctx: MessageContext
    ) -> None:
        """Handle initial area proposal request."""
        try:
            log.info(
                f"Scientist {self._scientist_id} handling area proposal request for domain: {message.domain}"
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

            log.info(f"Scientist {self._scientist_id} is parsing LLM response.")
            parsed = parse_llm_json_response(model_result.content)
            proposal_content = json.dumps(parsed["areas"])

            log.info(f"Scientist {self._scientist_id} publishing area proposal")
            await self.publish_message(
                ScientistAreaProposal(
                    scientist_id=self._scientist_id,
                    proposal=proposal_content,
                    round=0,
                ),
                topic_id=DefaultTopicId(),
            )

        except Exception as e:
            log.error(
                f"Error in Scientist {self._scientist_id} handle_area_proposal_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise

    @message_handler
    async def handle_revision_request(
        self, message: ScientistRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request from moderator."""
        try:
            if message.scientist_id != self._scientist_id:
                return  # Not for this scientist

            round_num = message.round
            log.info(
                f"Scientist {self._scientist_id} handling revision request for round {round_num}"
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

            log.info(f"Scientist {self._scientist_id} is parsing LLM response.")
            parsed = parse_llm_json_response(model_result.content)
            proposal_content = json.dumps(parsed["areas"])

            log.info(
                f"Scientist {self._scientist_id} publishing revised proposal for round {round_num}"
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
            log.error(
                f"Error in Scientist {self._scientist_id} handle_revision_request: {e}"
            )
            log.error(f"Traceback: {traceback.format_exc()}")
            raise
