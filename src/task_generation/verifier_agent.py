"""Source file for the verifier agent used in task verification."""

import json
import logging
import re
from typing import Any, Dict, Sequence, Tuple, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from src.task_generation.prompts import (
    MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    SYSTEM_TASK_VERIFICATION_PROMPT,
    USER_TASK_VERIFICATION_PROMPT,
)


logger = logging.getLogger(__name__)


class VerifierAgent(AssistantAgent):
    """Verifier agent for task verification."""

    def __init__(self, name: str, model_client: Any, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            model_client=model_client,
            **kwargs,
        )

    async def verify_task(
        self,
        candidate_output: Union[Dict[str, Any], str],
    ) -> Dict[str, Any]:
        """
        Verify the candidate task output against the provided blueprint and context.

        Args:
            candidate_output: The candidate task output to be verified.

        Returns
        -------
            A dictionary containing the verification results.
        """
        candidate_str = (
            json.dumps(candidate_output, indent=2, ensure_ascii=False)
            if isinstance(candidate_output, dict)
            else str(candidate_output)
        )

        user_prompt = USER_TASK_VERIFICATION_PROMPT.format(
            candidate_output=candidate_str,
        )
        task = SYSTEM_TASK_VERIFICATION_PROMPT + "\n\n" + user_prompt
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text)

    async def check_and_revise_mcq_option(
        self, candidate_question: str
    ) -> Tuple[Union[Dict[str, Any], str], str]:
        """
        Check the correctness of a candidate question.

        Args:
            candidate_question: The candidate problem to process.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT.format(
            candidate_question=candidate_question,
        )
        result = await self.run(task=task)
        text = self._extract_message(result.messages)
        return self._extract_message_content(text), task

    def _extract_message(self, messages: Sequence[Any]) -> str:
        """
        Extract text content from messages.

        Args:
            messages: List of message objects.

        Returns
        -------
            Extracted text content.
        """
        for m in reversed(messages):
            if isinstance(m, TextMessage):
                return m.content
            if hasattr(m, "content") and isinstance(m.content, str):
                return m.content
        return ""

    def _extract_message_content(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON content from the given text.

        Args:
            content: The text content to parse.

        Returns
        -------
            Parsed JSON content as a dictionary.
        """
        content = content.strip()
        if not content:
            return self._fallback("empty response")

        try:
            obj = json.loads(content)
            return obj if isinstance(obj, dict) else {"result": obj}
        except json.JSONDecodeError:
            pass

        blocks = re.findall(
            r"```json\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE
        )
        for b in blocks:
            try:
                obj = json.loads(b.strip())
                return obj if isinstance(obj, dict) else {"result": obj}
            except json.JSONDecodeError:
                continue

        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(content[start : end + 1])
                return obj if isinstance(obj, dict) else {"result": obj}
            except json.JSONDecodeError:
                pass

        return self._fallback("json parse failed")

    def _fallback(self, msg: str) -> Dict[str, Any]:
        """Return a fallback result in case of errors."""
        return {
            "overall_verdict": "fail",
            "json_format_valid": "fail",
            "mcq_integrity": "fail",
            "clarity_well_posed": "fail",
            "constraint_compliance": "fail",
            "explanation": f"System Error: {msg}",
            "question_evaluation": {},
        }
