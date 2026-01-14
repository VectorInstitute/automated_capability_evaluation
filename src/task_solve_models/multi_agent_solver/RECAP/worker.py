"""ReCAP worker agent: executes exactly one step and returns a structured result."""

from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any, Dict, List, Tuple

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from langfuse import Langfuse

from src.task_solve_models.multi_agent_solver.RECAP.messages import (
    RecapStepRequest,
    RecapStepResult,
)
from src.task_solve_models.multi_agent_solver.RECAP.prompts import (
    RECAP_STEP_PROMPT,
    RECAP_WORKER_SYSTEM,
)
from src.utils.json_utils import parse_llm_json_response
from src.utils.tools import python_calculator

log = logging.getLogger("task_solver.recap.worker")

MAX_MODEL_ATTEMPTS = 3
MAX_REASONING_STEPS = 5


def _to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _extract_python_blocks(text: str) -> List[str]:
    return re.findall(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)


@default_subscription
class RecapWorker(RoutedAgent):
    """A ReCAP step worker. It receives a step request and returns a step result."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        worker_id: str,
        langfuse_client: Langfuse,
    ) -> None:
        super().__init__(f"ReCAP Worker {worker_id}")
        self._model_client = model_client
        self._worker_id = worker_id
        self._langfuse_client = langfuse_client

    async def _call_with_optional_python_tool(
        self, system: str, user: str, ctx: MessageContext
    ) -> Tuple[str, bool]:
        """Run a short ReAct loop: execute python blocks if present."""
        messages: list[Any] = [
            SystemMessage(content=system),
            UserMessage(content=user, source="user"),
        ]
        used_tool = False

        for step_idx in range(MAX_REASONING_STEPS):
            resp = await self._model_client.create(
                messages,
                cancellation_token=ctx.cancellation_token,
            )
            content = str(getattr(resp, "content", "") or "").strip()
            if not content:
                break

            blocks = _extract_python_blocks(content)
            if blocks:
                used_tool = True
                code = blocks[0]
                output = python_calculator(code)
                messages.append(AssistantMessage(content=content, source="assistant"))
                # After tool output, force the model back into JSON mode.
                messages.append(
                    UserMessage(
                        content=(
                            f"Tool Output:\n{output}\n\n"
                            "Now return ONLY the JSON object for the requested schema "
                            "(keys: result, evidence, confidence, assumptions_used). "
                            "Do not include any code blocks."
                        ),
                        source="user",
                    )
                )
                continue

            # If we got here, there are no python blocks; return raw content.
            return content, used_tool

        # If we fell out of the loop, return last assistant content if any
        # We will make one final attempt to get JSON (many models get stuck emitting python forever).
        try:
            resp = await self._model_client.create(
                messages
                + [
                    UserMessage(
                        content=(
                            "FINAL: Return ONLY a JSON object now. "
                            "No markdown fences. No code. "
                            "Keys: result, evidence, confidence, assumptions_used."
                        ),
                        source="user",
                    )
                ],
                cancellation_token=ctx.cancellation_token,
                extra_create_args={"response_format": {"type": "json_object"}},
            )
            forced = str(getattr(resp, "content", "") or "").strip()
            if forced:
                return forced, used_tool
        except Exception:
            pass

        last_assistant = ""
        for m in reversed(messages):
            if isinstance(m, AssistantMessage):
                last_assistant = str(m.content)
                break
        return last_assistant.strip(), used_tool

    def _parse_step_json(self, text: str) -> Tuple[str, str, float, List[str]]:
        """Parse a JSON object from the model output."""
        parsed = parse_llm_json_response(text)
        result = str(parsed.get("result", "")).strip()
        evidence = str(parsed.get("evidence", "")).strip()
        confidence_raw = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        assumptions = parsed.get("assumptions_used", [])
        if not isinstance(assumptions, list):
            assumptions = [str(assumptions)]
        assumptions = [str(a).strip() for a in assumptions if str(a).strip()]
        return result, evidence, confidence, assumptions

    @message_handler
    async def handle_step_request(self, message: RecapStepRequest, ctx: MessageContext) -> None:
        with self._langfuse_client.start_as_current_span(
            name=f"recap_worker_{self._worker_id}_step_{message.step_id}"
        ) as span:
            try:
                if message.worker_id != self._worker_id:
                    return

                state_json = _to_pretty_json(message.state)
                step_json = _to_pretty_json(
                    {
                        "step_id": message.step_id,
                        "title": message.step_title,
                        "description": message.step_description,
                        "expected_output": message.expected_output,
                    }
                )
                
                # Add enforcement message if Python is required
                enforce_msg = ""
                if message.enforce_python:
                    enforce_msg = "\n\nCRITICAL: This step REQUIRES Python tool usage. Your previous attempt was rejected because Python was not used. You MUST use Python for this step. No exceptions."
                
                # Special enforcement for explicit calculation steps
                if '_calc' in message.step_id.lower():
                    explicit_enforce = "\n\nCRITICAL: This is an EXPLICIT CALCULATION STEP. You were given this step because Python was required but not used in a previous step. You MUST use Python tool NOW. Write Python code immediately. No exceptions. No mental math. Extract all numbers from the problem and compute using Python."
                    enforce_msg = enforce_msg + explicit_enforce if enforce_msg else explicit_enforce
                
                prompt = RECAP_STEP_PROMPT.format(state_json=state_json, step_json=step_json) + enforce_msg

                used_tool = False
                last_error: Exception | None = None

                for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
                    try:
                        raw, did_use_tool = await self._call_with_optional_python_tool(
                            RECAP_WORKER_SYSTEM, prompt, ctx
                        )
                        used_tool = used_tool or did_use_tool
                        if not raw:
                            raise ValueError("Empty worker response")
                        # If model still returns only code, treat as failure and retry.
                        if raw.lstrip().startswith("```") and "{" not in raw:
                            raise ValueError("Worker returned non-JSON (code only)")
                        result, evidence, confidence, assumptions = self._parse_step_json(raw)
                        if not result:
                            raise ValueError("Empty parsed result")

                        step_result = RecapStepResult(
                            task_id=message.task_id,
                            step_id=message.step_id,
                            worker_id=self._worker_id,
                            result=result,
                            evidence=evidence or "No evidence provided",
                            confidence=confidence,
                            assumptions_used=assumptions,
                            used_python_tool=used_tool,
                        )
                        await self.publish_message(step_result, topic_id=DefaultTopicId())

                        span.update(
                            metadata={
                                "step_id": message.step_id,
                                "worker_id": self._worker_id,
                                "used_python_tool": used_tool,
                                "confidence": confidence,
                            }
                        )
                        return
                    except Exception as exc:
                        last_error = exc
                        log.warning(
                            "ReCAP worker %s failed attempt %d/%d for %s: %s",
                            self._worker_id,
                            attempt,
                            MAX_MODEL_ATTEMPTS,
                            message.step_id,
                            exc,
                        )

                # Hard failure: still respond so controller can proceed
                fail = RecapStepResult(
                    task_id=message.task_id,
                    step_id=message.step_id,
                    worker_id=self._worker_id,
                    result="ERROR",
                    evidence=f"Worker failed: {last_error}",
                    confidence=0.0,
                    assumptions_used=[],
                    used_python_tool=used_tool,
                )
                await self.publish_message(fail, topic_id=DefaultTopicId())
            except Exception as exc:
                log.error("Worker %s crashed: %s", self._worker_id, exc)
                log.error(traceback.format_exc())
                span.update(metadata={"error": str(exc)})

