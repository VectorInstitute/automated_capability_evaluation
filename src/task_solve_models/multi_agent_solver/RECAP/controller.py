"""ReCAP controller agent: plan → step dispatch → refine loop."""

from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from src.task_solve_models.multi_agent_solver.RECAP.messages import (
    CanonicalState,
    RecapFinalSolution,
    RecapStep,
    RecapStepRequest,
    RecapStepResult,
    RecapTask,
)
from src.task_solve_models.multi_agent_solver.RECAP.prompts import (
    RECAP_CONTROLLER_SYSTEM,
    RECAP_FINALIZER_PROMPT,
    RECAP_PLANNER_PROMPT,
    RECAP_REFINER_PROMPT,
)
from src.utils.json_utils import parse_llm_json_response

log = logging.getLogger("task_solver.recap.controller")


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


@default_subscription
class RecapController(RoutedAgent):
    """Controller that owns canonical state and runs the ReCAP loop."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        output_dir: Path,
        worker_ids: List[str],
        langfuse_client: Langfuse,
        max_steps: int = 12,
    ) -> None:
        super().__init__("ReCAP Controller")
        self._model_client = model_client
        self._output_dir = output_dir
        self._worker_ids = worker_ids
        self._langfuse_client = langfuse_client
        self._max_steps = max_steps

        self._task: Optional[RecapTask] = None
        self._state: Optional[CanonicalState] = None
        self._step_trace: List[Dict[str, Any]] = []
        self._current_step_idx: int = 0
        self._step_buffer: Dict[str, List[RecapStepResult]] = {}

    async def _call_json(
        self, prompt: str, ctx: MessageContext
    ) -> Dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                resp = await self._model_client.create(
                    messages=[
                        SystemMessage(content=RECAP_CONTROLLER_SYSTEM),
                        UserMessage(content=prompt, source="user"),
                    ],
                    cancellation_token=ctx.cancellation_token,
                    extra_create_args={"response_format": {"type": "json_object"}},
                )
                content = str(getattr(resp, "content", "") or "").strip()
                if not content:
                    raise ValueError("Empty controller response")
                parsed = parse_llm_json_response(content)
                if not parsed:
                    raise ValueError("Controller returned non-dict JSON")
                return parsed
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Controller failed to produce JSON: {last_error}")

    def _current_step(self) -> Optional[RecapStep]:
        if not self._state:
            return None
        if self._current_step_idx >= len(self._state.plan):
            return None
        return self._state.plan[self._current_step_idx]

    async def _dispatch_step(self, step: RecapStep) -> None:
        assert self._state is not None
        # Clear buffer for this step
        if step.step_id not in self._step_buffer:
            self._step_buffer[step.step_id] = []
        for wid in self._worker_ids:
            req = RecapStepRequest(
                task_id=self._state.task_id,
                step_id=step.step_id,
                step_title=step.title,
                step_description=step.description,
                expected_output=step.expected_output,
                state=self._state.to_dict(),
                worker_id=wid,
                enforce_python=False,  # Not used for re-dispatch; refiner handles plan updates
            )
            await self.publish_message(req, topic_id=DefaultTopicId())

    async def _finalize(self, ctx: MessageContext) -> None:
        assert self._state is not None
        prompt = RECAP_FINALIZER_PROMPT.format(state_json=_pretty(self._state.to_dict()))
        parsed = await self._call_json(prompt, ctx)

        answer = str(parsed.get("answer", "null")).strip()
        numerical = parsed.get("numerical_answer", None)
        numerical_answer = "null" if numerical is None else str(numerical).strip()
        reasoning = str(parsed.get("reasoning", "")).strip() or "No reasoning provided"

        self._state.final_answer = answer
        self._state.numerical_answer = numerical_answer
        self._state.final_reasoning = reasoning

        final = RecapFinalSolution(
            task_id=self._state.task_id,
            problem=self._state.problem,
            task_type=self._state.task_type,
            answer=answer,
            numerical_answer=numerical_answer,
            reasoning=reasoning,
            state=self._state.to_dict(),
            step_trace=self._step_trace,
        )
        await self._save_final(final)

    async def _save_final(self, final_solution: RecapFinalSolution) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out = self._output_dir / f"{final_solution.task_id}_solution.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(final_solution.to_dict(), f, ensure_ascii=False, indent=2)
        log.info("Saved ReCAP final solution for %s to %s", final_solution.task_id, out)

    async def _plan(self, ctx: MessageContext) -> None:
        assert self._task is not None
        prompt = RECAP_PLANNER_PROMPT.format(
            problem_text=self._task.problem,
            task_type=self._task.task_type,
        )
        parsed = await self._call_json(prompt, ctx)
        goal = str(parsed.get("goal", "Produce the required final answer.")).strip()
        givens = parsed.get("givens", [])
        unknowns = parsed.get("unknowns", [])
        invariants = parsed.get("invariants", [])
        open_q = parsed.get("open_questions", [])
        plan_raw = parsed.get("plan", [])

        def _as_str_list(x: Any) -> List[str]:
            if not isinstance(x, list):
                return []
            return [str(i).strip() for i in x if str(i).strip()]

        steps: List[RecapStep] = []
        if isinstance(plan_raw, list):
            for i, s in enumerate(plan_raw, start=1):
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("step_id", f"S{i}")).strip() or f"S{i}"
                steps.append(
                    RecapStep(
                        step_id=sid,
                        title=str(s.get("title", f"Step {i}")).strip(),
                        description=str(s.get("description", "")).strip(),
                        expected_output=str(s.get("expected_output", "")).strip(),
                    )
                )

        # Guardrail: ensure there is at least one step.
        if not steps:
            steps = [
                RecapStep(
                    step_id="S1",
                    title="Solve",
                    description="Derive the final answer from the problem.",
                    expected_output="A concise final answer and any required numeric output.",
                )
            ]

        self._state = CanonicalState(
            task_id=self._task.task_id,
            problem=self._task.problem,
            task_type=self._task.task_type,
            goal=goal,
            givens=_as_str_list(givens),
            unknowns=_as_str_list(unknowns),
            invariants=_as_str_list(invariants),
            open_questions=_as_str_list(open_q),
            plan=steps,
        )

    @message_handler
    async def handle_task(self, message: RecapTask, ctx: MessageContext) -> None:
        with self._langfuse_client.start_as_current_span(
            name=f"recap_controller_task_{message.task_id}"
        ) as span:
            try:
                log.info("ReCAP Controller received task %s", message.task_id)
                self._task = message
                self._state = None
                self._step_trace = []
                self._current_step_idx = 0
                self._step_buffer = {}

                await self._plan(ctx)
                assert self._state is not None

                # Start with the first step.
                step = self._current_step()
                if step is None:
                    await self._finalize(ctx)
                    return
                await self._dispatch_step(step)

                span.update(
                    metadata={
                        "task_id": message.task_id,
                        "plan_len": len(self._state.plan),
                        "worker_ids": self._worker_ids,
                    }
                )
            except Exception as exc:
                log.error("Controller failed on task %s: %s", message.task_id, exc)
                log.error(traceback.format_exc())
                span.update(metadata={"error": str(exc)})

    @message_handler
    async def handle_step_result(self, message: RecapStepResult, ctx: MessageContext) -> None:
        with self._langfuse_client.start_as_current_span(
            name=f"recap_controller_step_{message.task_id}_{message.step_id}"
        ) as span:
            try:
                if not self._state:
                    return
                if message.task_id != self._state.task_id:
                    return

                step = self._current_step()
                if not step or message.step_id != step.step_id:
                    # Ignore out-of-order (old) results.
                    return

                buf = self._step_buffer.setdefault(step.step_id, [])
                buf.append(message)

                if len(buf) < len(self._worker_ids):
                    return

                # Validation: Check if step requires Python and filter candidates
                step_desc_lower = step.description.lower()
                requires_python = (
                    "must use python" in step_desc_lower
                    or "python tool" in step_desc_lower
                    or ("calculation" in step_desc_lower and "must" in step_desc_lower)
                )
                
                valid_candidates = list(buf)
                python_validation_note = ""
                if requires_python:
                    python_candidates = [c for c in buf if c.used_python_tool]
                    if python_candidates:
                        valid_candidates = python_candidates
                        log.info(
                            "Step %s requires Python: filtered %d/%d candidates that used Python",
                            step.step_id,
                            len(python_candidates),
                            len(buf),
                        )
                    else:
                        # No Python candidates - let refiner handle it by updating the plan
                        log.warning(
                            "Step %s requires Python but no candidates used it. Refiner will update plan to add explicit calculation step.",
                            step.step_id,
                        )
                        python_validation_note = f"\nWARNING: Step {step.step_id} required Python tool usage, but NO candidates used Python. You MUST update the plan to add a new explicit calculation step immediately after this step that breaks down the calculation and explicitly requires Python."
                        # Still proceed to refiner with all candidates - refiner will see the note and update plan

                # Validation: Check result format and reasonableness
                for candidate in valid_candidates:
                    # Check if result is empty
                    if not candidate.result or not candidate.result.strip():
                        log.warning(
                            "Step %s candidate from worker %s has empty result",
                            step.step_id,
                            candidate.worker_id,
                        )
                        continue
                    
                    # Check if expected output suggests numeric but result is not numeric
                    expected_lower = step.expected_output.lower()
                    if any(keyword in expected_lower for keyword in ["numeric", "number", "value", "calculate", "compute"]):
                        try:
                            # Try to extract a number from the result
                            num_match = re.search(r"[-+]?\d*\.?\d+", str(candidate.result))
                            if not num_match:
                                log.warning(
                                    "Step %s candidate from worker %s: expected numeric output but result '%s' contains no number",
                                    step.step_id,
                                    candidate.worker_id,
                                    candidate.result[:50],
                                )
                        except Exception:
                            pass

                # Refine/choose.
                # Always call refiner, regardless of worker count, to enable plan updates
                if not valid_candidates:
                    # Fallback: use all candidates if validation filtered everything out
                    valid_candidates = buf
                
                candidates_json = _pretty([c.to_dict() for c in valid_candidates])
                state_json = _pretty(self._state.to_dict())
                step_json = _pretty(step.to_dict())

                chosen_result = ""
                chosen_evidence = ""
                chosen_worker_id = ""
                updated_open_q: List[str] = self._state.open_questions
                updated_plan: List[RecapStep] = []
                
                # PHASE 1: Refiner selects best candidate (simplified - no plan updates)
                prompt = RECAP_REFINER_PROMPT.format(
                    state_json=state_json,
                    step_json=step_json,
                    candidates_json=candidates_json,
                    python_validation_note=python_validation_note,
                )
                parsed = await self._call_json(prompt, ctx)
                chosen_result = str(parsed.get("chosen_result", "")).strip()
                chosen_evidence = str(parsed.get("chosen_evidence", "")).strip()
                chosen_worker_id = str(parsed.get("chosen_worker_id", "")).strip()
                oq = parsed.get("updated_open_questions", self._state.open_questions)
                if isinstance(oq, list):
                    updated_open_q = [str(x).strip() for x in oq if str(x).strip()]
                else:
                    updated_open_q = self._state.open_questions

                # Update state for this step.
                step.done = True
                step.result = chosen_result
                self._state.open_questions = updated_open_q
                
                # PHASE 2: Controller handles plan updates (explicit step generation if Python was required but not used)
                if requires_python and not python_candidates:
                    # CRITICAL: Check if current step is already an explicit calculation step to prevent infinite recursion
                    is_already_explicit = '_calc' in step.step_id.lower()
                    
                    if is_already_explicit:
                        log.warning(
                            f"Step {step.step_id} is already an explicit calculation step "
                            f"but Python was not used. Not adding another explicit step to prevent infinite recursion."
                        )
                        # Don't add another explicit step - accept the result as-is to prevent recursion
                    else:
                        # Controller adds explicit calculation step as fallback
                        log.warning(
                            f"Step {step.step_id} required Python but no candidates used it. "
                            f"Controller adding explicit calculation step."
                        )
                        explicit_step = RecapStep(
                            step_id=f"{step.step_id}_calc",
                            title=f"Explicit calculation for {step.title}",
                            description=f"CRITICAL: This calculation MUST use Python tool. "
                                       f"Break down: {step.description}. "
                                       f"Extract all numbers and formulas, then compute using Python. "
                                       f"No mental math. No approximations.",
                            expected_output=step.expected_output
                        )
                        # Insert explicit step immediately after current step
                        remaining_plan = self._state.plan[self._current_step_idx + 1:]
                        self._state.plan = (
                            self._state.plan[: self._current_step_idx + 1] + 
                            [explicit_step] + 
                            remaining_plan
                        )
                        log.info(f"Added explicit calculation step {explicit_step.step_id} to plan")

                self._step_trace.append(
                    {
                        "step": step.to_dict(),
                        "candidates": [c.to_dict() for c in buf],
                        "chosen_worker_id": chosen_worker_id,
                        "chosen_result": chosen_result,
                        "chosen_evidence": chosen_evidence,
                    }
                )
                
                # Validation: Check if result is reasonable
                if chosen_result:
                    # For numeric expected outputs, verify result contains a number
                    expected_lower = step.expected_output.lower()
                    if any(keyword in expected_lower for keyword in ["numeric", "number", "value", "calculate", "compute"]):
                        if not re.search(r"[-+]?\d*\.?\d+", str(chosen_result)):
                            log.warning(
                                "Step %s result '%s' does not contain a number but expected numeric output",
                                step.step_id,
                                chosen_result[:50],
                            )

                self._current_step_idx += 1

                # Stop conditions
                if self._current_step_idx >= len(self._state.plan) or self._current_step_idx >= self._max_steps:
                    await self._finalize(ctx)
                    return

                next_step = self._current_step()
                if next_step is None:
                    await self._finalize(ctx)
                    return
                
                # Validate explicit steps are executed: Log if next step is explicit calculation
                if '_calc' in next_step.step_id.lower() or 'explicit calculation' in next_step.title.lower():
                    log.info(
                        f"Explicit calculation step {next_step.step_id} will be dispatched next. "
                        f"Title: {next_step.title}"
                    )

                await self._dispatch_step(next_step)
                span.update(metadata={"advanced_to_step": next_step.step_id})
            except Exception as exc:
                log.error("Controller step handling error: %s", exc)
                log.error(traceback.format_exc())
                span.update(metadata={"error": str(exc)})

