"""Tool-assisted task solver agent with code execution capabilities."""

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

from src.task_solver.messages import (
    AgentRevisionRequest,
    AgentSolution,
    TaskSolutionRequest,
    ToolAssistedAgentSolution,
)
from src.tools.toolkit import ScientificToolKit
from src.utils.json_utils import parse_llm_json_response
from src.utils.tool_assisted_prompts import (
    TOOL_ASSISTED_ROUND_1_PROMPT,
    TOOL_ASSISTED_SUBSEQUENT_ROUNDS_PROMPT,
    TOOL_ASSISTED_SYSTEM_MESSAGE,
)


log = logging.getLogger("task_solver.tool_assisted_scientist")

MAX_MODEL_ATTEMPTS = 3
MAX_CODE_EXECUTION_ATTEMPTS = 3  # Increased from 2 for better reliability


@default_subscription
class ToolAssistedScientist(RoutedAgent):
    """A scientist that solves tasks with code execution capabilities.

    This agent can execute Python code using SymPy, NumPy, and SciPy to assist
    in solving mathematical problems.

    Attributes
    ----------
    _model_client : ChatCompletionClient
        ChatCompletionClient for generating solutions via LLM.
    _scientist_id : str
        Unique identifier for this scientist agent in the debate.
    _langfuse_client : Langfuse
        Langfuse client for tracing and logging scientist activity.
    _toolkit : ScientificToolKit
        Toolkit for tool selection and code execution.
    """

    def __init__(
        self,
        model_client: ChatCompletionClient,
        scientist_id: str,
        langfuse_client: Langfuse,
        toolkit: ScientificToolKit,
    ) -> None:
        super().__init__(f"Tool-Assisted Scientist {scientist_id}")
        self._model_client = model_client
        self._scientist_id = scientist_id
        self._langfuse_client = langfuse_client
        self._toolkit = toolkit

    def _extract_solution_components(
        self, response: str
    ) -> tuple[str, str | None, str | None, str, str]:
        """Extract components from JSON response including code fields."""
        try:
            parsed = parse_llm_json_response(response)
            thought_raw = parsed.get("thought", response.strip())
            code = parsed.get("code")
            code_output = parsed.get("code_output")
            final_answer_raw = parsed.get("final_answer", "No clear answer provided")
            numerical_answer = parsed.get("numerical_answer")

            thought = (
                json.dumps(thought_raw, ensure_ascii=False)
                if isinstance(thought_raw, (dict, list))
                else str(thought_raw).strip()
            )
            final_answer = (
                json.dumps(final_answer_raw, ensure_ascii=False, indent=2)
                if isinstance(final_answer_raw, (dict, list))
                else str(final_answer_raw).strip()
            )

            # Handle code field
            if code is not None and code != "null":
                # JSON parser already handles escape sequences correctly
                code = str(code).strip()
                log.debug("Code (first 200 chars): %s", code[:200])
            else:
                code = ""

            # Handle code_output field
            if code_output is not None and code_output != "null":
                code_output = str(code_output).strip()
            else:
                code_output = ""

            # Handle numerical_answer
            if numerical_answer is not None:
                numerical_answer = str(numerical_answer)
            else:
                numerical_answer = "null"

            return thought, code, code_output, final_answer, numerical_answer

        except Exception as e:
            msg = f"Failed to parse JSON response: {e} \n Response: {response}"
            log.error(msg)
            log.error(traceback.format_exc())
            raise

    async def _generate_solution_with_code_execution(
        self, system_message: SystemMessage, user_message: UserMessage
    ) -> tuple[str, str | None, str | None, str, str]:
        """Generate solution with iterative code execution feedback.

        This method handles the iterative process of:
        1. Getting LLM response with potential code
        2. Executing code if present
        3. If execution fails, providing error feedback to LLM
        4. Repeating until code succeeds or max attempts exhausted
        5. Returning final solution components
        """
        conversation_history = [system_message, user_message]
        last_error: Exception | None = None
        
        # Track final components from last valid parse
        last_valid_components = None
        
        # Track total code execution attempts across all model responses
        total_exec_attempts = 0

        for model_attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
            try:
                # Get response from LLM
                log.debug(
                    "Scientist %s: LLM attempt %d/%d",
                    self._scientist_id,
                    model_attempt,
                    MAX_MODEL_ATTEMPTS,
                )
                response = await self._model_client.create(
                    conversation_history,
                    json_output=True,
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                log.warning(
                    "Tool-assisted scientist %s failed to get response on attempt %d: %s",
                    self._scientist_id,
                    model_attempt,
                    exc,
                )
                continue

            response_content = str(getattr(response, "content", "") or "").strip()
            if not response_content:
                last_error = ValueError("Empty response content")
                log.warning(
                    "Tool-assisted scientist %s received empty response on attempt %d",
                    self._scientist_id,
                    model_attempt,
                )
                continue

            log.debug(
                "Scientist %s: Received response (first 200 chars): %s",
                self._scientist_id,
                response_content[:200],
            )
            log.debug(
                "Scientist %s: Raw JSON with special chars visible (first 500 chars): %s",
                self._scientist_id,
                repr(response_content[:500]),
            )

            try:
                thought, code, code_output, final_answer, numerical_answer = (
                    self._extract_solution_components(response_content)
                )
                last_valid_components = (thought, code, code_output, final_answer, numerical_answer)

                # Log the extracted code to see if parsing is correct
                if code:
                    log.debug(
                        "Scientist %s: Extracted code (first 200 chars, repr): %s",
                        self._scientist_id,
                        repr(code[:200]),
                    )
                last_valid_components = (thought, code, code_output, final_answer, numerical_answer)

                # If no code (empty string), we're done
                if not code:
                    log.info(
                        "Scientist %s: No code to execute, returning solution",
                        self._scientist_id,
                    )
                    return thought, code, code_output, final_answer, numerical_answer

                # Code is present - attempt execution with retry loop
                log.info(
                    "Scientist %s: Code detected, starting execution loop",
                    self._scientist_id,
                )
                
                # Clean up common code issues before execution
                # Sometimes LLM generates code with literal \n that should be newlines
                if '\\n' in code:
                    # Replace literal \n with actual newlines (but be careful with string escapes)
                    # This is a heuristic fix for common JSON escaping issues
                    code = code.replace('\\n', '\n')
                    log.debug(
                        "Scientist %s: Applied \\n replacement to clean up code",
                        self._scientist_id,
                    )
                
                log.debug(
                    "Scientist %s: Generated code:\n%s\n%s\n%s",
                    self._scientist_id,
                    "-" * 60,
                    code,
                    "-" * 60,
                )

                # Increment total execution attempts
                total_exec_attempts += 1
                log.info(
                    "Scientist %s: Code execution attempt %d/%d",
                    self._scientist_id,
                    total_exec_attempts,
                    MAX_CODE_EXECUTION_ATTEMPTS,
                )
                
                execution_result = self._toolkit.execute_code(code)
                # Convert dict to object-like access
                execution_result = type('obj', (object,), {'success': execution_result['success'], 'output': execution_result['output'], 'error': execution_result.get('error')})()

                if execution_result.success:
                    code_output = execution_result.output
                    log.info(
                        "Scientist %s: Code execution successful",
                        self._scientist_id,
                    )
                    log.debug(
                        "Scientist %s: Code output:\n%s\n%s\n%s",
                        self._scientist_id,
                        "-" * 60,
                        code_output[:500],
                        "-" * 60,
                    )
                    return thought, code, code_output, final_answer, numerical_answer

                # Code execution failed
                code_output = f"ERROR: {execution_result.error}"
                log.warning(
                    "Scientist %s: Code execution failed (attempt %d/%d): %s",
                    self._scientist_id,
                    total_exec_attempts,
                    MAX_CODE_EXECUTION_ATTEMPTS,
                    execution_result.error,
                )
                log.debug(
                    "Scientist %s: Full error:\n%s\n%s\n%s",
                    self._scientist_id,
                    "-" * 60,
                    execution_result.error,
                    "-" * 60,
                )

                # If we've exhausted code execution attempts, break out
                if total_exec_attempts >= MAX_CODE_EXECUTION_ATTEMPTS:
                    log.error(
                        "Scientist %s: Max code execution attempts (%d) exhausted",
                        self._scientist_id,
                        MAX_CODE_EXECUTION_ATTEMPTS,
                    )
                    return thought, code, code_output, final_answer, numerical_answer

                # Provide feedback to LLM for another attempt
                log.info(
                    "Scientist %s: Providing error feedback to LLM for code correction",
                    self._scientist_id,
                )
                
                # Construct enhanced error feedback with actionable hints
                error_msg = str(execution_result.error)
                hints = []
                
                if "unterminated string" in error_msg.lower() or "eol while scanning" in error_msg.lower():
                    hints.append("- Use triple-quoted strings for multi-line output: print('''text''')")
                    hints.append("- Avoid backslashes in strings; use raw strings r'' if needed")
                    hints.append("- For print statements, use separate print() calls instead of \\n in strings")
                elif "syntaxerror" in error_msg.lower():
                    hints.append("- Review Python syntax, especially quotes, parentheses, and indentation")
                    hints.append("- Check for unmatched brackets or quotes")
                elif "nameerror" in error_msg.lower():
                    hints.append("- Ensure all imports are at the top of the code")
                    hints.append("- Check that all variables are defined before use")
                elif "importerror" in error_msg.lower() or "modulenotfounderror" in error_msg.lower():
                    hints.append("- Only use approved libraries: sympy, numpy, scipy, math, fractions, decimal")
                
                hints_text = "\n".join(hints) if hints else "- Carefully review the error message above"
                
                feedback_prompt = f"""Your previous code execution failed with the following error:

ERROR: {execution_result.error}

Failed code:
```python
{code}
```

ACTIONABLE FIXES:
{hints_text}

IMPORTANT: When writing code in JSON:
- Use simple print() statements on separate lines
- Avoid LaTeX notation in code comments or strings
- Use triple-quoted strings for multi-line output: print('''result''')
- Keep code focused on numerical computation only

Return your corrected solution in the same JSON format with fixed code."""

                feedback_message = UserMessage(
                    content=feedback_prompt,
                    source="user"
                )
                conversation_history.append(feedback_message)
                
                # Continue to next model attempt for corrected code

            except Exception as exc:
                last_error = exc
                log.warning(
                    "Tool-assisted scientist %s failed to parse response on attempt %d: %s",
                    self._scientist_id,
                    model_attempt,
                    exc,
                )
                log.debug("Full exception: %s", traceback.format_exc())
                continue

        # If we have a last valid parse, return it (even if code failed)
        if last_valid_components is not None:
            log.warning(
                "Scientist %s: Returning last valid components after exhausting attempts",
                self._scientist_id,
            )
            return last_valid_components

        raise RuntimeError(
            f"Tool-assisted scientist {self._scientist_id} could not obtain valid "
            f"response after {MAX_MODEL_ATTEMPTS} attempts"
        ) from last_error

    @message_handler
    async def handle_task_solution_request(
        self, message: TaskSolutionRequest, ctx: MessageContext
    ) -> None:
        """Handle initial task solution request with code execution."""
        with self._langfuse_client.start_as_current_span(
            name=f"tool_assisted_scientist_{self._scientist_id}_initial_solution"
        ) as span:
            try:
                msg = (
                    f"Tool-assisted scientist {self._scientist_id} handling initial "
                    f"solution request for task: {message.task_id}, "
                    f"capability: {message.capability_name}, area: {message.area_name}, "
                    f"round: {message.round_number}"
                )
                log.info(msg)
                span.update(
                    metadata={
                        "solution_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task_id,
                        "capability": message.capability_name,
                        "area": message.area_name,
                        "round": message.round_number,
                        "tool_assisted": True,
                    }
                )

                # Step 1: Prepare tool context for problem
                log.info(f"Scientist {self._scientist_id}: Preparing tool context")
                tool_context = await self._toolkit.prepare_tools(message.problem)
                
                span.update(
                    metadata={
                        "tool_selection_needs_tools": tool_context.get("needs_tools"),
                        "tool_selection_reasoning": tool_context.get("reasoning"),
                        "selected_tools": tool_context.get("selected_libraries", []),
                    }
                )
                
                # Step 2: Format tool context for prompt
                tool_context_str = self._toolkit.format_tool_context(tool_context)
                
                # Step 3: Create prompt with tool context
                prompt = TOOL_ASSISTED_ROUND_1_PROMPT.format(
                    problem_text=message.problem,
                    tool_context=tool_context_str
                )

                system_message = SystemMessage(content=TOOL_ASSISTED_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                (
                    thought,
                    code,
                    code_output,
                    final_answer,
                    numerical_answer,
                ) = await self._generate_solution_with_code_execution(
                    system_message, user_message
                )

                # Debug log before creating solution
                log.debug(
                    "Scientist %s: About to create solution - code present: %s, code_output present: %s",
                    self._scientist_id,
                    bool(code),
                    bool(code_output),
                )
                if code:
                    log.debug(
                        "Scientist %s: Code length: %d characters",
                        self._scientist_id,
                        len(code),
                    )

                # Create solution with code execution metadata
                solution = ToolAssistedAgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    numerical_answer=numerical_answer,
                    round_number=message.round_number,
                    capability_name=message.capability_name,
                    area_name=message.area_name,
                    code=code,
                    code_output=code_output,
                )

                # Debug log after creating solution
                log.debug(
                    "Scientist %s: Created solution - code in solution: %s, code_output in solution: %s",
                    self._scientist_id,
                    bool(solution.code),
                    bool(solution.code_output),
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "solution_generated": (
                            f"Tool-assisted scientist {self._scientist_id} generated "
                            f"solution for task {message.task_id}"
                        ),
                        "code_executed": bool(code),
                        "code_success": code_output and not code_output.startswith("ERROR:"),
                    }
                )

            except Exception as e:
                msg = (
                    f"Error in tool-assisted scientist {self._scientist_id} "
                    f"task solution request: {str(e)}"
                )
                log.error(msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": msg})

    @message_handler
    async def handle_agent_revision_request(
        self, message: AgentRevisionRequest, ctx: MessageContext
    ) -> None:
        """Handle revision request with code execution capabilities."""
        with self._langfuse_client.start_as_current_span(
            name=f"tool_assisted_scientist_{self._scientist_id}_round_{message.round_number}"
        ) as span:
            try:
                msg = (
                    f"Tool-assisted scientist {self._scientist_id} handling revision "
                    f"request for task: {message.task_id}, "
                    f"capability: {message.capability_name}, area: {message.area_name}, "
                    f"round: {message.round_number}"
                )
                log.info(msg)
                span.update(
                    metadata={
                        "revision_request_received": msg,
                        "scientist_id": self._scientist_id,
                        "task_id": message.task_id,
                        "round": message.round_number,
                        "num_other_solutions": len(message.other_solutions),
                        "tool_assisted": True,
                    }
                )

                other_solutions_text = "\n\n".join(
                    [
                        (
                            f"Scientist {sol['agent_id']}: Reasoning: {sol['thought']}, "
                            f"Final solution: {sol['final_answer']}"
                        )
                        for sol in message.other_solutions
                        if sol["agent_id"] != self._scientist_id
                    ]
                )

                prompt = TOOL_ASSISTED_SUBSEQUENT_ROUNDS_PROMPT.format(
                    other_solutions=other_solutions_text,
                    problem_text=message.problem,
                )

                system_message = SystemMessage(content=TOOL_ASSISTED_SYSTEM_MESSAGE)
                user_message = UserMessage(content=prompt, source="user")

                (
                    thought,
                    code,
                    code_output,
                    final_answer,
                    numerical_answer,
                ) = await self._generate_solution_with_code_execution(
                    system_message, user_message
                )

                solution = ToolAssistedAgentSolution(
                    agent_id=self._scientist_id,
                    task_id=message.task_id,
                    thought=thought,
                    final_answer=final_answer,
                    numerical_answer=numerical_answer,
                    round_number=message.round_number,
                    capability_name=message.capability_name,
                    area_name=message.area_name,
                    code=code,
                    code_output=code_output,
                )

                await self.publish_message(solution, topic_id=DefaultTopicId())

                span.update(
                    metadata={
                        "revision_generated": (
                            f"Tool-assisted scientist {self._scientist_id} generated "
                            f"revision for task {message.task_id}"
                        ),
                        "code_executed": bool(code),
                        "code_success": code_output and not code_output.startswith("ERROR:"),
                    }
                )

            except Exception as e:
                msg = (
                    f"Error in tool-assisted scientist {self._scientist_id} "
                    f"agent revision request: {str(e)}"
                )
                log.error(msg)
                log.error(traceback.format_exc())
                span.update(metadata={"error": msg})
