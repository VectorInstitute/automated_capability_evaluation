#!/usr/bin/env python3
"""V5 -- Single Agent with tool access.

One LLM call per problem for code generation (using the full scientific
toolkit), code execution via ``PythonExecutor``, then one LLM call for the
final answer incorporating the tool output.  No perspectives, no contrast.

Example
-------
    python -m src.task_solver.self_contrast.run_single_agent_tools \
        --model gpt-4o --batch-file evaluation_batch.json

    # vec-inf / vLLM endpoint:
    python -m src.task_solver.self_contrast.run_single_agent_tools \
        --model Qwen2.5-72B-Instruct --url http://10.1.1.29:8081/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.task_solver.self_contrast._runner_utils import (
    parse_single_agent_response,
    resolve_batch_file,
)
from src.task_solver.self_contrast.evaluator import (
    evaluate_batch,
    evaluate_result,
    print_summary,
    save_results,
)
from src.task_solver.self_contrast.model_client import LLMClient
from src.task_solver.self_contrast.prompts import (
    FORMAT_RULES,
    SINGLE_AGENT_SYSTEM_PROMPT,
    SINGLE_AGENT_TOOLS_ANSWER_PROMPT,
    SINGLE_AGENT_TOOLS_CODE_PROMPT,
    SINGLE_AGENT_TOOLS_CODE_SYSTEM_PROMPT,
    TOOL_LIBRARIES_DESCRIPTION,
)
from src.tools.definitions import PYTHON_SCIENTIFIC_TOOL
from src.tools.executor import PythonExecutor


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("self_contrast.v5")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset" / "XFinBench"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "Results"

ANSWER_FORMATS: Dict[str, str] = {
    "mcq": "The option letter (e.g., A, B, C)",
    "bool": "1.0 for True/Yes, 0.0 for False/No",
    "calcu": "The numerical value only (no units)",
}


def _build_libraries_description() -> str:
    """Build the library description string from tool definitions."""
    tool = PYTHON_SCIENTIFIC_TOOL
    libs = getattr(tool, "libraries", [])
    if not libs:
        return TOOL_LIBRARIES_DESCRIPTION

    lines = ["Available Python libraries:"]
    for lib in libs:
        name = getattr(lib, "name", "")
        desc = getattr(lib, "description", "")
        imp = getattr(lib, "import_name", "")
        funcs = getattr(lib, "common_functions", [])
        line = f"- {name}: {desc}"
        if imp:
            line += f" (import as: {imp})"
        if funcs:
            line += f" — common: {', '.join(funcs[:6])}"
        lines.append(line)
    return "\n".join(lines)


def _extract_code_block(text: str) -> Optional[str]:
    """Extract the first Python code block from markdown fences."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.strip().startswith(("import ", "from ", "print(", "def ", "#")):
        return text.strip()
    return None


class SingleAgentToolsSolver:
    """Single-agent solver with scientific toolkit access.

    For each problem:
    1. Generate Python code using the full scientific toolkit.
    2. Execute the code via PythonExecutor.
    3. Ask the LLM for a final JSON answer incorporating the tool output.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        executor: Optional[PythonExecutor] = None,
        force_json: bool = True,
        max_code_retries: int = 1,
    ) -> None:
        self.client = client
        self.executor = executor or PythonExecutor()
        self.force_json = force_json
        self.max_code_retries = max_code_retries
        self._libs_desc = _build_libraries_description()
        self._use_threads = getattr(client, "provider", "") not in (
            "ollama",
            "openai_compatible",
        )

    async def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a single problem: code generation -> execution -> answer."""
        task_type = problem.get("task") or problem.get("task_type") or ""
        question = problem.get("question", "")
        choices = problem.get("choice", "")
        format_rule = FORMAT_RULES.get(task_type, "")

        # --- Step 1: Generate code ---
        code_prompt = SINGLE_AGENT_TOOLS_CODE_PROMPT.format(
            question=question,
            libraries_description=self._libs_desc,
            format_rule=format_rule,
        )
        log.info("  requesting code ...")
        code_response = await self._call_llm(
            SINGLE_AGENT_TOOLS_CODE_SYSTEM_PROMPT, code_prompt, force_json=False
        )
        code = _extract_code_block(code_response)
        log.info("  code received (has_code=%s)", code is not None)

        # --- Step 2: Execute code ---
        tool_output = None
        if code:
            result = self.executor.execute(code)
            if result.success:
                tool_output = result.output.strip()
                log.info(
                    "  code executed OK (output=%s)",
                    tool_output[:80] if tool_output else "empty",
                )
            else:
                log.warning(
                    "  code execution failed: %s",
                    result.error[:120] if result.error else "unknown",
                )
                for attempt in range(self.max_code_retries):
                    retry_prompt = (
                        f"The following code failed:\n```python\n{code}\n```\n\n"
                        f"Error:\n{result.error}\n\n"
                        f"Fix the code and return only the corrected Python code block.\n"
                        f"Problem: {question}"
                    )
                    retry_resp = await self._call_llm(
                        SINGLE_AGENT_TOOLS_CODE_SYSTEM_PROMPT,
                        retry_prompt,
                        force_json=False,
                    )
                    code = _extract_code_block(retry_resp)
                    if code:
                        result = self.executor.execute(code)
                        if result.success:
                            tool_output = result.output.strip()
                            log.info(
                                "  retry %d OK (output=%s)",
                                attempt + 1,
                                tool_output[:80] if tool_output else "empty",
                            )
                            break
                        log.warning(
                            "  retry %d failed: %s",
                            attempt + 1,
                            result.error[:120] if result.error else "unknown",
                        )

        # --- Step 3: Final answer ---
        choices_section = f"Choices:\n{choices}\n" if choices else ""
        answer_format = ANSWER_FORMATS.get(task_type, "Your final answer")

        tool_output_section = ""
        if tool_output:
            tool_output_section = f"Python Tool Output:\n{tool_output}\n"

        answer_prompt = SINGLE_AGENT_TOOLS_ANSWER_PROMPT.format(
            question=question,
            choices_section=choices_section,
            tool_output_section=tool_output_section,
            answer_format=answer_format,
        )
        log.info("  requesting answer ...")
        answer_response = await self._call_llm(
            SINGLE_AGENT_SYSTEM_PROMPT, answer_prompt, force_json=self.force_json
        )

        prediction, reasoning = parse_single_agent_response(answer_response, task_type)

        return {
            "id": problem.get("id"),
            "prediction": prediction,
            "reasoning": reasoning,
            "ground_truth": problem.get("ground_truth"),
            "task_type": task_type,
            "raw_response": answer_response,
            "python_code": code,
            "python_output": tool_output,
        }

    async def _call_llm(self, system: str, user: str, *, force_json: bool) -> str:
        if self._use_threads:
            return await asyncio.to_thread(
                self.client.call, system, user, force_json=force_json
            )
        return self.client.call(system, user, force_json=force_json)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for V5 Single Agent + Tools."""
    parser = argparse.ArgumentParser(
        description="V5: Single Agent + Tools (code generation & execution per problem)."
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name.")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument(
        "--batch-file",
        type=str,
        default="evaluation_batch.json",
        help="Batch file name inside dataset-dir.",
    )
    parser.add_argument("--output", type=str, help="Output JSON file name.")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-problems", type=int)
    parser.add_argument("--url", type=str, help="Base URL for vec-inf / vLLM.")
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--max-code-retries", type=int, default=1)

    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument("--force-json", dest="force_json", action="store_true")
    json_group.add_argument("--no-force-json", dest="force_json", action="store_false")
    parser.set_defaults(force_json=True)
    return parser


async def run(args: argparse.Namespace) -> None:
    """Execute the V5 Single Agent + Tools pipeline."""
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else DEFAULT_DATASET_DIR
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_path = resolve_batch_file(args.batch_file, dataset_dir)
    with open(batch_path, "r", encoding="utf-8") as f:
        problems: List[Dict[str, Any]] = json.load(f)

    if args.max_problems:
        problems = problems[: args.max_problems]

    client = LLMClient(
        args.model,
        base_url=args.url,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    tool_cfg = PYTHON_SCIENTIFIC_TOOL
    executor = PythonExecutor(
        allowed_imports=tool_cfg.allowed_imports,
        timeout=int(tool_cfg.metadata.get("timeout", 30)),
    )

    solver = SingleAgentToolsSolver(
        client,
        executor=executor,
        force_json=args.force_json,
        max_code_retries=args.max_code_retries,
    )

    start = time.time()
    results: List[Dict[str, Any]] = []

    for idx, problem in enumerate(problems, start=1):
        pid = problem.get("id", f"problem_{idx}")
        log.info("Problem %s/%s: %s", idx, len(problems), pid)
        try:
            result = await solver.solve_problem(problem)
            result["is_correct"] = evaluate_result(result)
            results.append(result)
            log.info("  -> %s", "CORRECT" if result["is_correct"] else "INCORRECT")
        except Exception as exc:
            log.error("Error on %s: %s", pid, exc)

    elapsed = time.time() - start
    time_fmt = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    batch_metrics = evaluate_batch(results)
    extra = {
        "model": args.model,
        "method": "Single Agent + Tools",
        "temperature": args.temperature,
        "max_code_retries": args.max_code_retries,
        "batch_info": batch_path.stem,
        "execution_time_seconds": round(elapsed, 2),
        "execution_time_formatted": time_fmt,
    }

    safe_model = args.model.replace("/", "-").replace(":", "-")
    version_dir = results_dir / "v5_single_agent_tools" / safe_model
    version_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = version_dir / args.output
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = version_dir / f"{batch_path.stem}_T{args.temperature}_{ts}.json"

    save_results(batch_metrics, results, out_path, extra_metadata=extra)
    print_summary(
        batch_metrics,
        model=args.model,
        method="Single Agent + Tools (V5)",
        batch_info=batch_path.stem,
        execution_time=time_fmt,
    )


def main() -> None:
    """CLI entry point for V5 Single Agent + Tools."""
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
