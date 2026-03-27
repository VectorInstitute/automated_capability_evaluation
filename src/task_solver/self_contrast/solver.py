"""Core Self-Contrast solver.

Implements the three-phase pipeline (perspective solving, contrastive
comparison, adjudication) with configurable tool integration modes.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.task_solver.self_contrast.model_client import LLMClient
from src.task_solver.self_contrast.perspectives import PerspectiveResponse
from src.task_solver.self_contrast.prompts import (
    ADJUDICATION_SYSTEM_PROMPT,
    ADJUDICATION_USER_PROMPT,
    CODE_BLOCK_RETRY_PROMPT,
    CODE_RETRY_PROMPT,
    CONTRAST_SYSTEM_PROMPT,
    CONTRAST_USER_PROMPT,
    DEFAULT_FORMAT_RULE,
    FORMAT_RULES,
    PERSPECTIVE_ANSWER_REQUEST,
    PERSPECTIVE_CODE_REQUEST,
    PERSPECTIVE_CODE_SYSTEM_PROMPT,
    PERSPECTIVE_SYSTEM_PROMPT,
    TOOL_INTEGRATED_CODE_REQUEST,
    TOOL_LIBRARIES_DESCRIPTION,
    TOOL_PERSPECTIVE_CODE_REQUEST,
    TOOL_SELECTION_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)
from src.tools.definitions import PYTHON_SCIENTIFIC_TOOL
from src.tools.executor import PythonExecutor
try:
    from src.utils.json_utils import parse_llm_json_response
except Exception:
    from src.task_solver.self_contrast._json_utils import parse_llm_json_response

if TYPE_CHECKING:
    from src.tools.docs import ScientificDocRetriever


log = logging.getLogger(__name__)

LIBRARY_NAME_ALIASES = {
    "arch": "arch",
    "cmath": "cmath",
    "datetime": "datetime",
    "decimal": "decimal",
    "empyrical": "empyrical",
    "fractions": "fractions",
    "math": "math",
    "np": "numpy",
    "npf": "numpy_financial",
    "numpy": "numpy",
    "numpy_financial": "numpy_financial",
    "py_vollib": "py_vollib",
    "pyportfolioopt": "pypfopt",
    "pypfopt": "pypfopt",
    "qf_lib": None,
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sympy": "sympy",
}

FALLBACK_LIBRARY_DETAILS: Dict[str, Dict[str, Any]] = {
    "datetime": {
        "import_name": "datetime",
        "description": "Standard library date and time utilities.",
        "common_functions": ["datetime.datetime", "datetime.timedelta"],
        "use_cases": ["Cash-flow schedules", "Date arithmetic"],
    },
    "statsmodels": {
        "import_name": "statsmodels",
        "description": "Statistical models, regression, and time-series analysis.",
        "common_functions": [
            "statsmodels.api.OLS",
            "statsmodels.tsa.api.ARIMA",
            "statsmodels.stats.weightstats",
        ],
        "use_cases": ["Regression analysis", "Time-series modelling"],
    },
}


class SelfContrastSolver:
    """Self-Contrast solver with configurable tool integration.

    Parameters
    ----------
    client : LLMClient
        Lightweight model client for LLM calls.
    perspectives : List[Dict[str, Any]]
        Perspective definitions.  Each dict must have ``id``, ``label``,
        ``guidance``, and ``uses_tools`` keys.
    executor : Optional[PythonExecutor]
        Code executor for tool-augmented modes.  Required when
        *tool_mode* is not ``"none"``.
    tool_mode : str
        ``"none"``  -- V1: basic python only for calcu tasks.
        ``"all"``   -- V3: every perspective uses ``PythonExecutor``.
        ``"tool_perspective_only"`` -- V4: only perspectives with
        ``uses_tools=True`` use ``PythonExecutor``.
    prompt_repeat : int
        Repeat the user prompt N times (1/2/3) per arXiv:2512.14982.
    force_json : bool
        Request JSON output via ``response_format`` when supported.
    doc_retriever : Optional[ScientificDocRetriever]
        Optional local HTML documentation retriever for RAG-style module context.
    max_code_retries : int
        Maximum number of retry attempts after code execution errors.
    enable_tool_selection : bool
        Whether to run an LLM-based tool-selection step for tool-enabled modes.
    """

    def __init__(
        self,
        client: LLMClient,
        perspectives: List[Dict[str, Any]],
        *,
        executor: Optional[PythonExecutor] = None,
        tool_mode: str = "none",
        prompt_repeat: int = 1,
        force_json: bool = True,
        doc_retriever: Optional[ScientificDocRetriever] = None,
        max_code_retries: int = 0,
        enable_tool_selection: bool = False,
    ) -> None:
        self.client = client
        self.perspectives = perspectives
        self.executor = executor or PythonExecutor()
        self.tool_mode = tool_mode
        self.prompt_repeat = max(1, min(3, prompt_repeat))
        self.force_json = force_json
        self.doc_retriever = doc_retriever
        self.max_code_retries = max(0, max_code_retries)
        self.enable_tool_selection = enable_tool_selection
        self._library_configs = {
            config.name: config for config in PYTHON_SCIENTIFIC_TOOL.libraries
        }
        self._executor_lock = threading.Lock()
        self._use_threads = getattr(client, "provider", "") not in (
            "ollama",
            "openai_compatible",
        )

    async def _to_thread(self, func, *args, **kwargs):
        """Offload *func* to a thread for cloud APIs; call directly for local."""
        if self._use_threads:
            return await asyncio.to_thread(func, *args, **kwargs)
        return func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full Self-Contrast pipeline on a single problem.

        Returns a result dict compatible with the evaluator (contains
        ``id``, ``prediction``, ``ground_truth``, ``task_type``, and
        ``contrast_details``).
        """
        task_type = problem.get("task") or problem.get("task_type") or ""
        pid = problem.get("id", "?")
        problem_text = self._build_problem_text(problem)
        tool_context = await self._prepare_tool_context(problem_text, task_type)

        log.info("[%s] Phase 1: solving %d perspectives (sequential=%s)",
                 pid, len(self.perspectives),
                 not self._should_parallelize_perspective_calls())
        perspective_outputs = await self._run_perspective_round(
            problem_text,
            task_type,
            tool_context=tool_context,
        )
        log.info("[%s] Phase 1 done. Answers: %s", pid,
                 [r.answer for r in perspective_outputs])

        majority = self._select_majority_answer(perspective_outputs, task_type)

        if self._has_local_consensus(perspective_outputs, task_type):
            majority = self._select_majority_answer(perspective_outputs, task_type)
            log.info("[%s] Local consensus reached, skipping contrast.", pid)
            if majority is None:
                majority = self._select_any_answer(perspective_outputs, task_type)
            contrast = {
                "raw_response": "",
                "pairwise_discrepancies": {},
                "checklist": [],
                "skipped": True,
                "reasoning": (
                    "Perspective answers already aligned after local normalization. "
                    "Skipping contrast and adjudication."
                ),
            }
            final: Dict[str, Any] = {
                "answer": majority["answer"] if majority else None,
                "rationale": (
                    "Perspective answers already aligned after local normalization."
                ),
                "decision_source": "local_consensus",
                "vote_details": majority or {},
            }
        else:
            log.info("[%s] Phase 2: contrast (no local consensus).", pid)
            contrast = await self._contrast_discrepancies(
                problem_text, task_type, perspective_outputs
            )
            log.info("[%s] Phase 2 done.", pid)

            has_discrepancies = self._contrast_has_discrepancies(contrast)

            if not has_discrepancies:
                if majority is None:
                    majority = self._select_any_answer(perspective_outputs, task_type)
                final = {
                    "answer": majority["answer"] if majority else None,
                    "rationale": (
                        "Contrast found no discrepancies. Using available consensus."
                    ),
                    "decision_source": "contrast_ok",
                    "vote_details": majority or {},
                }
            else:
                final = await self._final_adjudication(
                    problem_text, task_type, perspective_outputs, contrast
                )
                final["decision_source"] = "contrast_adjudication"

        return {
            "id": problem.get("id"),
            "prediction": final.get("answer"),
            "ground_truth": problem.get("ground_truth"),
            "task_type": task_type,
            "contrast_details": {
                "perspectives": [r.to_dict() for r in perspective_outputs],
                "contrast": contrast,
                "final": final,
            },
        }

    # ------------------------------------------------------------------
    # Phase 1 – perspective solving
    # ------------------------------------------------------------------

    async def _solve_with_perspective(
        self,
        problem_text: str,
        task_type: str,
        perspective: Dict[str, Any],
        *,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> PerspectiveResponse:
        label = perspective.get("label", perspective.get("id", "?"))
        use_tools = self._should_use_tools(
            perspective,
            problem_text,
            task_type,
            tool_context=tool_context,
        )

        python_code: Optional[str] = None
        python_output: Optional[str] = None

        if use_tools:
            log.info("  [%s] requesting code ...", label)
            python_code = await self._request_code(
                problem_text,
                task_type,
                perspective,
                tool_context=tool_context,
            )
            log.info("  [%s] code received (has_code=%s)", label, python_code is not None)
            retry_count = 0
            while python_code:
                execution = await self._execute_code(python_code)
                python_output = execution["formatted_output"]
                if execution["success"] or retry_count >= self.max_code_retries:
                    break

                retry_count += 1
                retry_code = await self._request_code_retry(
                    problem_text,
                    perspective,
                    failed_code=python_code,
                    error_message=execution["error"] or execution["formatted_output"],
                    tool_context=tool_context,
                )
                if not retry_code or retry_code == python_code:
                    break
                python_code = retry_code

        log.info("  [%s] requesting answer ...", label)
        answer_response = await self._request_answer(
            problem_text, task_type, perspective, python_output=python_output
        )
        log.info("  [%s] answer received.", label)

        parsed = self._parse_answer_json(answer_response, task_type)
        if parsed.get("answer") is None and python_output is not None:
            tool_answer = self._fallback_answer(python_output, task_type)
            if tool_answer is not None:
                parsed = {
                    "answer": tool_answer,
                    "rationale": "Derived directly from Python tool output.",
                }

        return PerspectiveResponse(
            perspective_id=perspective["id"],
            label=perspective["label"],
            answer=parsed.get("answer"),
            rationale=parsed.get("rationale", ""),
            raw_response=answer_response,
            python_code=python_code,
            python_output=python_output,
        )

    def _should_use_tools(
        self,
        perspective: Dict[str, Any],
        problem_text: str,
        task_type: str,
        *,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        selection_allows_tools = True
        if self.enable_tool_selection and tool_context is not None:
            selection_allows_tools = bool(tool_context.get("needs_tools"))
        heuristic_requires_tools = self._calcu_heuristic(problem_text, task_type)
        if self.tool_mode == "all":
            return heuristic_requires_tools or selection_allows_tools
        if self.tool_mode == "tool_perspective_only":
            return bool(perspective.get("uses_tools")) and (
                heuristic_requires_tools or selection_allows_tools
            )
        # tool_mode == "none": use basic python only when heuristic fires
        return heuristic_requires_tools

    def _use_legacy_local_base_prompts(self) -> bool:
        provider = getattr(self.client, "provider", "")
        return self.tool_mode == "none" and provider in ("ollama", "openai_compatible")

    async def _request_code(
        self,
        problem_text: str,
        task_type: str,
        perspective: Dict[str, Any],
        *,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        libraries_description = self._resolve_libraries_description(tool_context)
        format_rule = FORMAT_RULES.get(task_type, DEFAULT_FORMAT_RULE)
        system_prompt = PERSPECTIVE_CODE_SYSTEM_PROMPT
        if perspective.get("uses_tools") or self.tool_mode == "all":
            if perspective.get("uses_tools"):
                prompt = TOOL_PERSPECTIVE_CODE_REQUEST.format(
                    guidance=perspective["guidance"],
                    problem_text=problem_text,
                    libraries_description=libraries_description,
                    format_rule=format_rule,
                )
            else:
                prompt = TOOL_INTEGRATED_CODE_REQUEST.format(
                    label=perspective["label"],
                    guidance=perspective["guidance"],
                    problem_text=problem_text,
                    libraries_description=libraries_description,
                    format_rule=format_rule,
                )
        else:
            if self._use_legacy_local_base_prompts():
                system_prompt = (
                    "You are a financial analyst. Provide only Python code to "
                    "compute the answer."
                )
                prompt = (
                    f"Perspective: {perspective['label']}.\n"
                    f"Guidance: {perspective['guidance']}\n\n"
                    "Problem:\n"
                    f"{problem_text}\n\n"
                    "Instructions:\n"
                    "- Write a single Python code block that computes the final "
                    "numeric answer.\n"
                    "- Use only the Python standard library (math is allowed).\n"
                    "- Print the final numeric answer.\n"
                    "- Output ONLY the Python code block, nothing else."
                )
            else:
                prompt = PERSPECTIVE_CODE_REQUEST.format(
                    label=perspective["label"],
                    guidance=perspective["guidance"],
                    problem_text=problem_text,
                    format_rule=format_rule,
                )

        prompt = self._apply_prompt_repetition(prompt)
        response = await self._to_thread(
            self.client.call, system_prompt, prompt
        )
        code = self._extract_python_code(response)
        if code:
            return code
        if self._use_legacy_local_base_prompts():
            log.info(
                "  [%s] no code block returned; skipping retry to match legacy local behavior.",
                perspective["label"],
            )
            return None

        retry_prompt = CODE_BLOCK_RETRY_PROMPT.format(
            label=perspective["label"],
            guidance=perspective["guidance"],
            problem_text=problem_text,
            libraries_description=libraries_description,
            previous_response=response or "(empty response)",
            format_rule=format_rule,
        )
        retry_prompt = self._apply_prompt_repetition(retry_prompt)
        retry_response = await self._to_thread(
            self.client.call, system_prompt, retry_prompt
        )
        code = self._extract_python_code(retry_response)
        if not code:
            log.warning(
                "No python code block found for perspective %s", perspective["id"]
            )
        return code

    async def _request_code_retry(
        self,
        problem_text: str,
        perspective: Dict[str, Any],
        *,
        failed_code: str,
        error_message: str,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        prompt = CODE_RETRY_PROMPT.format(
            label=perspective["label"],
            guidance=perspective["guidance"],
            problem_text=problem_text,
            libraries_description=self._resolve_libraries_description(tool_context),
            failed_code=failed_code,
            error_message=error_message,
        )
        prompt = self._apply_prompt_repetition(prompt)
        response = await self._to_thread(
            self.client.call, PERSPECTIVE_CODE_SYSTEM_PROMPT, prompt
        )
        code = self._extract_python_code(response)
        if not code:
            log.warning(
                "No retry code block found for perspective %s", perspective["id"]
            )
        return code

    async def _prepare_tool_context(
        self, problem_text: str, task_type: str
    ) -> Dict[str, Any]:
        if self.tool_mode == "none":
            return {
                "needs_tools": False,
                "selected_modules": [],
                "documentation": "",
                "reasoning": "Base Self-Contrast mode uses its built-in heuristic only.",
                "raw_response": "",
            }

        if self._calcu_heuristic(problem_text, task_type):
            return {
                "needs_tools": True,
                "selected_modules": [],
                "documentation": self._build_tool_context(),
                "reasoning": (
                    "Calculation-oriented tasks always enable tool execution for "
                    "reliable verification."
                ),
                "raw_response": "",
            }

        if not self.enable_tool_selection:
            return {
                "needs_tools": True,
                "selected_modules": [],
                "documentation": self._build_tool_context(),
                "reasoning": (
                    "Tool selection disabled. Tool-enabled perspectives may use the "
                    "full scientific toolkit."
                ),
                "raw_response": "",
            }

        return await self._run_tool_selection(problem_text)

    async def _run_tool_selection(self, problem_text: str) -> Dict[str, Any]:
        prompt = TOOL_SELECTION_PROMPT.format(
            problem_text=problem_text,
            tools_description=self._build_tool_selection_overview(),
        )
        response = await self._to_thread(
            self.client.call,
            TOOL_SELECTION_SYSTEM_PROMPT,
            prompt,
            force_json=self.force_json,
        )
        parsed = self._safe_parse_json(response)

        if not parsed:
            return {
                "needs_tools": True,
                "selected_modules": [],
                "documentation": self._build_tool_context(),
                "reasoning": (
                    "Tool selection response was invalid. Falling back to full toolkit "
                    "availability."
                ),
                "raw_response": response,
            }

        needs_tools = self._coerce_bool(parsed.get("tool_necessity"), default=True)
        selected_modules = self._sanitize_selected_modules(
            parsed.get("selected_modules", [])
        )
        documentation = (
            self._build_tool_context(selected_modules=selected_modules)
            if needs_tools
            else ""
        )
        reasoning = parsed.get("reasoning") or (
            "Selected toolkit libraries based on the problem requirements."
            if needs_tools
            else "Problem appears solvable without external computation."
        )
        return {
            "needs_tools": needs_tools,
            "selected_modules": selected_modules,
            "documentation": documentation,
            "reasoning": reasoning,
            "raw_response": response,
        }

    def _build_tool_selection_overview(self) -> str:
        if self.doc_retriever is not None:
            overview = self.doc_retriever.get_library_overview().strip()
            if overview:
                return overview

        lines = [
            "These libraries are available in the scientific toolkit:",
        ]
        for library_name in self._tool_library_order():
            lines.extend(self._format_library_entry(library_name))
        return "\n".join(lines)

    def _build_tool_context(
        self,
        *,
        selected_modules: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        lines = [
            "You have access to the scientific Python toolkit backed by "
            "PythonExecutor.",
            f"Execution timeout: {getattr(self.executor, '_timeout', 30)} seconds.",
        ]

        module_refs = self._format_selected_modules(selected_modules or [])
        if module_refs:
            lines.append(f"Suggested modules for this problem: {', '.join(module_refs)}")

        lines.append("Available libraries:")
        for library_name in self._tool_library_order(selected_modules):
            lines.extend(self._format_library_entry(library_name))

        doc_context = self._build_doc_context(selected_modules or [])
        if doc_context:
            lines.append("Local documentation context:")
            lines.append(doc_context)

        return "\n".join(lines)

    def _resolve_libraries_description(
        self, tool_context: Optional[Dict[str, Any]]
    ) -> str:
        if self.tool_mode == "none":
            return "Use only the Python standard library. ``math`` is allowed."
        if tool_context and tool_context.get("documentation"):
            return str(tool_context["documentation"])
        return TOOL_LIBRARIES_DESCRIPTION

    def _build_doc_context(self, selected_modules: List[Dict[str, str]]) -> str:
        if self.doc_retriever is None:
            return ""

        if not selected_modules:
            return self.doc_retriever.get_library_overview().strip()

        doc_blocks: List[str] = []
        seen_blocks = set()
        for item in selected_modules:
            library = item.get("library")
            module = item.get("module")
            if not library or not module:
                continue
            docs = self.doc_retriever.get_full_module_context(library, module).strip()
            if docs and not docs.startswith("Error:") and docs not in seen_blocks:
                seen_blocks.add(docs)
                doc_blocks.append(docs)
        return "\n\n".join(doc_blocks)

    def _tool_library_order(
        self,
        selected_modules: Optional[List[Dict[str, str]]] = None,
    ) -> List[str]:
        library_order = list(dict.fromkeys(PYTHON_SCIENTIFIC_TOOL.allowed_imports))
        if not selected_modules:
            return library_order

        selected_names = {
            item["library"]
            for item in selected_modules
            if item.get("library") in library_order
        }
        if not selected_names:
            return library_order
        return [name for name in library_order if name in selected_names]

    def _format_library_entry(self, library_name: str) -> List[str]:
        config = self._library_configs.get(library_name)
        if config is not None:
            details = [
                f"- {config.name}: {config.description}",
                f"  Import: {config.import_name}",
            ]
            if config.common_functions:
                details.append(
                    "  Common functions: " + ", ".join(config.common_functions)
                )
            if config.use_cases:
                details.append("  Use cases: " + "; ".join(config.use_cases))
            return details

        fallback = FALLBACK_LIBRARY_DETAILS.get(library_name)
        if fallback is None:
            return [f"- {library_name}: Available in the execution environment."]

        details = [
            f"- {library_name}: {fallback['description']}",
            f"  Import: {fallback['import_name']}",
        ]
        common_functions = fallback.get("common_functions") or []
        if common_functions:
            details.append("  Common functions: " + ", ".join(common_functions))
        use_cases = fallback.get("use_cases") or []
        if use_cases:
            details.append("  Use cases: " + "; ".join(use_cases))
        return details

    def _format_selected_modules(self, selected_modules: List[Dict[str, str]]) -> List[str]:
        formatted: List[str] = []
        for item in selected_modules:
            library = item.get("library")
            module = item.get("module")
            if not library:
                continue
            formatted.append(f"{library}.{module}" if module else library)
        return list(dict.fromkeys(formatted))

    def _sanitize_selected_modules(self, raw_modules: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_modules, list):
            return []

        selected_modules: List[Dict[str, str]] = []
        seen = set()
        for item in raw_modules:
            if not isinstance(item, dict):
                continue
            library = self._normalize_library_name(item.get("library", ""))
            if not library:
                continue
            module = self._normalize_module_name(item.get("module", ""))
            key = (library, module)
            if key in seen:
                continue
            seen.add(key)
            selected_modules.append({"library": library, "module": module})
        return selected_modules

    def _normalize_library_name(self, library_name: Any) -> Optional[str]:
        normalized = str(library_name).strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            return None
        canonical = LIBRARY_NAME_ALIASES.get(normalized, normalized)
        if canonical is None:
            return None
        if canonical in PYTHON_SCIENTIFIC_TOOL.allowed_imports:
            return canonical
        return None

    @staticmethod
    def _normalize_module_name(module_name: Any) -> str:
        normalized = str(module_name).strip().lower()
        if not normalized or normalized in {"all", "*", "none"}:
            return ""
        normalized = normalized.split(".", maxsplit=1)[0]
        return normalized.replace("-", "_").replace(" ", "_")

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return default

    async def _request_answer(
        self,
        problem_text: str,
        task_type: str,
        perspective: Dict[str, Any],
        *,
        python_output: Optional[str] = None,
    ) -> str:
        tool_section = ""
        if python_output is not None:
            tool_section = f"\nPython Tool Output:\n{python_output}\n"

        format_rule = FORMAT_RULES.get(task_type, DEFAULT_FORMAT_RULE)
        prompt = PERSPECTIVE_ANSWER_REQUEST.format(
            label=perspective["label"],
            guidance=perspective["guidance"],
            problem_text=problem_text,
            tool_output_section=tool_section,
            format_rule=format_rule,
        )
        prompt = self._apply_prompt_repetition(prompt)
        return await self._to_thread(
            self.client.call,
            PERSPECTIVE_SYSTEM_PROMPT,
            prompt,
            force_json=self.force_json,
        )

    # ------------------------------------------------------------------
    # Phase 2 – contrastive comparison
    # ------------------------------------------------------------------

    async def _contrast_discrepancies(
        self,
        problem_text: str,
        task_type: str,
        responses: List[PerspectiveResponse],
    ) -> Dict[str, Any]:
        labels = [chr(ord("A") + i) for i in range(len(responses))]
        formatted = self._format_responses(responses, labels)

        pairs = list(itertools.combinations(labels, 2))
        pairwise_keys = "\n".join(f'    "{a}_vs_{b}": ["..."],' for a, b in pairs)
        if pairwise_keys.endswith(","):
            pairwise_keys = pairwise_keys[:-1]

        prompt = CONTRAST_USER_PROMPT.format(
            n_perspectives=len(responses),
            problem_text=problem_text,
            formatted_solutions=formatted,
            pairwise_keys=pairwise_keys,
        )

        response = await self._to_thread(
            self.client.call,
            CONTRAST_SYSTEM_PROMPT,
            prompt,
            force_json=self.force_json,
        )
        parsed = self._safe_parse_json(response)
        return {
            "raw_response": response,
            "pairwise_discrepancies": parsed.get("pairwise_discrepancies", {}),
            "checklist": parsed.get("checklist", []),
        }

    # ------------------------------------------------------------------
    # Phase 3 – adjudication
    # ------------------------------------------------------------------

    async def _final_adjudication(
        self,
        problem_text: str,
        task_type: str,
        responses: List[PerspectiveResponse],
        contrast: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = [chr(ord("A") + i) for i in range(len(responses))]
        formatted = self._format_responses(responses, labels)

        checklist = contrast.get("checklist") or []
        checklist_text = "\n".join(f"- {item}" for item in checklist) or "- (none)"
        format_rule = FORMAT_RULES.get(task_type, DEFAULT_FORMAT_RULE)

        prompt = ADJUDICATION_USER_PROMPT.format(
            problem_text=problem_text,
            formatted_solutions=formatted,
            checklist_text=checklist_text,
            format_rule=format_rule,
        )

        response = await self._to_thread(
            self.client.call,
            ADJUDICATION_SYSTEM_PROMPT,
            prompt,
            force_json=self.force_json,
        )
        parsed = self._parse_answer_json(response, task_type)
        parsed["raw_response"] = response
        return parsed

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    async def _execute_code(self, code: str) -> Dict[str, Any]:
        return await self._to_thread(self._execute_code_sync, code)

    def _execute_code_sync(self, code: str) -> Dict[str, Any]:
        with self._executor_lock:
            try:
                result = self.executor.execute(code)
                if result.success:
                    return {
                        "success": True,
                        "output": result.output,
                        "error": None,
                        "formatted_output": result.output,
                    }
                log.warning("Code execution error: %s", result.error)
                return {
                    "success": False,
                    "output": result.output,
                    "error": result.error,
                    "formatted_output": f"Error: {result.error}",
                }
            except Exception as exc:
                log.warning("Code execution exception: %s", exc)
                return {
                    "success": False,
                    "output": "",
                    "error": str(exc),
                    "formatted_output": f"Error: {exc}",
                }

    # ------------------------------------------------------------------
    # Voting helpers
    # ------------------------------------------------------------------

    def _select_majority_answer(
        self,
        responses: List[PerspectiveResponse],
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        entries = self._build_answer_entries(responses, task_type)
        if not entries:
            return None

        pools = []
        python_entries = [e for e in entries if e["python_ok"]]
        if python_entries:
            pools.append(("python_majority", python_entries))
        pools.append(("majority", entries))

        for source, pool in pools:
            if task_type == "calcu":
                majority = self._select_calcu_majority(pool)
            else:
                majority = self._select_exact_majority(pool)
            if majority is not None:
                majority["source"] = source
                return majority
        return None

    def _select_any_answer(
        self,
        responses: List[PerspectiveResponse],
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        entries = self._build_answer_entries(responses, task_type)
        if not entries:
            return None

        preferred_entries = [e for e in entries if e["python_ok"]] or entries
        first = preferred_entries[0]
        return {
            "answer": first["display"],
            "source": "first_available",
            "votes": 1,
            "counts": {first["key"]: 1},
        }

    def _has_local_consensus(
        self,
        responses: List[PerspectiveResponse],
        task_type: str,
    ) -> bool:
        entries = self._build_answer_entries(responses, task_type)
        if len(entries) < 2:
            return False

        if task_type != "calcu":
            return len({entry["key"] for entry in entries}) == 1

        equivalent_groups = self._group_entries_by_relation(
            entries, self._entries_are_equivalent
        )
        if len(equivalent_groups) != 1:
            return False

        same_scale_groups = self._group_entries_by_relation(
            equivalent_groups[0], self._entries_share_same_scale
        )
        top_group = self._unique_top_group(same_scale_groups)
        return top_group is not None and len(top_group) >= 2

    def _build_answer_entries(
        self,
        responses: List[PerspectiveResponse],
        task_type: str,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for resp in responses:
            key, display = self._normalize_answer(resp.answer, task_type)
            if key is None or display is None:
                continue
            entry: Dict[str, Any] = {
                "key": key,
                "display": display,
                "python_ok": self._python_output_ok(resp.python_output),
            }
            if task_type == "calcu":
                entry["numeric"] = self._extract_number(display)
            entries.append(entry)
        return entries

    @staticmethod
    def _select_exact_majority(pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        for entry in pool:
            counts[entry["key"]] = counts.get(entry["key"], 0) + 1
        if not counts:
            return None

        top_key, top_count = max(counts.items(), key=lambda kv: kv[1])
        if list(counts.values()).count(top_count) > 1 or top_count < 2:
            return None

        display = next(entry["display"] for entry in pool if entry["key"] == top_key)
        return {
            "answer": display,
            "votes": top_count,
            "counts": counts,
        }

    def _select_calcu_majority(
        self, pool: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        equivalent_groups = self._group_entries_by_relation(
            pool, self._entries_are_equivalent
        )
        top_group = self._unique_top_group(equivalent_groups)
        if top_group is None or len(top_group) < 2:
            return None

        same_scale_groups = self._group_entries_by_relation(
            top_group, self._entries_share_same_scale
        )
        top_same_scale_group = self._unique_top_group(same_scale_groups)
        if top_same_scale_group is None or len(top_same_scale_group) < 2:
            return None

        representative = self._format_numeric_group(top_same_scale_group)
        counts = {
            entry["display"]: sum(
                1 for other in pool if other["display"] == entry["display"]
            )
            for entry in pool
        }
        return {
            "answer": representative,
            "votes": len(top_group),
            "same_scale_votes": len(top_same_scale_group),
            "counts": counts,
        }

    @staticmethod
    def _unique_top_group(
        groups: List[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not groups:
            return None
        top_size = max(len(group) for group in groups)
        top_groups = [group for group in groups if len(group) == top_size]
        if len(top_groups) != 1:
            return None
        return top_groups[0]

    def _group_entries_by_relation(
        self,
        entries: List[Dict[str, Any]],
        relation: Any,
    ) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        visited: set[int] = set()

        for idx in range(len(entries)):
            if idx in visited:
                continue

            queue = [idx]
            visited.add(idx)
            group_indices = [idx]
            while queue:
                current = queue.pop()
                for other in range(len(entries)):
                    if other in visited:
                        continue
                    if relation(entries[current], entries[other]):
                        visited.add(other)
                        queue.append(other)
                        group_indices.append(other)
            groups.append([entries[index] for index in group_indices])
        return groups

    def _entries_are_equivalent(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> bool:
        left_num = left.get("numeric")
        right_num = right.get("numeric")
        if left_num is None or right_num is None:
            return False
        return self._numeric_close(left_num, right_num) or self._numeric_scale_match(
            left_num, right_num
        )

    def _entries_share_same_scale(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> bool:
        left_num = left.get("numeric")
        right_num = right.get("numeric")
        if left_num is None or right_num is None:
            return False
        return self._numeric_close(left_num, right_num)

    @staticmethod
    def _numeric_close(left: float, right: float) -> bool:
        if left == right:
            return True
        abs_diff = abs(left - right)
        scale = max(abs(left), abs(right))
        if scale < 1e-9:
            return abs_diff <= 1e-9
        return abs_diff / scale <= 0.01

    def _numeric_scale_match(self, left: float, right: float) -> bool:
        if abs(left) < 1 <= abs(right):
            return self._numeric_close(left * 100, right)
        if abs(right) < 1 <= abs(left):
            return self._numeric_close(right * 100, left)
        return False

    @staticmethod
    def _format_numeric_group(group: List[Dict[str, Any]]) -> str:
        numeric_values = [
            entry["numeric"] for entry in group if entry.get("numeric") is not None
        ]
        if not numeric_values:
            return group[0]["display"]
        mean_value = sum(numeric_values) / len(numeric_values)
        if mean_value == 0:
            mean_value = 0.0
        return f"{mean_value:.10g}"

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _build_problem_text(self, problem: Dict[str, Any]) -> str:
        question = problem.get("question", "")
        choices = problem.get("choice", "")
        if choices:
            return f"Question: {question}\nChoices:\n{choices}"
        return f"Question: {question}"

    def _parse_answer_json(self, response: str, task_type: str) -> Dict[str, Any]:
        parsed = self._safe_parse_json(response)
        answer = parsed.get("answer")
        rationale = parsed.get("rationale") or parsed.get("reasoning") or ""
        if answer is None:
            answer = self._fallback_answer(response, task_type)
        return {"answer": answer, "rationale": rationale}

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        try:
            return parse_llm_json_response(response)
        except Exception:
            return {}

    def _fallback_answer(self, response: str, task_type: str) -> Optional[str]:
        text = response.strip()
        if task_type == "mcq":
            match = re.search(r"\b([A-Z])\b", text)
            if match:
                return match.group(1)
        if task_type == "bool":
            match = re.search(
                r"\b(1\.0|0\.0|1|0|true|false|yes|no)\b", text, re.IGNORECASE
            )
            if match:
                return match.group(1)
        if task_type == "calcu":
            num = self._extract_number(text)
            if num is not None:
                return str(num)
        return text or None

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        if text is None:
            return None
        clean = str(text).strip().replace(",", "").replace("$", "").replace("%", "")
        if clean.startswith("(") and clean.endswith(")"):
            clean = "-" + clean[1:-1]
        sci = re.search(r"[-+]?\d*\.?\d+[eE][-+]?\d+", clean)
        if sci:
            try:
                return float(sci.group())
            except ValueError:
                pass
        match = re.search(r"[-+]?\d*\.?\d+", clean)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_python_code(response: str) -> Optional[str]:
        match = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if not match:
            match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return code if code else None
        text = response.strip()
        if (
            text
            and "```" not in text
            and not text.lstrip().startswith("{")
            and (
                "print(" in text
                or text.startswith("import ")
                or text.startswith("from ")
                or "\nprint(" in text
            )
        ):
            return text
        return None

    def _apply_prompt_repetition(self, prompt: str) -> str:
        if self.prompt_repeat <= 1:
            return prompt
        if self.prompt_repeat == 2:
            return prompt + "\n\n" + prompt
        return (
            prompt
            + "\n\nLet me repeat that:\n\n"
            + prompt
            + "\n\nLet me repeat that one more time:\n\n"
            + prompt
        )

    def _format_responses(
        self, responses: List[PerspectiveResponse], labels: List[str]
    ) -> str:
        blocks = []
        for label, resp in zip(labels, responses):
            block = (
                f"{label}. {resp.label}\n"
                f"Answer: {resp.answer}\n"
                f"Rationale: {resp.rationale}\n"
            )
            if resp.python_output:
                block += f"Python Output: {resp.python_output}\n"
            blocks.append(block)
        return "\n".join(blocks)

    @staticmethod
    def _normalize_answer(  # noqa: PLR0911
        answer: Optional[str], task_type: str
    ) -> tuple[Optional[str], Optional[str]]:
        if answer is None:
            return None, None
        text = str(answer).strip()
        if not text:
            return None, None

        if task_type == "calcu":
            num = SelfContrastSolver._extract_number(text)
            if num is None:
                return None, None
            if abs(num) == 0:
                num = 0.0
            key = f"{num:.10g}"
            return key, key

        if task_type == "bool":
            lowered = text.lower()
            true_vals = {"1", "1.0", "true", "yes", "y", "t"}
            false_vals = {"0", "0.0", "false", "no", "n", "f"}
            if lowered in true_vals:
                return "1.0", "1.0"
            if lowered in false_vals:
                return "0.0", "0.0"
            return lowered, text

        if task_type == "mcq":
            match = re.search(r"[A-Z]", text.upper())
            if match:
                return match.group(0), match.group(0)
            return text.upper(), text.upper()

        return text, text

    @staticmethod
    def _python_output_ok(output: Optional[str]) -> bool:
        if not output:
            return False
        return "Error" not in output and "Traceback" not in output

    @staticmethod
    def _contrast_has_discrepancies(contrast: Dict[str, Any]) -> bool:
        pairwise = contrast.get("pairwise_discrepancies")
        if not isinstance(pairwise, dict):
            return True
        return any(items for items in pairwise.values())

    def _should_parallelize_perspective_calls(self) -> bool:
        provider = getattr(self.client, "provider", "")
        # vec-inf / Ollama-style endpoints usually back a single serving queue.
        # Serializing perspective calls reduces queue buildup and timeout risk.
        return provider not in ("ollama", "openai_compatible")

    async def _run_perspective_round(
        self,
        problem_text: str,
        task_type: str,
        *,
        tool_context: Optional[Dict[str, Any]],
    ) -> List[PerspectiveResponse]:
        if self._should_parallelize_perspective_calls():
            return list(
                await asyncio.gather(
                    *[
                        self._solve_with_perspective(
                            problem_text,
                            task_type,
                            perspective,
                            tool_context=tool_context,
                        )
                        for perspective in self.perspectives
                    ]
                )
            )

        outputs: List[PerspectiveResponse] = []
        for perspective in self.perspectives:
            outputs.append(
                await self._solve_with_perspective(
                    problem_text,
                    task_type,
                    perspective,
                    tool_context=tool_context,
                )
            )
        return outputs

    # ------------------------------------------------------------------
    # Phase 2.5 – reflective debate
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calcu_heuristic(problem_text: str, task_type: str) -> bool:
        """Heuristic: use python when the problem is clearly computational."""
        if task_type == "calcu":
            return True
        text = problem_text.lower()
        has_number = bool(re.search(r"\d", text))
        calc_keywords = [
            "calculate",
            "compute",
            "find",
            "how much",
            "how many",
            "what is",
            "total",
            "sum",
            "difference",
            "increase",
            "decrease",
            "percentage",
            "percent",
            "rate",
            "ratio",
            "growth",
            "yield",
            "npv",
            "irr",
            "duration",
            "return",
            "profit",
            "loss",
            "cash flow",
        ]
        return has_number and any(kw in text for kw in calc_keywords)
