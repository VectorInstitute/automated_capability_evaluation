"""Prompt templates for the Self-Contrast framework.

All prompts used across the four solver variants are defined here so they
can be iterated on in one place.
"""

from __future__ import annotations

from typing import Dict


# ---------------------------------------------------------------------------
# Format rules per task type
# ---------------------------------------------------------------------------

FORMAT_RULES: Dict[str, str] = {
    "mcq": "Answer must be a single option letter (e.g., A, B, C).",
    "bool": "Answer must be 1.0 for True/Yes or 0.0 for False/No.",
    "calcu": "Answer must be a numeric value only (no units).",
}

DEFAULT_FORMAT_RULE = "Answer must be the final response only."

# ---------------------------------------------------------------------------
# Perspective solving
# ---------------------------------------------------------------------------

PERSPECTIVE_SYSTEM_PROMPT = (
    "You are a financial analyst. Provide a concise answer with a brief rationale."
)

PERSPECTIVE_CODE_SYSTEM_PROMPT = (
    "You are a financial analyst. Return exactly one runnable Python code block "
    "that computes the answer. Do not include prose or JSON."
)

PERSPECTIVE_CODE_REQUEST = """\
Perspective: {label}.
Guidance: {guidance}

Problem:
{problem_text}

Instructions:
- Write a single Python code block that computes the final answer.
- Use only the Python standard library. ``math`` is allowed.
- Use only values and assumptions stated in the problem. Do not invent example inputs.
- Print the final answer in the requested format.
- {format_rule}
- Output ONLY the Python code block, nothing else."""

PERSPECTIVE_ANSWER_REQUEST = """\
Perspective: {label}
Guidance: {guidance}

Problem:
{problem_text}
{tool_output_section}
Return JSON only in this format:
{{ "answer": "...", "rationale": "brief rationale" }}
{format_rule}
If Python Tool Output is provided, treat it as the authoritative computation result.
Do not ignore, contradict, or restate a different quantity than the tool output.
If the tool output is a decimal rate but the question asks for a percentage/rate/return,
convert it before giving the final answer.
Keep the rationale short (1-3 sentences)."""

# ---------------------------------------------------------------------------
# Tool-augmented perspective (V3 / V4)
# ---------------------------------------------------------------------------

TOOL_LIBRARIES_DESCRIPTION = """\
You have access to a Python execution environment with these libraries:
- numpy (as np) -- arrays, linear algebra, random
- scipy -- optimization, integration, interpolation, statistics
- sympy -- symbolic math, equation solving, calculus
- math, cmath, fractions, decimal -- standard library math
- numpy_financial -- financial functions (npv, irr, pmt, etc.)
- py_vollib -- option pricing (Black-Scholes, Greeks)
- pypfopt -- portfolio optimization (efficient frontier, HRP)
- empyrical -- performance & risk metrics (Sharpe, max drawdown)
- arch -- volatility modelling (GARCH)
- statsmodels -- statistical models, time series
- datetime -- date manipulation"""

TOOL_PERSPECTIVE_CODE_REQUEST = """\
Perspective: Tool-Assisted
Guidance: {guidance}

Problem:
{problem_text}

{libraries_description}

Instructions:
- Write a single Python code block that computes the final answer.
- Use the libraries above as needed.
- Use only values and assumptions stated in the problem. Do not invent example inputs.
- Print the final answer in the requested format.
- {format_rule}
- Output ONLY the Python code block, nothing else."""

TOOL_INTEGRATED_CODE_REQUEST = """\
Perspective: {label}
Guidance: {guidance}

Problem:
{problem_text}

{libraries_description}

Instructions:
- Write a single Python code block that computes the final answer.
- Use the libraries above as needed for precise computation.
- Use only values and assumptions stated in the problem. Do not invent example inputs.
- Print the final answer in the requested format.
- {format_rule}
- Output ONLY the Python code block, nothing else."""

CODE_BLOCK_RETRY_PROMPT = """\
Perspective: {label}
Guidance: {guidance}

Problem:
{problem_text}

{libraries_description}

Your previous reply did not contain a runnable Python code block.

Previous reply:
{previous_response}

Instructions:
- Return exactly one runnable Python code block.
- Do not include prose, JSON, bullet points, or markdown outside the code block.
- Use only values and assumptions stated in the problem. Do not invent example inputs.
- Print the final answer in the requested format.
- {format_rule}"""

TOOL_SELECTION_SYSTEM_PROMPT = (
    "You are an expert scientific programmer. Decide whether computational "
    "tools are needed and which libraries or modules are most relevant."
)

TOOL_SELECTION_PROMPT = """\
Analyze the following problem and identify the specific library modules that are
most relevant for solving it.

Problem:
{problem_text}

Available toolkit:
{tools_description}

Return JSON only in this format:
{{
  "tool_necessity": true,
  "reasoning": "brief explanation",
  "selected_modules": [
    {{
      "library": "library_name",
      "module": "module_name"
    }}
  ]
}}

Guidelines:
- Set "tool_necessity" to true only if code execution or programmatic
  verification would be useful.
- Keep "selected_modules" empty when tools are not needed.
- Prefer library and module names that match the available toolkit exactly."""

CODE_RETRY_PROMPT = """\
Perspective: {label}
Guidance: {guidance}

Problem:
{problem_text}

{libraries_description}

Your previous code produced an error.

Failed code:
```python
{failed_code}
```

Error:
{error_message}

Instructions:
- Fix the code so it runs in the available environment.
- Keep the solution concise and print only the final answer.
- Output ONLY a single corrected Python code block, nothing else."""

# ---------------------------------------------------------------------------
# Contrast step
# ---------------------------------------------------------------------------

CONTRAST_SYSTEM_PROMPT = (
    "You are a meticulous auditor. Compare multiple solution attempts "
    "and identify only substantive discrepancies."
)

CONTRAST_USER_PROMPT = """\
We have {n_perspectives} independent solutions to the same problem, each from \
a distinct perspective.
Compare them to find discrepancies, then produce a checklist to resolve them.

Problem:
{problem_text}

Solutions:
{formatted_solutions}

Tasks:
1) Compare each pair of solutions. List disagreements in assumptions, \
formulas, intermediate values, or final answers.
2) Synthesize a consolidated checklist of concrete verification items \
needed to resolve discrepancies.

Important:
- Ignore differences in wording, explanation style, or level of detail.
- If two answers are numerically equivalent up to rounding, do not mark a discrepancy.
- If one answer is a decimal rate and another is the percentage form of the same rate,
  do not mark that alone as a discrepancy.

Return JSON only in this format:
{{
  "pairwise_discrepancies": {{
{pairwise_keys}
  }},
  "checklist": ["...", "..."]
}}"""

# ---------------------------------------------------------------------------
# Final adjudication
# ---------------------------------------------------------------------------

ADJUDICATION_SYSTEM_PROMPT = (
    "You are the final adjudicator. Use the checklist to reconcile "
    "discrepancies and produce the best final answer."
)

ADJUDICATION_USER_PROMPT = """\
Problem:
{problem_text}

Solutions:
{formatted_solutions}

Discrepancy Checklist:
{checklist_text}

Use the checklist to reconcile differences and provide the final answer.
Prefer tool-backed computations when they ran successfully.
Do not treat mere rounding, formatting, or decimal-vs-percentage presentation of the
same quantity as a substantive disagreement.
Return JSON only in this format:
{{ "answer": "...", "rationale": "brief rationale" }}
{format_rule}"""

# ---------------------------------------------------------------------------
# Single-agent baseline (V2)
# ---------------------------------------------------------------------------

SINGLE_AGENT_SYSTEM_PROMPT = (
    "You are a financial expert. Solve the given problem accurately. "
    "Follow the specific format for the answer."
)

SINGLE_AGENT_USER_PROMPT = """\
Question: {question}
{choices_section}
Provide your answer in the following JSON format:
{{
  "reasoning": "Your step-by-step reasoning here",
  "answer": "{answer_format}"
}}
IMPORTANT: Return only valid JSON. Do not include markdown formatting \
like ```json ... ```."""

# ---------------------------------------------------------------------------
# Single-agent + tools (V5)
# ---------------------------------------------------------------------------

SINGLE_AGENT_TOOLS_CODE_SYSTEM_PROMPT = (
    "You are a financial analyst with access to a scientific Python toolkit. "
    "Return exactly one runnable Python code block that computes the answer. "
    "Do not include prose or JSON."
)

SINGLE_AGENT_TOOLS_CODE_PROMPT = """\
Problem:
{question}

{libraries_description}

Instructions:
- Write a single Python code block that computes the final answer.
- Use the libraries above as needed for precise computation.
- Use only values and assumptions stated in the problem.
- Print the final answer in the requested format.
- {format_rule}
- Output ONLY the Python code block, nothing else."""

SINGLE_AGENT_TOOLS_ANSWER_PROMPT = """\
Question: {question}
{choices_section}
{tool_output_section}
Using the computation above (if any), provide your final answer.
Provide your answer in the following JSON format:
{{
  "reasoning": "Your step-by-step reasoning here",
  "answer": "{answer_format}"
}}
IMPORTANT: Return only valid JSON. Do not include markdown formatting \
like ```json ... ```."""
