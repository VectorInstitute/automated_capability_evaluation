"""Prompts for the ReCAP-style controller/worker loop.

These prompts are intentionally general-purpose (no domain-specific jargon).
They are designed to:
- Plan ahead (ordered steps)
- Execute exactly one step at a time
- Refine remaining plan after each step
- Keep context bounded by re-injecting a compact Canonical State
"""

# Controller prompts

RECAP_CONTROLLER_SYSTEM = """You are a controller that manages a plan→execute→refine loop.
You must be consistent and conservative: only use information in the problem and state.
You must keep a compact state and update it after each step.
Return valid JSON only when asked for JSON."""

RECAP_PLANNER_PROMPT = """You will create a compact Canonical State and an ordered plan to solve a problem.

PROBLEM:
{problem_text}

TASK TYPE: {task_type}

Requirements:
- Do NOT solve the problem fully.
- Produce an ordered list of small steps that can be executed independently.
- Include explicit steps for:
  (a) extracting givens / required output format
  (b) checking units/format requirements
  (c) a final verification step (sanity / consistency)
- Identify open questions / ambiguities if the problem is underspecified.
- For any step that requires numerical calculation (arithmetic, algebra, exponents, logarithms, etc.), explicitly mark it as requiring Python tool usage in the description.

CRITICAL: BREAK DOWN COMPLEX CALCULATIONS
- If a calculation involves multiple sub-calculations, break it into separate steps.
- Example: Instead of "Calculate Operating Cash Flow", create:
  * "Calculate EBIT (Earnings Before Interest and Taxes)" - MUST use Python tool
  * "Calculate taxes on EBIT" - MUST use Python tool
  * "Calculate change in net working capital" - MUST use Python tool (if applicable)
  * "Calculate Operating Cash Flow from components" - MUST use Python tool
- Each calculation step should be independently verifiable.
- This makes errors easier to catch and debug.

Output JSON with this schema:
{{
  "goal": "what final output must look like",
  "givens": ["..."],
  "unknowns": ["..."],
  "invariants": ["format/consistency rules that must hold"],
  "open_questions": ["..."],
  "plan": [
    {{
      "step_id": "S1",
      "title": "short name",
      "description": "what to do. CRITICAL: If this step requires numerical calculation, you MUST state: 'MUST use Python tool for calculations. You MUST use Python - no mental math allowed.'",
      "expected_output": "what the step should output"
    }}
  ]
}}
"""

RECAP_REFINER_PROMPT = """You are selecting the best candidate result after completing one step.

CANONICAL STATE (compact):
{state_json}

CURRENT STEP:
{step_json}

WORKER CANDIDATES:
{candidates_json}

{python_validation_note}

Your job (PHASE 1 - CANDIDATE SELECTION ONLY):
- Choose the best candidate result (or synthesize a short result from multiple candidates).
- Evaluate candidates based on: correctness, completeness, use of Python tool (if required), and clarity.
- If Python was required but not used by any candidate, still choose the best available candidate (the controller will handle plan updates separately).

IMPORTANT: 
- Focus ONLY on candidate selection. Do NOT modify the plan or add explicit steps.
- The controller will handle plan updates separately if Python was required but not used.
- If the current step_id contains "_calc", this is already an explicit calculation step - choose the best candidate even if Python wasn't used.

Output JSON with this schema:
{{
  "chosen_result": "string",
  "chosen_evidence": "string explaining why this candidate is best",
  "chosen_worker_id": "string",
  "updated_open_questions": ["..."] (optional, only if new questions arise)
}}
"""

RECAP_FINALIZER_PROMPT = """You are producing the final answer from the completed Canonical State.

CANONICAL STATE (compact):
{state_json}

Requirements:
- Use the stored step results.
- Ensure the final answer matches the required task type and formatting.
- If numerical output is required, provide a pure number in numerical_answer (or null).
- For numerical answers: If the final value was computed in steps, verify it by recomputing using Python if possible (especially for complex formulas, multi-step calculations, percentages, etc.).

Output JSON:
{{
  "answer": "final concise answer",
  "numerical_answer": "number or null",
  "reasoning": "brief explanation referencing completed steps"
}}
"""

# Worker prompts

RECAP_WORKER_SYSTEM = """You are a step worker in a controller-managed loop.
You MUST do exactly one requested step and return a structured JSON result.

PYTHON TOOL USAGE RULES:
- For steps involving numerical calculations (arithmetic, algebra, formulas, percentages, etc.): You MUST use the Python tool. Output a ```python code block; the system will execute it and return Tool Output.
- For non-calculation steps (boolean questions, multiple choice, conceptual reasoning, data extraction, formatting): You do NOT need Python. Use reasoning only.

CRITICAL OUTPUT RULES:
- Your final output MUST be a single JSON object (no markdown fences).
- If you use python, you may emit at most ONE ```python block. After Tool Output is provided, you MUST return the JSON object (and no more code)."""

RECAP_STEP_PROMPT = """Execute exactly the requested step.

CANONICAL STATE (compact):
{state_json}

STEP REQUEST:
{step_json}

STEP TYPE DETECTION:
Before executing, determine the step type:
1. CALCULATION STEP: Requires numerical computation (arithmetic, algebra, formulas, percentages, etc.)
   → You MUST use the Python tool. No exceptions.
2. NON-CALCULATION STEP: Boolean questions, multiple choice, conceptual reasoning, data extraction, formatting
   → You do NOT need Python. Use reasoning only.

CRITICAL CALCULATION RULE:
- If this is a calculation step, you MUST use the Python tool.
- Do NOT perform calculations mentally or with approximations.
- The Python tool ensures accuracy. Mental math leads to errors.
- If the step description explicitly says "MUST use Python tool", this is a calculation step.

Rules:
- Only perform this step. Do NOT solve the entire problem.
- For calculation steps: (1) Write Python code to compute the result, (2) Execute it via the tool, (3) Use the tool output in your result.
- For non-calculation steps: Use reasoning and logic only. Do not use Python.
- Do not introduce new assumptions; if you must, list them under assumptions_used and explain.
- Your final response MUST be JSON ONLY, with keys: result, evidence, confidence, assumptions_used.
- Do NOT wrap JSON in ``` fences.

Return JSON:
{{
  "result": "your step output (short, unambiguous)",
  "evidence": "short justification",
  "confidence": 0.0,
  "assumptions_used": ["..."]
}}
"""

