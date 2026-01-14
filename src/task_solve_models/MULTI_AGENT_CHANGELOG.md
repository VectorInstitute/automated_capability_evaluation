# Multi-Agent System Changelog

## 2025-12-04: Initial Modularization

### 1. Codebase Refactoring
*   **Created `src/task_solve_models/solver.py`**: Encapsulated single-agent solving logic.
*   **Created `src/task_solve_models/evaluator.py`**: Centralized evaluation logic with robust handling for boolean, MCQ, and numerical answers.
*   **Created `src/task_solve_models/run_single_agent.py`**: Script for running baseline single-agent evaluations.
*   **Created `src/task_solve_models/run_multi_agent.py`**: Script for orchestrating the multi-agent debate system.
*   **Directory Structure**: Moved dataset to `src/task_solve_models/dataset/XFinBench` and results to `src/task_solve_models/Results`.
*   **Problem Solved**: Separated concerns to allow independent development and testing of single-vs-multi agent systems.

## 2025-12-04: Multi-Model Support & Answer Field

### 1. Dynamic Model Configuration
*   **Updated `run_multi_agent.py`**: Added support for specifying different models for each agent via command-line arguments.
*   **Problem Solved**: Enabled heterogeneous agent setups (e.g., mixing GPT-4o and Gemini) to leverage different model strengths.

### 2. Structured "Answer" Field
*   **Updated `messages.py`**: Added `answer` and `numerical_answer` fields to `AgentSolution` and `FinalSolution`.
*   **Updated `prompts.py`**: Requested a concise `answer` field separate from verbose reasoning.
*   **Updated `scientist.py` & `moderator.py`**: Implemented logic to parse and prioritize these new fields.
*   **Problem Solved**: Improved evaluation accuracy by separating the "verdict" (e.g., "A", "True") from the "explanation," reducing false negatives where the answer was buried in text.

### 3. JSON Parsing Robustness
*   **Updated `scientist.py`**: Implemented regex-based repair for common JSON errors.
*   **Problem Solved**: Addressed initial crashes caused by models (like Claude 3 Haiku) generating malformed JSON with missing commas.

## 2025-12-04: Robustness and Format Improvements

### 1. Structured Output Enforcement (JSON Mode)
*   **Modified `scientist.py` & `moderator.py`**: Updated model calls to use `extra_create_args={"response_format": {"type": "json_object"}}`.
*   **Problem Solved**: Eliminated the majority of "JSON Parse Error" crashes by forcing compatible models (GPT-4o, Gemini) to output valid JSON syntax.

### 2. Error Handling & Format Fixes
*   **JSON Repair**: Kept as a fallback mechanism.
*   **Field Parsing**: Prioritized strict `answer` field parsing.
*   **Problem Solved**: Further reduced technical failures and ensured evaluation focused on the intended answer field.

## 2025-12-04: Prompt Improvements for Coordination & Accuracy

### 1. Moderator "Judge" Logic (Prompt Only)
*   **Updated `TASK_MODERATOR_SYSTEM_MESSAGE`**: Instructed Moderator to act as a **JUDGE** and break ties based on reasoning quality.
*   **Problem Solved**: Resolved infinite loop / stalemate scenarios (e.g., `vali_34`) where agents would politely swap sides or refuse to agree, ensuring a decision is reached.

### 2. \"Two-Step\" Reasoning & \"Stickiness\" (Prompt Only)
*   **Updated `TASK_SOLVER_SYSTEM_MESSAGE`**: Enforced \"Think first, then answer\" and added \"Stickiness\" instructions.
*   **Problem Solved**: Aimed to improve calculation accuracy and prevent \"polite flip-flopping\" where agents abandoned correct answers too easily.

## 2025-12-04: \"Formatter Agent\" Architecture (Robust 2-Step Generation)

### 1. Two-Step Generation Pipeline
*   **Modified `scientist.py`**: Refactored `_generate_solution_payload` to use two steps: (1) Free-text reasoning, (2) Strict JSON formatting.
*   **Problem Solved**: Permanently fixed persistent JSON crashes on complex math problems (e.g., `vali_40`, `vali_44`). Models often failed to generate valid JSON when the reasoning contained complex LaTeX or unescaped characters; this architecture separates the \"thinking\" from the \"formatting.\"

### 2. Prompt Architecture Update
*   **Updated `prompts.py`**: Simplified solver prompts and added dedicated formatter prompts.
*   **Problem Solved**: Reduced cognitive load on the solver model, allowing it to focus on the math/logic without being constrained by JSON syntax rules during the reasoning phase.

## 2025-12-05: Robustness, Precision, and Formatting Fixes

### 1. Unit & Precision Handling
*   **Updated `prompts.py`**: Added explicit instructions to `TASK_SOLVER_FORMATTER_PROMPT` to check for requested units (e.g., %, decimal).
*   **Updated `prompts.py`**: Added instruction to `TASK_SOLVER_SYSTEM_MESSAGE` to avoid intermediate rounding.
*   **Problem Solved**: Addressed failures where models returned decimals instead of percentages (`0.08` vs `8.0`) and reduced calculation errors due to premature rounding.

### 2. System Stability Fixes
*   **Updated `scientist.py`**: Implemented a retry loop (3 attempts) for the two-step generation process.
*   **Updated `moderator.py`**: Fixed a crash (`KeyError: slice`) caused by the moderator model occasionally returning nested JSON objects instead of strings for the solution field.
*   **Problem Solved**: Significantly improved pipeline reliability by handling transient model failures (empty responses, malformed JSON) and ensuring the moderator can always process the final output.

## 2025-12-05: Cost Optimization & Round Limiting

### 1. Default Max Rounds Reduced
*   **Updated `run_multi_agent.py`**: Changed default `max_rounds` from 3 to 1.
*   **Problem Solved**: Reduced unnecessary API costs and runtime for tasks where consensus is often reached early or further debate yields diminishing returns.

### 2. Adaptive Scientist Calls (Round 0 Optimization)
*   **Updated `scientist.py`**: Modified `_generate_solution_payload` to use a single-step (1 API call) generation for Round 0, while maintaining the robust two-step (2 API calls) process for subsequent rounds.
*   **Problem Solved**: Cut the cost of the initial "consensus check" phase in half (from 4 calls to 2 calls per task) without sacrificing the robustness needed for complex revisions in later rounds.

## 2025-12-05: Moderator Prompt Improvements

### 1. Urgency & Round Awareness
*   **Updated `prompts.py` & `moderator.py`**: Injected current/max round info into the moderator prompt.
*   **Problem Solved**: Forces the moderator to act decisively in the final round (Judge mode) rather than requesting another round, preventing stalemates.

### 2. Unit Verification ("Unit Police")
*   **Updated `prompts.py`**: Added explicit instruction for the Moderator to verify that the `numerical_answer` matches the requested unit (e.g., percent vs decimal).
*   **Problem Solved**: Reduces "technically correct but wrong format" errors.

### 3. Precise Consensus Definition
*   **Updated `prompts.py`**: Defined numerical consensus as matching within "1% relative difference" to handle minor floating-point variations.
*   **Problem Solved**: Prevents unnecessary debate rounds over negligible differences.

## 2025-12-05: Reversion of Cost Optimizations

### 1. Max Rounds Increase
*   **Updated `run_multi_agent.py`**: Increased `max_rounds` from 1 back to 2.
*   **Reasoning**: Preliminary tests showed `max_rounds=1` was too aggressive and hurt performance by cutting off debates prematurely.

### 2. Re-enable Two-Step Generation for Round 0
*   **Updated `scientist.py`**: Reverted the single-step optimization. Now uses the robust two-step generation (Reasoning + Formatter) for ALL rounds, including Round 0.
*   **Reasoning**: Single-step generation in Round 0 often led to lower quality reasoning or formatting errors, undermining the initial consensus check.

## 2025-12-05: Fault Tolerance Improvements

### 1. Scientist Retry Logic with Backoff
*   **Updated `scientist.py`**: Added a small exponential backoff (sleep) to the retry loop for API calls.
*   **Reasoning**: Helps resolve transient "Empty response" or rate limit errors from the LLM provider (e.g., Gemini) by giving the system a moment to recover before retrying.

### 2. Moderator "Error Filtering"
*   **Updated `moderator.py`**: Modified `_check_simple_consensus` to ignore solutions that contain "ERROR" or "Failed to generate".
*   **Reasoning**: Prevents a single agent failure (e.g., Agent B crashing) from dragging down the entire task. If Agent A provides a valid answer and Agent B fails, the Moderator now proceeds with Agent A's answer instead of failing the task.

## 2025-12-05: Task-Specific Prompting & Enforcement

### 1. Task Type Propagation
*   **Updated `messages.py`**: Added `task_type` field to all message classes (`Task`, `AgentSolution`, etc.).
*   **Updated `run_multi_agent.py`**: Now extracts `task` type from the JSON dataset and passes it into the system.
*   **Problem Solved**: Enables agents to know if they are solving a Boolean, MCQ, or Calculation task.

### 2. Dynamic Scientist Instructions
*   **Updated `scientist.py`**: Injects specific strict formatting instructions into the prompt based on `task_type`.
    *   *Boolean*: "MUST be exactly 'True' or 'False'."
    *   *MCQ*: "MUST be exactly one uppercase letter."
*   **Problem Solved**: Reduces "Format Mismatch" errors where agents output text explanations instead of the required format (e.g., `vali_120`, `vali_131`).

### 3. Moderator Format Enforcement
*   **Updated `moderator.py`**: Added `_enforce_format` method that runs before saving the final answer. It heuristically cleans up answers (e.g., mapping "Yes" -> "True", "Choice A" -> "A") based on the `task_type`.
*   **Problem Solved**: Acts as a final safety net to ensure the output format matches the evaluation expectations, fixing cases where the LLM is correct in reasoning but slightly off in format.

## 2025-12-08: Batch 9 Results & Critical Fixes

### 1. Batch 9 Performance
*   **Result**: The Multi-Agent system achieved **75% accuracy**, significantly outperforming the Single Agent (GPT-4o) baseline of **65%**.
*   **Analysis**: The improvements in "Task Type Enforcement" were highly effective, ensuring Boolean tasks returned strict "True/False" outputs and MCQ tasks returned single letters. The Moderator correctly acted as a "Judge" in tie-breaker scenarios (e.g., `vali_160`), overruling incorrect agents based on superior reasoning (Poole's Analysis).

### 2. Moderator Stability Fix (`AttributeError`)
*   **Updated `moderator.py`**: Patched the `_enforce_format` method to safely cast inputs to strings before processing.
*   **Problem Solved**: Prevented system crashes (e.g., `vali_175`) caused by the LLM returning raw integers or booleans in the JSON response, which the previous code failed to handle.

## 2025-12-08: Local LLM Support (Ollama Integration)

### 1. Native Ollama Support
*   **Updated `run_multi_agent.py` & `run_single_agent.py`**: Implemented automatic detection for local model names (`llama`, `mistral`, `qwen`, `phi`, `deepseek`).
*   **Feature**: When these model names are used (e.g., `--model llama3`), the system bypasses the remote API client and connects directly to the local Ollama server at `localhost:11434`.
*   **Benefit**: Enables completely offline, private, and free execution of the evaluation pipeline using open-source models, without requiring changes to the core utility libraries.

### 2. Prompt Improvement: Moderator Format Alignment
*   **Updated `TASK_MODERATOR_SYSTEM_MESSAGE`**: Added explicit instruction for the Moderator to cross-check the final answer against the original question's requested format and units.
*   **Problem Solved**: Improves accuracy by catching instances where agents calculate the correct value but output it in the wrong format (e.g., "Yes" instead of "True", or incorrect unit scaling), ensuring the final output matches evaluation expectations.

## 2025-12-15: System Stability & Robustness

### 1. Moderator Retry Logic
*   **Updated `moderator.py`**: Implemented a retry loop with exponential backoff for the Moderator's consensus check (similar to the Scientist's logic).
*   **Problem Solved**: Fixed the "Empty JSON content" crashes (e.g., `vali_197`) caused by transient API failures or empty responses from the Moderator model. This ensures the debate pipeline is resilient to occasional model hiccups.

## 2025-12-15: Code Execution Tool ("Python Calculator")

### 1. Tool Implementation
*   **Created `src/utils/tools.py`**: Implemented a safe `python_calculator` function that executes code in a sandbox, capturing standard output and error messages.
*   **Updated `scientist.py`**: Upgraded the agent's reasoning process to use a "ReAct" loop. The agent can now write Python code blocks (e.g. ` ```python ... ``` `), which the system automatically executes, feeding the output back to the agent for further reasoning.
*   **Updated `prompts.py`**: Added explicit instructions to `TASK_SOLVER_SYSTEM_MESSAGE` on how to use the Python calculator.
*   **Problem Solved**: Directly addresses "Calculation Hallucinations" by giving agents a reliable calculator. Instead of guessing arithmetic or complex formulas, agents can now run code to verify their answers before finalizing them.

## 2025-12-15: Map-Reduce / Coordinator-Worker Architecture

### 1. Strategic Planning ("Chain of Thought")
*   **Updated `moderator.py`**: Implemented a planning step where the Moderator first analyzes the problem and generates a step-by-step guidance plan (`moderator_guidance`) before engaging any Scientists.
*   **Updated `prompts.py`**: Added `TASK_MODERATOR_BREAKDOWN_PROMPT` for the planning phase and updated scientist prompts to include the Moderator's guidance.
*   **Problem Solved**: Replaces "blind" parallel solving with a coordinated strategy. The Moderator acts as an Architect, breaking down complex problems (e.g., "Step 1: Convert units", "Step 2: Use Black-Scholes") so Scientists have a clear roadmap.

### 2. Single-Round Efficiency
*   **Updated `run_multi_agent.py`**: Changed default `max_rounds` to 1.
*   **Architecture Shift**: Moved from a multi-round debate loop to a single-pass "Plan -> Execute -> Synthesize" flow.
*   **Benefit**: Significant speedup and cost reduction. By leveraging the Moderator's intelligence upfront to guide the Scientists, we avoid the need for multiple rounds of correction and debate.

## 2025-12-16: Batch 13 Reliability Fixes (Prompt + Tooling)

### 1. Remove Scientist JSON-Output Conflict
*   **Updated `prompts.py`**: Removed the legacy instruction that forced Scientists to output valid JSON.
*   **Problem Solved**: Prevents GPT-style models from wrapping solutions in JSON blocks, which previously caused regex extraction failures (e.g., "No answer found") in the single-call Scientist flow.

### 2. Stabilize Python Calculator Tool
*   **Updated `src/utils/tools.py`**: Increased `python_calculator` timeout from 5s to 10s and added `scipy.stats.norm` to the default execution globals.
*   **Problem Solved**: Reduces intermittent tool timeouts and improves support for common finance/statistics calculations (e.g., normal CDF usage in option pricing).

## 2025-12-18: Add Third Scientist (Local Llama via Ollama)

### 1. Three-Scientist Multi-Agent Runner
*   **Updated `run_multi_agent.py`**: Added `--model-scientist-c` and registered `TaskSolverScientistC` (scientist_id="C").
*   **Updated `run_multi_agent.py`**: Increased `num_solvers` from 2 to 3 so the Moderator waits for all three solutions each round.
*   **Benefit**: Adds diversity and reduces tie/judge-overreach cases by enabling majority dynamics (A/B/C) when models disagree.

## 2025-12-18: Moderator Extraction & Consensus Robustness

### 1. Robust Answer Extraction
*   **Updated `moderator.py`**: Enhanced `_extract_consensus_components` to handle cases where the model nests the final answer/numerical answer inside the `final_solution` field (as a JSON string).
*   **Problem Solved**: Fixed "NONE" output failures (e.g., `vali_292`) where agents were correct but the Moderator failed to extract the value from its own synthesis reasoning.

### 2. Improved Simple Consensus Logic
*   **Updated `moderator.py`**: Added `clean_answer` and `clean_num` helper functions to `_check_simple_consensus`.
*   **Feature**: Standardizes strings by removing trailing punctuation (periods, stars) and stripping unit symbols/formatting before comparison.
*   **Benefit**: Increases the likelihood of early consensus by ignoring negligible formatting differences (e.g., "7.04" vs "7.04.") that previously forced an LLM judge call.

## 2025-12-18: Execution Time Tracking

### 1. Performance Metrics
*   **Updated `run_single_agent.py` & `run_multi_agent.py`**: Added execution time tracking using `time.time()` to measure total runtime.
*   **Feature**: Records both human-readable format (e.g., "5m 42s") and precise seconds (e.g., 342.18) in the metrics section of results JSON.
*   **Benefit**: Enables performance comparison across different configurations (local vs cloud models, single vs multi-agent, different scientist counts) and helps identify bottlenecks.

## 2025-12-18: Evaluation Batch Creation

### 1. Comprehensive Test Set
*   **Created `create_evaluation_batch.py`**: Script to generate a diverse 50-question evaluation set from the full validation set.
*   **Created `evaluation_batch.json`**: Contains 50 questions (19 calcu, 8 mcq, 23 bool) representing all question types including LaTeX tables, figures, and text.
*   **Selection Criteria**: Stratified sampling across 15 financial capabilities (TU, TR, NM, FF, SP, and combinations) to ensure comprehensive coverage.
*   **Benefit**: Provides a standardized benchmark for comparing different model configurations and system architectures without cherry-picking easy or hard questions.

## 2025-12-18: Robust Extraction Logic & Benchmark Results

### 1. Robust Single-Agent Extraction
*   **Updated `solver.py`**: Completely overhauled the extraction process for single-agent runs. It now uses a 3-step hierarchy:
    1.  **Strict JSON block parsing** (with fixes for unescaped newlines).
    2.  **Regex key-value extraction** (recovers `"answer"` from malformed/truncated JSON).
    3.  **Raw fallback** (treats entire response as the answer if no keys are found).
*   **Benefit**: Massive accuracy recovery. **Claude 3.5 Haiku jumped +20%** and **Gemini 3 Pro Preview jumped +16%** in accuracy simply by correctly parsing their reasoning chains.

### 2. Standardized Leaderboard (50 Question Evaluation Batch)
*   **Execution**: Completed a full benchmark of 10 models (top-tier cloud + local Ollama models) on the new `evaluation_batch.json`.
*   **Results**:
    *   **Champion**: `gemini-2.5-pro` (**84%**) - excels at detailed step-by-step reasoning.
    *   **Challenger**: `gemini-3-pro-preview` (**78%**) - demonstrates strong reasoning but is significantly slower.
    *   **Speed Demon**: `gpt-5.1` (**68%**) - fastest top-tier model (~3 minutes for 50 questions).
    *   **Local King**: `qwen2.5 (7B)` (**48%**) - outperforming some cloud models in logic tasks.
*   **Benefit**: Establishes a definitive baseline for the project, showing that while size matters, extraction logic and reasoning depth are equally critical for financial accuracy.

## 2025-12-18: Multi-Agent Benchmark (r0 vs r1)

### 1. High-Performance Configuration
*   **Scientists**: `gemini-2.5-pro`, `claude-opus-4-5-20251101`, `gpt-5.1`
*   **Moderator**: `gemini-3-pro-preview`
*   **Run Details**: Initiated two benchmark runs on the 50-question `evaluation_batch.json` to compare performance between:
    *   **Max Rounds = 0**: "Smart Voting" mode (single-pass judgment).
    *   **Max Rounds = 1**: "Refinement" mode (two-pass debate).
*   **Benefit**: This will establish if the "Smart Voting" logic is sufficient for high-tier models or if the extra debate round significantly boosts accuracy on complex financial tasks.

### 2. Prompt Reversion (Mandatory Tool Use)
*   **Action**: Reverted the `TASK_SOLVER_SYSTEM_MESSAGE` to the **Optional Tool Use** version.
*   **Reasoning**: Mandatory tool usage caused a performance drop (**84% ➡️ 82%** in R1 and **82% ➡️ 76%** in R0). Models became conceptually "lazy," focusing on the code output for sub-steps but failing on higher-level financial logic (e.g., annualization of returns).
*   **Insight**: High-tier models like Gemini 2.5 Pro are more effective when allowed to choose their own "reasoning mode" (mental math vs. Python) based on the task complexity.

## 2025-12-20: Strategic Architect & Auditor Workflow (v3)

### 1. Strategic Protocol Shift
*   **Updated `prompts.py`**: Implemented a new "Architecture & Audit" interaction loop:
    1.  **Moderator (Architect)**: Now explicitly proposes a "Hypothesis Strategy" and identifies potential technical pitfalls before solvers begin.
    2.  **Scientist (Auditor)**: Now mandated to start with a "Strategy Weighting" (1-5 confidence) and identify logic gaps in the Moderator's proposal before executing their solution.
*   **Benefit**: Massive logic recovery. The system now catches "Annualization" traps and "Textbook Assumptions" by forcing a critical review phase before calculation.

### 2. New Benchmark Record (86%)
*   **Result**: The **Smart Voting (R0)** configuration achieved **86% accuracy** (43/50), surpassing the best single-agent and multi-agent debate (R1) scores.
*   **Key Insight**: For ultra-high-tier models (Gemini 2.5 Pro + Claude 4.5 Opus), the first refined thought—informed by a strategic audit—is more accurate than a multi-round debate. R1 (84%) actually showed slight "polite consensus" drift compared to the more decisive R0.

## 2025-12-25: ReCAP Solver (Plan → Step → Refine)

### 1. New ReCAP Implementation (Controller/Worker)
*   **Created `src/task_solve_models/multi_agent_solver/RECAP/`**: Added a ReCAP-style implementation with:
    *   `controller.py`: Controller that owns a compact Canonical State and runs a plan → dispatch step → refine loop.
    *   `worker.py`: Step worker that executes exactly one step and returns a structured `StepResult`.
    *   `messages.py`: Canonical State + step request/result schemas.
    *   `prompts.py`: Separate prompt file for planner/refiner/finalizer + step worker prompts.
*   **Benefit**: Enables bounded-context, step-wise solving (structured state re-injection) without requiring 3 parallel “full-solution” scientists.

### 2. New Runner Script
*   **Created `src/task_solve_models/run_recap.py`**: A new runner that mirrors `run_multi_agent.py` output shape (`metrics` + per-task `results`) while executing the ReCAP loop and saving per-task step traces under `Results/recap_outputs/`.

### 3. Output Format Hardening (Tool + JSON)
*   **Updated `RECAP/prompts.py` & `RECAP/worker.py`**: Hardened the worker loop so that even when python tool blocks are used, the final output returns **JSON only** (no markdown fences), using an explicit “return JSON now” instruction and a forced JSON response_format call as a fallback.
*   **Updated `RECAP/controller.py`**: Added retry logic to ensure controller plan/refine/finalize calls always yield parseable JSON dicts.
*   **Problem Solved**: Prevents step workers from getting stuck returning only code blocks (which previously caused JSON parse failures and incorrect evaluations).

### 4. Initial ReCAP Results (68% Accuracy)
*   **Result**: ReCAP with GPT-5.1 (controller + worker) achieved **68% accuracy** (34/50) on evaluation batch.
*   **Key Finding**: **0/16 incorrect answers used Python tool** - all failures relied on mental/approximate calculations.
*   **Error Analysis**: 14 calculation errors (87.5%), 2 boolean errors (12.5%). Large percentage differences (52.3%, 90.1%) in some cases indicating formula/calculation errors rather than rounding.

## 2025-12-25: ReCAP Python Tool Enforcement (v2)

### 1. Mandatory Python Tool for Calculations
*   **Updated `RECAP_STEP_PROMPT`**: Added explicit rule: "If this step requires ANY numerical calculation, you MUST use the Python tool. Do NOT perform calculations mentally or with approximations."
*   **Updated `RECAP_PLANNER_PROMPT`**: Instructed planner to explicitly mark calculation steps as requiring Python tool usage.
*   **Updated `RECAP_FINALIZER_PROMPT`**: Added verification step to recompute final numerical answers using Python when possible.
*   **Rationale**: Analysis showed 0% Python tool usage in all 16 incorrect answers, with mental math leading to calculation errors (e.g., bond pricing 52.3% off, OCF missing working capital changes).

### 2. Expected Impact
*   **Target**: Reduce calculation errors by 50-70%, improving accuracy from 68% to 80-85%.
*   **Focus**: Address the root cause (no tool usage) rather than just formula errors.

### 3. Prompt Strengthening (v2.1)
*   **Updated `RECAP_WORKER_SYSTEM`**: Changed from "You may use python as tool if you want" to "You MUST use the Python tool for any step that involves numerical calculations."
*   **Rationale**: The optional language ("may use if you want") was undermining the mandatory requirement in `RECAP_STEP_PROMPT`. Making it mandatory in the system message ensures consistency and stronger enforcement.
*   **Result**: Workers are now explicitly told Python is mandatory for calculations at the system level, not just in the step prompt.

### 4. Performance Degradation (v3 - 58% Accuracy)
*   **Result**: ReCAP v3 achieved **58% accuracy** (29/50), down from 68% in v1.
*   **Key Finding**: Python tool usage improved (9/21 failures used Python vs 3/17 in v2), but new errors appeared:
    *   More boolean/MCQ errors (4 boolean, 1 MCQ vs 1 boolean in v2) - possibly from confusion about Python requirement
    *   Workers still ignoring "MUST use Python" in step descriptions (vali_143 S7, vali_294 S6)
    *   Formula errors persist even when Python is used correctly (vali_143: missing working capital in OCF)
*   **Conclusion**: Prompt-only enforcement is insufficient. System-level validation needed.

## 2025-12-25: ReCAP System-Level Validation (v4)

### 1. Controller-Level Python Validation (#1)
*   **Updated `controller.py`**: Added validation in `handle_step_result` that:
    *   Detects if step description requires Python (checks for "must use python", "python tool", or "calculation" + "must")
    *   Filters candidates to only those that used Python when required
    *   Re-dispatches step with `enforce_python=True` flag if no candidates used Python
*   **Updated `RecapStepRequest`**: Added `enforce_python` field to pass enforcement flag to workers.
*   **Updated `worker.py`**: Adds explicit enforcement message to prompt when `enforce_python=True`.
*   **Benefit**: System-level enforcement ensures Python is actually used, not just requested.

### 2. Explicit Non-Calculation Instructions (#5)
*   **Updated `RECAP_WORKER_SYSTEM`**: Clarified that Python is mandatory ONLY for numerical calculations, and NOT needed for boolean/MCQ/conceptual steps.
*   **Updated `RECAP_STEP_PROMPT`**: Added "STEP TYPE DETECTION" section that explicitly distinguishes calculation vs non-calculation steps.
*   **Benefit**: Reduces confusion that may have caused boolean/MCQ errors in v3.

### 3. Step Result Validation (#6)
*   **Updated `controller.py`**: Added validation checks before accepting step results:
    *   Validates result is not empty
    *   Checks if expected numeric output actually contains a number
    *   Logs warnings for suspicious results
*   **Benefit**: Catches format errors and unreasonable results early, preventing propagation.

### 4. Expected Impact
*   **Target**: Improve accuracy from 58% to 70-75% by enforcing Python usage and catching errors early.
*   **Focus**: System-level enforcement + validation, not just prompt guidance.

## 2025-12-25: ReCAP Proper Refiner-Based Plan Updates (v5)

### 1. Refiner Handles Plan Updates (Proper ReCAP Method)
*   **Updated `controller.py`**: Removed controller-level step insertion and re-dispatch logic. Controller now only validates Python usage and passes warnings to refiner.
*   **Updated `RECAP_REFINER_PROMPT`**: Added explicit instructions for refiner to update plan when Python is required but not used. Refiner now adds explicit calculation steps to the plan.
*   **Removed**: `_step_retry_count` tracking and `enforce_python` re-dispatch mechanism from controller.
*   **Benefit**: Follows proper ReCAP architecture where the refiner refines the plan based on step results, not the controller inserting steps directly.

### 2. How It Works
*   **Controller**: Detects if step requires Python but no candidates used it → logs warning and passes note to refiner.
*   **Refiner**: Sees warning → updates plan to add explicit calculation step that breaks down the calculation and requires Python.
*   **Execution**: Controller continues with updated plan, executing the new explicit step normally.
*   **Benefit**: Plan-based recovery that follows ReCAP's design pattern of plan refinement, not controller intervention.

### 3. Expected Impact
*   **Target**: Improve accuracy by ensuring Python is used through proper plan refinement.
*   **Focus**: Refiner-based plan updates instead of controller re-dispatch, aligning with ReCAP methodology.

## 2025-12-25: ReCAP Single-Worker Refiner Fix (v6)

### 1. Fixed Single-Worker Refiner Bypass
*   **Updated `controller.py`**: Removed the single-worker bypass that prevented refiner from being called. Now refiner is **always called** regardless of worker count (1, 2, or 3).
*   **Problem Solved**: With 1 worker, the refiner was never called, so Python validation warnings were ignored and plan updates never happened. This caused v5 accuracy to drop to 62%.
*   **Benefit**: Consistent behavior across all worker counts. Plan refinement and Python validation now work with 1 worker.

### 2. How It Works Now
*   **Before**: `if len(self._worker_ids) == 1:` → bypass refiner, use candidate directly
*   **After**: Always call refiner → refiner sees Python validation warnings → refiner updates plan if needed
*   **Result**: Single-worker runs now benefit from plan refinement, just like multi-worker runs.

### 3. Expected Impact
*   **Target**: Restore accuracy to v4 levels (66%) or better by enabling plan updates with 1 worker.
*   **Testing**: Running parallel tests with 1 worker and 2 workers to compare performance.

## 2025-12-25: ReCAP Infinite Recursion Fix + Explicit Step Enforcement (v8)

### 1. Fixed Infinite Recursion Bug
*   **Updated `controller.py`**: Added check to prevent adding explicit calculation steps when the current step is already an explicit step (contains `_calc` in step_id). This prevents infinite recursion where explicit steps keep generating more explicit steps.
*   **Updated `RECAP_REFINER_PROMPT`**: Added instruction to check if current step is already explicit before adding another explicit step.
*   **Problem Solved**: System was creating recursive explicit steps (S4_calc → S4_calc_calc → S4_calc_calc_calc...) wasting time and steps.
*   **Benefit**: Prevents infinite loops and allows system to proceed even when explicit steps fail.

### 2. Strengthened Worker Prompt for Explicit Steps (#5)
*   **Updated `worker.py`**: Added special enforcement message when step_id contains `_calc`. Workers now receive explicit instruction that this is an explicit calculation step and MUST use Python immediately.
*   **Message**: "CRITICAL: This is an EXPLICIT CALCULATION STEP. You were given this step because Python was required but not used in a previous step. You MUST use Python tool NOW. Write Python code immediately. No exceptions."
*   **Benefit**: Increases likelihood that workers will actually use Python in explicit calculation steps.

### 3. Expected Impact
*   **Target**: Improve accuracy from 60-64% to 70-75% by preventing recursion and enforcing Python usage in explicit steps.
*   **Focus**: Fix infinite recursion + strengthen explicit step enforcement.

## 2025-12-25: ReCAP Simplified Refiner + Decomposed Calculations (v9)

### 1. Simplified Refiner - Two-Phase Approach (#3)
*   **Updated `RECAP_REFINER_PROMPT`**: Refiner now focuses ONLY on candidate selection (Phase 1). Removed all plan update logic from refiner.
*   **Updated `controller.py`**: Controller now handles plan updates separately (Phase 2). Explicit step generation is now purely controller-driven, not refiner-driven.
*   **Problem Solved**: Refiner was doing too much (choosing candidates, updating plans, adding explicit steps), causing confusion and errors. The complexity made it hard to debug issues.
*   **Benefit**: Clearer separation of concerns. Refiner = candidate selection. Controller = plan management. Easier to debug and maintain.

### 2. Break Down Complex Calculations (#5)
*   **Updated `RECAP_PLANNER_PROMPT`**: Added explicit instruction to break down complex calculations into smaller, independently verifiable steps.
*   **Example**: Instead of "Calculate Operating Cash Flow", planner should create:
    * "Calculate EBIT" - MUST use Python tool
    * "Calculate taxes on EBIT" - MUST use Python tool
    * "Calculate change in net working capital" - MUST use Python tool (if applicable)
    * "Calculate Operating Cash Flow from components" - MUST use Python tool
*   **Problem Solved**: Large calculation steps made errors hard to catch. If one part of a complex calculation was wrong, the entire step failed without visibility into which sub-calculation was wrong.
*   **Benefit**: Each sub-step is independently verifiable. Errors are easier to catch and debug. Workers can focus on one calculation at a time.

### 3. Expected Impact
*   **Target**: Improve accuracy from 60% to 70-75% by simplifying refiner logic and making calculations more granular and debuggable.
*   **Focus**: Architectural simplification + calculation decomposition.

