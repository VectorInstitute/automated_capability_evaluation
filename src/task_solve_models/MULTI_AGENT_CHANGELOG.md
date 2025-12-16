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
