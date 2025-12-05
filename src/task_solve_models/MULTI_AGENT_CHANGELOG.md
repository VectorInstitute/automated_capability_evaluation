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

### 2. "Two-Step" Reasoning & "Stickiness" (Prompt Only)
*   **Updated `TASK_SOLVER_SYSTEM_MESSAGE`**: Enforced "Think first, then answer" and added "Stickiness" instructions.
*   **Problem Solved**: Aimed to improve calculation accuracy and prevent "polite flip-flopping" where agents abandoned correct answers too easily.

## 2025-12-04: "Formatter Agent" Architecture (Robust 2-Step Generation)

### 1. Two-Step Generation Pipeline
*   **Modified `scientist.py`**: Refactored `_generate_solution_payload` to use two steps: (1) Free-text reasoning, (2) Strict JSON formatting.
*   **Problem Solved**: Permanently fixed persistent JSON crashes on complex math problems (e.g., `vali_40`, `vali_44`). Models often failed to generate valid JSON when the reasoning contained complex LaTeX or unescaped characters; this architecture separates the "thinking" from the "formatting."

### 2. Prompt Architecture Update
*   **Updated `prompts.py`**: Simplified solver prompts and added dedicated formatter prompts.
*   **Problem Solved**: Reduced cognitive load on the solver model, allowing it to focus on the math/logic without being constrained by JSON syntax rules during the reasoning phase.
