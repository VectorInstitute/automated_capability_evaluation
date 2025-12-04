# Multi-Agent System Changelog

This is a log for the modifications of the multi-agent solver.

## [Unreleased]
- **Script Updates**:
    - Updated `run_multi_agent.py` to support specifying different models for different agents via command-line arguments (e.g., `--model-scientist-b`).
    - Updated `run_multi_agent.py` to use the new structured `answer` field from the debate result for evaluation, prioritizing it over the verbose `solution`.
- **Enhanced JSON Output Structure**:
    - Updated `AgentSolution` and `FinalSolution` dataclasses in `messages.py` to include a dedicated `answer` field. This field captures the concise solution (e.g., "A", "True", "42") separate from the verbose `final_answer`.
    - Modified `TASK_SOLVER_ROUND_1_PROMPT`, `TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT`, and `TASK_MODERATOR_CONSENSUS_PROMPT` in `prompts.py` to explicitly instruct agents to provide this `answer` field.
- **Scientist Agent Improvements**:
    - Updated `Scientist` in `scientist.py` to parse and populate the `answer` field from the LLM's JSON response.
- **Moderator Agent Improvements**:
    - Updated `Moderator` in `moderator.py` to extract the `answer` field during consensus checks.
    - Improved `_check_simple_consensus` logic to prioritize exact matches on the `answer` field, improving consensus reliability for MCQ and boolean tasks.
    - Updated `_save_final_solution` to include the `answer` field in the final JSON output for easier downstream evaluation.
