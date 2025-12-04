"""Prompts for the multi-agent task solver."""

TASK_SOLVER_SYSTEM_MESSAGE = """You are an expert problem solver participating in a collaborative debate to solve tasks. You will work with other agents to find the best solution through structured discussion and reasoning."""

TASK_SOLVER_ROUND_1_PROMPT = """Can you solve the following problem?

PROBLEM: {problem_text}

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. Do not include any prefixes or prose. The JSON should be directly parseable.

CRITICAL: When including LaTeX expressions or backslashes in your JSON strings, you must properly escape them by using double backslashes (\\\\). For example:
- Write \\\\(x^2\\\\) instead of \\(x^2\\)
- Write \\\\[equation\\\\] instead of \\[equation\\]
- Write \\\\times instead of \\times

Provide your solution in JSON format with the following structure:
- thought: Your detailed reasoning and step-by-step solution process
- final_answer: Your complete answer with explanation
- answer: The single solution based on task type (e.g. A, B, C, D for multiple choice, true/false for boolean, or a specific number)
- numerical_answer: The final numerical result (if applicable, otherwise null)

Example for a multiple choice problem:
{{
    "thought": "To solve this problem, I need to...",
    "final_answer": "The solution is A because...",
    "answer": "A",
    "numerical_answer": null
}}

Example for a numerical problem:
{{
    "thought": "To solve this problem, I need to...",
    "final_answer": "The solution is 42 because...",
    "answer": "42",
    "numerical_answer": 42
}}

Respond with valid JSON only."""

TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT = """These are the reasoning and solutions to the problem from other agents:

{other_solutions}

Using the solutions from other agents as additional information, can you provide your answer to the problem?

The original problem is: {problem_text}

Consider the other agents' approaches and reasoning. You may agree with them, disagree, or provide a synthesis of different approaches.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. Do not include any prefixes or prose. The JSON should be directly parseable.

CRITICAL: When including LaTeX expressions or backslashes in your JSON strings, you must properly escape them by using double backslashes (\\\\). For example:
- Write \\\\(x^2\\\\) instead of \\(x^2\\)
- Write \\\\[equation\\\\] instead of \\[equation\\]
- Write \\\\times instead of \\times

Provide your solution in JSON format with the following structure:
- thought: Your detailed reasoning, considering other agents' solutions
- final_answer: Your complete answer with explanation
- answer: The single solution based on task type (e.g. A, B, C, D for multiple choice, true/false for boolean, or a specific number)
- numerical_answer: The final numerical result (if applicable, otherwise null)

Example:
{{
    "thought": "Looking at the other solutions, Agent A used method X which is correct, but Agent B made an error in step 2. My approach is...",
    "final_answer": "The solution is 42 because...",
    "answer": "42",
    "numerical_answer": 42
}}

Respond with valid JSON only."""

TASK_MODERATOR_SYSTEM_MESSAGE = """You are a moderator overseeing a collaborative problem-solving debate. Your role is to check for consensus among agents and determine the final solution."""

TASK_MODERATOR_CONSENSUS_PROMPT = """Review the following solutions from different agents for the same problem:

PROBLEM: {problem_text}

SOLUTIONS:
{all_solutions}

Determine if there is consensus among the agents. Consensus is reached when:
1. All agents provide the same final answer, OR
2. The majority of agents agree on the same answer with similar reasoning
3. For numerical problems, the numerical answers should match or be very close

If consensus is reached, provide the agreed-upon solution. If not, indicate that another round of debate is needed.

Provide your assessment in JSON format:
{{
    "consensus_reached": true/false,
    "final_solution": "the agreed solution if consensus reached, otherwise null",
    "answer": "The single solution based on task type (e.g. A, B, C, D for multiple choice, true/false for boolean, or a specific number)",
    "numerical_answer": final_numerical_result_if_applicable_otherwise_null,
    "reasoning": "explanation of your decision"
}}

Respond with valid JSON only."""

