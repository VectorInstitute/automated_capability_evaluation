"""Prompts for the debate-based agentic area, capability, and task generation."""

# =============================================================================
# AREA GENERATION PROMPTS
# =============================================================================

AREA_SCIENTIST_INITIAL_PROMPT = """You are Scientist {scientist_id}. You are an expert in evaluating large language models (LLMs) in the domain of {domain}. Your task is to independently propose a list of {num_areas} high-level, non-overlapping **capability areas** that collectively cover the space of skills relevant to this domain.

Each area should:
- Represent a broad but distinct dimension of LLM competence.
- Be clearly distinct from the other proposed areas (no overlap).
- Contain enough conceptual room to allow for multiple fine-grained capabilities in the next stage.

For each area, provide:
1. A short name (a few words).
2. A 2–3 sentence description that defines its boundaries and justifies its inclusion.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Please return your proposal and your thoughts and reasoning in the following format:
{{
  "thought": "Your reasoning and thought process here",
  "areas": {{
    "area_0": {{
      "name": "Area Name",
      "description": "Area description"
    }},
    ...
  }}
}}"""

AREA_SCIENTIST_REVISION_PROMPT = """You are Scientist {scientist_id}. You are reviewing the merged set of capability areas proposed by the Moderator.

Moderator's Proposal:
{moderator_proposal}

Please review the proposed areas carefully and suggest any of the following:
- Minor refinements or clarifications to area descriptions.
- Proposed merges/splits where you see overlap or conceptual drift.
- Additions of missing areas or removal of unneeded ones.

Keep your feedback constructive and focused on improving clarity, coverage, and non-overlap. Avoid unnecessary changes.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Return your revised proposal and your thoughts and reasoning with the following format:
{{
  "thought": "Your reasoning and thought process here",
  "areas": {{
    "area_0": {{
      "name": "Area Name",
      "description": "Area description"
    }},
    ...
  }}
}}
"""

AREA_MODERATOR_MERGE_PROMPT = """You are the Moderator. Two scientist agents have independently proposed a list of high-level capability areas for evaluating large language models in the domain of {domain}.

Below are their proposals:

Scientist A Proposal:
{scientist_a_proposal}

Scientist B Proposal:
{scientist_b_proposal}

Your task is to merge their proposals into a unified set of {num_final_areas} areas. In doing so:
- Eliminate overlaps and redundant areas.
- Justify any removals, merges, or renamings.
- Ensure that the final set is mutually exclusive and collectively exhaustive for this domain.

Explain how you merge the above proposals. Be thoughtful and concise in your output.
You will then submit this merged proposal for review by the scientist agents. If either scientist provides substantive suggestions, you may revise the proposal and initiate another round of review.
If you judge the merged set to be clear, comprehensive, and non-overlapping, and scientists are have reached consensus, you may declare the area design finalized.
To finalize, set the "finalized" field in your JSON response to true, otherwise set it to false.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Present the merged areas and your thoughts and reasoning in the following format:
{{
  "thought": "Your reasoning and thought process here",
  "areas": {{
    "area_0": {{
      "name": "Area Name",
      "description": "Area description"
    }},
    ...
  }},
  "finalized": true
}}
"""

# =============================================================================
# CAPABILITY GENERATION PROMPTS
# =============================================================================

CAPABILITY_SCIENTIST_INITIAL_PROMPT = """You are Scientist {scientist_id}. You have been assigned the area: "{area_name}".
Area Description: {area_description}
Your task is to propose {num_capabilities} specific, **non-overlapping capabilities** within this area that test different aspects of LLM performance.

Each capability should:
- Be clearly within the scope of the area.
- Be distinct from the others (no overlap).
- Be testable via concrete tasks in later stages.

Provide each capability with:
1. A concise name (lowercase_with_underscores).
2. A 2–3 sentence description justifying its purpose.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Please return your proposal and your thoughts and reasoning in the following format:

{{
  "thought": "Your reasoning and thought process here",
  "capabilities": {{
    "capability_0": {{
      "name": "capability_name",
      "description": "Capability description",
      "area": "{area_name}"
    }},
    ...
  }}
}}
"""

CAPABILITY_SCIENTIST_REVISION_PROMPT = """You are Scientist {scientist_id}. The Moderator has proposed a merged list of capabilities for the area "{area_name}".

Moderator's Proposal:
{moderator_proposal}

Please review and revise the merged capability list by:
- Clarifying or refining capability descriptions.
- Flagging capabilities that may be overlapping or vague.
- Proposing any additions or deletions if you believe something important is missing or redundant.

Detail the modifications you make to the above proposal and explain your reasoning.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Return the updated list in the following format:

{{
  "thought": "Your reasoning and thought process here",
  "capabilities": {{
    "capability_0": {{
      "name": "capability_name",
      "description": "Capability description",
      "area": "{area_name}"
    }},
    ...
  }}
}}"""

CAPABILITY_MODERATOR_MERGE_PROMPT = """You are the Moderator. Two scientist agents have independently proposed a list of capabilities within the capability area: "{area_name}".

Below are their proposals:

Scientist A Proposal:
{scientist_a_proposal}

Scientist B Proposal:
{scientist_b_proposal}

Your task is to merge these proposals into a unified set of capabilities for the area. In doing so:
- Eliminate redundancy and overlapping capabilities.
- Ensure all capabilities are clearly within the scope of the area.
- Ensure all capabilities are distinct from one another.
- Improve clarity and precision in naming and descriptions, where needed.

You will then submit this merged capability list for review by the scientist agents. If either scientist provides substantive suggestions, you may revise the list and initiate another round of review.
If, after incorporating feedback or upon review, you judge the merged set to be clear, comprehensive, and non-overlapping within the area, you may declare the capability design finalized.
To finalize, set the "finalized" field to true, otherwise set it to false.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Present the merged capabilities in the following format:
{{
  "thought": "Your reasoning and thought process here",
  "capabilities": {{
    "capability_0": {{
      "name": "capability_name",
      "description": "Capability description",
      "area": "{area_name}"
    }},
    ...
  }},
  "finalized": true
}}
"""

# =============================================================================
# TASK GENERATION PROMPTS
# =============================================================================

TASK_SCIENTIST_PROBLEM_SYSTEM_PROMPT = """You are Scientist {scientist_id}, an expert in designing tasks for evaluating a given capability. You will be shown the capability's name, description, domain, and a few sample tasks. Your goal is to propose novel, diverse, and non-trivial task problems that assess different aspects of this capability.

You will be particularly rewarded for:
- Ensuring clear alignment with the capability,
- Avoiding overlap or redundancy,
- Proposing tasks that vary in difficulty and structure.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. The JSON should be directly parseable.

Please return your proposal and your thoughts and reasoning in the following format:
{{
  "thought": "Your reasoning and thought process about the kind of tasks you're proposing",
  "problems": {{
    "problem_0": "TASK_TEXT_1",
    "problem_1": "TASK_TEXT_2",
    ...
  }}
}}

Make sure:
- All tasks are within the scope of the capability.
- Tasks are phrased as standalone problem descriptions, without any answers or solutions.
- LaTeX strings are properly escaped (e.g., \\\\[2x + 3 = 11\\\\]).
- Each task is distinct from the others and covers a different aspect or sub-skill."""

TASK_SCIENTIST_PROBLEM_USER_PROMPT = """Design {num_problems} tasks for the following capability:

Name: {capability_name}
Description: {capability_description}
Domain: {capability_domain}
Sample tasks:
{sample_tasks_text}"""


TASK_MODERATOR_PROBLEM_SYSTEM_PROMPT = """You are the Moderator overseeing capability-based task design. Your task is to review proposed tasks from multiple scientist agents and synthesize a final, high-quality task set for the capability.

Your responsibilities:
- Eliminate any task that is not clearly aligned with the capability.
- Merge or remove tasks that are redundant or overly similar.
- Ensure that the final set of tasks is diverse, non-trivial, and tests different facets of the capability.
- Select only the highest quality tasks that best represent the capability.

IMPORTANT: Return your response as raw JSON only. Do not wrap it in markdown code blocks or add any formatting. Do not include any prefixes or prose. The JSON should be directly parseable.

CRITICAL: When including LaTeX expressions or backslashes in your JSON strings, you must properly escape them by using double backslashes (\\\\). For example:
- Write \\\\(x^2\\\\) instead of \\(x^2\\)
- Write \\\\[equation\\\\] instead of \\[equation\\]
- Write \\\\times instead of \\times

Please return your curation and your thoughts and reasoning in the following format:
{
  "thought": "Your reasoning and curation plan here",
  "final_tasks": {
    "task_1": "<FINAL_TASK_1>",
    "task_2": "<FINAL_TASK_2>",
    ...
  }
}"""

TASK_MODERATOR_PROBLEM_USER_PROMPT = """Below is a capability and task proposals from multiple scientist agents. Curate the final task set by filtering, editing, or merging as needed.

Name: {capability_name}
Description: {capability_description}
Domain: {capability_domain}

Proposed Tasks:
{problems_text}"""

# =============================================================================
# TASK SOLVING DEBATE PROMPTS
# =============================================================================

TASK_SOLVER_SYSTEM_MESSAGE = """You are an expert problem solver participating in a collaborative debate to solve tasks. You will work with other agents to find the best solution through structured discussion and reasoning."""

TASK_SOLVER_ROUND_1_PROMPT = """Can you solve the following problem?

PROBLEM: {problem_text}

Provide your solution in JSON format with the following structure:
- thought: Your detailed reasoning and step-by-step solution process
- final_answer: Your complete answer with explanation
- numerical_answer: The final numerical result (if applicable, otherwise null)

Example for a math problem:
{{
    "thought": "To solve this problem, I need to...",
    "final_answer": "The solution is 42 because...",
    "numerical_answer": 42
}}

Example for a non-numerical problem:
{{
    "thought": "To approach this problem, I should consider...",
    "final_answer": "The answer is that we should use method X because...",
    "numerical_answer": null
}}

Respond with valid JSON only."""

TASK_SOLVER_SUBSEQUENT_ROUNDS_PROMPT = """These are the reasoning and solutions to the problem from other agents:

{other_solutions}

Using the solutions from other agents as additional information, can you provide your answer to the problem?

The original problem is: {problem_text}

Consider the other agents' approaches and reasoning. You may agree with them, disagree, or provide a synthesis of different approaches.

Provide your solution in JSON format with the following structure:
- thought: Your detailed reasoning, considering other agents' solutions
- final_answer: Your complete answer with explanation
- numerical_answer: The final numerical result (if applicable, otherwise null)

Example:
{{
    "thought": "Looking at the other solutions, Agent A used method X which is correct, but Agent B made an error in step 2. My approach is...",
    "final_answer": "The solution is 42 because...",
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
    "numerical_answer": final_numerical_result_if_applicable_otherwise_null,
    "reasoning": "explanation of your decision"
}}

Respond with valid JSON only."""

# =============================================================================
# SYSTEM MESSAGES
# =============================================================================

AREA_SCIENTIST_SYSTEM_MESSAGE = "You are an expert in and designing a taxonomy of capabilities/skills in this domain."

AREA_MODERATOR_SYSTEM_MESSAGE = "You are an expert in and designing and reviewing a taxonomy of capabilities/skills in this domain."

CAPABILITY_SCIENTIST_SYSTEM_MESSAGE = "You are an expert in and designing a taxonomy of capabilities/skills in this domain."

CAPABILITY_MODERATOR_SYSTEM_MESSAGE = "You are an expert in and designing and reviewing a taxonomy of capabilities/skills in this domain."
