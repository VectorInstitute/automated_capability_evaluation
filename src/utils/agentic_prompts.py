"""Prompts for the debate-based agentic capability generation."""

CAPABILITY_GENERATION_AGGREGATOR_PROMPT = """
Generate {num_gen_capabilities} new, interesting capabilities within the {domain} domain for the "{capability_area}" area.
Each item must have: 'name', 'description', 'instructions', and 'tasks'.
The 'tasks' field must be a list of exactly 3 items; each item must have 'problem' and 'answer'.
"""


CAPABILITY_GENERATION_SCIENTIST_PROMPT = """
You are an expert in designing capabilities to assess the abilities of large language models (LLMs). Your goal is to create novel, diverse capabilities that can reveal the breadth and depth of LLMs' skills within the specified domain. You will be particularly rewarded for uncovering capabilities that could reveal surprising abilities or failures of LLMs. Valid capabilities will be added to a capability archive. In each generation, previously accepted capabilities for the specified domain will be provided as context.

Each capability must provide the following fields:
- "name": A concise, descriptive label (lowercase, no spaces, e.g., "math_competition_algebra").
- "description": A clear explanation of what the capability evaluates.
- "instructions": Initial instructions shown to the subject model. Provide clean natural language instructions; do NOT include any Python code.
- "tasks": Exactly 3 tasks, each an object with:
  - "problem": The task text
  - "answer": The expected answer

Guidelines for instructions:
1. Enforce the final answer in the form "ANSWER: $ANSWER" ONLY for capabilities that expect a short and objective answer. Do not include this for long-form tasks (explanations, proofs, essays, etc.).
2. The instructions should not contain any code or references to implementation details. They should be general-purpose and domain-appropriate.

Return ONLY JSON with a 'capabilities' list with exactly as many items as requested.

In each capability, include exactly these fields: name, description, instructions, tasks.
Do not download additional data from the internet or access the file system.
Be creative and design capabilities that can distinguish between models with varying levels of expertise, but ensure that the capability remains relevant to the domain.
Also ensure that the proposed capabilities are distinct.
Your response will be automatically parsed so ensure it adheres to the specified format.
"""
