CAPABILITY_GENERATION_SYSTEM_PROMPT = """
You are an expert in designing capabilities to assess the abilities of large language models (LLMs). Your goal is to create novel, diverse capabilities that can reveal the breadth and depth of LLMs’ skills within the specified domain. You will be particularly rewarded for uncovering capabilities that could reveal surprising abilities or failures of LLMs. Valid capabilities will be added to a capability archive. In each generation, previously accepted capabilities for the specified domain will be provided as context.

Each capability should be designed according to the METR Standard, which requires the following Python format:
```python
class Capability:
    @staticmethod
    def repr_tasks() -> dict[str, dict]:
        # Returns a mapping from task ids to arbitrary data that is specific to each capability.
        # Each capability should contain 3 distinct tasks in the dictionary, labelled "1", "2" and "3".
        # ... implementation ...

    @staticmethod
    def get_instructions(t: dict) -> str:
        # This method should return a string containing initial capability instructions.
        # ... implementation ...

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        # ‘score‘ is called after the subject model submits a solution.
        # Return 0 on failure and 1 on success.
        # Throwing an exception will result in a score of 0.
        # ... implementation ...
```

Respond precisely in the following format, including the JSON start and end markers:

THOUGHT: <THOUGHT>
RESPONSE JSON:
{
    "capability_0": <JSON>,
    "capability_1": <JSON>,
    ...
}

In <THOUGHT>, briefly think and reason about what kind of capability you want to propose.
In <JSON>, provide a JSON response of the new capability with the following fields:
- "name": A concise, descriptive label (lowercase, no spaces, e.g., "math_competition_algebra").
- "description": A clear explanation of what the capability entails (e.g., The capability consists of challenging competition mathematics problems in algebra).
- "domain": The domain to which the capability belongs to (e.g., math, physics, etc.).
- "class": The fully implemented Python code for the Capability class. This should be easily human-readable.

All values in the JSON should be strings. Do not download additional data from the internet or access the file system.

Be creative and design capabilities that can distinguish between models with varying levels of expertise, but ensure that the capability remains relevant to the domain. Also ensure that the proposed capabilities ARE DISTINCT compared to the previous capabilities. Previous seed capabilities will be provided in the same JSON format as above. Whereas, only capability names will be provided for previously generated capabilities.

Your response will be automatically parsed so ensure it adheres to the specified format.
"""  # noqa: D100

CAPABILITY_GENERATION_USER_PROMPT = """
Summary of previous capabilities from the {domain} domain is given below:
Seed capabilities:
{seed_capabilities}

Previously generated capabilities:
{prev_capabilities}

Generate {num_gen_capabilities} new interesting capabilities within the {domain} domain.
"""

TASK_GENERATION_SYSTEM_PROMPT = """
You are an expert in designing tasks for a given capability. The name, description, {zero_or_few_shot_patch} for the capability will be provided. You will be particularly rewarded for designing diverse tasks spanning a wide range of difficulty levels for the given capability.

Respond precisely in the following format, including the JSON start and end markers:

THOUGHT: <THOUGHT>
RESPONSE JSON:
{response_json_format}

In <THOUGHT>, briefly think and reason about what kind of tasks you want to propose.
In <STR>, provide a string containing the task text.

Be careful to make sure that all proposed tasks are unique. Also ensure that all tasks are within the scope of the given capability. If the text includes mathematical symbols or equations, ensure they are appropriately formatted using LaTeX.

Your response will be automatically parsed so ensure it adheres to the specified format.
"""

TASK_GENERATION_USER_PROMPT = """
Design tasks for the following capability:

Name: {capability_name}
Description: {capability_description}
Domain: {capability_domain}
{zero_or_few_shot_patch}
Generate {num_gen_tasks} new tasks for the given capability.
"""

TASK_GENERATION_ZERO_OR_FEW_SHOT_PATCH = {
    "zero_shot": {"sys": "and domain", "user": ""},
    "few_shot": {
        "sys": "domain and a few sample tasks",
        "user": "Sample tasks:\n{capability_sample_tasks}\n",
    },
}

TASK_GENERATION_RESPONSE_JSON_FORMAT = """
{
    "task_1": <STR>,
    "task_2": <STR>,
    ...
}""".strip("\n")


TASK_SOLVER_SYSTEM_PROMPT = """
You are an expert in completing tasks for the {capability_name} capability in the {capability_domain} domain. Complete the given task by carefully following the provided instructions.
"""
