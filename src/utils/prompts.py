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
The following points should be considered while designing the Capability class:
1. The instructions should enforce the final answer in the form "ANSWER: $ANSWER" ONLY for those capabilities that expect a short and objective answer. It should not be part of instructions for capabilities with long-form, detailed and subjective answers for e.g. explanations, proofs, essays, etc.
2. If the instructions enforces the final answer in the form "ANSWER: $ANSWER", then the score function should first call a helper function to parse the submission string and extract the answer:
```python
def parse_submission(submission: str) -> str:
    # Parse the submission string to extract the answer based on the "ANSWER" keyword.
    # Return an empty string if no match is found.
```
3. The score function should use a helper function that uses LLM as a judge to score the submission:
```python
def evaluate_with_llm_judge(
    submission: str,
    target: str,
) -> bool:
    # Evaluate the submission using an LLM judge.
```
4. DO NOT re-implement the `parse_submission()` or `evaluate_with_llm_judge()` helper functions.

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

Be creative and design capabilities that can distinguish between models with varying levels of expertise, but ensure that the capability remains relevant to the domain. Also ensure that the proposed capabilities ARE DISTINCT compared to the existing capabilities. Names of all existing capabilities will be provided.

Your response will be automatically parsed so ensure it adheres to the specified format.
"""  # noqa: D100

CAPABILITY_GENERATION_USER_PROMPT = """
A sample capability JSON is provided below. The names of all existing capabilities are also provided.

Sample capability:
{sample_capability_json}

Existing capability names:
{prev_capabilities}

Generate {num_gen_capabilities} new, interesting capabilities within the {domain} domain.
"""

HIERARCHICAL_CAPABILITY_GENERATION_USER_PROMPT = """
A sample capability JSON is provided below. The names of all existing capabilities are also provided.

Sample capability:
{{sample_capability_json}}

Existing capability names:
{{prev_capabilities}}

Generate {{num_gen_capabilities}} new, interesting capabilities for the "{capability_area}" area within the {{domain}} domain.
"""

HIERARCHICAL_CAPABILITY_AREAS_GENERATION_USER_PROMPT = """
You are an expert in designing capabilities to assess the abilities of large language models (LLMs). Identify {num_areas} broad and diverse areas for capability generation for the {domain} domain. The areas should be relevant to the {domain} domain, should be high level and should not overlap with each other.

Respond precisely in the following format:

RESPONSE JSON:
{response_json_format}
"""

CAPABILITY_AREAS_GENERATION_RESPONSE_JSON_FORMAT = """
{
    "area_0": <STR>,
    "area_1": <STR>,
    ...
}""".strip("\n")

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

ANSWER_JUDGEMENT_SYSTEM_PROMPT = """
You are an expert in evaluating answers to problems for the {capability_domain} domain. Your goal is to determine whether the provided answer correctly and completely solves the given problem. You must carefully analyze the problem and the answer, and provide a judgement along with your reasoning.

Respond precisely in the following format:

THOUGHT: <THOUGHT>
JUDGEMENT:
<JUDGEMENT>

In <THOUGHT>, briefly explain your reasoning process for evaluating the answer.
In <JUDGEMENT>, respond with "yes" if the answer correctly and completely solves the problem, otherwise respond with "no".

Be objective and thorough in your evaluation. Ensure that your reasoning is clear and directly supports your judgement.
"""

ANSWER_JUDGEMENT_USER_PROMPT = """
Evaluate the following problem and answer for the {capability_name} capability in the {capability_domain} domain:

Problem: {problem}
Answer: {answer}

Determine if the answer correctly and completely solves the problem. Provide your reasoning and judgement.
"""
