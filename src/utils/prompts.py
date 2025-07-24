CAPABILITY_GENERATION_SYSTEM_PROMPT = """
You are an expert in designing capabilities to assess the abilities of foundation models.
Your goal is to create novel, diverse capabilities that can reveal the breadth and depth of a foundation model's skills within the specified domain.
You will be particularly rewarded for a comprehensive design of capabilities.
Valid capabilities will be added to a capability archive.
In each generation, previously accepted capabilities for the specified domain will be provided as context.

Each capability should be designed according to the METR Standard, which requires the following Python format:
```python
class Capability:
    @staticmethod
    def repr_tasks() -> dict[str, dict]:
        # Returns a mapping from task ids to arbitrary data that is specific to each capability.
        # Each capability should contain 3 distinct tasks in the dictionary, labelled "1", "2" and "3".
        # Each task should be a dictionary with the following keys:
        # - "problem": The problem or question text.
        # - "answer": The expected answer to the problem.
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
3. The score function should use a helper function that uses a large language model (LLM) as a judge to score the submission:
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
- "name": A concise, descriptive label (lowercase, no spaces, e.g., "personalized_budget_planning").
- "description": A clear explanation of what the capability entails (e.g., "Ability to generate a realistic monthly budget tailored to an individual's income, fixed and variable expenses, and financial goals. Requires understanding spending categories, prioritization, and basic cash flow allocation.").
- "domain": The domain to which the capability belongs to (e.g., personal finance, math, etc.).
- "class": The fully implemented Python code for the Capability class. This should be easily human-readable.

Do not download additional data from the internet or access the file system.

Be creative and design capabilities that can distinguish between different levels of expertise, but ensure that the capability remains relevant to the domain.
Also ensure that the proposed capabilities ARE DISTINCT compared to the existing capabilities.
Names of all existing capabilities will be provided.

Your response will be automatically parsed so ensure it adheres to the specified format.
"""  # noqa: D100

CAPABILITY_GENERATION_USER_PROMPT = """
A sample capability JSON is provided below. The names of all existing capabilities are also provided.

Sample capability:
{sample_capability_json}

Existing capability names:
{prev_capabilities}

Generate {num_gen_capabilities} new capabilities within the {domain} domain that are **semantically and functionally distinct** from the existing capabilities.
"""

HIERARCHICAL_CAPABILITY_GENERATION_USER_PROMPT = """
A sample capability JSON is provided below. The names of all existing capabilities are also provided.

Sample capability:
{{sample_capability_json}}

Existing capability names:
{{prev_capabilities}}

Generate {{num_gen_capabilities}} new capabilities for the "{capability_area}" area within the {{domain}} domain that do not overlap with the existing capabilities.
"""

HIERARCHICAL_CAPABILITY_AREAS_GENERATION_USER_PROMPT = """
You are an expert in designing capabilities to assess the abilities of foundation models.
For the domain of {domain}, identify {num_areas} high-level, broad, diverse, and non-overlapping areas for capability generation.
Each area should cover {num_capabilities_per_area} capabilities, which will be generated in the next step.
Aim for each area to cover a broad subdomain or skill cluster within the domain.

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

SCORE_BASED_NEW_CAPABILITY_DISCOVERY_USER_PROMPT = """
A sample capability JSON is provided below. Additionally, the names of all existing capabilities and their respective scores for the subject LLM are provided.

Sample capability:
{sample_capability_json}

Existing capability names and scores:
{prev_capabilities_and_scores}

Design a new capability that pushes the boundaries of the subject LLM's abilities. The proposed capability should specifically target areas where the LLM has demonstrated weaknesses or borderline performance. Ensure the capability is unique compared to the existing ones and aligns with the {domain} domain.
"""

KNN_BASED_NEW_CAPABILITY_DISCOVERY_USER_PROMPT = """
A sample capability JSON is provided below. Additionally, the names of {num_input_capabilities} existing capabilities are provided.

Sample capability:
{sample_capability_json}

Existing capability names:
{prev_capabilities}

Design a new capability that is semantically close to the provided {num_input_capabilities} capabilities. The proposed capability should align with the {domain} domain and should be unique compared to the existing ones.
"""

TASK_GENERATION_SYSTEM_PROMPT = """
You are an expert in designing tasks for a given capability. The name, description, {zero_or_few_shot_patch} for the capability will be provided. Ensure designed tasks are diverse spanning a wide range of difficulty levels for the given capability.

Respond precisely in the following format, including the JSON start and end markers:

THOUGHT: <THOUGHT>
RESPONSE JSON:
{response_json_format}

In <THOUGHT>, briefly think and reason about what kind of tasks you want to propose.
In <STR>, provide a string containing the task text.

Be careful to make sure that all proposed tasks are unique. Also ensure that all tasks are within the scope of the given capability.
Avoid simple rewordings or duplicates in structure. Tasks should test different angles of the capability.
If the text includes mathematical symbols or equations, ensure they are appropriately formatted using LaTeX. Ensure the single backlash "\\" included in a LateX string is escaped as "\\\\". For example, the LaTeX string "$\\[2x + 3 = 11\\]$" should be formatted as "$\\\\[2x + 3 = 11\\\\]$" in the task text.

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

TASK_GENERATION_SYSTEM_PROMPT_V2 = """
You are an expert in designing tasks for a given capability. The name, description, {zero_or_few_shot_patch} for the capability will be provided. You will be particularly rewarded for designing diverse tasks spanning a wide range of difficulty levels for the given capability.

Respond precisely in the following format, including the JSON start and end markers:

THOUGHT: <THOUGHT>
RESPONSE JSON:
{response_json_format}

In <THOUGHT>, briefly think and reason about what kind of tasks you want to propose.
In <STR>, provide a string containing the task text.

Be careful to make sure that all proposed tasks are unique. Also ensure that all tasks are within the scope of the given capability.
Avoid simple rewordings or duplicates in structure. Tasks should test different angles of the capability.

If the text includes mathematical symbols or equations, ensure they are appropriately formatted using LaTeX. Ensure the single backlash "\\" included in a LateX string is escaped as "\\\\". For example, the LaTeX string "$\\[2x + 3 = 11\\]$" should be formatted as "$\\\\[2x + 3 = 11\\\\]$" in the task text.

Ensure that the tasks you design are diverse and span a wide range of difficulty levels. Include tasks that test basic, intermediate, and advanced understanding of the capability. Strive to create tasks that challenge different aspects of the capability to ensure comprehensive evaluation.

Your response will be automatically parsed so ensure it adheres to the specified format.
"""

TASK_GENERATION_USER_PROMPT_V2 = """
Design tasks for the following capability:

Name: {capability_name}
Description: {capability_description}
Domain: {capability_domain}
{zero_or_few_shot_patch}
Design {num_gen_tasks} tasks that are diverse in format and difficulty, and collectively cover multiple dimensions of the capability's skill requirements.
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
You are an expert in completing tasks for the {capability_name} capability in the {capability_domain} domain.
Complete the given task by carefully following the provided instructions.
"""

ANSWER_JUDGEMENT_SYSTEM_PROMPT = """
You are an expert in evaluating answers to problems for the {capability_domain} domain. Your goal is to determine whether the provided answer correctly and completely solves the given problem. You must carefully analyze the problem and the answer, and provide a judgement along with your reasoning. "Correctly and completely" means the answer must be accurate, sufficient, and aligned with the task's expectations.

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
