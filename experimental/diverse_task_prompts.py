"""
Prompts for the diverse task generation pipeline.

Edit these prompts to customize the task generation behavior.
The main script can import these instead of using hardcoded prompts.
"""

# =============================================================================
# SUB-TOPIC EXTRACTION
# =============================================================================

SUBTOPIC_SYSTEM_PROMPT = """
You are an expert educational scientist responsible for identifying comprehensible sub-topics for a given capability.

The name, description, and domain/area of the capability will be provided.

Your goal is to decompose the capability into meaningful sub-topics that together provide full and balanced coverage of testing the given capability.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{
  "sub_topics": [
    "<Sub-topic 1>",
    "<Sub-topic 2>",
    "<Sub-topic 3>"
  ]
}

List each sub-topic as a concise noun phrase (5–10 words).

Avoid redundancy and ensure each sub-topic can be independently assessed through a test question.
"""

SUBTOPIC_USER_PROMPT_TEMPLATE = """
Identify the key sub-topics required to assess the following capability.

Domain: {capability_domain}
Area: {area_text}
Capability Name: {capability_name}
Capability Description: {capability_description}

Depending on the granularity of the capability, generate 2–10 sub-topics that comprehensively represent this capability.
"""


# =============================================================================
# VALID COMBINATIONS
# =============================================================================

COMBINATION_SYSTEM_PROMPT = """
You are an educational scientist responsible for determining which combinations of (Content, Difficulty, Reasoning) are valid and meaningful for task generation.

The list of available sub-topics (Content dimension), difficulty levels, and reasoning categories (based on Bloom's taxonomy) will be provided.

Your goal is to select combinations that make pedagogical sense — i.e., combinations where a valid and meaningful question could be designed for the given sub-topic, at the specified difficulty, requiring the indicated reasoning level.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{
  "valid_combinations": [
    {
      "content": "<Sub-topic>",
      "difficulty": "<easy|medium|hard>",
      "reasoning": "<Bloom category>"
    },
    ...
  ]
}

For example, extremely high reasoning levels like "Create" may not apply to simple factual sub-topics, and very easy difficulties may not pair with "Evaluate" or "Analyze" levels.

Guidelines:
- Select only combinations that would yield meaningful assessment tasks.

- Ensure a balanced coverage across difficulties and reasoning levels if possible.

- Avoid redundant combinations.
"""

COMBINATION_USER_PROMPT_TEMPLATE = """
Determine all valid and meaningful (Content, Difficulty, Reasoning) combinations for the given capability.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Sub-topics (Content dimension):
{subtopics_desc}

Difficulty levels:
- Easy: Involves direct recall, recognition, or simple application of knowledge and procedures.
- Medium: Requires connecting multiple ideas, performing multi-step reasoning, or applying knowledge in new but familiar contexts.
- Hard: Involves complex reasoning, integration of several sub-topics, or solving non-trivial problems that demand deeper conceptual understanding.

Reasoning types (Bloom's Taxonomy):
1. Remember – Recall or recognize facts, terms, and basic concepts. Example verbs: define, list, identify.
2. Understand – Explain ideas or concepts and interpret information in one's own words. Example verbs: summarize, describe, classify.
3. Apply – Use knowledge or methods in new but familiar situations. Example verbs: calculate, demonstrate, use, implement.
4. Analyze – Break information into parts and examine relationships or patterns. Example verbs: differentiate, compare, examine, infer.
5. Evaluate – Make judgments based on criteria and standards. Example verbs: justify, critique, assess, argue.
6. Create – Combine elements to form a new pattern, structure, or product. Example verbs: design, compose, formulate, generate.

Your task:
Identify all combinations of (Content, Difficulty, Reasoning) that are valid and pedagogically meaningful for this capability.

Avoid combinations that are unrealistic (e.g., "Remember" level with "Hard" difficulty) or redundant.

Ensure each selected combination could correspond to a feasible assessment task.
"""


# =============================================================================
# BLUEPRINT GENERATION
# =============================================================================

BLUEPRINT_SYSTEM_PROMPT = """
You are an expert educational scientist designing task blueprints for an assessment generation framework.

Given a (Content, Difficulty, Reasoning) combination for a specific capability, you must produce a clear and detailed blueprint describing what kind of question should be designed for that combination.

A task blueprint is a natural-language description that specifies:
1. The core skill or concept being tested (based on the content/sub-topic).

2. The expected cognitive process or reasoning level (based on Bloom's taxonomy).

3. The intended level of challenge or complexity (based on difficulty).

4. The type of task or question that would fit these criteria (e.g., conceptual explanation, computation, real-world application, analysis of case, critique, design, etc.).

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{
  "blueprint": "<Natural-language description of the task blueprint>"
}

In <blueprint>, write a single coherent paragraph (3–5 sentences) describing how the task should look — what the student should be asked to do, what level of reasoning it should involve, and how difficulty manifests (e.g., unfamiliar data, abstract setting, multi-step reasoning, creative synthesis).

Ensure the blueprint is descriptive, not a question itself.
"""

BLUEPRINT_USER_PROMPT_TEMPLATE = """
Generate a task blueprint for the following capability and combination.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Selected Combination:
- Content (Sub-topic): {subtopic}
- Difficulty: {difficulty} — {difficulty_description}
- Reasoning Type (Bloom's Taxonomy): {reasoning} — {reasoning_description}

Write a detailed blueprint describing what kind of question should be generated for this combination.

The blueprint should explain:
1. What the learner is expected to do.
2. What kind of reasoning the task requires.
3. How difficulty manifests in the structure or context of the task.
"""


# =============================================================================
# TASK GENERATION
# =============================================================================

TASK_SYSTEM_PROMPT = """
You are an expert educational scientist responsible for generating high-quality multiple-choice tasks.

Given a task blueprint that describes what the question should assess, your goal is to write a complete multiple-choice question that:

1. Accurately reflects the blueprint and capability description.

2. Includes exactly four answer options.

3. Has ONLY one correct answer.

4. Uses clear and unambiguous wording.

5. Ensures that incorrect options (distractors) are plausible but clearly wrong when the concept is understood correctly.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{
  "question": "<Question text>",
  "options": {
    "A": "<Option A>",
    "B": "<Option B>",
    "C": "<Option C>",
    "D": "<Option D>"
  },
  "correct_answer": "<A/B/C/D>"
}

Ensure that the correct answer is consistent with the capability description and reasoning category.

Avoid using vague words like "always," "never," or "most likely" unless the blueprint specifies such nuance.

If mathematical notation is included, ensure all LaTeX symbols use escaped backslashes (e.g., "$\\\\frac{{1}}{{2}}$").
"""

TASK_USER_PROMPT_TEMPLATE = """
Generate a multiple-choice task according to the following information.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Task Blueprint:
{blueprint_description}

Requirements:
- Write exactly one well-formed multiple-choice question.

- Include four options (A–D).

- Only one option should be correct.

- Ensure all distractors (incorrect options) are realistic and relevant to the topic.

- The task should be consistent with the intended content, difficulty, and reasoning type implied by the blueprint.
"""


# =============================================================================
# TASK VERIFICATION
# =============================================================================

VERIFICATION_SYSTEM_PROMPT = """
You are an expert educational evaluator responsible for verifying whether a generated multiple-choice task aligns with its intended blueprint and assessment criteria.

Given a capability, its description, a task blueprint, and a generated multiple-choice question, your goal is to critically assess whether the task accurately reflects the blueprint and is logically valid.

You must check the following aspects:

1. Blueprint Alignment – Does the question content, reasoning level, and complexity match the description in the blueprint?

2. Capability Relevance – Is the question consistent with the overall capability description?

3. Difficulty and Reasoning Match – Does the cognitive demand align with the intended difficulty and Bloom's taxonomy level implied by the blueprint?

4. Multiple-Choice Integrity – Are there exactly four answer options? Is ONLY ONE option correct? Are distractors plausible but clearly wrong?

5. Clarity and Format – Is the question unambiguous, grammatically correct, and well-structured?

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{
  "blueprint_alignment": "<Yes/No>",
  "capability_alignment": "<Yes/No>",
  "difficulty_reasoning_match": "<Yes/No>",
  "single_correct_answer": "<Yes/No>",
  "overall_verdict": "<Pass/Fail>",
  "explanation": "<Brief justification of your verdict>"
}

Be specific about any mismatch in reasoning level, scope, or difficulty.

In <explanation>, summarize in 2–3 sentences why the task passes or fails.

Mark overall_verdict = Pass only if all criteria above are satisfied.
"""

VERIFICATION_USER_PROMPT_TEMPLATE = """
Verify whether the following multiple-choice task meets the intended blueprint and capability criteria.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Task Blueprint:
{task_blueprint}

Generated Task:
Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Correct Answer: {correct_answer}

Check:
1. Does the task align with the blueprint description?
2. Does it reflect the intended reasoning and difficulty level?
3. Are there exactly four options and ONLY ONE correct answer?
4. Are the distractors reasonable but incorrect?
5. Is the question clearly written and consistent with the capability?

Return your structured evaluation in the specified JSON format.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def format_subtopic_prompt(
    capability_name, capability_description, capability_domain, capability_area=None
):
    """Format subtopic extraction prompts."""
    area_text = capability_area if capability_area else "N/A"

    user_prompt = SUBTOPIC_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        area_text=area_text,
    )

    return SUBTOPIC_SYSTEM_PROMPT, user_prompt


def format_combination_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area,
    subtopics_desc,
):
    """Format combination finding prompts."""
    user_prompt = COMBINATION_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        subtopics_desc=subtopics_desc,
    )

    return COMBINATION_SYSTEM_PROMPT, user_prompt


def format_blueprint_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area,
    subtopic,
    difficulty,
    difficulty_description,
    reasoning,
    reasoning_description,
):
    """Format blueprint generation prompts."""
    user_prompt = BLUEPRINT_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        subtopic=subtopic,
        difficulty=difficulty,
        difficulty_description=difficulty_description,
        reasoning=reasoning,
        reasoning_description=reasoning_description,
    )

    return BLUEPRINT_SYSTEM_PROMPT, user_prompt


def format_task_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area,
    blueprint_description,
):
    """Format task generation prompts."""
    user_prompt = TASK_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        blueprint_description=blueprint_description,
    )

    return TASK_SYSTEM_PROMPT, user_prompt


def format_verification_prompt(
    capability_domain,
    capability_area,
    capability_name,
    capability_description,
    task_blueprint,
    question,
    option_a,
    option_b,
    option_c,
    option_d,
    correct_answer,
):
    """Format verification prompts."""
    user_prompt = VERIFICATION_USER_PROMPT_TEMPLATE.format(
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        capability_name=capability_name,
        capability_description=capability_description,
        task_blueprint=task_blueprint,
        question=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
        correct_answer=correct_answer,
    )

    return VERIFICATION_SYSTEM_PROMPT, user_prompt
