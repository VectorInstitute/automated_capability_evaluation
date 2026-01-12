"""
Prompts for the base pipeline stages.

This module contains all prompts used by the base (non-agentic) pipeline:
- Stage 1: Area generation
- Stage 2: Capability generation
- Stage 3: Task generation (sub-topics, combinations, blueprints, questions, options)
- Stage 4: Solution generation
- Stage 5: Task validation

Edit these prompts to customize generation behavior.
"""

# =============================================================================
# AREA GENERATION (Stage 1)
# =============================================================================

AREAS_GENERATION_USER_PROMPT = """
You are an expert in designing capabilities to assess the abilities of foundation models.
For the domain of {domain}, identify {num_areas} high-level, broad, diverse, and non-overlapping areas for capability generation.
Each area should cover {num_capabilities_per_area} capabilities, which will be generated in the next step.
Aim for each area to cover a broad subdomain or skill cluster within the domain.

Respond in the following JSON format:

{response_json_format}
"""

AREAS_GENERATION_RESPONSE_JSON_FORMAT = """
{{
    "areas": [
        <STR>,
        <STR>,
        ...
    ]
}}"""


# =============================================================================
# CAPABILITY GENERATION (Stage 2)
# =============================================================================

CAPABILITY_GENERATION_SYSTEM_PROMPT = """
You are an expert in designing capabilities to assess the abilities of foundation models.
Your goal is to create novel, diverse capabilities that can reveal the breadth and depth of a foundation model's skills within the specified domain.
You will be particularly rewarded for a comprehensive design of capabilities.
Valid capabilities will be added to a capability archive.
In each generation, previously accepted capabilities for the specified domain will be provided as context.

Respond precisely in the following JSON format:

{
    "thought": <STR>,
    "capabilities": [
        {
            "name": <STR>,
            "description": <STR>
        },
        ...
    ]
}

In "thought", briefly think and reason about what kind of capabilities you want to propose.
In "capabilities", provide an array of new capability objects with the following fields:
- "name": A concise, descriptive label (lowercase, underscores for spaces, e.g., "personalized_budget_planning").
- "description": A clear and detailed explanation of what the capability entails, including the skills and knowledge required (e.g., "Ability to generate a realistic monthly budget tailored to an individual's income, fixed and variable expenses, and financial goals. Requires understanding spending categories, prioritization, and basic cash flow allocation.").

Do not download additional data from the internet or access the file system.

Be creative and design capabilities that can distinguish between different levels of expertise, but ensure that the capability remains relevant to the domain.
Also ensure that the proposed capabilities ARE DISTINCT compared to the existing capabilities.
Names of all existing capabilities will be provided.

Your response will be automatically parsed so ensure it adheres to the specified format.
"""

CAPABILITY_GENERATION_USER_PROMPT = """
The names of all existing capabilities are provided below.

Existing capability names:
{prev_capabilities}

Generate {num_capabilities} new capabilities for the "{area}" area within the {domain} domain that do not overlap with the existing capabilities.
"""


# =============================================================================
# SUB-TOPIC EXTRACTION (Stage 3 - Step 1)
# =============================================================================

SUBTOPIC_SYSTEM_PROMPT = """
You are an expert in {capability_domain} responsible for identifying comprehensible sub-topics for a given capability.

A domain is a broad subject area (e.g., Mathematics), an area is a specialized field within that domain (e.g., Linear Algebra), and a capability is a specific topic within that area (e.g., representing graphs using matrices).

The name, description, and domain/area of the capability will be provided.

Your goal is to decompose the capability into meaningful sub-topics that together provide full and balanced coverage of testing the given capability. Generate between {min_subtopics} and {max_subtopics} sub-topics depending on the granularity and scope of the capability.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{{{{
  "sub_topics": [
    "<Sub-topic 1>",
    "<Sub-topic 2>",
    "<Sub-topic 3>",
    ...
  ]
}}}}

List each sub-topic as a concise noun phrase (5–10 words).

Avoid redundancy and ensure each sub-topic can be independently assessed through a test question.
"""

SUBTOPIC_USER_PROMPT_TEMPLATE = """
Identify the key sub-topics required to assess the following capability.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Depending on the granularity of the capability, generate {min_subtopics}–{max_subtopics} sub-topics that comprehensively represent this capability.
"""


# =============================================================================
# VALID COMBINATIONS
# =============================================================================

COMBINATION_SYSTEM_PROMPT = """
You are an expert in {capability_domain} responsible for determining which combinations of (Content, Difficulty, Reasoning) are valid and meaningful for task generation.

A domain is a broad subject area (e.g., Mathematics), an area is a specialized field within that domain (e.g., Linear Algebra), a capability is a specific concept or topic within that area (e.g., representing graphs using matrices), and a sub-topic is a concrete skill of that capability that can be assessed (e.g., constructing an adjacency matrix for a given graph).

The list of available sub-topics (Content dimension), difficulty levels, and reasoning categories (based on Bloom's taxonomy) will be provided.

Your goal is to select combinations that make pedagogical sense — i.e., combinations where a valid and meaningful question could be designed for the given sub-topic, at the specified difficulty, requiring the indicated reasoning level.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{{{{
  "valid_combinations": [
    {{{{
      "content": "<Sub-topic>",
      "difficulty": "<easy|medium|hard>",
      "reasoning": "<Bloom category>"
    }}}},
    ...
  ]
}}}}

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
{content_list}

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

A domain is a broad subject area (e.g., Mathematics), an area is a specialized field within that domain (e.g., Linear Algebra), a capability is a specific concept or topic within that area (e.g., representing graphs using matrices), and a sub-topic is a concrete skill of that capability that can be assessed (e.g., constructing an adjacency matrix for a given graph).

Given a (Content (sub-topic), Difficulty, Reasoning) combination for a specific capability, you must produce a clear and detailed blueprint describing what kind of question should be designed for that combination.

A task blueprint is a natural-language description that specifies:
1. The core skill or concept being tested (based on the content/sub-topic).

2. The expected cognitive process or reasoning level (based on Bloom's taxonomy).

3. The intended level of challenge or complexity (based on difficulty).

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{{{{
  "blueprint": "<Natural-language description of the task blueprint>"
}}}}

In <blueprint>, write a single coherent paragraph (3–5 sentences) describing how the task should look — what the task evaluates, what the student should be asked to do, what level of reasoning it should involve (based on the bloom's taxonomy provided below), and how difficulty manifests (e.g., unfamiliar data, abstract setting, multi-step reasoning, creative synthesis).

Reasoning types (Bloom's Taxonomy):
1. Remember – Recall or recognize facts, terms, and basic concepts. Example verbs: define, list, identify.
2. Understand – Explain ideas or concepts and interpret information in one's own words. Example verbs: summarize, describe, classify.
3. Apply – Use knowledge or methods in new but familiar situations. Example verbs: calculate, demonstrate, use, implement.
4. Analyze – Break information into parts and examine relationships or patterns. Example verbs: differentiate, compare, examine, infer.
5. Evaluate – Make judgments based on criteria and standards. Example verbs: justify, critique, assess, argue.
6. Create – Combine elements to form a new pattern, structure, or product. Example verbs: design, compose, formulate, generate.

Ensure the blueprint is descriptive, not a question itself.
"""

BLUEPRINT_USER_PROMPT_TEMPLATE = """
Generate a task blueprint for the following capability and combination.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Selected Combination:
- Content (Sub-topic): {content_value}
- Difficulty: {difficulty_value} — {difficulty_definition}
- Reasoning Type (Bloom's Taxonomy): {reasoning_value} — {reasoning_definition}

Write a detailed blueprint describing what kind of question should be generated for this combination.

The blueprint should explain:
1. What the learner is expected to do.
2. What kind of reasoning the task requires.
3. How difficulty manifests in the structure or context of the task.

Respond in the following JSON format:
{{{{
  "blueprint": "<Natural-language description of the task blueprint>"
}}}}
"""


# =============================================================================
# QUESTION GENERATION (Stage 3 - Step 1)
# =============================================================================

QUESTION_SYSTEM_PROMPT = """
You are an expert educational scientist responsible for generating high-quality assessment questions.

A domain is a broad subject area (e.g., Mathematics), an area is a specialized field within that domain (e.g., Linear Algebra), a capability is a specific concept or topic within that area (e.g., representing graphs using matrices), and a sub-topic is a concrete skill of that capability that can be assessed (e.g., constructing an adjacency matrix for a given graph).

Given a task blueprint that describes what the question should assess, your goal is to write a clear, well-formed question that:

1. Accurately reflects the blueprint and capability description.

2. Is suitable for a multiple-choice format (will have options generated separately).

3. Uses clear and unambiguous wording.

4. Has a single, objectively correct answer.

IMPORTANT: Generate ONLY the question text. Do NOT include any answer options.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{{{{
  "question": "<Question text>"
}}}}

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

Avoid using vague words like "always," "never," or "most likely" unless the blueprint specifies such nuance.

If mathematical notation is included, ensure all LaTeX symbols use escaped backslashes (e.g., "$\\\\frac{1}{2}$").
"""

QUESTION_USER_PROMPT_TEMPLATE = """
Generate a question according to the following information.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Task Blueprint:
{task_blueprint}

Requirements:
- Write exactly one well-formed question.

- The question should be suitable for multiple-choice format.

- The question should have a single, objectively correct answer.

- The question should be consistent with the intended content, difficulty, and reasoning type implied by the blueprint.

- Do NOT include any answer options - only the question text.
"""


# =============================================================================
# OPTIONS GENERATION (Stage 3 - Step 2)
# =============================================================================

OPTIONS_SYSTEM_PROMPT = """
You are an expert educational scientist responsible for generating high-quality multiple-choice options.

Given a question, your goal is to generate exactly four answer options (A, B, C, D) where:

1. Exactly ONE option is the correct answer.

2. The other three options (distractors) are plausible but clearly wrong when the concept is understood correctly.

3. All options are realistic and relevant to the topic.

4. Options are distinct and unambiguous.

IMPORTANT: Do NOT indicate which option is correct. The correct answer will be determined separately by solving the problem.

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{{{{
  "options": {{{{
    "A": "<Option A>",
    "B": "<Option B>",
    "C": "<Option C>",
    "D": "<Option D>"
  }}}}
}}}}

Guidelines for creating good distractors:
- Common misconceptions related to the topic
- Errors from typical calculation mistakes
- Partially correct answers that miss key aspects
- Answers that would result from misreading or misunderstanding the question

Ensure all options are of similar length and format to avoid giving hints about the correct answer.

If mathematical notation is included, ensure all LaTeX symbols use escaped backslashes (e.g., "$\\\\frac{1}{2}$").
"""

OPTIONS_USER_PROMPT_TEMPLATE = """
Generate four multiple-choice options for the following question.

Domain: {capability_domain}
Area: {capability_area}
Capability Name: {capability_name}
Capability Description: {capability_description}

Question:
{question}

Requirements:
- Generate exactly four options (A, B, C, D).

- Exactly one option must be the correct answer.

- The other three options should be plausible distractors.

- All options should be relevant to the topic.

- Do NOT indicate which option is correct.
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
{{{{
  "blueprint_alignment": "<Yes/No>",
  "capability_alignment": "<Yes/No>",
  "difficulty_reasoning_match": "<Yes/No>",
  "single_correct_answer": "<Yes/No>",
  "overall_verdict": "<Pass/Fail>",
  "explanation": "<Brief justification of your verdict>"
}}}}

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
Question: {question_text}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}


Check:
1. Does the task align with the blueprint description?
2. Does it reflect the intended reasoning and difficulty level?
3. Are there exactly four options and ONLY ONE correct answer?
4. Are the distractors reasonable but incorrect?
5. Is the question clearly written and consistent with the capability?

Return your structured evaluation in the specified JSON format.
"""


# =============================================================================
# SOLUTION GENERATION (Stage 4)
# =============================================================================

SOLUTION_SYSTEM_PROMPT = """
You are an expert in {capability_domain} responsible for solving multiple-choice questions.

Given a multiple-choice question with exactly four options (A, B, C, D), your goal is to:
1. Carefully analyze the question and all options
2. Apply your knowledge of the domain to determine the correct answer
3. Provide clear reasoning for your answer

Respond precisely in the following format, including the JSON start and end markers:

RESPONSE JSON:
{{{{
  "answer": "<A/B/C/D>",
  "reasoning": "<Step-by-step explanation of why this answer is correct>"
}}}}

Guidelines:
- Consider each option carefully before making your decision
- Explain why the correct answer is right
- If relevant, briefly explain why other options are incorrect
- Be precise and confident in your answer
"""

SOLUTION_USER_PROMPT_TEMPLATE = """
Solve the following multiple-choice question.

Domain: {capability_domain}
Area: {capability_area}
Capability: {capability_name}
Capability Description: {capability_description}

Question:
{task_text}

Analyze the question and provide your answer with reasoning.
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def format_subtopic_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area=None,
    min_subtopics=3,
    max_subtopics=8,
):
    """Format subtopic extraction prompts.

    Args:
        capability_name: Name of the capability
        capability_description: Description of the capability
        capability_domain: Domain name
        capability_area: Area name (optional)
        min_subtopics: Minimum number of subtopics to generate
        max_subtopics: Maximum number of subtopics to generate
    """
    area_text = capability_area if capability_area else "N/A"

    system_prompt = SUBTOPIC_SYSTEM_PROMPT.format(
        capability_domain=capability_domain,
        min_subtopics=min_subtopics,
        max_subtopics=max_subtopics,
    )

    user_prompt = SUBTOPIC_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=area_text,
        min_subtopics=min_subtopics,
        max_subtopics=max_subtopics,
    )

    return system_prompt, user_prompt


def format_combination_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area,
    content_list,
):
    """Format combination finding prompts."""
    system_prompt = COMBINATION_SYSTEM_PROMPT.format(
        capability_domain=capability_domain,
    )

    user_prompt = COMBINATION_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        content_list=content_list,
    )

    return system_prompt, user_prompt


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
        content_value=subtopic,
        difficulty_value=difficulty,
        difficulty_definition=difficulty_description,
        reasoning_value=reasoning,
        reasoning_definition=reasoning_description,
    )

    return BLUEPRINT_SYSTEM_PROMPT, user_prompt


def format_question_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area,
    blueprint_description,
):
    """Format question generation prompts (Stage 3 - Step 1).

    Args:
        capability_name: Name of the capability
        capability_description: Description of the capability
        capability_domain: Domain name
        capability_area: Area name
        blueprint_description: The blueprint describing what to assess

    Returns
    -------
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = QUESTION_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        task_blueprint=blueprint_description,
    )

    return QUESTION_SYSTEM_PROMPT, user_prompt


def format_options_prompt(
    capability_name,
    capability_description,
    capability_domain,
    capability_area,
    question,
):
    """Format options generation prompts (Stage 3 - Step 2).

    Args:
        capability_name: Name of the capability
        capability_description: Description of the capability
        capability_domain: Domain name
        capability_area: Area name
        question: The question text to generate options for

    Returns
    -------
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = OPTIONS_USER_PROMPT_TEMPLATE.format(
        capability_name=capability_name,
        capability_description=capability_description,
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        question=question,
    )

    return OPTIONS_SYSTEM_PROMPT, user_prompt


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
        question_text=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

    return VERIFICATION_SYSTEM_PROMPT, user_prompt


def format_solution_prompt(
    capability_domain,
    capability_area,
    capability_name,
    capability_description,
    task_text,
):
    """Format solution generation prompts.

    Args:
        capability_domain: Domain name
        capability_area: Area name
        capability_name: Capability name
        capability_description: Capability description
        task_text: The full task text (question + options)

    Returns
    -------
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = SOLUTION_SYSTEM_PROMPT.format(
        capability_domain=capability_domain,
    )

    user_prompt = SOLUTION_USER_PROMPT_TEMPLATE.format(
        capability_domain=capability_domain,
        capability_area=capability_area if capability_area else "N/A",
        capability_name=capability_name,
        capability_description=capability_description,
        task_text=task_text,
    )

    return system_prompt, user_prompt
