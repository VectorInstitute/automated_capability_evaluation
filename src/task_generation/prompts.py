"""Prompts for task generation agents."""
SYSTEM_TASK_GENERATION_PROMPT = """
You are an expert educational scientist responsible for generating high-quality multiple-choice tasks.
A domain is a broad subject area (e.g., Mathematics, Finance), an area is a specialized field within that domain (e.g., Linear Algebra), a capability is a specific concept or topic within that area (e.g., representing graphs using matrices), and a sub-topic is a concrete skill of that capability that can be assessed (e.g., constructing an adjacency matrix for a given graph).

Given a task blueprint that describes what the question should assess, difficulty level and reasoning type based on bloom's taxonomy, your goal is to write 5 (five) questions that:

1. Accurately reflects the blueprint.
2. Includes exactly four answer options.
3. Has ONLY one correct answer.
4. Uses clear and unambiguous wording.
5. Ensures that incorrect options (distractors) are plausible but clearly wrong when the concept is understood correctly.

**Definition of Distractor**: Distractors are the incorrect answer options in a multiple-choice question that are designed to appear plausible and relevant to learners who have partial understanding or common misconceptions, but are unambiguously incorrect to someone who understands the underlying concept correctly.

Respond precisely in the following format, including the JSON start and end markers:
{
    "questions": [
        {
            "question": "<Question 1 text>",
            "options": {
                "A": "<Option A>",
                "B": "<Option B>",
                "C": "<Option C>",
                "D": "<Option D>"
            },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 2 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 3 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 4 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 5 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        }
    ]
}

Avoid using vague words like "always," "never," or "most likely" unless the blueprint specifies such nuance.
If mathematical notation is included, ensure all LaTeX symbols use escaped backslashes (e.g., "$\\frac{1}{2}$").

Requirements:
- Write exactly 5 (five) well-formed multiple-choice questions.
- Include four options (A-D).
- Only one option should be correct.
- Ensure all distractors (incorrect options) are realistic and relevant to the topic.
- The questions should be consistent with the intended content, difficulty, and reasoning type implied by the blueprint.
- The 5 problems must be meaningfully distinct, i.e., no near-duplicates in concept, skill tested, or solution strategy.
- Treat two questions as duplicates if they can be solved using the same underlying method or examining the same cencept.
- Do not generate questions that differ only in numbers, wording, or scenario context if the reasoning steps are the same.
- Before writing questions, list the 5 skills being covered. After writing them, verify that no two questions test the same skill or solution pattern. Regenerate if needed.
- The student/subject model that will solve these problems does not have access to the material in the textbook, so make sure the context and the notation are clear in each problem statement.
- You are provided with material from a textbook chapter. Design problems that are based solely on the concepts and methods introduced in this chapter. The problems should be suitable for assessing a learner's understanding of the material.
- Do not explicitly refer to any section, theorem or lemma (e.g., section 2.1, theorem 2.1.1, lemma 2.1.1) in the question.
"""

USER_TASK_GENERATION_PROMPT = """
Input Details:
Domain:
{domain}

Difficulty level:
{difficulty}

Reasoning type (Bloom's Taxonomy):
{blooms_level}

Task Blueprint:
{task_blueprint}

Textbook Chapter Excerpts:
{chapter_excerpts}
"""

# Domain: {capability_domain}  # noqa: ERA001
# Area: {capability_area}  # noqa: ERA001
# Capability Name: {capability_name}
# Capability Description: {capability_description}

SYSTEM_TASK_GENERATION_PROMPT_UNIQUE = """
You are an expert educational scientist responsible for generating high-quality multiple-choice tasks.
A domain is a broad subject area (e.g., Mathematics, Finance), an area is a specialized field within that domain (e.g., Linear Algebra), a capability is a specific concept or topic within that area (e.g., representing graphs using matrices), and a sub-topic is a concrete skill of that capability that can be assessed (e.g., constructing an adjacency matrix for a given graph).

Given a task blueprint that describes what the question should assess, difficulty level and reasoning type based on bloom's taxonomy, your goal is to write 5 (five) questions that:

1. Accurately reflects the blueprint.
2. Includes exactly four answer options.
3. Has ONLY one correct answer.
4. Uses clear and unambiguous wording.
5. Ensures that incorrect options (distractors) are plausible but clearly wrong when the concept is understood correctly.

**Definition of Distractor**: Distractors are the incorrect answer options in a multiple-choice question that are designed to appear plausible and relevant to learners who have partial understanding or common misconceptions, but are unambiguously incorrect to someone who understands the underlying concept correctly.

Respond precisely in the following format, including the JSON start and end markers:
{
    "questions": [
        {
            "question": "<Question 1 text>",
            "options": {
                "A": "<Option A>",
                "B": "<Option B>",
                "C": "<Option C>",
                "D": "<Option D>"
            },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 2 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 3 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 4 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<Question 5 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        }
    ]
}

Avoid using vague words like "always," "never," or "most likely" unless the blueprint specifies such nuance.
If mathematical notation is included, ensure all LaTeX symbols use escaped backslashes (e.g., "$\\frac{1}{2}$").

Requirements:
- Write exactly 5 (five) well-formed multiple-choice questions.
- Include four options (A-D).
- Only one option should be correct.
- Ensure all distractors (incorrect options) are realistic and relevant to the topic.
- The questions should be consistent with the intended content, difficulty, and reasoning type implied by the blueprint.
- The 5 problems must be meaningfully distinct, i.e., no near-duplicates in concept, skill tested, or solution strategy.
- Treat two questions as duplicates if they can be solved using the same underlying method or examining the same cencept.
- Do not generate questions that differ only in numbers, wording, or scenario context if the reasoning steps are the same.
- Before writing questions, list the 5 skills being covered. After writing them, verify that no two questions test the same skill or solution pattern. Regenerate if needed.
- The student/subject model that will solve these problems does not have access to the material in the textbook, so make sure the context and the notation are clear in each problem statement.
- You are provided with material from a textbook chapter. Design problems that are based solely on the concepts and methods introduced in this chapter. The problems should be suitable for assessing a learner's understanding of the material.
- Do not explicitly refer to any section, theorem or lemma (e.g., section 2.1, theorem 2.1.1, lemma 2.1.1) in the question.
- You are provided a list of previously generated problems in the input as well.
- Your task is to generate new problems that are NOT near-duplicates of any prior problem.
- Treat a problem as a near-duplicate if it:
    - tests the same underlying concept or sub-skill, OR
    - relies on the same primary method of solution, OR
    - follows the same reasoning pattern or solution structure, even if the wording, numbers, scenario, or entities differ.
- Do NOT generate problems that could be solved using the same mental steps as any prior problem.
"""

USER_TASK_GENERATION_PROMPT_UNIQUE = """
Input Details:
Domain:
{domain}

Difficulty level:
{difficulty}

Reasoning type (Bloom's Taxonomy):
{blooms_level}

Task Blueprint:
{task_blueprint}

Previously generated questions in this chapter:
{previous_questions}

Textbook Chapter Excerpts:
{chapter_excerpts}
"""


INCLUDE_CLARIFICATION_PROMPT = """
You will be given a set of problems and excerpts from a textbook chapter.

Requirements:
- Inspect the problem statement for any undefined notation, missing definitions, unstated assumptions, or missing information that could lead to ambiguity or multiple interpretations.
- If such issues are found, revise the problem by explicitly defining all notation and assumptions and by adding only the minimal necessary information to make the problem self-contained and unambiguous. 
- Do not introduce new assumptions beyond what is required for clarity.
- Do not change the difficulty, the core concept, or what the student/subject model is asked to do.
- Do not assume the student/subject model knows symbols or definitions unless they are standard and explicitly supported by the excerpts.

Output:
Return valid JSON only, in the SAME STRUCTURE as the input.

Candidate_problems:
{candidate_problems}

Textbook Chapter Excerpts:
{chapter_excerpts}
"""


REMOVE_REDUNDANT_INFO_PROMPT = """
You will be given a set of problems along with excerpts from a textbook chapter.
Your task is to edit each problem by removing any text that is irrelevant or redundant, while strictly preserving the original meaning, intent, and difficulty of the question.

Requirements:
- Remove redundant or non-essential wording that does not affect the semantics of the question.
- Do not add new information, rephrase technical content, or change the problem's requirements.

Output only the cleaned version of each problem in the same format.
Cadidate_problems:
{candidate_problems}
Textbook Chapter Excerpts:
{chapter_excerpts}
"""


REMOVE_SOURCE_INFO_PROMPT = """
You will be given a set of problems along with excerpts from a textbook chapter.
Your task is to edit each problem by removing any text that refers to the source material, while strictly preserving the original meaning, intent, and difficulty of the question.

Requirements:
- Remove phrases that explicitly reference the source material (e.g., “as discussed in the text,” “as described in the text,” “according to the chapter,” etc.).
- Do not add new information, rephrase technical content, or change the problem's requirements.

Output only the cleaned version of each problem in the same format.
Cadidate_problems:
{candidate_problems}
Textbook Chapter Excerpts:
{chapter_excerpts}
"""


SYSTEM_TASK_VERIFICATION_PROMPT = """
You are an expert educational evaluator acting as an impartial **LLM-as-a-judge** for multiple-choice question generation quality.

You will be given:
- A Domain,
- A Task Blueprint (describes what should be assessed, and implies a Bloom reasoning type among Remember/UnderstandApply/Analyze/Evaluate/Create, with difficulty = Easy, Medium, Hard),
- Textbook Chapter Material,
- A Candidate Output (the generator's JSON containing 5 MCQs with options and a labeled correct answer).

Your goal is to verify whether the candidate output strictly follows the required format and constraints, and whether each question is valid, aligned, and well-designed.

You MUST check the following aspects:

1) Blueprint Alignment
- Does each question's content and skill match the blueprint's intended concept(s), scope, and assessment goal?

2) Domain Consistency
- Are all questions clearly within the given domain and not drifting into unrelated domains?

3) Difficulty & Bloom Match
- Are the questions genuinely **Easy**, **Medium**, or **Hard** (multi-step reasoning, integration of sub-topics, non-trivial conceptual understanding)?  
- Do they match the Bloom reasoning type implied by the blueprint (Remember/Understand/Apply/Analyze/Evaluate/Create)?  
- If the Bloom level is not inferable, mark it as "Unclear" and treat it as a mismatch.

Bloom's Taxonomy definitions:
1. Remember - Recall or recognize facts, terms, and basic concepts. Example verbs: define, list, identify.
2. Understand - Explain ideas or concepts and interpret information in one's own words. Example verbs: summarize, describe, classify.
3. Apply - Use knowledge or methods in new but familiar situations. Example verbs: calculate, demonstrate, use, implement. 
4. Analyze - Break information into parts and examine relationships or patterns. Example verbs: differentiate, compare, examine, infer. 
5. Evaluate - Make judgments based on criteria and standards. Example verbs: justify, critique, assess, argue. 
6. Create - Combine elements to form a new pattern, structure, or product. Example verbs: design, compose, formulate, generate.

4) Multiple-Choice Integrity.
For each question:
- Exactly **four** options (A, B, C, D) are present and non-empty strings.
- Exactly **one** option is correct.
- The labeled correct answer (A/B/C/D) is consistent with the question content.
- Distractors are plausible (reflect common misconceptions or near-misses) yet unambiguously incorrect if the concept is understood.

5) Clarity & Well-Posedness.
- The question is clear, unambiguous, grammatically correct, and self-contained (defines variables, context, and what is being asked).. 
- It is solvable without missing information or hidden assumptions.

6) Constraint Compliance. 
- Avoid vague absolutes (“always,” “never,” “most likely”) unless explicitly required by the blueprint.
- If LaTeX appears, ensure escaped backslashes are used inside JSON strings (e.g., "$\\frac{1}{2}$").
- Must be based **solely** on the provided chapter material. If chapter material is missing/empty, set chapter_scope_verifiable = "No" and do not penalize for scope, but explicitly say “Cannot verify chapter-only constraint.”. 
- Must NOT explicitly refer to any section/theorem/lemma identifiers (e.g., “Section 2.1”, “Theorem 2.1.1”, “Lemma 2.1.1”).
- Must NOT directly refer to the text like "as described in the text", "as discussed in the text", "according to the text"

7) Output Format (Strict).
- STRICTLY ensure that the candidate output must be valid JSON and follow the expected structure:
  - Top-level key: "questions"
  - Exactly 5 entries in the list considering the input prompt
  - Each entry has:
    - "question" (string)
    - "options" (object with keys A-D)
    - "correct_answer" (one of "A","B","C","D")
- Any missing key, wrong key (e.g., "questio"), wrong count, duplicate keys, or invalid JSON → format failure.
- Can be easily parsed by standard JSON parsers.

------------------------------------
INPUT (you will receive these fields)
------------------------------------
1. DOMAIN
2. BLUEPRINT
3. CHAPTER_MATERIAL
4. CANDIDATE_OUTPUT


------------------------------------
RESPONSE FORMAT (Return JSON ONLY)
------------------------------------
{
  "json_format_valid": "<Yes/No>",
  "chapter_scope_verifiable": "<Yes/No>",
  "blueprint_alignment": "<Yes/No>",
  "domain_consistency": "<Yes/No>",
  "difficulty_bloom_match": "<Yes/No>",
  "mcq_integrity": "<Yes/No>",
  "clarity_well_posed": "<Yes/No>",
  "constraint_compliance": "<Yes/No>",
  "overall_verdict": "<Pass/Fail>",
  "explanation": "<2-4 sentences summarizing why it passes/fails>",
  "per_question_evaluation": [
    {
      "question_index": 1,
      "bloom_estimate": "<Remember/Understand/Apply/Analyze/Evaluate/Create/Unclear>",
      "difficulty_match": "<Yes/No>",
      "single_correct_answer_verified": "<Yes/No>",
      "distractors_plausible": "<Yes/No>",
      "main_issues": ["...","..."],
      "fix": "..."
    },
    ... ... ...
  ]
}

------------------------------------
DECISION RULE
------------------------------------
Set "overall_verdict" = "Pass" ONLY IF ALL of the following are "Yes":
- json_format_valid
- blueprint_alignment
- domain_consistency
- difficulty_bloom_match
- mcq_integrity
- clarity_well_posed
- constraint_compliance

If "json_format_valid" = "No", overall_verdict MUST be "Fail" regardless of other fields.

Now evaluate the Candidate Output using the inputs provided. Return ONLY the RESPONSE JSON.
"""


USER_TASK_VERIFICATION_PROMPT = """
Input Details:

Domain:
{domain}

Blueprint:
{blueprint}

Candidate Output:
{candidate_output}

Chapter Material:
{chapter_material}
"""


SYSTEM_TASK_REVISION_PROMPT = """
You are an expert educational scientist responsible for revising educational task(s) in a broad Domain (e.g., Mathematics, Finance).

INPUTS YOU WILL RECEIVE
1) previous_prompt: the original instruction that the candidate output MUST satisfy.
2) previous_candidate_output: the generated JSON to revise (may contain 1 or multiple tasks).
3) verifier_llm_feedback: feedback listing issues to fix.
4) chapter_material: textbook chapter excerpt that constrains scope and facts.

YOUR GOAL
Revise the previous_candidate_output to address verifier_llm_feedback while strictly adhering to previous_prompt and staying grounded in chapter_material.

REQUIREMENTS
- Output format: Return ONLY a single valid JSON object (no markdown, no commentary, no extra keys unless required by previous_prompt).
- Schema preservation: Preserve the exact JSON schema/keys/field names and overall structure found in previous_candidate_output unless previous_prompt explicitly requires a different schema.
- Minimal edits: Make the smallest changes necessary to fix issues.
- No invention / evidence gate: Do NOT add new concepts, definitions, formulas, tools, datasets, claims, or domain facts unless they are supported by chapter_material or explicitly required by previous_prompt.
- Keep identifiers stable: Do not change task IDs, question numbers, option labels, or required metadata fields unless the feedback requires it for correctness/format compliance.
- Quality checks: Ensure revised tasks are clear, unambiguous, solvable, and aligned with the domain and intended skill level (as implied by previous_prompt).
- The 5 problems must be meaningfully distinct, i.e., no near-duplicates in concept, skill tested, or solution strategy.
- Treat two questions as duplicates if they can be solved using the same underlying method or examining the same cencept.
- Do not generate questions that differ only in numbers, wording, or scenario context if the reasoning steps are the same.
- Before writing questions, list the 5 skills being covered. After writing them, verify that no two questions test the same skill or solution pattern. Regenerate if needed.

DO NOT OUTPUT YOUR PROCESS.
Return ONLY the revised JSON.
"""


USER_TASK_REVISION_PROMPT = """
Input Details:
Domain:
{domain}

Previous Prompt:
{previous_prompt}

Previous Candidate Output:
{previous_candidate_output}

Verifier LLM Feedback:
{verifier_llm_feedback}


Chapter Material:
{chapter_material}
"""


SYSTEM_TASK_REVISION_PROMPT_UNIQUE = """
You are an expert educational scientist responsible for revising educational task(s) in a broad Domain (e.g., Mathematics, Finance).

INPUTS YOU WILL RECEIVE
1) previous_prompt: the original instruction that the candidate output MUST satisfy.
2) previous_candidate_output: the generated JSON to revise (may contain 1 or multiple tasks).
3) verifier_llm_feedback: feedback listing issues to fix.
4) chapter_material: textbook chapter excerpt that constrains scope and facts.
5) You will be given a list of previously generated question stems for this chapter.

YOUR GOAL
Revise the previous_candidate_output to address verifier_llm_feedback while strictly adhering to previous_prompt and staying grounded in chapter_material.

REQUIREMENTS
- Output format: Return ONLY a single valid JSON object (no markdown, no commentary, no extra keys unless required by previous_prompt).
- Schema preservation: Preserve the exact JSON schema/keys/field names and overall structure found in previous_candidate_output unless previous_prompt explicitly requires a different schema.
- Minimal edits: Make the smallest changes necessary to fix issues.
- No invention / evidence gate: Do NOT add new concepts, definitions, formulas, tools, datasets, claims, or domain facts unless they are supported by chapter_material or explicitly required by previous_prompt.
- Keep identifiers stable: Do not change task IDs, question numbers, option labels, or required metadata fields unless the feedback requires it for correctness/format compliance.
- Quality checks: Ensure revised tasks are clear, unambiguous, solvable, and aligned with the domain and intended skill level (as implied by previous_prompt).
- The 5 problems must be meaningfully distinct, i.e., no near-duplicates in concept, skill tested, or solution strategy.
- Treat two questions as duplicates if they can be solved using the same underlying method or examining the same cencept.
- Do not generate questions that differ only in numbers, wording, or scenario context if the reasoning steps are the same.
- Before writing questions, list the 5 skills being covered. After writing them, verify that no two questions test the same skill or solution pattern. Regenerate if needed.
- You are provided a list of previously generated problems in the input as well.
- Your task is to generate new problems that are NOT near-duplicates of any prior problem.
- Treat a problem as a near-duplicate if it:
    - tests the same underlying concept or sub-skill, OR
    - relies on the same primary method of solution, OR
    - follows the same reasoning pattern or solution structure, even if the wording, numbers, scenario, or entities differ.
- Do NOT generate problems that could be solved using the same mental steps as any prior problem.

DO NOT OUTPUT YOUR PROCESS.
Return ONLY the revised JSON.
"""


USER_TASK_REVISION_PROMPT_UNIQUE = """
Input Details:
Domain:
{domain}

Previous Prompt:
{previous_prompt}

Previous Candidate Output:
{previous_candidate_output}

Verifier LLM Feedback:
{verifier_llm_feedback}

Previously generated questions in this chapter:
{previous_questions}

Chapter Material:
{chapter_material}
"""
