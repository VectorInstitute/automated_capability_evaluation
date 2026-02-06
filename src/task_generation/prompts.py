"""Prompts for task generation agents."""

# Here, SYSTEM_TASK_GENERATION_PROMPT was for the first 5 questions (Easy-Remember)
# when the questions list were empty.
SYSTEM_TASK_GENERATION_PROMPT_INST = """
You are an expert educational scientist responsible for generating high-quality multiple-choice tasks.

Given a task blueprint that describes what the question should assess, difficulty level and reasoning type based on bloom's taxonomy, your goal is to write 5 (five) questions that:

1. Accurately reflects the blueprint.
2. Includes exactly four answer options.
3. Has ONLY one correct answer.
4. Uses clear and unambiguous wording.
5. Ensures that incorrect options (distractors) are plausible but clearly wrong when the concept is understood correctly.

**Definition of Distractor**: Distractors are the incorrect answer options in a multiple-choice question that are designed to appear plausible and relevant to learners who have partial understanding or common misconceptions, but are unambiguously incorrect to someone who understands the underlying concept correctly.

Avoid using vague words like "always," "never," or "most likely" unless the blueprint specifies such nuance.
If mathematical notation is included, ensure all LaTeX symbols use escaped backslashes (e.g., "$\\frac{1}{2}$").

Requirements:
- Write exactly 5 (five) well-formed multiple-choice questions.
- Include four options (A-D).
- Only one option should be correct.
- Ensure all distractors (incorrect options) are realistic and relevant to the topic.
- The questions should be consistent with the intended content, difficulty, and reasoning type implied by the blueprint.
- The 5 questions must be meaningfully distinct, i.e., no near-duplicates in concept, skill tested, or solution strategy. Treat two questions as near-duplicate if they can be solved using the same underlying method or examining the same cencept.
- Do not generate questions that differ only in numbers, wording, or scenario context if the reasoning steps are the same.
- Before writing questions, list the 5 skills being covered. After writing them, verify that no two questions test the same skill or solution pattern. Regenerate if needed.
- The student/subject model that will solve these questions does not have access to the material in the textbook, so make sure the context and the notation are clear in each question statement and the problem is self-contained.
- You are provided with material from a textbook chapter. Design questions that are based solely on the concepts and methods introduced in this chapter. The questions should be suitable for assessing a learner's understanding of the material.
- Do not explicitly refer to any section, theorem or lemma (e.g., section 2.1, theorem 2.1.1, lemma 2.1.1) in the question.
"""

OUT_FORMAT_EXAMPLE = """
Respond EXACTLY in the following format, including the JSON start and end markers:
{
    "questions": [
        {
            "question": "<question 1 text>",
            "options": {
                "A": "<Option A>",
                "B": "<Option B>",
                "C": "<Option C>",
                "D": "<Option D>"
            },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<question 2 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<question 3 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<question 4 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        },
        {
            "question": "<question 5 text>",
            "options": { ... },
            "correct_answer": "<A/B/C/D>"
        }
    ]
}
"""

SYSTEM_TASK_GENERATION_PROMPT = (
    SYSTEM_TASK_GENERATION_PROMPT_INST + "\n" + OUT_FORMAT_EXAMPLE
)

USER_TASK_GENERATION_PROMPT = """
Input Details:
Difficulty level:
{difficulty}

Reasoning type (Bloom's Taxonomy):
{blooms_level}

Task Blueprint:
{task_blueprint}

Textbook Chapter Excerpts:
{chapter_excerpts}
"""

# SYSTEM_TASK_GENERATION_PROMPT_UNIQUE is used
# for all the later combinations.
SYSTEM_TASK_GENERATION_PROMPT_EXTRA_INST = """
- You are provided a list of previously generated questions in the input as well.
- Your task is to generate new questions that are NOT near-duplicates of any prior question.
- Treat a question as a near-duplicate if it:
    - tests the same underlying concept or sub-skill, OR
    - relies on the same primary method of solution, OR
    - follows the same reasoning pattern or solution structure, even if the wording, numbers, scenario, or entities differ.
- Do NOT generate questions that could be solved using the same mental steps as any prior question.
"""

SYSTEM_TASK_GENERATION_PROMPT_UNIQUE = (
    SYSTEM_TASK_GENERATION_PROMPT_INST
    + "\n"
    + SYSTEM_TASK_GENERATION_PROMPT_EXTRA_INST
    + "\n"
    + OUT_FORMAT_EXAMPLE
)

USER_TASK_GENERATION_PROMPT_UNIQUE = """
Input Details:
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
You will be given a question in the input.

Requirements:
- Inspect the question statement for any undefined notation, missing definitions, unstated assumptions, or missing information that could lead to ambiguity or multiple interpretations.
- If such issues are found, update the question by explicitly defining all notation and assumptions and by adding only the minimal necessary information to make the question self-contained and unambiguous.
- Do not introduce new assumptions beyond what is required for clarity.
- Do not change the difficulty, the core concept, or what the student/subject model is asked to do.
- Do not assume the student/subject model knows symbols or definitions unless they are standard and widely known.

Output:
Return valid JSON only, in the SAME STRUCTURE as the input.

Input Details:
Candidate_question:
{candidate_question}
"""


REMOVE_REDUNDANT_INFO_PROMPT = """
You will be given a question in the input.
Your task is to edit the question by removing any text that is irrelevant or redundant, while strictly preserving the original meaning, intent, and difficulty of the question.

Requirements:
- Remove redundant or non-essential wording that does not affect the semantics of the question.
- Do not add new information, rephrase technical content, or change the question's requirements.

Output:
Return valid JSON only, in the SAME STRUCTURE as the input.

Input Details:
Candidate_question:
{candidate_question}
"""


REMOVE_SOURCE_INFO_PROMPT = """
You will be given a question in the input.
Your task is to edit the question by removing any text that refers to the source material, while strictly preserving the original meaning, intent, and difficulty of the question.

Requirements:
- Remove phrases that explicitly reference the source material (e.g., “as discussed in the text,” “as described in the text,” “according to the chapter,”, "as discussed in Section X, Y" etc.).
- Do not add new information, rephrase technical content, or change the question's requirements.

Output:
Return valid JSON only, in the SAME STRUCTURE as the input.

Input Details:
Candidate_question:
{candidate_question}
"""


SOUNDNESS_CHECK_PROMPT = """
You will be given a question in the input.
Your task is to edit the question only as needed to make it sound and well-posed, while strictly preserving the original meaning, intent, and difficulty of the question.

Definition of soundness:
A question is sound if it (1) makes logical sense as stated, (2) is grammatically and semantically clear, (3) has no internal contradictions, (4) is phrased correctly, and 5) is complete.

Requirements:
- Edit any wording that makes the question unclear, incorrect, ill-formed, or logically inconsistent.
- Edit any grammatical errors that affect clarity.
- Edit the question if it is incomplete or missing essential information needed to understand what is being asked.
- Do NOT change what is being asked, do NOT change the difficulty, and do NOT introduce new constraints beyond what is necessary for soundness.
- If the question is already sound, return it UNCHANGED.

Output:
Return valid JSON only, in the SAME STRUCTURE as the input.

Input Details:
Candidate_question:
{candidate_question}
"""


MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT = """
You will be given a multiple-choice question in the input.
Your task is to solve it from scratch and verify the answer key.

Multiple-Choice Integrity definition:
A multiple-choice question has integrity if (1) exactly one option is correct, (2) the provided correct answer label matches that unique correct option, and (3) all other options are definitively incorrect.

Requirements:
- Solve the question from scratch using only the information in the question.
- Verify whether the provided correct_answer label matches the uniquely correct option.
- Check each option (A/B/C/D) to confirm that no other option could also be correct under reasonable interpretation.
- If multiple options could be correct OR none is correct:
  1) Revise ONLY the options (do NOT change the question) so that exactly one option is correct.
     - If multiple options are correct: edit options so only one remains correct.
     - If no option is correct: edit one option so it becomes correct.
  2) Update the correct_answer accordingly.
- Do NOT change the question text or difficulty. Keep the question exactly as-is.

Return valid JSON only, in the SAME STRUCTURE as the input.
- If no revision is needed: return the input JSON unchanged.
- If revision is needed: return the same JSON structure, with updated "options" and "correct_answer" only.

Input Details:
Candidate Question:
{candidate_question}
"""


SYSTEM_TASK_VERIFICATION_PROMPT = """
You are an expert educational evaluator acting as an impartial **LLM-as-a-judge** for multiple-choice question generation quality.

You will be given:
- A Candidate Output (a response from question designer agent, in JSON format containing a MCQ with options and a labeled correct answer).

Your goal is to verify whether the candidate output strictly follows the required format and constraints, and whether each question is valid, self-contained, and well-designed.

You MUST check the following aspects:

1) Multiple-Choice Integrity.
For the question:
- Exactly **four** options (A, B, C, D) are present and non-empty strings.
- Distractors are plausible (reflect common misconceptions or near-misses) yet unambiguously incorrect if the concept is understood.

2) Clarity & Well-Posedness.
- The question is clear, unambiguous, grammatically correct, and self-contained (defines variables, context, and what is being asked).
- It is solvable without missing information or hidden assumptions.

3) Constraint Compliance.
- Avoid vague absolutes (“always,” “never,” “most likely”) unless explicitly required by the blueprint.
- If LaTeX appears, ensure escaped backslashes are used inside JSON strings (e.g., "$\\frac{1}{2}$").
- Must NOT explicitly refer to any section/theorem/lemma identifiers (e.g., “Section 2.1”, “Theorem 2.1.1”, “Lemma 2.1.1”).
- Must NOT directly refer to the text like "as described in the text", "as discussed in the text", "according to the text"

4) Output Format (Strict).
- STRICTLY ensure that the candidate output must be valid JSON and follow the expected structure:
  - Top-level key: "questions"
  - Exactly one entry in the list considering the input prompt
  - The single entry has:
    - "question" (string)
    - "options" (object with keys A-D)
    - "correct_answer" (one of "A","B","C","D")
- Any missing key, wrong key (e.g., "questio"), wrong count, duplicate keys, or invalid JSON should result in format failure.
- VERY IMPORTANT: Ensure that the question can be easily parsed by standard JSON parsers.

Respond EXACTLY in the following format, including the JSON start and end markers:
{
  "json_format_valid": "<Yes/No>",
  "mcq_integrity": "<Yes/No>",
  "clarity_well_posed": "<Yes/No>",
  "constraint_compliance": "<Yes/No>",
  "overall_verdict": "<Pass/Fail>",
  "explanation": "<2-4 sentences summarizing why it passes/fails>",
  "question_evaluation": {
        "distractors_plausible": "<Yes/No>",
        "main_issues": ["...","..."],
        "fix": "..."
    }
}

------------------------------------
DECISION RULE
------------------------------------
Set "overall_verdict" = "Pass" ONLY IF ALL of the following are "Yes":
- json_format_valid
- mcq_integrity
- clarity_well_posed
- constraint_compliance

If "json_format_valid" = "No", overall_verdict MUST be "Fail" regardless of other fields.

VERY IMPORTANT: Now evaluate the Candidate Output using the inputs provided. Return ONLY the RESPONSE JSON.
"""


USER_TASK_VERIFICATION_PROMPT = """
Input Details:

Candidate Output:
{candidate_output}
"""


SYSTEM_TASK_REVISION_PROMPT = """
You are an expert educational scientist responsible for revising educational task(s) in a broad Domain (e.g., Mathematics, Finance).

INPUTS YOU WILL RECEIVE
1) `Previous Candidate Output`: the generated question in JSON format to revise.
2) `Verifier LLM Feedback`: feedback listing issues to fix.
3) chapter_material: textbook chapter excerpt that constrains scope and facts.
4) You will be given a list of previously generated question stems for this chapter.

YOUR GOAL
Revise the `Previous Candidate Output` to address `Verifier LLM Feedback` and staying grounded in chapter_material.

REQUIREMENTS
- Output format: Return ONLY a single valid JSON object (no markdown, no commentary, no extra keys).
- Schema preservation: Preserve the exact JSON schema/keys/field names and overall structure found in `Previous Candidate Output`.
- Minimal edits: Make the smallest changes necessary to fix issues.
- No invention / evidence gate: Do NOT add new concepts, definitions, formulas, tools, datasets, claims, or domain facts unless they are supported by chapter_material.
- Quality checks: Ensure revised tasks are clear, unambiguous, solvable, and intended skill level.
- The question must be meaningfully distinct, i.e., no near-duplicates in concept, skill tested, or solution strategy.
- Compare the question with previously generated questions. Treat two questions as near-duplicates if they can be solved using the same underlying method or examining the same concept.
- Do not update the question such that it differ only in numbers, wording, or scenario context if the reasoning steps are the same.
- You are provided a list of previously generated questions in the input as well.
- Your task is to update the question that are NOT duplicate or near-duplicate of any prior question.
- Treat a question as a near-duplicate if it:
    - tests the same underlying concept or sub-skill, OR
    - relies on the same primary method of solution, OR
    - follows the same reasoning pattern or solution structure, even if the wording, numbers, scenario, or entities differ.
- Do NOT update the question such that it could be solved using the same mental steps as any prior question.

DO NOT OUTPUT YOUR PROCESS.
Return ONLY the revised JSON.
"""


USER_TASK_REVISION_PROMPT = """
Input Details:
Previous Candidate Output:
{previous_candidate_output}

Verifier LLM Feedback:
{verifier_llm_feedback}

Previously generated questions in this chapter:
{previous_questions}

Chapter Material:
{chapter_material}
"""
