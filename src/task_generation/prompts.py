"""Prompts for task generation agents."""

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

# SYSTEM Prompt for extracting structured knowledge from chapter excerpts.
SYSTEM_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT = """
You are extracting detailed structured knowledge from a textbook chapter excerpt.

PHASE 1 — Extract and Structure Knowledge
From the external source:
1) Identify:
- Core concepts
- Definitions
- Theorems or rules
- Procedures
- Algorithms
- Derived relationships
- Subtle constraints or caveats

2) Construct (internally) a dependency graph of how concepts rely on each other.

OUTPUT RULES:
- Output VALID JSON ONLY.
- Do NOT include any markdown fences.
- Do NOT include any commentary outside JSON.

Respond EXACTLY in the following format, including the JSON start and end markers:
{
  "core_concepts": [{"name": "...", "description": "..."}],
  "definitions": [{"term": "...", "definition": "..."}],
  "theorems_or_rules": [{"name": "...", "statement": "...", "conditions": ["..."], "implications": ["..."]}],
  "procedures": [{"name": "...", "steps": ["..."], "inputs": ["..."], "outputs": ["..."], "common_pitfalls": ["..."]}],
  "algorithms": [{"name": "...", "goal": "...", "steps": ["..."], "constraints": ["..."]}],
  "derived_relationships": [{"relationship": "...", "explanation": "..."}],
  "subtle_constraints_or_caveats": ["..."],
  "dependency_graph": {
    "nodes": [{"id": "C1", "label": "...", "type": "concept|definition|rule|procedure|algorithm"}],
    "edges": [{"from": "C1", "to": "C2", "relation": "requires|depends_on|uses"}]
  }
}
"""

# Prompt for the user to provide chapter excerpts for knowledge extraction.
USER_CHAPTER_KNOWLEDGE_SUMMARY_PROMPT = """
Input Details:
Chapter Material:
{chapter_excerpts}
"""


# SYSTEM Prompt for generating a question and solution graph from chapter knowledge.
SYSTEM_GRAPH_TASK_GENERATION_PROMPT_INST = """
You are an expert problem designer and subject-matter specialist.
Your task is to design an extremely challenging multiple-choice problem grounded in the material provided in the external source below (e.g., a book chapter). The problem must be correct, unambiguous, and verifiably solvable using only knowledge from that source.
The problem must be difficult enough that even a strong model might fail to solve it reliably. The problem must be unique and different from the provided list of previously designed problems from the same source.

Method You Must Follow (Solution-Trace-Driven Design):
You must follow a solution-first design approach, consisting of the following phases:

Phase 1 — FIRST, in an incremental fashion construct a hard solution deep reasoning graph with some nodes fan-in/fan-out > 1. To make the problem challenging, if possible design graphs with width > 1. The reasoning graph, G = (V, E) which is a directed acyclic graph consists of nodes (V), where each node V[i] represents an intermediate solution, and an edge E[i, j] represents an operation by applying a core concept, definition, theorem, algorithm, etc. (the above knowledge list) from the external source content to node V[i] to obtain V[j].

Before writing the problem in text, explicitly construct a solution trace graph that:
1. Requires multiple non-trivial reasoning steps.
2. Combines two or more distinct concepts or results from different parts of the provided chapter.
3. Includes at least one of the following:
 - A non-obvious dependency
 - A hidden constraint
 - A delayed-use intermediate result
 - A reasoning-mode shift (e.g., conceptual → algebraic → conceptual)
4. Has at least one plausible but incorrect alternative reasoning path.
5. Cannot be solved by a single direct formula lookup.

The trace must be logically correct and lead to a unique final answer.
You must ensure:
 - Every step is justified using only the provided source.
 - The trace is internally consistent.
 - No external knowledge beyond the provided source is required.

Phase 2 — Design the Problem From the Trace
Now construct a self-contained multiple-choice problem such that:
1. Solving the problem correctly requires following the designed trace (or an equally complex equivalent).
2. The problem statement:
 - Does NOT reference any section numbers, subsections, example numbers, or explicit mentions of the chapter structure.
 - Does NOT say “according to the chapter” or similar phrases.
 - Is fully self-contained.
 - Defines all necessary notation.
 - Includes all required assumptions.

3. The problem cannot be solved by trivial pattern matching.
4. The reasoning chain is necessary for correctness.

Phase 3 — Construct High-Quality Answer Choices
The problem must have exactly 5 answer options:
A.
B.
C.
D.
E. None of the above.

Requirements for answer choices:
1. Exactly one option must be correct.
2. Distractors must be:
 - Derived from realistic but incorrect reasoning paths.
 - Based on common misunderstandings of the material.
 - Close in structure or value to the correct answer.
 - Not trivially eliminable.

3. “None of the above” must be a viable option (i.e., the other options should not trivially rule it out).
4. Avoid obviously absurd or dimensionally inconsistent distractors.
5. Do not include meta-commentary in the options.

Phase 4 — Internal Verification
Before outputting:
1. Independently verify the solution step-by-step.
2. Check that:
 - The problem is unambiguous.
 - The answer is uniquely correct.
 - No shortcut makes the problem trivial.
 - The reasoning genuinely requires multiple structured steps.
3. Ensure the problem depends only on the provided source.

Required Output:
Output must contain the following sections in order:
1. The constructed solution graph specifying nodes and edges
2. The Problem (Provide the complete, self-contained multiple-choice problem here.)
3. Answer Choices
    A.
    B.
    C.
    D.
    E. None of the above.
4. Correct Answer (Provide only the letter.)
5. Complete Solution
Provide a fully rigorous, step-by-step solution that follows the intended reasoning trace.
Do NOT reference any section numbers or structural elements of the source in the solution.

Difficulty Requirements
The problem must:
- Require at least 4–6 logically connected reasoning steps.
- Combine multiple concepts.
- Be resistant to shallow pattern matching.
- Be non-trivial even for a strong model.

Do not include references to the content (section number, theorem number, examples, etc.) in the problem. Each problem must be self-contained.
"""

# Prompt for task output format instruction with an example.
OUT_FORMAT_EXAMPLE_GRAPH_TASK = """
IMPORTANT OUTPUT REQUIREMENT:
Respond EXACTLY in the following format, including the JSON start and end markers:
{
  "solution_graph": {
    "nodes": [{"id": "V1", "content": "..."}, {"id": "V2", "content": "..."}],
    "edges": [{"from": "V1", "to": "V2", "operation": "..."}]
  },
  "question": "<self-contained MCQ stem>",
  "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "None of the above" },
  "correct_answer": "<one of: A|B|C|D|E>",
  "complete_solution": "<rigorous step-by-step solution text>"
}
"""

# Final SYSTEM prompt for task generation (initial stage)
SYSTEM_GRAPH_TASK_GENERATION_PROMPT = SYSTEM_GRAPH_TASK_GENERATION_PROMPT_INST + "\n" + OUT_FORMAT_EXAMPLE_GRAPH_TASK

# Final USER prompt for task generation (initial stage)
USER_GRAPH_TASK_GENERATION_PROMPT = """
Input Details:
Textbook Chapter Excerpts:
{chapter_excerpts}

Textbook Chapter Knowledge Summary:
{chapter_knowledge_text}
"""

# Final SYSTEM prompt for task generation with uniqueness and anti-duplication.
SYSTEM_GRAPH_TASK_GENERATION_PROMPT_UNIQUE = SYSTEM_GRAPH_TASK_GENERATION_PROMPT + "\n" + SYSTEM_TASK_GENERATION_PROMPT_EXTRA_INST + "\n" + OUT_FORMAT_EXAMPLE_GRAPH_TASK

# Final USER prompt for task generation with uniqueness and anti-duplication.
USER_GRAPH_TASK_GENERATION_PROMPT_UNIQUE = """
Input Details:
Previously generated questions in this chapter:
{previous_questions}

Textbook Chapter Excerpts:
{chapter_excerpts}

Textbook Chapter Knowledge Summary:
{chapter_knowledge_text}
"""


# Prompt for including clarification information.
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

# SYSTEM Prompt for verifying and revising MCQ integrity based on solution trace and chapter knowledge.  # noqa: W505
SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT = """
You will be given ONE multiple-choice question in JSON format.

You are also given:
- Textbook chapter excerpts (ground truth).
- A structured chapter knowledge summary.
- A solution trace (solution_graph)
- A complete solution text (step by step solution) that follows the trace.

Your job is NOT to solve the problem independently from scratch.
Instead, you must validate and (if needed) REPAIR the PROVIDED solution trace so that it is:
- internally consistent,
- grounded ONLY in the provided chapter material,
- and matches exactly ONE answer option A–E.

You MUST enforce:
1) Trace validity:
   - Each step/node follows from prior steps/nodes and the described operation.
   - No missing jumps or unjustified claims.
   - The final answer implied by the trace is explicit.

2) Option consistency:
   - Exactly ONE option among A–E matches the trace’s final answer.
   - correct_answer must point to that uniquely matching option.
   - All other options must NOT match the trace final answer.

Option E rule:
- Option E MUST remain EXACTLY: "None of the above".
- If any of A–D matches the trace final answer, then E must be incorrect.
- If none of A–D matches, then E must be the ONLY correct option.

-------------------------
WHAT TO DO IF ISSUES EXIST
-------------------------

A) If the solution trace is invalid / inconsistent / incomplete:
   - REPAIR the trace so it becomes valid and grounded in the chapter excerpts.
   - Update "question" ONLY if necessary to align with the repaired trace and to keep the problem self-contained.
     - Do NOT reference section numbers or “as discussed in the chapter”.
     - Do NOT change the intended difficulty downward.
     - Keep the same “type” of problem; do not switch to a totally different concept.
   - Update "complete_solution" to match the repaired trace.
   - Update "options" and "correct_answer" so exactly one option matches the repaired final answer.

B) If the trace is valid BUT:
   - correct_answer does not point to the matching option, OR
   - multiple options match, OR
   - no option matches,
   then revise ONLY:
   - "options" (if needed), and/or
   - "correct_answer"
   Do NOT change: "question", "solution_graph", "solution_steps", "complete_solution".

-------------------------
REVISION POLICY
-------------------------
- Prefer minimal edits.
- Keep distractors plausible (realistic incorrect reasoning paths).
- Do NOT introduce any facts, definitions, or formulas not supported by the chapter excerpts/knowledge.

DO NOT OUTPUT YOUR PROCESS.

IMPORTANT OUTPUT REQUIREMENT. Follow this EXACTLY:
Respond EXACTLY in the following format, including the JSON start and end markers:
{
  "solution_graph": {
    "nodes": [{"id": "V1", "content": "..."}, {"id": "V2", "content": "..."}],
    "edges": [{"from": "V1", "to": "V2", "operation": "..."}]
  },
  "question": "<self-contained MCQ stem>",
  "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "None of the above" },
  "correct_answer": "<one of: A|B|C|D|E>",
  "complete_solution": "<rigorous step-by-step solution text>"
}
"""

# USER Prompt for verifying and revising MCQ integrity based on solution trace and chapter knowledge.  # noqa: W505
USER_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT = """
Input Details:
Textbook Chapter Excerpts:
{chapter_excerpts}

Textbook Chapter Knowledge Summary:
{chapter_knowledge_text}

Candidate Question JSON:
{candidate_question}

Solution Trace:
{solution_trace}

Full Solution JSON:
{solution_full}
"""


# Prompt for removing redundant information.
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

# Prompt for removing source information.
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

# Prompt for soundness check and repair.
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

# SYSTEM Prompt for MCQ integrity check (format, clarity, constraints), JSON validity check, and Latex format check.  # noqa: W505
SYSTEM_TASK_VERIFICATION_PROMPT = """
You are an expert educational evaluator acting as an impartial **LLM-as-a-judge** for multiple-choice question generation quality.

You will be given:
- A Candidate Output (a response from question designer agent, in JSON format containing a MCQ with options and a labeled correct answer).

Your goal is to verify whether the candidate output strictly follows the required format and constraints, and whether each question is valid, self-contained, and well-designed.

You MUST check the following aspects:

1) Multiple-Choice Integrity.
For the question:
- Exactly **five** options (A, B, C, D, E) are present and non-empty strings.
- Option "E" is labeled as "None of the above".
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
  - Top-level is a single JSON object (NOT a list)
  - It has:
    - "question" (string)
    - "options" (object with keys A-E)
    - "correct_answer" (one of "A","B","C","D","E")
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

# USER Prompt for verification step.
USER_TASK_VERIFICATION_PROMPT = """
Input Details:

Candidate Output:
{candidate_output}
"""

# SYSTEM Prompt for MCQ repair based on verifier feedback (format, correctness).
SYSTEM_TASK_REVISION_PROMPT_MCQ_FIX = """
You are an expert educational scientist and MCQ quality auditor.

INPUTS YOU WILL RECEIVE
1) Previous Candidate Output: a single MCQ in JSON (may be imperfect).
2) Verifier LLM Feedback: issues to fix (MCQ correctness, integrity, ambiguity, constraints, etc.).
3) chapter_material: textbook chapter excerpt that constrains scope and facts.
4) chapter_knowledge_text: structured knowledge summary of the chapter.
5) solution_trace: the reasoning trace/solution graph associated with the question.
6) previous_questions: a list of previously generated questions for this chapter (anti-dup).

YOUR GOAL
Repair the MCQ so it is fully consistent with the provided solution_trace and grounded ONLY in chapter_material and chapter_knowledge_text.
The solution_trace is the strongest constraint: the revised question must be solvable via the trace, and the correct answer option must match the trace’s final answer.

WHAT YOU ARE ALLOWED TO CHANGE
- You MAY edit: "question", "options", and/or "correct_answer" as needed to match the trace and satisfy the verifier feedback.
- You MAY edit wording to make the question self-contained and unambiguous.
- You MAY minimally adjust distractors to remain plausible but incorrect.

WHAT YOU MUST NOT CHANGE
- Do NOT introduce any facts, definitions, formulas, or claims not supported by chapter_material / chapter_knowledge_text.
- Do NOT change the underlying concept/skill tested to a different one.
- Do NOT produce a near-duplicate of any item in previous_questions (same concept + same solution method/reasoning pattern).

STRICT OUTPUT RULES
- Output ONLY a single valid JSON object (no markdown, no commentary).
- Preserve the JSON key names exactly: "question", "options", "correct_answer".
- "options" must contain exactly five keys: "A","B","C","D","E".
- "E" MUST be exactly: "None of the above".
- "correct_answer" must be exactly one of: "A","B","C","D","E".
- If the previous JSON includes extra fields (e.g., solution_graph, complete_solution), preserve them unless the verifier feedback requires updating them for consistency with the repaired MCQ.

DO NOT OUTPUT YOUR PROCESS.
Return ONLY the revised JSON object.

Respond EXACTLY in the following format, including the JSON start and end markers:
{
  "question": "<self-contained MCQ stem>",
  "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "None of the above" },
  "correct_answer": "<one of: A|B|C|D|E>"
}
"""

# USER Prompt for MCQ repair based on verifier feedback (format, correctness).
USER_TASK_REVISION_PROMPT_MCQ_FIX = """
Input Details:
Previous Candidate Output:
{previous_candidate_output}

Verifier LLM Feedback:
{verifier_llm_feedback}

Previously generated questions in this chapter:
{previous_questions}

Chapter Material:
{chapter_material}

Chapter Knowledge Summary:
{chapter_knowledge_text}

Solution Trace:
{solution_trace}
"""

# SYSTEM Prompt for JSON format repair (if the candidate output is not valid JSON).
SYSTEM_TASK_REVISION_PROMPT_JSON_ONLY = """
You are a JSON repair tool.

INPUTS YOU WILL RECEIVE
1) Previous Candidate Output: an MCQ that may be malformed JSON (or valid JSON but wrong formatting).
2) Verifier LLM Feedback: indicates formatting/JSON issues.
3) The original MCQ content must be preserved exactly.

YOUR GOAL
Fix ONLY the JSON formatting so that the output is valid JSON and can be parsed by a standard JSON parser.

WHAT YOU MUST DO
- Produce a single valid JSON object that matches the intended schema.
- Preserve the question text, all option texts, and the correct answer EXACTLY as they appear in the input (character-for-character), except for changes strictly required to make it valid JSON (e.g., escaping quotes, backslashes, newlines).

WHAT YOU MUST NOT DO
- Do NOT change the meaning, wording, punctuation, capitalization, numbers, symbols, or spacing of any content fields (question/options/correct_answer).
- Do NOT add, remove, or rename keys beyond what is required to restore valid JSON.
- Do NOT “improve” writing, grammar, clarity, options, or correctness. This task is format repair ONLY.

REQUIRED OUTPUT SCHEMA
Output ONLY a single valid JSON object (no markdown, no commentary) with:
- "question": string
- "options": object with keys "A","B","C","D","E" (each a string)
- "correct_answer": one of "A","B","C","D","E"

If these keys exist in the input, preserve them. If the input contains additional keys, preserve them exactly as-is as well (only fixing JSON syntax/escaping).

DO NOT OUTPUT YOUR PROCESS.
Return ONLY the repaired JSON.

Respond EXACTLY in the following format, including the JSON start and end markers:
{
  "question": "<self-contained MCQ stem>",
  "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "None of the above" },
  "correct_answer": "<one of: A|B|C|D|E>"
}
"""

# USER Prompt for JSON format repair (if the candidate output is not valid JSON).
USER_TASK_REVISION_PROMPT_JSON_ONLY = """
Input Details:

Previous Candidate Output:
{previous_candidate_output}

Verifier LLM Feedback:
{verifier_llm_feedback}
"""
