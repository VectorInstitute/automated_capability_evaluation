"""Centralized prompts for all Wikipedia-related scripts."""

# System prompts
SYSTEM_PROMPT_MATH_CAPABILITIES = "You are an expert in mathematical capabilities."
SYSTEM_PROMPT_MATH_TAXONOMIST = (
    "You are an expert mathematical taxonomist. "
    "Your task is to map a single math problem to the most appropriate area from a provided list. "
    "Output must be EXACTLY one of the given area names, or 'none' if no reasonable match exists. "
    "Do not include explanations or extra words."
)
SYSTEM_PROMPT_CAPABILITY_EVALUATION = """You are an expert in mathematics and capability evaluation. Your task is to create concise, informative summaries of mathematical concepts and capabilities.

Given a detailed description of a mathematical concept or capability, provide a clear, concise summary that captures the essential meaning and scope. The summary should be:
- Informative and accurate
- Concise ONLY ONE SENTENCE
- Written in clear, accessible language
- Focused on the core concept and its applications

Examples of good summaries:
- "Capability focusing on field theory, including solving problems related to field extensions, minimal polynomials, and degrees of extensions."
- "Capability that involves solving problems in ring theory including identification of ring properties and operations, testing the structure of rings."
- "Capability that asks the model to simplify algebraic expressions by reducing them to their simplest form. Involves collecting like terms and basic algebraic manipulations."

Provide only the summary, without any additional commentary or formatting."""
SYSTEM_PROMPT_CATEGORIZATION = """You are an expert in mathematics and capability evaluation. Your task is to categorize mathematical concepts and capabilities into one of 10 predefined mathematical areas.

Given a description of a mathematical concept or capability, determine which of the following 10 mathematical areas it best belongs to:

1. Algebra and Functions
2. Arithmetic and Number Theory
3. Calculus and Analysis
4. Differential Equations and Dynamical Systems
5. Discrete Mathematics and Combinatorics
6. Geometry and Spatial Reasoning
7. Linear Algebra and Matrix Theory
8. Mathematical Logic and Set Theory
9. Mathematical Modeling and Applications
10. Probability and Statistics

Return ONLY the exact area name from the list above, nothing else."""


# User prompts - functions that generate user prompts
def get_wikipedia_to_generated_prompt(
    wikipedia_cap_name: str, wikipedia_cap_description: str, capabilities_list: str
) -> str:
    """Generate prompt for matching Wikipedia capability to generated capabilities."""
    return f"""You are an expert in mathematical capabilities. Determine which generated capability best matches the given Wikipedia capability.

Wikipedia Capability:
Name: {wikipedia_cap_name}
Description: {wikipedia_cap_description}

Available Generated Capabilities:
{capabilities_list}

Instructions:
- Compare the Wikipedia capability with each available capability.
- Return the exact capability name if ANY of the following is true:
  * The Wikipedia capability and the available capability describe the same concept, OR
  * The Wikipedia capability is a SUBSET/PART of the available capability (i.e., the available capability includes the Wikipedia capability as one of its components or subskills), OR
  * The available capability is a broader superset that clearly contains the Wikipedia capability.
- Prefer the most specific matching capability when multiple candidates qualify.
- Return "none" only if no capability clearly contains or equals the Wikipedia capability.

Answer with only the capability name or "none":"""


def get_generated_to_wikipedia_prompt(
    generated_cap_name: str, generated_cap_description: str, capabilities_list: str
) -> str:
    """Generate prompt for matching generated capability to Wikipedia capabilities."""
    return f"""You are an expert in mathematical capabilities. Find the Wikipedia capability that most closely matches the generated capability.

Generated Capability:
Name: {generated_cap_name}
Description: {generated_cap_description}

Available Wikipedia Capabilities:
{capabilities_list}

Instructions:
- Compare the generated capability with each available Wikipedia capability.
- Return the exact Wikipedia capability name if ANY of the following is true:
  * The generated capability and the Wikipedia capability describe the same concept, OR
  * The generated capability is a SUBSET/PART of the Wikipedia capability (i.e., the Wikipedia capability includes the generated capability as one of its components or subskills), OR
  * The Wikipedia capability is a broader superset that clearly contains the generated capability.
- Prefer the most specific matching capability when multiple candidates qualify.
- Return "none" only if no Wikipedia capability clearly contains or equals the generated capability.

Answer with only the Wikipedia capability name or "none":"""


def get_area_categorization_prompt(area_bullets: str, question: str) -> str:
    """Generate prompt for categorizing a question into a mathematical area."""
    return f"""Available mathematical areas (choose exactly one):
{area_bullets}

Problem:
{question}

Instructions:
- Return ONLY the exact area name from the list above
- Prefer the closest match even if imperfect; avoid 'none' unless clearly unrelated
- Do not add punctuation or extra text

Answer:"""


def get_capability_summary_prompt(description: str) -> str:
    """Generate prompt for summarizing a mathematical capability."""
    return f"Please provide a concise summary of this mathematical concept:\n\n{description}"


def get_capability_categorization_prompt(description: str) -> str:
    """Generate prompt for categorizing a mathematical capability."""
    return f"Please categorize this mathematical concept into one of the 10 areas listed above:\n\n{description}"
