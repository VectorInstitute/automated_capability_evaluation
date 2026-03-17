"""Constants for diverse task generation."""

BLOOMS_TAXONOMY = {
    "Remember": {
        "description": "Recall or recognize facts, terms, and basic concepts. Example verbs: define, list, identify."
    },
    "Understand": {
        "description": "Explain ideas or concepts and interpret information in one's own words. Example verbs: summarize, describe, classify."
    },
    "Apply": {
        "description": "Use knowledge or methods in new but familiar situations. Example verbs: calculate, demonstrate, use, implement."
    },
    "Analyze": {
        "description": "Break information into parts and examine relationships or patterns. Example verbs: differentiate, compare, examine, infer."
    },
    "Evaluate": {
        "description": "Make judgments based on criteria and standards. Example verbs: justify, critique, assess, argue."
    },
    "Create": {
        "description": "Combine elements to form a new pattern, structure, or product. Example verbs: design, compose, formulate, generate."
    },
}

DIFFICULTY_LEVELS = {
    "easy": {
        "description": "Involves direct recall, recognition, or simple application of knowledge and procedures."
    },
    "medium": {
        "description": "Requires connecting multiple ideas, performing multi-step reasoning, or applying knowledge in new but familiar contexts."
    },
    "hard": {
        "description": "Involves complex reasoning, integration of several sub-topics, or solving non-trivial problems that demand deeper conceptual understanding."
    },
}
