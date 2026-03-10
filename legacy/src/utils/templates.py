"""
Module containing a template for the Capability class.

Includes methods for representing tasks, getting instructions, and
a placeholder for a capability score function.
"""

# Modified based on METR TaskFamily class: https://github.com/METR/task-template/blob/main/my_task/my_task.py#L17
CAPABILITY_CLASS_TEMPLATE = """
class Capability:
    @staticmethod
    def repr_tasks() -> dict[str, dict]:
        return {capability_tasks_dict}

    @staticmethod
    def get_instructions(t: dict) -> str:
        return {capability_instructions}

    @staticmethod
    {capability_score_func}
"""

INSPECT_EVALS_SCRIPT_FILE_TEMPLATE = '''
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr, CORRECT, INCORRECT
from inspect_ai.solver import TaskState, generate, prompt_template


USER_PROMPT_TEMPLATE = """
{prompt_template}
""".strip()


{score_func_str}


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        correct = await _score(
            t={score_func_t_dict_str},
            submission=state.output.completion,
        )
        return Score(
            value=CORRECT if correct else INCORRECT,
            explanation=state.output.completion,
        )

    return score


@task
def {capability_name}() -> Task:
    """Inspect task implementing the {capability_name} capability."""
    return Task(
        dataset=json_dataset(
            "{capability_name}/dataset.jsonl",
            FieldSpec(
                input="problem",
                target="answer",
                id="id",
                metadata={dataset_metadata_keys},
            ),
        ),
        solver=[
            prompt_template(
                template=USER_PROMPT_TEMPLATE,
            ),
            generate(),
        ],
        scorer=custom_scorer(),
    )
'''

INSPECT_EVALS_README_FILE_TEMPLATE = """
# {capability_name}

{capability_description}
"""

INSPECT_EVALS_INIT_FILE_TEMPLATE = """
from .{capability_name} import {capability_name}

__all__ = [
    "{capability_name}",
]
"""
