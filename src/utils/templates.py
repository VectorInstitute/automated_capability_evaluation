"""
Module containing a template for the TaskFamily class.

Includes methods for representing samples, getting instructions, and
a placeholder for a task score function.
"""

# Modified based on METR TaskFamily class: https://github.com/METR/task-template/blob/main/my_task/my_task.py#L17
TASK_CLASS_TEMPLATE = """
class Task:
    @staticmethod
    def repr_samples() -> dict[str, dict]:
        return {task_data_samples_dict}

    @staticmethod
    def get_instructions(t: dict) -> str:
        return {task_instructions}

    @staticmethod
    {task_score_func}
"""
