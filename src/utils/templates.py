"""
Module containing a template for the Capability class.

Includes methods for representing samples, getting instructions, and
a placeholder for a capability score function.
"""

# Modified based on METR TaskFamily class: https://github.com/METR/task-template/blob/main/my_task/my_task.py#L17
CAPABILITY_CLASS_TEMPLATE = """
class Capability:
    @staticmethod
    def repr_samples() -> dict[str, dict]:
        return {capability_data_samples_dict}

    @staticmethod
    def get_instructions(t: dict) -> str:
        return {capability_instructions}

    @staticmethod
    {capability_score_func}
"""
