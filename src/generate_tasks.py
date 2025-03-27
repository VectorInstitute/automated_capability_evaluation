from typing import Any, Dict  # noqa: D100

from capability import Capability
from model import Model


def generate_tasks_using_llm(
    capability: Capability,
    scientist_llm: Model,
    sys_prompt: str,
    user_prompt: str,
    num_tasks: int,
    scientist_llm_gen_cfg: Dict[str, Any],
) -> None:
    """
    Generate `num_tasks` tasks for the given capability.

    Generate tasks for the given capability
    using the scientist LLM model based on the following approach:
    <Approach>

    Args
    ----
        capability (Capability): The capability to generate tasks for.
        scientist_llm (Model): The scientist LLM model.
        sys_prompt (str): The system prompt for generating tasks.
        user_prompt (str): The user prompt for generating tasks.
        num_tasks (int): The number of tasks to generate.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
    """
    # TODO: Implement the function with the following components
    # # Approach 1
    # 1. First generate task questions. This can be done in two ways:
    #   a. Single run to generate all `num_tasks` (Nt) questions
    #       - input tokens: Pt
    #       - output tokens: Nt * Qt, where Qt is the mean # tokens in a question
    #   b. Multiple runs to generate `num_tasks` (Nt)
    #      questions in batches of `num_tasks_per_run` (Ntr)
    #       - input tokens: Pt * (Nt / Ntr)
    #       - output tokens: Nt * Qt
    # 2. Filter out similar/ill-formatted questions
    # 3. Then obtain task answers by:
    #   a. prompting the scientist LLM to solve these selected questions
    #   b. using a group of (less capable) models to solve
    #      these questions and then selecting the majority answer
    #   c. using a scoring function
    #
    # # Approach 2
    # 1. Generate task questions and answers together in a single run.
    #    Again, this can be done in two ways described above.
    # 2. Filter out similar/ill-formatted question/asnwer pairs
    # 3. Verify each pair by:
    #   a. prompting the scientist LLM to function as a judge
    #   b. using a group of (less capable) models to judge and
    #      then selecting the majority answer

    raise NotImplementedError
