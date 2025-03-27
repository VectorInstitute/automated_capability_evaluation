import os  # noqa: D100
from typing import Any, Dict, List

from model import Model
from utils.constants import BASE_ARTIFACTS_DIR
from utils.prompts import TASK_GENERATION_SYSTEM_PROMPT, TASK_GENERATION_USER_PROMPT


def generate_tasks_using_llm(
    capability_src_dir: str,
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
        capability_src_dir (str): The directory containing the capability files.
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


def generate_tasks(
    domain: str,
    capabilities: List[str],
    scientist_llm: Model,
    num_tasks: int,
    scientist_llm_gen_cfg: Dict[str, Any],
    **kwargs: Any,
) -> None:
    """
    Generate `num_tasks` tasks for all given capabilities.

    Generate tasks for all given capabilities
    using the scientist LLM model.

    Args
    ----
        domain (str): The domain name.
        capabilities (List[str]): The list of capability names to generate tasks for.
        scientist_llm (Model): The scientist LLM model.
        num_tasks (int): The number of tasks to generate.
        scientist_llm_gen_cfg (Dict[str, Any]): The generation configuration
            for the scientist LLM.
    """
    if "trial_run" in kwargs:
        capability_dir = os.path.join(
            BASE_ARTIFACTS_DIR, f"capabilities_{kwargs['run_id']}", domain
        )
    else:
        capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "capabilities", domain)

    # TODO: Run this asynchronosly
    # Generate tasks for each capability
    for capability in capabilities:
        generate_tasks_using_llm(
            capability_src_dir=os.path.join(capability_dir, capability),
            scientist_llm=scientist_llm,
            sys_prompt=TASK_GENERATION_SYSTEM_PROMPT,
            user_prompt=TASK_GENERATION_USER_PROMPT,
            num_tasks=num_tasks,
            scientist_llm_gen_cfg=scientist_llm_gen_cfg,
        )
