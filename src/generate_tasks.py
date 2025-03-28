import json  # noqa: D100
from typing import Any, Dict, List, Tuple

from capability import Capability
from model import Model
from utils.capability_utils import extract_and_parse_response
from utils.prompts import (
    PROBLEM_GENERATION_RESPONSE_JSON_FORMAT,
    PROBLEM_GENERATION_SYSTEM_PROMPT,
    PROBLEM_GENERATION_USER_PROMPT,
    PROBLEM_GENERATION_ZERO_OR_FEW_SHOT_PATCH,
)


def get_problem_generation_prompt(
    capability: Capability,
    num_gen_problems: int,
    few_shot: bool = False,
    sample_problems: List[Dict[str, Any]] | None = None,
) -> Tuple[str, str]:
    """
    Generate the system and user prompts for problem generation.

    Generate the system and user prompts for problem generation
    based on the given capability, number of problems to generate,
    and the few-shot setting.

    Args
    ----
        capability (Capability): The capability to generate problems for.
        num_gen_problems (int): The number of problems to generate.
        few_shot (bool, optional): The few-shot setting. Defaults to False.
        sample_problems (List[Dict[str, Any]] | None, optional): The sample problems
            to use. Defaults to None.

    Returns
    -------
        Tuple[str, str]: The system and user prompts.
    """
    assert (few_shot and (sample_problems is not None)) or (not few_shot), (
        "Few-shot setting is enabled but no sample problems are provided."
    )
    prompt_type = "few_shot" if few_shot else "zero_shot"
    sys_prompt = PROBLEM_GENERATION_SYSTEM_PROMPT.format(
        zero_or_few_shot_patch=PROBLEM_GENERATION_ZERO_OR_FEW_SHOT_PATCH[prompt_type][
            "sys"
        ],
        response_json_format=PROBLEM_GENERATION_RESPONSE_JSON_FORMAT,
    )
    user_zero_or_few_shot_patch = PROBLEM_GENERATION_ZERO_OR_FEW_SHOT_PATCH[
        prompt_type
    ]["user"]
    if few_shot and sample_problems is not None:
        user_zero_or_few_shot_patch = user_zero_or_few_shot_patch.format(
            capability_sample_problems=json.dumps(
                {f"problem_{elm['id']}": elm["problem"] for elm in sample_problems},
                indent=4,
            ),
        )
    user_prompt = PROBLEM_GENERATION_USER_PROMPT.format(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.domain,
        zero_or_few_shot_patch=user_zero_or_few_shot_patch,
        num_gen_problems=num_gen_problems,
    )
    return sys_prompt, user_prompt


def generate_tasks_using_llm(
    capability: Capability,
    scientist_llm: Model,
    num_tasks: int,
    scientist_llm_gen_cfg_task_gen: Dict[str, Any],
    scientist_llm_gen_cfg_task_solve: Dict[str, Any],
    **kwargs: Any,
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
        num_tasks (int): The number of tasks to generate.
        scientist_llm_gen_cfg_task_gen (Dict[str, Any]): The generation configuration
            for task generation using the scientist LLM.
        scientist_llm_gen_cfg_task_solve (Dict[str, Any]): The generation configuration
            for solving tasks using the scientist LLM.
        **kwargs (Any): Additional arguments for task generation.
    """
    # TODO: Implement the function with the following components
    # # Approach 2
    # 1. Generate task problems and answers together in a single run.
    #    Again, this can be done in two ways described above.
    # 2. Filter out similar/ill-formatted problem/asnwer pairs
    # 3. Verify each pair by:
    #   a. prompting the scientist LLM to function as a judge
    #   b. using a group of (less capable) models to judge and
    #      then selecting the majority answer

    # ==== Approach 1 ====
    # 1. First generate task problems. This can be done in two ways:
    #   a. Single run to generate all `num_tasks` (Nt) problems
    #       - input tokens: Pt
    #       - output tokens: Nt * Qt, where Qt is the mean # tokens in a problem
    #   b. Multiple runs to generate `num_tasks` (Nt)
    #      problems in batches of `num_tasks_per_run` (Ntr)
    #       - input tokens: Pt * (Nt / Ntr)
    #       - output tokens: Nt * Qt
    # 2. Filter out similar/ill-formatted problems
    # 3. Then obtain task answers by:
    #   a. prompting the scientist LLM to solve these selected problems
    #   b. using a group of (less capable) models to solve
    #      these problems and then selecting the majority answer
    #   c. using a scoring function

    # Generate task problems
    # Extract sample problems from representative tasks
    sample_problems = capability.get_repr_tasks()
    for task in sample_problems:
        # Remove the answer
        task.pop("answer", None)

    # Generate new problems using the scientist LLM
    sys_prompt, user_prompt = get_problem_generation_prompt(
        capability=capability,
        num_gen_problems=num_tasks,
        few_shot=kwargs.get("few_shot", True),
        sample_problems=sample_problems,
    )
    response, task_gen_metadata = scientist_llm.generate(
        sys_prompt=sys_prompt,
        user_prompt=user_prompt,
        generation_config=scientist_llm_gen_cfg_task_gen,
    )
    # Print the output
    print(f"Model: {scientist_llm.get_model_name()}")
    print(f"Output:\n\n{response}\n\n")
    print(f"Metadata: {task_gen_metadata}")
    parsed_response = extract_and_parse_response(response)
    new_problems = parsed_response["parsed_response"]
    # Combine with sample problems to get the full set of problems
    start_id = len(sample_problems) + 1
    all_problems = sample_problems + [
        {"id": (start_id + idx), "problem": new_problems[idx]}
        for idx in range(len(new_problems))
    ]

    # Solve task and generate answers
    solved_tasks, task_solver_metadata = capability.solve_tasks(
        tasks=all_problems,
        llm=scientist_llm,
        gen_cfg=scientist_llm_gen_cfg_task_solve,
    )
    print(json.dumps(solved_tasks, indent=4))
    print(task_solver_metadata)
