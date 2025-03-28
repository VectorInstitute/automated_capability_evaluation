import json  # noqa: D100
from typing import Any, Dict, List, Tuple

from capability import Capability
from model import Model
from utils.capability_utils import extract_and_parse_response
from utils.prompts import (
    TASK_GENERATION_RESPONSE_JSON_FORMAT,
    TASK_GENERATION_SYSTEM_PROMPT,
    TASK_GENERATION_USER_PROMPT,
    TASK_GENERATION_ZERO_OR_FEW_SHOT_PATCH,
)


def get_task_generation_prompt(
    capability: Capability,
    num_gen_tasks: int,
    few_shot: bool = False,
    sample_tasks: List[Dict[str, Any]] | None = None,
) -> Tuple[str, str]:
    """
    Generate the system and user prompts for task generation.

    Generate the system and user prompts for task generation
    based on the given capability, number of tasks to generate,
    and the few-shot setting.

    Args
    ----
        capability (Capability): The capability to generate tasks for.
        num_gen_tasks (int): The number of tasks to generate.
        few_shot (bool, optional): The few-shot setting. Defaults to False.
        sample_tasks (List[Dict[str, Any]] | None, optional): The sample tasks
            to use. Defaults to None.

    Returns
    -------
        Tuple[str, str]: The system and user prompts.
    """
    assert (few_shot and (sample_tasks is not None)) or (not few_shot), (
        "Few-shot setting is enabled but no sample tasks are provided."
    )
    prompt_type = "few_shot" if few_shot else "zero_shot"
    sys_prompt = TASK_GENERATION_SYSTEM_PROMPT.format(
        zero_or_few_shot_patch=TASK_GENERATION_ZERO_OR_FEW_SHOT_PATCH[prompt_type][
            "sys"
        ],
        response_json_format=TASK_GENERATION_RESPONSE_JSON_FORMAT,
    )
    user_zero_or_few_shot_patch = TASK_GENERATION_ZERO_OR_FEW_SHOT_PATCH[prompt_type][
        "user"
    ]
    if few_shot and sample_tasks is not None:
        user_zero_or_few_shot_patch = user_zero_or_few_shot_patch.format(
            capability_sample_tasks=json.dumps(
                {f"task_{elm['id']}": elm["problem"] for elm in sample_tasks},
                indent=4,
            ),
        )
    user_prompt = TASK_GENERATION_USER_PROMPT.format(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.domain,
        zero_or_few_shot_patch=user_zero_or_few_shot_patch,
        num_gen_tasks=num_gen_tasks,
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
    # Extract sample tasks from representative tasks
    sample_tasks = capability.get_repr_tasks()
    for task in sample_tasks:
        # Remove the answer
        task.pop("answer", None)

    # Generate new tasks using the scientist LLM
    sys_prompt, user_prompt = get_task_generation_prompt(
        capability=capability,
        num_gen_tasks=num_tasks,
        few_shot=kwargs.get("few_shot", True),
        sample_tasks=sample_tasks,
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
    new_tasks = parsed_response["parsed_response"]
    # Combine with sample tasks to get the full set of tasks
    start_id = len(sample_tasks) + 1
    all_tasks = sample_tasks + [
        {"id": (start_id + idx), "problem": new_tasks[idx]}
        for idx in range(len(new_tasks))
    ]

    # Solve task and generate answers
    solved_tasks, task_solver_metadata = capability.solve_tasks(
        tasks=all_tasks,
        llm=scientist_llm,
        gen_cfg=scientist_llm_gen_cfg_task_solve,
    )
    print(json.dumps(solved_tasks, indent=4))
    print(task_solver_metadata)
