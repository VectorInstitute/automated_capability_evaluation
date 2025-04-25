import json  # noqa: D100
import logging
from typing import Any, Dict, List, Tuple

from langsmith import tracing_context
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt

from capability import Capability
from model import Model
from utils import constants, prompts
from utils.capability_utils import extract_and_parse_response


logger = logging.getLogger(__name__)


def get_task_generation_prompt(
    capability: Capability,
    num_gen_tasks: int,
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
        sample_tasks (List[Dict[str, Any]] | None, optional): The sample tasks
            to use. Defaults to None.

    Returns
    -------
        Tuple[str, str]: The system and user prompts.
    """
    prompt_type = "few_shot" if sample_tasks is not None else "zero_shot"
    sys_prompt = prompts.TASK_GENERATION_SYSTEM_PROMPT.format(
        zero_or_few_shot_patch=prompts.TASK_GENERATION_ZERO_OR_FEW_SHOT_PATCH[
            prompt_type
        ]["sys"],
        response_json_format=prompts.TASK_GENERATION_RESPONSE_JSON_FORMAT,
    )
    user_zero_or_few_shot_patch = prompts.TASK_GENERATION_ZERO_OR_FEW_SHOT_PATCH[
        prompt_type
    ]["user"]
    if sample_tasks is not None:
        user_zero_or_few_shot_patch = user_zero_or_few_shot_patch.format(
            capability_sample_tasks=json.dumps(
                {f"task_{elm['id']}": elm["problem"] for elm in sample_tasks},
                indent=4,
            ),
        )
    user_prompt = prompts.TASK_GENERATION_USER_PROMPT.format(
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
    num_tasks_buffer: float,
    scientist_llm_gen_cfg_task_gen: Dict[str, Any],
    scientist_llm_gen_cfg_task_solve: Dict[str, Any],
    scientist_llm_gen_cfg_task_verify: Dict[str, Any],
    solve_sample_tasks: bool = False,
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
        num_tasks_buffer (float): Fraction of additional tasks to generate
            to account for filtering in the verification step.
        scientist_llm_gen_cfg_task_gen (Dict[str, Any]): The generation configuration
            for task generation using the scientist LLM.
        scientist_llm_gen_cfg_task_solve (Dict[str, Any]): The generation configuration
            for solving tasks using the scientist LLM.
        scientist_llm_gen_cfg_task_verify (Dict[str, Any]): The generation configuration
            for verifying tasks using the scientist LLM.
        solve_sample_tasks (bool, optional): Whether to solve sample tasks.
        **kwargs (Any): Additional arguments for task generation.
    """
    # TODO: Implement Approach 2 (low priority)
    # # Approach 2
    # 1. Generate task problems and answers together in a single run.
    #    Again, this can be done in two ways described above.
    # 2. Filter out similar/ill-formatted problem/asnwer pairs
    # 3. Verify each pair by:
    #   a. prompting the scientist LLM to function as a judge
    #   b. using a group of (less capable) models to judge and
    #      then selecting the majority answer

    # Generate task problems
    # Extract sample tasks from representative tasks
    sample_tasks = capability.get_repr_tasks()

    # Calculate the number of tasks to generate
    num_tasks = int(num_tasks * (1 + num_tasks_buffer))

    # Generate new tasks using the scientist LLM
    sys_prompt, user_prompt = get_task_generation_prompt(
        capability=capability,
        num_gen_tasks=num_tasks,
        sample_tasks=sample_tasks if kwargs.get("few_shot", True) else None,
    )

    # Generate tasks
    logger.info(f"Generating {num_tasks} tasks for {capability.name} ...")
    num_attempts = kwargs.get(
        "tasks_gen_retry_attempts", constants.DEFAULT_TASK_GENERATION_RETRY_ATTEMPTS
    )
    try:
        # Retry the generation process if a JSONDecodeError occurs
        # Common errors:
        # - json.decoder.JSONDecodeError: Invalid \escape: line 3 column 133
        for attempt in Retrying(
            stop=stop_after_attempt(num_attempts),
            retry=retry_if_exception_type(json.decoder.JSONDecodeError),
        ):
            with attempt:
                with tracing_context(
                    enabled=True,
                    tags=["generate_tasks_using_llm"],
                    metadata={
                        "ls_provider": scientist_llm.model_provider,
                        "ls_model_name": scientist_llm.get_model_name(
                            with_provider=False
                        ),
                        "ls_model_type": "chat",
                        "exp_id": kwargs.get("run_id"),
                        "capability_name": capability.name,
                        "domain": capability.domain,
                        "num_tasks": num_tasks,
                        **{
                            f"ls_{k}": v
                            for k, v in scientist_llm_gen_cfg_task_gen.items()
                        },
                    },
                ):
                    response, task_gen_metadata = scientist_llm.generate(
                        sys_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        generation_config=scientist_llm_gen_cfg_task_gen,
                    )
                parsed_response = extract_and_parse_response(response)
                new_tasks = parsed_response["parsed_response"]
    except RetryError as e:
        logger.error(
            f"Error generating tasks after {num_attempts}: {e.last_attempt.result()}"
        )
        raise e

    # Analyze tokens metadata for task problems generation
    tokens_summary = {
        "total_input_tokens": task_gen_metadata["input_tokens"],
        "total_output_tokens": task_gen_metadata["output_tokens"],
        "total_tokens": task_gen_metadata["input_tokens"]
        + task_gen_metadata["output_tokens"],
        "input_tokens_per_task": int(
            task_gen_metadata["input_tokens"] / len(new_tasks)
        ),
        "output_tokens_per_task": int(
            task_gen_metadata["output_tokens"] / len(new_tasks)
        ),
        "total_tokens_per_task": int(
            (task_gen_metadata["input_tokens"] + task_gen_metadata["output_tokens"])
            / len(new_tasks)
        ),
    }
    logger.info(
        f"Task problems generation tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
    )

    # Solve task and generate answers
    # Set starting ID for new tasks
    start_id = len(capability.get_tasks()) + 1
    all_tasks = [
        {"id": str(start_id + idx), "problem": new_tasks[idx]}
        for idx in range(len(new_tasks))
    ]
    # Add sample tasks if solving them
    if solve_sample_tasks:
        for task in sample_tasks:
            # Remove the answer
            task.pop("answer", None)
        all_tasks = sample_tasks + all_tasks
    solved_tasks, task_solver_metadata = capability.solve_tasks(
        tasks=all_tasks,
        llm=scientist_llm,
        gen_cfg=scientist_llm_gen_cfg_task_solve,
        run_id=kwargs.get("run_id"),
    )

    # Analyze tokens metadata for task solving
    total_input_tokens = sum([v["input_tokens"] for v in task_solver_metadata.values()])
    total_output_tokens = sum(
        [v["output_tokens"] for v in task_solver_metadata.values()]
    )
    tokens_summary = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_tokens_per_task": int(total_input_tokens / len(solved_tasks)),
        "output_tokens_per_task": int(total_output_tokens / len(solved_tasks)),
        "total_tokens_per_task": int(
            (total_input_tokens + total_output_tokens) / len(solved_tasks)
        ),
    }
    logger.info(f"Task solving tokens summary:\n{json.dumps(tokens_summary, indent=4)}")

    (successful_tasks, failed_tasks), task_judge_metadata = verify_solved_tasks(
        tasks=solved_tasks,
        capability=capability,
        llm=scientist_llm,
        gen_cfg=scientist_llm_gen_cfg_task_verify,
        run_id=kwargs.get("run_id"),
    )
    logger.info(
        f"{len(successful_tasks)}/{len(solved_tasks)} tasks passed the verification."
    )

    # Analyze tokens metadata for task verification
    total_input_tokens = sum([v["input_tokens"] for v in task_judge_metadata.values()])
    total_output_tokens = sum(
        [v["output_tokens"] for v in task_judge_metadata.values()]
    )
    tokens_summary = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_tokens_per_task": int(total_input_tokens / len(solved_tasks)),
        "output_tokens_per_task": int(total_output_tokens / len(solved_tasks)),
        "total_tokens_per_task": int(
            (total_input_tokens + total_output_tokens) / len(solved_tasks)
        ),
    }
    logger.info(
        f"Task verification tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
    )

    capability.add_and_update_tasks(
        tasks=successful_tasks,
        failed_tasks=failed_tasks if failed_tasks else None,
    )


def verify_solved_tasks(
    tasks: List[Dict[str, Any]],
    capability: Capability,
    llm: Model,
    gen_cfg: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[Any, Any]]:
    """
    Verify the solved tasks using the given LLM.

    Args
    ----
        tasks (List[Dict[str, Any]]): The list of tasks to verify.
        capability (Capability): The capability to which the tasks belong.
        llm (Model): The LLM model to use for verification.
        gen_cfg (Dict[str, Any]): The generation configuration for the LLM.
        **kwargs (Any): Additional arguments for verification.

    Returns
    -------
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing
            the list of successful tasks and the list of failed tasks.
    """
    successful_tasks = []
    failed_tasks = []
    metadata = {}

    sys_prompt = prompts.ANSWER_JUDGEMENT_SYSTEM_PROMPT.format(
        capability_domain=capability.domain,
    )

    for task in tasks:
        logger.info(f"Verifying task {task['id']} ...")
        user_prompt = prompts.ANSWER_JUDGEMENT_USER_PROMPT.format(
            capability_name=capability.name,
            capability_domain=capability.domain,
            problem=task["problem"],
            answer=task["answer"],
        )
        with tracing_context(
            enabled=True,
            tags=["verify_solved_tasks"],
            metadata={
                "ls_provider": llm.model_provider,
                "ls_model_name": llm.get_model_name(with_provider=False),
                "ls_model_type": "chat",
                "exp_id": kwargs.get("run_id"),
                "capability_name": capability.name,
                "domain": capability.domain,
                "task_id": task["id"],
                **{f"ls_{k}": v for k, v in gen_cfg.items()},
            },
        ):
            response, _metadata = llm.generate(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                generation_config=gen_cfg,
            )
        try:
            parsed_response = extract_and_parse_response(
                response,
                has_thought=True,
                parse_kw="JUDGEMENT",
                response_type="str_yes_no",
            )
        except AssertionError as e:
            # Tag as fail where the response is not "yes" or "no"
            parsed_response = {
                "parsed_response": "no",
                "thought": str(e),
            }
        verdict_str = parsed_response["parsed_response"]
        task["verification"] = {
            "verdict": verdict_str,
            "reason": parsed_response["thought"],
        }
        if verdict_str == "yes":
            successful_tasks.append(task)
        else:
            failed_tasks.append(task)
        metadata[task["id"]] = _metadata

    return (successful_tasks, failed_tasks), metadata
