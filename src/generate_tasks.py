import asyncio  # noqa: D100
import json
import logging
from typing import Any, Dict, List, Tuple

from langsmith import tracing_context
from tenacity import Retrying, stop_after_attempt

from capability import Capability, CapabilityState
from model import Model
from utils import constants, prompts
from utils.capability_utils import extract_and_parse_response


logger = logging.getLogger(__name__)


def get_task_generation_prompt(
    capability: Capability,
    num_gen_tasks: int,
    sample_tasks: List[Dict[str, Any]] | None = None,
    version: str = "v1",
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
        version (str, optional): The version of the prompt to use.
            Defaults to "v1".

    Returns
    -------
        Tuple[str, str]: The system and user prompts.
    """
    if version == "v1":
        sys_prompt_template = prompts.TASK_GENERATION_SYSTEM_PROMPT
        user_prompt_template = prompts.TASK_GENERATION_USER_PROMPT
    elif version == "v2":
        sys_prompt_template = prompts.TASK_GENERATION_SYSTEM_PROMPT_V2
        user_prompt_template = prompts.TASK_GENERATION_USER_PROMPT_V2
    else:
        raise ValueError(
            f"Invalid version: {version}. Supported versions are v1 and v2."
        )

    prompt_type = "few_shot" if sample_tasks is not None else "zero_shot"
    sys_prompt = sys_prompt_template.format(
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
    user_prompt = user_prompt_template.format(
        capability_name=capability.name,
        capability_description=capability.description,
        capability_domain=capability.domain,
        zero_or_few_shot_patch=user_zero_or_few_shot_patch,
        num_gen_tasks=num_gen_tasks,
    )
    return sys_prompt, user_prompt


def is_task_generation_required(
    capability: Capability,
    target_num_tasks: int,
    regenerate: bool = False,
    regenerate_if_partially_completed: bool = False,
) -> bool:
    """
    Check if task generation is required based on the capability state.

    Args
    ----
        capability (Capability): The capability to check.
        target_num_tasks (int): The target number of tasks.
        regenerate (bool, optional): Whether to regenerate tasks.
            Defaults to False.
        regenerate_if_partially_completed (bool, optional): Whether to
            regenerate tasks if the task generation is partially completed.
            Defaults to False.

    Returns
    -------
        bool: True if task generation is required, False otherwise.
    """
    task_gen_required = True
    if capability.get_state().name == CapabilityState.FILTERED_OUT.name:
        logger.warning(
            f"Capability {capability.name} is filtered out. Hence, skipping task generation."
        )
    elif (
        capability.get_state().name
        == CapabilityState.TASK_GENERATION_PARTIALLY_COMPLETED.name
        and not regenerate_if_partially_completed
        and not regenerate
    ):
        logger.warning(
            f"[{capability.name}] Task generation is partially completed with {len(capability.get_tasks())}/{target_num_tasks} tasks. "
            "In order to regenerate all tasks, set either 'regenerate_if_partially_completed' or 'regenerate' to True."
        )
        task_gen_required = False
    elif (
        capability.get_state().name == CapabilityState.TASK_GENERATION_COMPLETED.name
        and not regenerate
    ):
        logger.warning(
            f"[{capability.name}] Task generation is already completed with {len(capability.get_tasks())}/{target_num_tasks} tasks. "
            "In order to regenerate all tasks, set 'regenerate' to True."
        )
        task_gen_required = False
    return task_gen_required


def generate_tasks_using_llm(
    capability: Capability,
    scientist_llm: Model,
    num_tasks: int,
    num_tasks_buffer: float,
    scientist_llm_gen_cfg_task_gen: Dict[str, Any],
    scientist_llm_gen_cfg_task_solve: Dict[str, Any],
    scientist_llm_gen_cfg_task_verify: Dict[str, Any],
    solve_sample_tasks: bool = False,
    regenerate: bool = False,
    regenerate_if_partially_completed: bool = False,
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
        regenerate (bool, optional): Whether to regenerate tasks.
        regenerate_if_partially_completed (bool, optional): Whether to
            regenerate tasks if the task generation is partially complete
            (not reached target number of tasks).
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

    # Calculate the number of tasks to generate
    target_num_tasks = num_tasks
    num_tasks = int(num_tasks * (1 + num_tasks_buffer))

    # Check if the capability is in a state that allows task generation
    if not is_task_generation_required(
        capability=capability,
        target_num_tasks=target_num_tasks,
        regenerate=regenerate,
        regenerate_if_partially_completed=regenerate_if_partially_completed,
    ):
        logger.warning(f"[{capability.name}] Skipping task generation.")
        return

    # Generate task problems
    # Extract sample tasks from representative tasks
    sample_tasks = capability.get_repr_tasks()

    # Fetch prompts for task generation
    sys_prompt, user_prompt = get_task_generation_prompt(
        capability=capability,
        num_gen_tasks=num_tasks,
        sample_tasks=sample_tasks if kwargs.get("few_shot", True) else None,
        version=kwargs.get("task_gen_prompt_version", "v1"),
    )
    # Generate new tasks
    logger.info(f"Generating {num_tasks} tasks for {capability.name} ...")
    num_attempts = kwargs.get(
        "tasks_gen_retry_attempts", constants.DEFAULT_TASK_GENERATION_RETRY_ATTEMPTS
    )
    try:
        # Retry the generation process if an error occurs
        # Common errors:
        # - json.decoder.JSONDecodeError: Invalid \escape: line 3 column 133
        for attempt in Retrying(
            stop=stop_after_attempt(num_attempts),
            reraise=True,
        ):
            with attempt:
                # Update the seed for each attempt
                scientist_llm_gen_cfg_task_gen["seed"] += 1
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
    except Exception as e:
        error_msg = (
            f"Error generating tasks for capability {capability.name}: {repr(e)}"
        )
        logger.error(error_msg)
        logger.error(f"[{capability.name}] Response:\n{response}")
        # Set capability state to task generation failed
        capability.set_state(
            state_str=constants.C_STATE_TASK_GENERATION_FAILED_STR,
            reason=error_msg,
        )
        return

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
        f"[{capability.name}] Task problems generation tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
    )

    # Solve task and generate answers
    # Set starting ID for new tasks
    start_id = (
        max([int(elm["id"]) for elm in capability.get_tasks(include_failed=True)]) + 1
    )
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
    (solved_tasks, unsolved_tasks), task_solver_metadata = capability.solve_tasks(
        tasks=all_tasks,
        llm=scientist_llm,
        gen_cfg=scientist_llm_gen_cfg_task_solve,
        run_id=kwargs.get("run_id"),
        concurrency=kwargs.get(
            "concurrency_task_solver", constants.DEFAULT_TASK_SOLVER_CONCURRENCY
        ),
    )
    logger.info(
        f"[{capability.name}] {len(solved_tasks)}/{len(all_tasks)} tasks were solved successfully."
    )

    # Analyze tokens metadata for task solving
    total_input_tokens = sum(
        [v.get("input_tokens", 0) for v in task_solver_metadata.values()]
    )
    total_output_tokens = sum(
        [v.get("output_tokens", 0) for v in task_solver_metadata.values()]
    )
    tokens_summary = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "input_tokens_per_task": int(total_input_tokens / len(all_tasks)),
        "output_tokens_per_task": int(total_output_tokens / len(all_tasks)),
        "total_tokens_per_task": int(
            (total_input_tokens + total_output_tokens) / len(all_tasks)
        ),
    }
    logger.info(
        f"[{capability.name}] Task solving tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
    )

    # TODO: Include reasoning along with the answer in the prompt
    (successful_tasks, failed_tasks), task_judge_metadata = verify_solved_tasks(
        tasks=solved_tasks,
        capability=capability,
        llm=scientist_llm,
        gen_cfg=scientist_llm_gen_cfg_task_verify,
        run_id=kwargs.get("run_id"),
        concurrency=kwargs.get(
            "concurrency_task_verifier", constants.DEFAULT_TASK_VERIFIER_CONCURRENCY
        ),
    )
    logger.info(
        f"[{capability.name}] {len(successful_tasks)}/{len(solved_tasks)} tasks passed the verification."
    )

    # Analyze tokens metadata for task verification
    total_input_tokens = sum([v["input_tokens"] for v in task_judge_metadata.values()])
    total_output_tokens = sum(
        [v["output_tokens"] for v in task_judge_metadata.values()]
    )
    if solved_tasks:
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
            f"[{capability.name}] Task verification tokens summary:\n{json.dumps(tokens_summary, indent=4)}"
        )
    else:
        logger.warning(
            f"[{capability.name}] No tasks were solved. Hence, task verification was skipped."
        )

    # Failed tasks consist of both unsolved tasks and tasks which failed verification
    failed_tasks = failed_tasks + unsolved_tasks

    # TODO: Handle scenario when representative tasks are not solved
    #   or fail verification
    capability.add_and_update_tasks(
        tasks=sorted(successful_tasks, key=lambda x: int(x["id"])),
        failed_tasks=sorted(failed_tasks, key=lambda x: int(x["id"]))
        if failed_tasks
        else None,
        seed=kwargs.get("seed", constants.DEFAULT_RANDOM_SEED),
    )

    if len(successful_tasks) < target_num_tasks:
        warning_msg = (
            f"[{capability.name}] Only {len(successful_tasks)} tasks were successfully solved and verified. "
            + f"Target number of tasks not reached: {target_num_tasks}. "
            + "It is recommended to increase the buffer."
        )
        logger.warning(warning_msg)
        capability.set_state(
            state_str=constants.C_STATE_TASK_GENERATION_PARTIALLY_COMPLETED_STR,
            reason=warning_msg,
        )
    else:
        logger.info(
            f"[{capability.name}] {len(successful_tasks)}/{len(all_tasks)} tasks were successfully solved and verified."
        )
        capability.set_state(
            state_str=constants.C_STATE_TASK_GENERATION_COMPLETED_STR,
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

    async def _verify_task(
        task: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logger.info(f"[{capability.name}] Verifying task {task['id']} ...")
        user_prompt = prompts.ANSWER_JUDGEMENT_USER_PROMPT.format(
            capability_name=capability.name,
            capability_domain=capability.domain,
            problem=task["problem"],
            answer=task["answer"],
        )
        try:
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
                response, _metadata = await llm.async_generate(
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    generation_config=gen_cfg,
                )
        except Exception as e:
            logger.warning(
                f"[{capability.name}] Error verifying task {task['id']}: {repr(e)}"
            )
            verdict_str = "no"
            verdict_reason = f"Failed to verify task: {repr(e)}"
            _metadata = {}
        else:
            try:
                parsed_response = extract_and_parse_response(
                    response,
                    has_thought=True,
                    parse_kw="JUDGEMENT",
                    response_type="str_yes_no",
                )
            except Exception as e:
                logger.warning(
                    f"[{capability.name}] Error parsing response for task {task['id']}: {repr(e)}"
                )
                parsed_response = {
                    "parsed_response": "no",
                    "thought": f"Failed to verify task: {repr(e)}",
                }
            verdict_str = parsed_response["parsed_response"]
            verdict_reason = parsed_response["thought"]
        finally:
            task["verification"] = {
                "verdict": verdict_str,
                "reason": verdict_reason,
            }
        return task, _metadata

    async def _verify_all_tasks(concurrency_limit: int) -> None:
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def _limited_verify_task(
            task: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            async with semaphore:
                return await _verify_task(task)

        results = await asyncio.gather(
            *(_limited_verify_task(task) for task in tasks),
        )
        for result in results:
            task, _metadata = result
            if task["verification"]["verdict"] == "yes":
                successful_tasks.append(task)
            else:
                failed_tasks.append(task)
            metadata[task["id"]] = _metadata

    asyncio.run(
        _verify_all_tasks(kwargs.get("concurrency", constants.DEFAULT_CONCURRENCY))
    )

    return (successful_tasks, failed_tasks), metadata
