import hydra  # noqa: D100
from omegaconf import DictConfig

from generate_capabilities import generate_capabilities

# from lbo import generate_new_capability
from model import Model


def check_cfg(cfg: DictConfig) -> None:
    """
    Check configuration compatibility.

    Args
    ----
        cfg (DictConfig): The provided configuration.
    """
    assert cfg.capabilities_cfg.num_gen_capabilities > 0
    assert cfg.capabilities_cfg.num_gen_capabilities_per_run > 0
    assert (
        cfg.capabilities_cfg.num_gen_capabilities
        >= cfg.capabilities_cfg.num_gen_capabilities_per_run
    ), (
        "The total number of capabilities to generate must be greater than or equal to the number of capabilities to generate per run."
    )
    # log warning
    rem_c = (
        cfg.capabilities_cfg.num_gen_capabilities
        % cfg.capabilities_cfg.num_gen_capabilities_per_run
    )
    additional_c = cfg.capabilities_cfg.num_gen_capabilities_per_run - rem_c
    if rem_c != 0:
        print(f"{additional_c} capabilities will be generated.")


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Run the model with the specified configuration.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    check_cfg(cfg)

    run_id = f"{cfg.scientist_llm.name}_T{cfg.capabilities_cfg.num_gen_capabilities}_R{cfg.capabilities_cfg.num_gen_capabilities_per_run}"

    # Initialize the scientist LLM model
    scientist_llm = Model(cfg.scientist_llm.name)
    scientist_llm_gen_cfg = cfg.scientist_llm.generation_cfg

    # Stage 1. Generate initial capabilities
    capabilities = generate_capabilities(
        domain=cfg.capabilities_cfg.domain,
        num_capabilities=cfg.capabilities_cfg.num_gen_capabilities,
        num_capabilities_per_run=cfg.capabilities_cfg.num_gen_capabilities_per_run,
        scientist_llm=scientist_llm,
        num_seed_capabilities=cfg.capabilities_cfg.num_seed_capabilities,
        scientist_llm_gen_cfg=scientist_llm_gen_cfg.capability_generation,
        exclude_seed_capability_names=["grade_school_math_word_problems"],
        run_id=run_id,
        trial_run=cfg.exp_cfg.trial_run,
    )
    # capabilities = filter_capabilities(capabilities)
    print(capabilities)

    # # TODO: Only used for testing, remove this block later ========================
    # if cfg.exp_cfg.trial_run:
    #     # Set the base capability directory
    #     base_capability_dir = os.path.join(
    #         BASE_ARTIFACTS_DIR, f"capabilities_{run_id}", cfg.capabilities_cfg.domain
    #     )
    #     os.makedirs(base_capability_dir, exist_ok=True)

    #     # Fetch previously generated capabilities, if any
    #     capabilities = _get_previous_capabilities(capability_dir=base_capability_dir)
    # # =============================================================================

    # # Stage 2. Generate tasks and evaluate subject model on initial capabilities
    # num_lbo_runs = cfg.lbo_cfg.num_lbo_runs
    # if cfg.lbo_cfg.pipeline_id == "nearest_neighbour":
    #     # For pipeline 1 (pipeline_id=="nearest_neighbour"), the set of
    #     # generated capabilities are split into two sets
    #     train_capabilities, candidate_capabilities = get_lbo_train_set(
    #         input_data=capabilities,
    #         train_frac=cfg.lbo_cfg.train_frac,
    #         min_train_size=cfg.lbo_cfg.min_train_size,
    #     )
    #     if num_lbo_runs > len(candidate_capabilities):
    #         print(
    #             f"Warning: Number of LBO runs ({num_lbo_runs}) exceeds the number of "
    #             + f"candidate capabilities ({len(candidate_capabilities)}). "
    #             + f"Setting the number of LBO runs to {len(candidate_capabilities)}."
    #         )
    #         num_lbo_runs = len(candidate_capabilities)
    # elif cfg.lbo_cfg.pipeline_id == "discover_new":
    #     # For pipeline 2 (pipeline_id=="discover_new"), use all generated capabilities
    #     # for training
    #     train_capabilities = capabilities
    #     candidate_capabilities = None

    # # Initialize the subject LLM model
    # subject_llm = Model(cfg.subject_llm.name)
    # subject_llm_gen_cfg = dict(cfg.subject_llm.generation_cfg)
    # subject_llm_gen_cfg.update(
    #     {
    #         "limit": cfg.capabilities_cfg.num_eval_tasks_per_capability,
    #     }
    # )

    # # TODO: Run this asynchronosly
    # for capability in train_capabilities:
    #     # Generate tasks for each capability
    #     generate_tasks_using_llm(
    #         capability=capability,
    #         scientist_llm=scientist_llm,
    #         num_tasks=cfg.capabilities_cfg.num_gen_tasks_per_capability,
    #         scientist_llm_gen_cfg_task_gen=scientist_llm_gen_cfg.task_generation,
    #         scientist_llm_gen_cfg_task_solve=scientist_llm_gen_cfg.task_solve,
    #         solve_sample_tasks=True,
    #         few_shot=cfg.capabilities_cfg.task_gen_few_shot,
    #     )
    #     # Evaluate subject LLM on each capability
    #     capability.evaluate([subject_llm], [subject_llm_gen_cfg])

    #     # TODO: Only used for testing, remove this block later ==============
    #     if cfg.exp_cfg.trial_run:
    #         break
    #     # ===================================================================

    # # Stage 3. Use LBO to generate new capabilities
    # for lbo_run_id in range(num_lbo_runs):
    #     new_capability = generate_new_capability(
    #         capabilities=train_capabilities,
    #         subject_llm_name=cfg.subject_llm.name,
    #         capabilities_pool=candidate_capabilities,
    #         pipeline_id=cfg.lbo_cfg.pipeline_id,
    #         lbo_run_id=lbo_run_id,
    #     )
    #     # Generate tasks for new capability
    #     generate_tasks_using_llm(
    #         capability=new_capability,
    #         scientist_llm=scientist_llm,
    #         sys_prompt=TASK_GENERATION_SYSTEM_PROMPT,
    #         user_prompt=TASK_GENERATION_USER_PROMPT,
    #         num_tasks=cfg.capabilities_cfg.num_gen_tasks_per_capability,
    #         scientist_llm_gen_cfg_task_gen=scientist_llm_gen_cfg.task_generation,
    #         scientist_llm_gen_cfg_task_solve=scientist_llm_gen_cfg.task_solve,
    #         solve_sample_tasks=True,
    #         few_shot=cfg.capabilities_cfg.task_gen_few_shot,
    #     )
    #     # Evaluate subject LLM on new capability
    #     new_capability.evaluate([subject_llm], [subject_llm_gen_cfg])
    #     # Add new capability to train capabilities list
    #     train_capabilities.append(new_capability)
    #     # Remove new capability from candidate capabilities
    #     # for pipeline 1
    #     if candidate_capabilities is not None:
    #         candidate_capabilities.remove(new_capability)

    # new_capabilities = train_capabilities[-num_lbo_runs:]
    # print(f"New capabilities: {new_capabilities}")


if __name__ == "__main__":
    main()
