import hydra  # noqa: D100
from omegaconf import DictConfig

from generate_capabilities import filter_capabilities, generate_capabilities
from model import Model


# from capability import evaluate_capabilities
# from generate_tasks import generate_tasks
# from lbo import generate_new_capability


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

    # Stage 1. Generate initial capabilities
    capabilities = generate_capabilities(
        domain=cfg.capabilities_cfg.domain,
        num_capabilities=cfg.capabilities_cfg.num_gen_capabilities,
        num_capabilities_per_run=cfg.capabilities_cfg.num_gen_capabilities_per_run,
        scientist_llm=scientist_llm,
        num_seed_capabilities=cfg.capabilities_cfg.num_seed_capabilities,
        scientist_llm_gen_cfg=cfg.scientist_llm.gen_cfg,
        run_id=run_id,
        trial_run=cfg.exp_cfg.trial_run,
    )
    capabilities = filter_capabilities(capabilities)
    print(capabilities)

    # # Stage 2. Generate tasks and evaluate subject model on initial capabilities
    # # Initialize the subject LLM model
    # subject_llm = Model(cfg.subject_llm.name)
    # generate_tasks(
    #     domain=cfg.capabilities_cfg.domain,
    #     capabilities=capabilities,
    #     scientist_llm=scientist_llm,
    #     num_tasks=cfg.tasks_cfg.num_tasks,
    #     scientist_llm_gen_cfg=cfg.scientist_llm.gen_cfg,
    #     run_id=run_id,
    #     trial_run=cfg.exp_cfg.trial_run,
    # )
    # evaluate_capabilities(
    #     domain=cfg.capabilities_cfg.domain,
    #     capabilities=capabilities,
    #     subject_llms=[subject_llm],
    #     run_id=run_id,
    #     trial_run=cfg.exp_cfg.trial_run,
    # )

    # # Stage 3. Use LBO to generate new capabilities
    # for lbo_run_id in range(cfg.lbo_cfg.num_lbo_runs):
    #     new_capability = generate_new_capability(
    #         domain=cfg.capabilities_cfg.domain,
    #         capabilities=capabilities,
    #         subject_llm_name=cfg.subject_llm.name,
    #         run_id=run_id,
    #         trial_run=cfg.exp_cfg.trial_run,
    #         lbo_run_id=lbo_run_id,
    #     )
    #     # Generate tasks for new capability
    #     generate_tasks(
    #         domain=cfg.capabilities_cfg.domain,
    #         capabilities=[new_capability],
    #         scientist_llm=scientist_llm,
    #         num_tasks=cfg.tasks_cfg.num_tasks,
    #         scientist_llm_gen_cfg=cfg.scientist_llm.gen_cfg,
    #         run_id=run_id,
    #         trial_run=cfg.exp_cfg.trial_run,
    #     )
    #     # Evaluate subject LLM on new capability
    #     evaluate_capabilities(
    #         domain=cfg.capabilities_cfg.domain,
    #         capabilities=[new_capability],
    #         subject_llms=[subject_llm],
    #         run_id=run_id,
    #         trial_run=cfg.exp_cfg.trial_run,
    #     )
    #     # Add new capability to capabilities list
    #     capabilities.append(new_capability)

    # new_capabilities = capabilities[-cfg.lbo_cfg.num_lbo_runs:]
    # print(f"New capabilities: {new_capabilities}")


if __name__ == "__main__":
    main()
