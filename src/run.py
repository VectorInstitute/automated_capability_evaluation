import hydra  # noqa: D100
from omegaconf import DictConfig

from generate_initial_capabilities import generate_capabilities


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

    run_id = f"{cfg.generator_model.name}_T{cfg.capabilities_cfg.num_gen_capabilities}_R{cfg.capabilities_cfg.num_gen_capabilities_per_run}"

    _ = generate_capabilities(
        domain=cfg.capabilities_cfg.domain,
        num_capabilities=cfg.capabilities_cfg.num_gen_capabilities,
        num_capabilities_per_run=cfg.capabilities_cfg.num_gen_capabilities_per_run,
        scientist_llm=cfg.generator_model.name,
        num_seed_capabilities=cfg.capabilities_cfg.num_seed_capabilities,
        scientist_llm_gen_cfg=cfg.generator_model.gen_cfg,
        run_id=run_id,
        trial_run=cfg.exp_cfg.trial_run,
    )


if __name__ == "__main__":
    main()
