import hydra  # noqa: D100
from omegaconf import DictConfig

from generate_initial_capabilities import generate_capabilities


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Run the model with the specified configuration.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    run_id = f"{cfg.generator_model.name}_{cfg.capabilities_cfg.num_gen_capabilities}"
    _ = generate_capabilities(
        domain=cfg.capabilities_cfg.domain,
        num_capabilities=cfg.capabilities_cfg.num_gen_capabilities,
        scientist_llm=cfg.generator_model.name,
        num_seed_capabilities=cfg.capabilities_cfg.num_seed_capabilities,
        scientist_llm_gen_cfg=cfg.generator_model.gen_cfg,
        run_id=run_id,
        trial_run=cfg.exp_cfg.trial_run,
    )


if __name__ == "__main__":
    main()
