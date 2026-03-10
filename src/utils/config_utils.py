"""Configuration validation helpers."""

import logging

from omegaconf import DictConfig


def check_cfg(cfg: DictConfig, logger: logging.Logger) -> None:
    """Validate required configuration fields and warn on generation rounding."""
    assert getattr(cfg, "exp_cfg", None) is not None, "exp_cfg must be set."
    assert getattr(cfg.exp_cfg, "exp_id", ""), "exp_id must be set in exp_cfg."
    assert getattr(cfg, "global_cfg", None) is not None, "global_cfg must be set."
    assert getattr(cfg.global_cfg, "output_dir", ""), (
        "global_cfg.output_dir must be set."
    )
    assert getattr(cfg.global_cfg, "domain", ""), "global_cfg.domain must be set."
    assert getattr(cfg.global_cfg, "pipeline_type", None) is not None, (
        "global_cfg.pipeline_type must be set."
    )
    assert cfg.capabilities_cfg.num_capabilities > 0
    assert cfg.capabilities_cfg.num_gen_capabilities_per_run > 0
    num_capabilities = int(
        cfg.capabilities_cfg.num_capabilities
        * (1 + cfg.capabilities_cfg.num_capabilities_buffer)
    )
    assert num_capabilities >= cfg.capabilities_cfg.num_gen_capabilities_per_run, (
        "The total number of capabilities to generate must be greater than or equal to the number of capabilities to generate per run."
    )
    rem_c = num_capabilities % cfg.capabilities_cfg.num_gen_capabilities_per_run
    additional_c = cfg.capabilities_cfg.num_gen_capabilities_per_run - rem_c
    if rem_c != 0:
        logger.warning(f"{additional_c} additional capabilities might be generated.")
