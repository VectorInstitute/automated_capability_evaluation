"""Stage 0: Experiment and domain setup.

This stage initializes the experiment and creates domain metadata.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.schemas.domain_schemas import Domain
from src.schemas.experiment_schemas import Experiment
from src.schemas.io_utils import save_domain, save_experiment
from src.schemas.metadata_schemas import PipelineMetadata
from src.utils.data_utils import check_cfg
from src.utils.timestamp_utils import iso_timestamp


logger = logging.getLogger(__name__)


def run_stage0(cfg: DictConfig) -> None:
    """Stage 0: Experiment and domain setup.

    Creates experiment.json and domain/domain.json files.

    Args:
        cfg: Configuration object
    """
    check_cfg(cfg, logger)
    exp_id = cfg.exp_cfg.exp_id
    output_base_dir = Path(cfg.global_cfg.output_dir)
    domain_name = cfg.global_cfg.domain
    pipeline_type = cfg.global_cfg.pipeline_type

    logger.info(
        "Stage 0: exp_id=%s | domain=%s | output_base_dir=%s | pipeline_type=%s",
        exp_id,
        domain_name,
        output_base_dir,
        pipeline_type,
    )

    domain_id = "domain_000"
    domain_obj = Domain(
        domain_name=domain_name,
        domain_id=domain_id,
        domain_description=None,
    )

    # Convert entire config to dictionary for experiment configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_obj = Experiment(
        experiment_id=exp_id,
        domain=domain_name,
        domain_id=domain_id,
        pipeline_type=pipeline_type,
        configuration=config_dict,
    )

    metadata = PipelineMetadata(
        experiment_id=exp_id,
        output_base_dir=str(output_base_dir),
        timestamp=iso_timestamp(),
        input_stage_tag=None,
        output_stage_tag=None,
        resume=False,
    )

    save_experiment(
        experiment=experiment_obj,
        metadata=metadata,
        output_path=output_base_dir / exp_id / "experiment.json",
    )
    save_domain(
        domain=domain_obj,
        metadata=metadata,
        output_path=output_base_dir / exp_id / "domain" / "domain.json",
    )

    logger.info(
        "Stage 0: saved experiment and domain artifacts under %s", output_base_dir
    )
