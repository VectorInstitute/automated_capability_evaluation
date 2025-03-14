import os  # noqa: D100

import hydra
from omegaconf import DictConfig

from capability import get_capability_repr_with_score, select_seed_capabilities
from model import Model
from utils.prompts import (
    CAPABILITY_GENERATION_SYSTEM_PROMPT,
    CAPABILITY_GENERATION_USER_PROMPT,
)


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Run the model with the specified configuration.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    # Select seed capabilities
    base_dir = cfg.capabilities_cfg.capabilities_dir
    domain = cfg.capabilities_cfg.domain
    seed_capability_dir = os.path.join(base_dir, "seed_capabilities", domain)

    include_capabilities = ["math_grade_school"]
    seed_capabilities = select_seed_capabilities(
        seed_capability_dir,
        cfg.capabilities_cfg.num_seed_capabilities,
        include_capabilities=include_capabilities,
    )

    # Set system message
    sys_msg = CAPABILITY_GENERATION_SYSTEM_PROMPT

    # Create an instance of the Model class with the specified model name
    model = Model(
        model_name=cfg.generator_model.name,
        sys_msg=sys_msg,
    )

    # Obtain capability scores for candidate model and get capability representations
    candidate_model = cfg.candidate_model.name
    seed_capabilities_repr = [
        get_capability_repr_with_score(capability, candidate_model)
        for capability in seed_capabilities
    ]

    # Model input
    sample_input = CAPABILITY_GENERATION_USER_PROMPT.format(
        prev_capabilities="\n".join(seed_capabilities_repr), domain=domain
    )

    # Generate output using the model with specified generation arguments
    gen_cfg = cfg.generator_model.gen_cfg
    output, metadata = model.generate(
        prompt=sample_input,
        generation_config=gen_cfg,
    )

    # Print the output
    print(f"Model: {model.get_model_name()}")
    print(f"Output:\n\n{output}\n\n")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
