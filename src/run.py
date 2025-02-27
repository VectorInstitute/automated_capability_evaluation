import os  # noqa: D100

import hydra
from omegaconf import DictConfig

from model import Model
from task import get_task_repr_with_score, select_seed_tasks
from utils.prompts import TASK_GENERATION_SYSTEM_PROMPT, TASK_GENERATION_USER_PROMPT


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig) -> None:
    """
    Run the model with the specified configuration.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    # Select seed tasks
    base_dir = os.path.dirname(os.path.abspath(__file__))
    domain = cfg.tasks_cfg.domain
    seed_task_dir = os.path.join(base_dir, "seed_tasks", domain)

    include_tasks = ["math_grade_school"]
    seed_tasks = select_seed_tasks(
        seed_task_dir, cfg.tasks_cfg.num_seed_tasks, include_tasks=include_tasks
    )

    # Set system message
    sys_msg = TASK_GENERATION_SYSTEM_PROMPT

    # Create an instance of the Model class with the specified model name
    model = Model(
        model_name=cfg.generator_model.name,
        sys_msg=sys_msg,
    )

    # Obtain task scores for candidate model and get task representations
    candidate_model = cfg.candidate_model.name
    seed_tasks_repr = [
        get_task_repr_with_score(task, candidate_model) for task in seed_tasks
    ]

    # Model input
    sample_input = TASK_GENERATION_USER_PROMPT.format(
        prev_tasks="\n".join(seed_tasks_repr), domain=domain
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
