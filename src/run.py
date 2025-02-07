import hydra
from omegaconf import DictConfig

from model import Model
from task import Task

from utils.prompts import TASK_GENERATION_SYSTEM_PROMPT, TASK_GENERATION_USER_PROMPT


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig):
    """
    Run the model with the specified configuration.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    domain = "math"
    math_task = Task(
        cfg.tasks.task_cfgs.mathematics,
    )
    gsm8k_task = Task(
        cfg.tasks.task_cfgs.gsm8k,
    )

    # Set system message
    sys_msg = TASK_GENERATION_SYSTEM_PROMPT

    # Create an instance of the Model class with the specified model name
    model = Model(
        model_name=cfg.model.name,
        sys_msg=sys_msg,
    )

    # Model input
    sample_input = TASK_GENERATION_USER_PROMPT.format(prev_tasks="\n".join([str(math_task), str(gsm8k_task)]), domain=domain)

    # Generate output using the model with specified generation arguments
    gen_cfg = cfg.gen_cfg
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
