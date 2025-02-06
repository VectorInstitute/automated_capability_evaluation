import hydra
from omegaconf import DictConfig

from model import Model
from task import MathTask

from pprint import pprint


@hydra.main(version_base=None, config_path="cfg", config_name="run_cfg")
def main(cfg: DictConfig):
    """
    Run the model with the specified configuration.

    Args:
        cfg (DictConfig): Configuration for the model.
    """
    math_task = MathTask(
        cfg.tasks.task_cfgs.mathematics,
    )
    print(math_task)

    # # Create an instance of the Model class with the specified model name
    # model = Model(
    #     model_name=cfg.model.name,
    #     **cfg.prompt_cfg,
    # )

    # # Define a sample input
    # sample_input = "Are LLMs capable of generating evaluation benchmarks for self-evaluation?"

    # # Generate output using the model with specified generation arguments
    # gen_cfg = cfg.gen_cfg
    # output, metadata = model.generate(
    #     prompt=sample_input,
    #     generation_config=gen_cfg,
    # )

    # # Print the output
    # print(f"Output: {output}")
    # print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main()
