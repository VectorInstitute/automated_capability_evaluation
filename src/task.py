import importlib  # noqa: D100
import json
import os
import sys
from typing import Any, Dict

from src.model import Model
from src.utils.data_utils import load_data


class TaskSeedDataset:
    """
    A class to represent a task seed dataset with its configuration.

    Attributes
    ----------
    name : str
        The name of the task seed dataset.
    description : str
        The description of the task seed dataset.
    domain : str
        The domain of the task seed dataset.
    family : str
        The family of the task seed dataset.
    instructions : str
        The instructions for evaluating samples in the dataset.
    _cfg : dict
        The configuration dictionary for the task seed dataset.
    _data : Dataset
        The set of samples associated with the task seed dataset.

    Methods
    -------
    _load_dataset() -> None
        Loads the dataset based on the configuration.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.name = cfg["name"]
        self.description = cfg["description"]
        self.domain = cfg["domain"]
        self.family = cfg["family"]
        self.instructions = cfg["instructions"]

        self._cfg = cfg

        self._load_dataset()

    def _load_dataset(self) -> None:
        self._data = load_data(
            dataset_name=self._cfg["data_args"]["source"], **self._cfg["data_args"]
        )


class Task:
    """
    A class to represent a task.

    Attributes
    ----------
    source_dir : str
        The directory where the task files are located.
    name : str
        The name of the task.
    description : str
        A description of the task.
    domain : str
        The domain of the task.
    family : str
        The family of the task.
    instructions : str
        Instructions for the task.
    is_seed : bool
        Indicates if the task is a seed task.
    task_repr_class : type
        The class representing the task.

    Methods
    -------
    _load_task_json() -> None
        Loads the task configuration from a JSON file.
    _load_task_repr_class() -> None
        Loads the task representation class from a Python file.
    _to_dict() -> Dict[str, Any]
        Converts the task attributes to a dictionary.
    to_json_str() -> str
        Converts the task to a JSON string.
    __str__() -> str
        Returns a JSON string representation of the task.
    """

    def __init__(self, task_dir: str) -> None:
        self.source_dir = task_dir
        self._load_task_json()
        self._load_task_repr_class()

    def _load_task_json(self) -> None:
        with open(os.path.join(self.source_dir, "task.json"), "r") as f:
            _cfg = json.load(f)
        self.name = _cfg["task_name"]
        self.description = _cfg["task_description"]
        self.domain = _cfg["task_domain"]
        self.family = _cfg["task_family"]
        self.instructions = _cfg["task_instructions"]
        # TODO: Store data is stored in json or elsewhere?
        self._data = _cfg["task_data"]
        # Check if the task is a seed task, use source_dataset as indicator
        self.is_seed = "source_dataset" in _cfg

    def _load_task_repr_class(self) -> None:
        # Borrowed from: https://github.com/conglu1997/ACD/blob/main/generate_acd_tasks.py#L103C1-L117C23
        module_path = os.path.join(self.source_dir, "task.py")
        module_name = f"task_{os.path.basename(self.source_dir)}"
        task_module = import_from_path(module_name, module_path)
        task_class = task_module.Task
        # Check task class has required methods.
        for method in ["repr_samples", "get_instructions", "score"]:
            if not hasattr(task_class, method):
                raise AttributeError(f"Task class must define a {method} method.")
        self.task_repr_class = task_class

        with open(module_path, "r") as f:
            self.task_repr_class_str = f.read()

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "family": self.family,
            "class": self.task_repr_class_str,
        }

    def to_json_str(self) -> str:
        """
        Convert the task to a JSON string.

        Returns
        -------
        str
            A JSON string representation of the task.
        """
        return json.dumps(self._to_dict(), indent=4)

    def __str__(self) -> str:
        """
        Return a JSON string representation of the task.

        Returns
        -------
        str
            A JSON string representation of the task.
        """
        return self.to_json_str()

    def embed(self, embed_model: Model) -> None:
        """
        Embed the task using the provided model.

        Parameters
        ----------
        embed_model : Model
            The model to use for embedding the task.
        """
        raise NotImplementedError

    def evaluate_using_inspect(self, model: Model) -> None:  # noqa: D102
        # evaluate the task using inspect-evals
        raise NotImplementedError


def import_from_path(module_name: str, file_path: str) -> Any:
    """
    Import a module from a specified file path.

    Args:
        module_name (str): The name to assign to the imported module.
        file_path (str): The file path to the module to be imported.

    Returns
    -------
        Module: The imported module.

    Raises
    ------
        AssertionError: If the module specification or loader cannot be found.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
