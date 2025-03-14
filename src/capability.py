import importlib  # noqa: D100
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List

from src.model import Model
from src.utils.capability_utils import read_score_inspect_json
from src.utils.data_utils import load_data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_CAPABILITIES_SCORE_DIR = os.path.join(BASE_DIR, "seed_capabilities_results")
NON_SEED_CAPABILITIES_SCORE_DIR = os.path.join(BASE_DIR, "capabilities_results")


class CapabilitySeedDataset:
    """
    A class to represent a capability seed dataset with its configuration.

    Attributes
    ----------
    name : str
        The name of the capability seed dataset.
    description : str
        The description of the capability seed dataset.
    domain : str
        The domain of the capability seed dataset.
    family : str
        The family of the capability seed dataset.
    instructions : str
        The instructions for evaluating tasks in the dataset.
    _cfg : dict
        The configuration dictionary for the capability seed dataset.
    _data : Dataset
        The set of tasks associated with the capability seed dataset.

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


class Capability:
    """
    A class to represent a capability.

    Attributes
    ----------
    source_dir : str
        The directory where the capability files are located.
    name : str
        The name of the capability.
    description : str
        A description of the capability.
    domain : str
        The domain of the capability.
    family : str
        The family of the capability.
    instructions : str
        Instructions for the capability.
    is_seed : bool
        Indicates if the capability is a seed capability.
    capability_repr_class : type
        The class representing the capability.

    Methods
    -------
    _load_capability_json() -> None
        Loads the capability configuration from a JSON file.
    _load_capability_repr_class() -> None
        Loads the capability representation class from a Python file.
    _to_dict() -> Dict[str, Any]
        Converts the capability attributes to a dictionary.
    to_json_str() -> str
        Converts the capability to a JSON string.
    __str__() -> str
        Returns a JSON string representation of the capability.
    """

    def __init__(self, capability_dir: str) -> None:
        self.source_dir = capability_dir
        self._load_capability_json()
        self._load_capability_repr_class()

        self.score_dir = (
            SEED_CAPABILITIES_SCORE_DIR
            if self.is_seed
            else NON_SEED_CAPABILITIES_SCORE_DIR
        )

    def _load_capability_json(self) -> None:
        with open(os.path.join(self.source_dir, "capability.json"), "r") as f:
            _cfg = json.load(f)
        self.name = _cfg["capability_name"]
        self.description = _cfg["capability_description"]
        self.domain = _cfg["capability_domain"]
        self.family = _cfg["capability_family"]
        self.instructions = _cfg["capability_instructions"]
        # TODO: Store data is stored in json or elsewhere?
        self._data = _cfg["capability_data"]
        # Check if the capability is a seed capability, use source_dataset as indicator
        self.is_seed = "source_dataset" in _cfg

    def _load_capability_repr_class(self) -> None:
        # Borrowed from: https://github.com/conglu1997/ACD/blob/main/generate_acd_tasks.py#L103C1-L117C23
        module_path = os.path.join(self.source_dir, "capability.py")
        module_name = f"capability_{os.path.basename(self.source_dir)}"
        capability_module = import_from_path(module_name, module_path)
        capability_class = capability_module.Capability
        # Check capability class has required methods.
        for method in ["repr_tasks", "get_instructions", "score"]:
            if not hasattr(capability_class, method):
                raise AttributeError(f"Capability class must define a {method} method.")
        self.capability_repr_class = capability_class

        with open(module_path, "r") as f:
            capability_repr_class_str = f.read()
        newline = "\n"
        self.capability_repr_class_str = (
            f"```python{newline}{capability_repr_class_str.strip(newline)}{newline}```"
        )

    def load_scores(self, scores_dir: str | None = None) -> Dict[str, float]:
        """
        Load scores from JSON files in the specified directory.

        Args
        ----
            scores_dir (str | None): The directory containing the score files.

        Returns
        -------
            Dict[str, float]: A dictionary where the keys are model names and
            the values are the scores.
        """
        scores_dir = scores_dir if scores_dir else self.score_dir
        scores_dict = defaultdict(float)
        for model in os.listdir(scores_dir):
            scores_file = os.path.join(
                scores_dir, model, self.domain, f"{self.name}.json"
            )
            if os.path.isfile(scores_file):
                scores_dict[model] = read_score_inspect_json(scores_file)
        return scores_dict

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "family": self.family,
            "class": self.capability_repr_class_str,
        }

    def to_json_str(self) -> str:
        """
        Convert the capability to a JSON string.

        Returns
        -------
        str
            A JSON string representation of the capability.
        """
        return json.dumps(self._to_dict(), indent=4)

    def __str__(self) -> str:
        """
        Return a JSON string representation of the capability.

        Returns
        -------
        str
            A JSON string representation of the capability.
        """
        return self.to_json_str()

    def embed(self, embed_model: Model) -> None:
        """
        Embed the capability using the provided model.

        Parameters
        ----------
        embed_model : Model
            The model to use for embedding the capability.
        """
        raise NotImplementedError

    def evaluate_using_inspect(self, model: Model) -> None:  # noqa: D102
        # evaluate the capability using inspect-evals
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


def select_seed_capabilities(
    seed_capability_dir: str,
    num_seed_capabilities: int = -1,
    include_capabilities: List[str] | None = None,
    random_seed: int = 42,
) -> List[Capability]:
    """
    Select `num_seed_capabilities` seed capabilities from the specified directory.

    Args
    ----
        seed_capability_dir (str): The directory containing the seed capabilities.
        num_seed_capabilities (int): The number of seed capabilities to select.
        include_capabilities (List[str] | None): A list of capability names to include.
        random_seed (int): The seed for the random number generator.

    Returns
    -------
        List[Capability]: A list of capability objects.
    """
    random.seed(random_seed)

    selected_seed_capabilities = []
    all_seed_capability_paths = os.listdir(seed_capability_dir)

    # Select all capabilities if num_seed_capabilities is -1
    if num_seed_capabilities == -1:
        num_seed_capabilities = len(all_seed_capability_paths)
        include_capabilities = None

    # Force include some capabilities
    if include_capabilities is not None:
        assert num_seed_capabilities >= len(include_capabilities), (
            "Number of seed capabilities is less than the number of capabilities to include."
        )
        for capability_name in include_capabilities:
            capability = Capability(os.path.join(seed_capability_dir, capability_name))
            selected_seed_capabilities.append(capability)
            all_seed_capability_paths.remove(capability_name)
        num_seed_capabilities -= len(include_capabilities)

    # TODO: Enhance the selection criterion
    for capability_path in random.sample(
        all_seed_capability_paths, num_seed_capabilities
    ):
        capability = Capability(os.path.join(seed_capability_dir, capability_path))
        selected_seed_capabilities.append(capability)

    return selected_seed_capabilities


def get_capability_repr_with_score(capability: Capability, model_name: str) -> str:
    """
    Get the capability representation with score for the specified model.

    Args
    ----
        capability (Capability): The capability to get the representation for.
        model_name (str): The name of the model to use for scoring the capability.

    Returns
    -------
        str: A JSON string containing the capability representation and score.
    """
    model_score = capability.load_scores()[model_name]
    capability_dict = capability._to_dict()
    capability_dict["score"] = model_score
    return json.dumps(capability_dict, indent=4)
