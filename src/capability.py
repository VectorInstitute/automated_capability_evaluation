import importlib  # noqa: D100
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

from src.utils.capability_utils import parse_python_class_str, read_score_inspect_json
from src.utils.constants import (
    BASE_ARTIFACTS_DIR,
    NON_SEED_CAPABILITIES_SCORE_DIR,
    SEED_CAPABILITIES_SCORE_DIR,
)
from src.utils.data_utils import load_data


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

    @classmethod
    def from_dict(cls, capability_dict: Dict[str, Any], base_dir: str) -> "Capability":
        """
        Create a Capability object from a dictionary.

        Args
        ----
        capability_dict (Dict[str, Any]): A dictionary containing
            the capability attributes.

        Returns
        -------
        Capability: A Capability object created from the dictionary.
        """
        for key in ["name", "description", "domain", "class"]:
            if key not in capability_dict:
                raise ValueError(f"Capability dictionary must contain a {key} key.")

        c_dict = capability_dict.copy()
        c_dir = os.path.join(base_dir, c_dict["name"])
        if os.path.exists(c_dir):
            raise FileExistsError(
                f"Capability directory already exists: {c_dir}. Please initialize capability directly using the directory: ```Capability(<capability_dir>)```."
            )
        os.makedirs(c_dir, exist_ok=False)

        # Extract instructions and tasks from the capability python class
        python_class_str = parse_python_class_str(c_dict.pop("class"))
        with open(os.path.join(c_dir, "capability.py"), "w") as f:
            f.write(python_class_str)
        c_module = _import_from_path(
            f"capability_{c_dict['name']}", os.path.join(c_dir, "capability.py")
        )
        c_obj = c_module.Capability()
        initial_tasks = list(c_obj.repr_tasks().values())
        template_instructions = c_obj.get_instructions({"problem": '{t["problem"]}'})
        template_instructions = f'f"""{template_instructions}"""'

        c_dict.update(
            {
                "capability_name": c_dict.pop("name"),
                "capability_description": c_dict.pop("description"),
                "capability_domain": c_dict.pop("domain"),
                "capability_instructions": template_instructions,
                "capability_data": initial_tasks,
            }
        )
        with open(os.path.join(c_dir, "capability.json"), "w") as f:
            json.dump(c_dict, f, indent=4)

        return cls(c_dir)

    def _load_capability_json(self) -> None:
        with open(os.path.join(self.source_dir, "capability.json"), "r") as f:
            _cfg = json.load(f)
        self.name = _cfg["capability_name"]
        self.description = _cfg["capability_description"]
        self.domain = _cfg["capability_domain"]
        self.instructions = _cfg["capability_instructions"]
        # TODO: Store data is stored in json or elsewhere?
        self._data = _cfg["capability_data"]
        # Check if the capability is a seed capability, use source_dataset as indicator
        self.is_seed = "source_dataset" in _cfg

    def _load_capability_repr_class(self) -> None:
        # Borrowed from: https://github.com/conglu1997/ACD/blob/main/generate_acd_tasks.py#L103C1-L117C23
        module_path = os.path.join(self.source_dir, "capability.py")
        module_name = f"capability_{os.path.basename(self.source_dir)}"
        capability_module = _import_from_path(module_name, module_path)
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

    def encode(self, encoder_model: Any) -> None:
        """
        Encode the capability using the provided encoder model.

        Args
        ----
        encoder_model : Any
            The model to use for encoding the capability.
        """
        # TODO: Implement capability encoding
        self.encoding = None
        raise NotImplementedError

    def _create_inspect_file(self) -> None:
        """
        Implement pipeline to evaluate the capability using the inspect framework.

        This involves converting the METR format to inspect solvers and scorers.
        """
        raise NotImplementedError

    def _evaluate_using_inspect(self, subject_llm: str) -> None:  # noqa: D102
        """
        Evaluate subject LLM on the capability using the inspect framework.

        Args
        ----
        subject_llm : str
            The name of the LLM to use for evaluation.
        """
        raise NotImplementedError

    def evaluate(self, subject_llms: List[str]) -> None:
        """
        Evaluate the provided subject LLMs on the capability.

        Args
        ----
        subject_llms : List[str]
            The name of the LLMs to use for evaluation.
        """
        # TODO: Run asynchronosly
        for model in subject_llms:
            self._evaluate_using_inspect(model)


def _import_from_path(module_name: str, file_path: str) -> Any:
    """
    Import a module from a specified file path.

    This is a helper function for loading the capability.py file as a module.

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


def evaluate_capabilities(
    domain: str,
    capabilities: List[str],
    subject_llms: List[str],
    **kwargs: Dict[str, Any],
) -> None:
    """
    Evaluate the subject LLMs on the capabilities.

    Args
    ----
        domain (str): The domain name.
        capabilities (List[str]): The list of capabilities to evaluate on.
        subject_llms (List[str]): The list of subject LLMs to evaluate.
    """
    if "trial_run" in kwargs:
        capability_dir = os.path.join(
            BASE_ARTIFACTS_DIR, f"capabilities_{kwargs['run_id']}", domain
        )
    else:
        capability_dir = os.path.join(BASE_ARTIFACTS_DIR, "capabilities", domain)

    # TODO: Run this asynchronosly
    for capability_name in capabilities:
        cap = Capability(os.path.join(capability_dir, capability_name))
        cap.evaluate(subject_llms=subject_llms)
