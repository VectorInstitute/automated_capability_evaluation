"""
Contains unit tests for the Capability class.

The tests included in this module are:
- `test_create_capability`: Verifies the creation of a Capability object and
    checks its attributes.
- `test_capability_to_json_str`: Tests the serialization of a
    Capability object to a JSON string.

The Capability class is expected to be imported from the `src.capability` module.

Attributes
----------
capability_cfg : dict
    A dictionary containing the configuration
    for creating a Capability object.
capability : Capability
    An instance of the Capability class created
    using the `capability_cfg` configuration.

Functions
---------
test_create_capability()
    Test the creation of a Capability object and verify its attributes.
test_capability_to_json_str()
    Test the serialization of a Capability object
    to a JSON string and verify its content.
"""

import json
import os

from src.capability import Capability, CapabilitySeedDataset


# Define a capability seed dataset configuration and create an object
capability_seed_dataset_cfg = {
    "name": "mathematics",
    "description": "Solve mathematical problems.",
    "domain": "math",
    "family": "competition",
    "data_args": {
        "source": "qwedsacf/competition_math",
        "split": "train",
        "subset": "",
        "streaming": False,
        "num_repr_tasks": 3,
    },
    "instructions": "Solve the following mathematical problems.",
    "test_args": {
        "size": 12500,
    },
}
capability_seed_dataset = CapabilitySeedDataset(capability_seed_dataset_cfg)

# Define a capability configuration and create an object
capability_cfg = {
    "name": "math_competition_algebra",
    "description": "The math_competition_algebra capability consists of 2931 challenging competition mathematics problems in algebra. Each problem has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. It has 5 levels.\n",
    "domain": "math",
    "family": "competition",
    "instructions": """f\"\"\"Solve the following algebra math problem step by step. The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.\\n\\nProblem: {t[\"problem\"]}\\n\\nRemember to put your answer on its own line at the end in the form \"ANSWER:$ANSWER\" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\\\boxed command.\"\"\"""",
    "path": "seed_capabilities/math/math_competition_algebra",
    "scores_path": "seed_capabilities_scores",
    "scores": {
        "c4ai-command-r-plus": 0.34288121314237574,
        "gpt-4o": 0.8289806234203876,
    },
}
test_dir = os.path.dirname(os.path.abspath(__file__))
capability = Capability(os.path.join(test_dir, capability_cfg["path"]))


def test_create_capability_seed_dataset():
    """
    Test the creation of a capability seed dataset object.

    This test verifies that the capability seed dataset object is created successfully
    with the correct attributes and data. It checks the following:
    - The capability seed dataset's name matches the expected name
    from the configuration.
    - The capability seed dataset's description matches the expected description
    from the configuration.
    - The capability seed dataset's domain matches the expected domain
    from the configuration.
    - The capability seed dataset's family matches the expected family
    from the configuration.
    - The capability seed dataset's instructions match the expected instructions
    from the configuration.
    - The size of the capability seed dataset's data matches the expected size
    from the configuration.
    """
    # Check if the capability object is created successfully
    assert capability_seed_dataset.name == capability_seed_dataset_cfg["name"]
    assert (
        capability_seed_dataset.description
        == capability_seed_dataset_cfg["description"]
    )
    assert capability_seed_dataset.domain == capability_seed_dataset_cfg["domain"]
    assert capability_seed_dataset.family == capability_seed_dataset_cfg["family"]
    assert (
        capability_seed_dataset.instructions
        == capability_seed_dataset_cfg["instructions"]
    )
    assert (
        len(capability_seed_dataset._data)
        == capability_seed_dataset_cfg["test_args"]["size"]
    )


def test_create_capability():
    """
    Test the creation of a capability object.

    This test verifies that the capability object is created successfully
    with the correct attributes and data. It checks the following:
    - The capability's source directory matches the expected path
    from the configuration.
    - The capability's name matches the expected name from the configuration.
    - The capability's description matches the expected description
    from the configuration.
    - The capability's domain matches the expected domain from the configuration.
    - The capability's family matches the expected family from the configuration.
    - The capability's instructions match the expected instructions
    from the configuration.
    - The capability's representation class string is of type string.
    """
    # Check if the capability object is created successfully
    assert capability.source_dir == os.path.join(test_dir, capability_cfg["path"])
    assert capability.name == capability_cfg["name"]
    assert capability.description == capability_cfg["description"]
    assert capability.domain == capability_cfg["domain"]
    assert capability.family == capability_cfg["family"]
    assert capability.instructions == capability_cfg["instructions"]
    assert isinstance(capability.capability_repr_class_str, str)
    assert capability.capability_repr_class_str.startswith("```python\n")
    assert capability.capability_repr_class_str.endswith("\n```")


def test_capability_to_json_str():
    """
    Test the serialization of a capability object to a JSON string.

    This test verifies that the `to_json_str` method of the capability object correctly
    serializes the capability into a JSON string. It checks the following:
    - The result of `to_json_str` is a string.
    - The JSON string contains the keys "name", "description", "domain",
      "family" and "class".
    """
    # Check if the capability object can be serialized to JSON string
    capability_repr_json_str = capability.to_json_str()
    assert isinstance(capability_repr_json_str, str)

    capability_repr_json = json.loads(capability_repr_json_str)
    assert "name" in capability_repr_json
    assert "description" in capability_repr_json
    assert "domain" in capability_repr_json
    assert "family" in capability_repr_json
    assert "class" in capability_repr_json


def test_capability_load_scores():
    """
    Test the `load_scores` method of the `capability` object.

    This test verifies that the `load_scores` method correctly loads the scores
    from the specified directory and returns them as a dictionary. It checks the
    following:
    - The returned object is a dictionary.
    - The length of the returned dictionary matches the expected number of scores.
    - Each model in the expected scores is present in the returned dictionary.
    - The scores for each model match the expected scores.
    """
    scores_dir = os.path.join(test_dir, capability_cfg["scores_path"])
    scores_dict = capability.load_scores(scores_dir)
    assert isinstance(scores_dict, dict)
    assert len(scores_dict) == len(capability_cfg["scores"])
    for model, score in capability_cfg["scores"].items():
        assert model in scores_dict
        assert scores_dict[model] == score
