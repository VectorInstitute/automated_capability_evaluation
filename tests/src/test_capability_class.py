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
import shutil
from unittest.mock import MagicMock, patch

import pytest

from src.capability import Capability, CapabilitySeedDataset, _import_from_path
from src.utils.capability_utils import extract_and_parse_response, run_inspect_evals
from src.utils.constants import BASE_INSPECT_EVALS_DIR


# Define a capability seed dataset configuration and create an object
capability_seed_dataset_cfg = {
    "name": "mathematics",
    "description": "Solve mathematical problems.",
    "domain": "math",
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
    "name": "competition_math_algebra",
    "description": "The competition_math_algebra capability consists of 2931 challenging competition mathematics problems in algebra. Each problem has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. It has 5 levels.\n",
    "domain": "math",
    "instructions": """f\"\"\"Solve the following algebra math problem step by step. The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.\\n\\nProblem: {t[\"problem\"]}\\n\\nRemember to put your answer on its own line at the end in the form \"ANSWER:$ANSWER\" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\\\boxed command.\"\"\"""",
    "path": "seed_capabilities/math/competition_math_algebra",
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
    - The capability's instructions match the expected instructions
    from the configuration.
    - The capability's representation class string is of type string.
    """
    # Check if the capability object is created successfully
    assert capability.source_dir == os.path.join(test_dir, capability_cfg["path"])
    assert capability.name == capability_cfg["name"]
    assert capability.description == capability_cfg["description"]
    assert capability.domain == capability_cfg["domain"]
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
      and "class".
    """
    # Check if the capability object can be serialized to JSON string
    capability_repr_json_str = capability.to_json_str()
    assert isinstance(capability_repr_json_str, str)

    capability_repr_json = json.loads(capability_repr_json_str)
    assert "name" in capability_repr_json
    assert "description" in capability_repr_json
    assert "domain" in capability_repr_json
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


def test_create_capability_from_dict():
    """
    Test the initialization of a capability object from a dictionary.

    This test verifies that a capability object can be initialized from a dictionary
    representation of the capability. It checks the following:
    - The capability object is created successfully from the dictionary.
    - The attributes of the capability match the values in the dictionary.
    """
    try:
        gen_capability_dict = {
            "name": "math_mathematics_combinatorial_proofs",
            "description": "The math_mathematics_combinatorial_proofs capability consists of 1500 challenging combinatorial proof problems. These problems require the model to provide rigorous combinatorial arguments to prove identities or count specific configurations.",
            "domain": "math",
            "class": '```python\nclass Capability:\n    @staticmethod\n    def repr_tasks() -> dict[str, dict]:\n        return {\n    "1": {\n        "problem": "Prove that the number of ways to choose 2 elements from a set of n elements is equal to the number of ways to choose n-2 elements from the same set.",\n        "answer": "\\\\binom{n}{2} = \\\\binom{n}{n-2}"\n    },\n    "2": {\n        "problem": "Show that for any positive integer n, the sum of the first n odd numbers equals n^2.",\n        "answer": "1 + 3 + 5 + ... + (2n-1) = n^2"\n    },\n    "3": {\n        "problem": "Demonstrate that \\sum_{k=0}^{n} \\binom{n}{k} = 2^n using a combinatorial argument.",\n        "answer": "\\\\sum_{k=0}^{n} \\binom{n}{k} = 2^n"\n    }\n}\n\n    @staticmethod\n    def get_instructions(t: dict) -> str:\n        return f"""Provide a combinatorial proof for the following problem. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is your proof or explanation.\\n\\nProblem: {t["problem"]}\\n\\nRemember to put your proof or explanation on its own line at the end in the form "ANSWER:$ANSWER" (without quotes) where $ANSWER is your proof or explanation."""\n\n    @staticmethod\n    def score(t: dict, submission: str) -> float | None:\n        return 1.0 if submission.lower().strip() == t["answer"].lower().strip() else 0.0\n```',
        }
        gen_capability_tests_dir = os.path.join(
            test_dir, "capabilities_t1", gen_capability_dict["domain"]
        )
        os.makedirs(gen_capability_tests_dir, exist_ok=True)

        capability = Capability.from_dict(
            capability_dict=gen_capability_dict, base_dir=gen_capability_tests_dir
        )
        assert capability.name == gen_capability_dict["name"]
        assert capability.description == gen_capability_dict["description"]
        assert capability.domain == gen_capability_dict["domain"]
        shutil.rmtree(os.path.join(test_dir, "capabilities_t1"))
    except Exception as e:
        if os.path.exists(os.path.join(test_dir, "capabilities_t1")):
            # Clean up the test directory if it exists
            shutil.rmtree(os.path.join(test_dir, "capabilities_t1"))
        raise e


def test_extract_and_parse_response():
    """
    Test the extract_and_parse_response function.

    This test verifies that the extract_and_parse_response function correctly
    extracts and parses the response from a given dummy response string.
    The dummy response string contains a "thought" and a "response_json" with
    multiple entries. The test checks that the extracted response dictionary
    has the same "thought" and the same number of parsed responses as the
    dummy response dictionary.

    Assertions:
        - The "thought" in the extracted response dictionary matches the
          "thought" in the dummy response dictionary.
        - The number of parsed responses in the extracted response dictionary
          matches the number of entries in the "response_json" of the dummy
          response dictionary.
    """
    dummy_response_dict = {
        "thought": "<THOUGHT>",
        "response_json": {
            "0": {},
            "1": {},
            "2": {},
        },
    }
    dummy_response = f"""THOUGHT:\n{dummy_response_dict["thought"]}\n\nRESPONSE JSON:\n{json.dumps(dummy_response_dict["response_json"])}"""
    extracted_response_dict = extract_and_parse_response(dummy_response)
    assert extracted_response_dict["thought"] == dummy_response_dict["thought"]
    assert len(extracted_response_dict["parsed_response"]) == len(
        dummy_response_dict["response_json"]
    )


def test_add_and_update_tasks_new_task():
    """
    Test the add_and_update_tasks method of the Capability class.

    This test verifies that the add_and_update_tasks method correctly
    adds and updates tasks in the capability. It checks the following:
    - The number of tasks in the capability is updated correctly.
    - The tasks are added and updated as expected.
    - The capability's representation string remains unchanged.
    """
    try:
        capability_path = (
            "capabilities_t2/math/competition_math_combinatorial_challenges"
        )
        # Create a copy
        shutil.copytree(
            os.path.join(test_dir, capability_path),
            os.path.join(test_dir, f"copy_{capability_path}"),
            dirs_exist_ok=True,
        )
        # Read the capability configuration from the copy
        capability = Capability(os.path.join(test_dir, f"copy_{capability_path}"))
        # Create a list of new tasks to add
        new_tasks = [
            {"id": "4", "problem": "Problem 4", "answer": "Answer 4"},
        ]
        # Add and update tasks in the capability
        capability.add_and_update_tasks(tasks=new_tasks)
        # Check if the number of tasks is updated correctly and
        # the update doesnt affect the representation string
        original_capability = Capability(os.path.join(test_dir, capability_path))
        assert (len(original_capability._data) + len(new_tasks)) == len(
            capability._data
        )
        assert (
            capability.capability_repr_class_str
            == original_capability.capability_repr_class_str
        )
        # Clean up
        shutil.rmtree(os.path.join(test_dir, f"copy_{capability_path.split('/')[0]}"))
    except Exception as e:
        if os.path.exists(
            os.path.join(test_dir, f"copy_{capability_path.split('/')[0]}")
        ):
            # Clean up the capability path if it exists
            shutil.rmtree(
                os.path.join(test_dir, f"copy_{capability_path.split('/')[0]}")
            )
        raise e


def test_add_and_update_tasks_repr_tasks():
    """
    Test the add_and_update_tasks method of the Capability class.

    Test the add_and_update_tasks method of the Capability class.
    This test verifies that the add_and_update_tasks method correctly
    updates representative tasks in the capability. It checks the following:
    - The number of tasks in the capability is updated correctly.
    - The tasks are updated as expected.
    - The capability's representation string is updated.
    """
    try:
        capability_path = (
            "capabilities_t2/math/competition_math_combinatorial_challenges"
        )
        # Create a copy
        shutil.copytree(
            os.path.join(test_dir, capability_path),
            os.path.join(test_dir, f"copy_{capability_path}"),
            dirs_exist_ok=True,
        )
        # Read the capability configuration from the copy
        capability = Capability(os.path.join(test_dir, f"copy_{capability_path}"))
        # Create a list of new tasks to add
        repr_tasks = [
            {"id": "1", "problem": "Problem 1", "answer": "Answer 1"},
            {"id": "2", "problem": "Problem 2", "answer": "Answer 2"},
            {"id": "3", "problem": "Problem 3", "answer": "Answer 3"},
        ]
        repr_tasks.sort(key=lambda x: x["id"])
        # Add and update tasks in the capability
        capability.add_and_update_tasks(tasks=repr_tasks)
        # Check if the existing tasks are updated correctly
        original_capability = Capability(os.path.join(test_dir, capability_path))
        assert len(original_capability._data) == len(capability._data)
        assert capability._data[0]["problem"] == repr_tasks[0]["problem"]
        assert capability._data[1]["problem"] == repr_tasks[1]["problem"]
        assert capability._data[2]["problem"] == repr_tasks[2]["problem"]
        assert capability._data[0]["answer"] == repr_tasks[0]["answer"]
        assert capability._data[1]["answer"] == repr_tasks[1]["answer"]
        assert capability._data[2]["answer"] == repr_tasks[2]["answer"]
        assert (
            capability.capability_repr_class_str
            != original_capability.capability_repr_class_str
        )
        # Clean up
        shutil.rmtree(os.path.join(test_dir, f"copy_{capability_path.split('/')[0]}"))
    except Exception as e:
        if os.path.exists(
            os.path.join(test_dir, f"copy_{capability_path.split('/')[0]}")
        ):
            # Clean up the capability path if it exists
            shutil.rmtree(
                os.path.join(test_dir, f"copy_{capability_path.split('/')[0]}")
            )
        raise e


def test__create_inspect_file_default_judge():
    """
    Test the _create_inspect_file method of the Capability class.

    This test verifies that the _create_inspect_file method correctly
    creates the inspect file and its associated files. It checks the following:
    - The inspect file is created in the specified path.
    - The inspect file contains the expected functions and attributes.
    - The utils.py file contains the expected judge LLM and generation
      configuration.
    """
    try:
        capability_path = (
            "capabilities_t2/math/competition_math_combinatorial_challenges"
        )
        capability = Capability(os.path.join(test_dir, capability_path))
        capability.name = f"test_{capability.name}"
        inspect_path = os.path.join(BASE_INSPECT_EVALS_DIR, capability.name)
        os.makedirs(inspect_path)
        capability._create_inspect_file(
            path=inspect_path,
            judge_llm_name=None,
            judge_llm_gen_args=None,
        )
        # TODO: Add assertions to check if the file is created correctly
        # Check if all required files are created
        assert os.path.exists(os.path.join(inspect_path, "dataset.jsonl"))
        assert os.path.exists(os.path.join(inspect_path, "__init__.py"))
        assert os.path.exists(os.path.join(inspect_path, "README.md"))
        assert os.path.exists(os.path.join(inspect_path, "utils.py"))
        assert os.path.exists(os.path.join(inspect_path, f"{capability.name}.py"))
        # Check the format and contents of the main script file
        inspect_module = _import_from_path(
            module_name=f"{capability.name}_inspect_eval_script",
            file_path=os.path.join(inspect_path, f"{capability.name}.py"),
        )
        assert hasattr(inspect_module, "_score")
        assert hasattr(inspect_module, "custom_scorer")
        assert hasattr(inspect_module, capability.name)
        with open(os.path.join(inspect_path, f"{capability.name}.py"), "r") as f:
            content = f.read()
        assert "async def _score" in content
        assert "correct = await evaluate_with_llm_judge" in content
        assert f"from {capability.name}.utils" in content
        # Check the contents of the utils.py file
        with open(os.path.join(inspect_path, "utils.py"), "r") as f:
            content = f.read()
        assert 'INSPECT_JUDGE_LLM = "openai/gpt-4o-mini"' in content
        assert "INSPECT_JUDGE_LLM_GEN_CONFIG: Dict[str, Any] = {}" in content
        shutil.rmtree(inspect_path)
    except Exception as e:
        if os.path.exists(inspect_path):
            # Clean up the inspect path if it exists
            shutil.rmtree(inspect_path)
        raise e


def test__create_inspect_file_w_judge():
    """
    Test the _create_inspect_file method of the Capability class with a custom judge.

    This test verifies that the _create_inspect_file method correctly
    creates the inspect file and its associated files when a custom judge
    is provided. It checks the following:
    - The utils.py file contains the expected judge LLM and generation
      configuration.
    """
    try:
        capability_path = (
            "capabilities_t2/math/competition_math_combinatorial_challenges"
        )
        capability = Capability(os.path.join(test_dir, capability_path))
        capability.name = f"test_{capability.name}"
        inspect_path = os.path.join(BASE_INSPECT_EVALS_DIR, capability.name)
        os.makedirs(inspect_path)
        dummy_judge_llm_name = "openai/gpt-4o"
        dummy_judge_llm_gen_args = {
            "temperature": 0.7,
            "max_tokens": 100,
        }
        capability._create_inspect_file(
            path=inspect_path,
            judge_llm_name=dummy_judge_llm_name,
            judge_llm_gen_args=dummy_judge_llm_gen_args,
        )
        # Check the contents of the utils.py file
        with open(os.path.join(inspect_path, "utils.py"), "r") as f:
            content = f.read()
        assert f'INSPECT_JUDGE_LLM = "{dummy_judge_llm_name}"' in content
        assert (
            f"INSPECT_JUDGE_LLM_GEN_CONFIG: Dict[str, Any] = {json.dumps(dummy_judge_llm_gen_args, indent=4)}"
            in content
        )
        shutil.rmtree(inspect_path)
    except Exception as e:
        if os.path.exists(inspect_path):
            # Clean up the inspect path if it exists
            shutil.rmtree(inspect_path)
        raise e


def test_run_inspect_evals_success():
    """
    Test the `run_inspect_evals` function for a successful evaluation.

    This test verifies that the function correctly runs the inspect evals
    command and processes the results when the evaluation is successful.
    """
    path = "capabilities/math/algebra"
    model = MagicMock()
    model.get_model_name.side_effect = (
        lambda with_provider: "local-model" if with_provider else "model"
    )
    model.model_provider = "local"
    model.model_url = "http://localhost:8000"
    log_dir = "logs"

    mock_eval_log = MagicMock()
    mock_eval_log.status = "success"
    mock_eval_log.stats.model_usage = {
        "openai-model": MagicMock(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            reasoning_tokens=30,
        )
    }

    with (
        patch("src.utils.capability_utils.inspect_eval", return_value=[mock_eval_log]),
        patch(
            "src.utils.capability_utils.traceable",
            side_effect=lambda *args, **kwargs: lambda f: f,
        ),
    ):
        run_inspect_evals(path, model, log_dir)

    model.get_model_name.assert_any_call(with_provider=True)
    model.get_model_name.assert_any_call(with_provider=False)


def test_run_inspect_evals_error():
    """
    Test the `run_inspect_evals` function for an evaluation error.

    This test verifies that the function raises a ValueError when the
    evaluation fails with an error status.
    """
    path = "capabilities/math/algebra"
    model = MagicMock()
    model.get_model_name.side_effect = (
        lambda with_provider: "openai-model" if with_provider else "model"
    )
    model.model_provider = "openai"
    log_dir = "logs"

    mock_eval_log = MagicMock()
    mock_eval_log.status = "error"
    mock_eval_log.error = "Evaluation failed"

    with (
        patch("src.utils.capability_utils.inspect_eval", return_value=[mock_eval_log]),
        patch(
            "src.utils.capability_utils.traceable",
            side_effect=lambda *args, **kwargs: lambda f: f,
        ),
        pytest.raises(ValueError, match="Error running inspect evals"),
    ):
        run_inspect_evals(path, model, log_dir)


def test_run_inspect_evals_local_model():
    """
    Test the `run_inspect_evals` function for a local model.

    This test verifies that the function correctly sets and resets the
    `OPENAI_BASE_URL` environment variable when using a local model.
    """
    path = "capabilities/math/algebra"
    model = MagicMock()
    model.get_model_name.side_effect = (
        lambda with_provider: "local-model" if with_provider else "model"
    )
    model.model_provider = "local"
    model.model_url = "http://localhost:8000"
    log_dir = "logs"

    mock_eval_log = MagicMock()
    mock_eval_log.status = "success"
    mock_eval_log.stats.model_usage = {}

    with (
        patch("src.utils.capability_utils.inspect_eval", return_value=[mock_eval_log]),
        patch(
            "src.utils.capability_utils.traceable",
            side_effect=lambda *args, **kwargs: lambda f: f,
        ),
        patch.dict(
            "os.environ", {"ORIGINAL_OPENAI_BASE_URL": "https://api.openai.com"}
        ),
    ):
        run_inspect_evals(path, model, log_dir)
        assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com"


def test_run_inspect_evals_with_kwargs():
    """
    Test the `run_inspect_evals` function with additional keyword arguments.

    This test verifies that the function correctly processes additional
    keyword arguments and includes them in the metadata.
    """
    path = "capabilities/math/algebra"
    model = MagicMock()
    model.get_model_name.side_effect = (
        lambda with_provider: "openai-model" if with_provider else "model"
    )
    model.model_provider = "openai"
    log_dir = "logs"
    kwargs = {"temperature": 0.7, "max_tokens": 100}

    mock_eval_log = MagicMock()
    mock_eval_log.status = "success"
    mock_eval_log.stats.model_usage = {}

    with (
        patch("src.utils.capability_utils.inspect_eval", return_value=[mock_eval_log]),
        patch(
            "src.utils.capability_utils.traceable",
            side_effect=lambda *args, **kwargs: lambda f: f,
        ),
    ):
        run_inspect_evals(path, model, log_dir, **kwargs)
