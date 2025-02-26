"""
Contains unit tests for the Task class.

The tests included in this module are:
- `test_create_task`: Verifies the creation of a Task object and checks its attributes.
- `test_task_to_json_str`: Tests the serialization of a Task object to a JSON string.

The Task class is expected to be imported from the `src.task` module.

Attributes
----------
task_cfg : dict
    A dictionary containing the configuration for creating a Task object.
task : Task
    An instance of the Task class created using the `task_cfg` configuration.

Functions
---------
test_create_task()
    Test the creation of a Task object and verify its attributes.
test_task_to_json_str()
    Test the serialization of a Task object to a JSON string and verify its content.
"""

import json
import os

from src.task import Task, TaskSeedDataset


# Define a task seed dataset configuration and create an object
task_seed_dataset_cfg = {
    "name": "mathematics",
    "description": "Solve mathematical problems.",
    "domain": "math",
    "family": "competition",
    "data_args": {
        "source": "qwedsacf/competition_math",
        "split": "train",
        "subset": "",
        "streaming": False,
        "num_repr_samples": 3,
    },
    "instructions": "Solve the following mathematical problems.",
    "test_args": {
        "size": 12500,
    },
}
task_seed_dataset = TaskSeedDataset(task_seed_dataset_cfg)

# Define a task configuration and create an object
task_cfg = {
    "name": "math_competition_algebra",
    "description": "The math_competition_algebra task consists of 2931 challenging competition mathematics problems in algebra. Each problem has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. It has 5 levels.\n",
    "domain": "math",
    "family": "competition",
    "instructions": """f\"\"\"Solve the following algebra math problem step by step. The last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the problem.\\n\\nProblem: {t[\"problem\"]}\\n\\nRemember to put your answer on its own line at the end in the form \"ANSWER:$ANSWER\" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\\\boxed command.\"\"\"""",
    "path": "seed_tasks/math/math_competition_algebra",
}
test_dir = os.path.dirname(os.path.abspath(__file__))
task = Task(os.path.join(test_dir, task_cfg["path"]))


def test_create_task_seed_dataset():
    """
    Test the creation of a task seed dataset object.

    This test verifies that the task seed dataset object is created successfully
    with the correct attributes and data. It checks the following:
    - The task seed dataset's name matches the expected name from the configuration.
    - The task seed dataset's description matches the expected description
    from the configuration.
    - The task seed dataset's domain matches the expected domain from the configuration.
    - The task seed dataset's family matches the expected family from the configuration.
    - The task seed dataset's instructions match the expected instructions
    from the configuration.
    - The size of the task seed dataset's data matches the expected size
    from the configuration.
    """
    # Check if the task object is created successfully
    assert task_seed_dataset.name == task_seed_dataset_cfg["name"]
    assert task_seed_dataset.description == task_seed_dataset_cfg["description"]
    assert task_seed_dataset.domain == task_seed_dataset_cfg["domain"]
    assert task_seed_dataset.family == task_seed_dataset_cfg["family"]
    assert task_seed_dataset.instructions == task_seed_dataset_cfg["instructions"]
    assert len(task_seed_dataset._data) == task_seed_dataset_cfg["test_args"]["size"]


def test_create_task():
    """
    Test the creation of a task object.

    This test verifies that the task object is created successfully with the correct
    attributes and data. It checks the following:
    - The task's source directory matches the expected path from the configuration.
    - The task's name matches the expected name from the configuration.
    - The task's description matches the expected description from the configuration.
    - The task's domain matches the expected domain from the configuration.
    - The task's family matches the expected family from the configuration.
    - The task's instructions match the expected instructions from the configuration.
    - The task's representation class string is of type string.
    """
    # Check if the task object is created successfully
    assert task.source_dir == os.path.join(test_dir, task_cfg["path"])
    assert task.name == task_cfg["name"]
    assert task.description == task_cfg["description"]
    assert task.domain == task_cfg["domain"]
    assert task.family == task_cfg["family"]
    assert task.instructions == task_cfg["instructions"]
    assert isinstance(task.task_repr_class_str, str)


def test_task_to_json_str():
    """
    Test the serialization of a task object to a JSON string.

    This test verifies that the `to_json_str` method of the task object correctly
    serializes the task into a JSON string. It checks the following:
    - The result of `to_json_str` is a string.
    - The JSON string contains the keys "name", "description", "domain",
      "family" and "class".
    """
    # Check if the task object can be serialized to JSON string
    task_repr_json_str = task.to_json_str()
    assert isinstance(task_repr_json_str, str)

    task_repr_json = json.loads(task_repr_json_str)
    assert "name" in task_repr_json
    assert "description" in task_repr_json
    assert "domain" in task_repr_json
    assert "family" in task_repr_json
    assert "class" in task_repr_json
