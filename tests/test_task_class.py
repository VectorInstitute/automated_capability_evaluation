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

from src.task import Task


# Define a task configuration and create a task object
task_cfg = {
    "name": "mathematics",
    "description": "Solve mathematical problems.",
    "domain": "math",
    "data_args": {
        "source": "qwedsacf/competition_math",
        "split": "train",
        "subset": "",
        "streaming": False,
        "num_repr_samples": 3,
    },
    "test_args": {
        "size": 12500,
    },
}
task = Task(task_cfg)


def test_create_task():
    """
    Test the creation of a task object.

    This test verifies that the task object is created successfully with the
    correct attributes and data. It checks the following:
    - The task's name matches the expected name from the configuration.
    - The task's description matches the expected description from the configuration.
    - The task's domain matches the expected domain from the configuration.
    - The size of the task's data matches the expected size from the configuration.
    - The number of representative samples in the task matches the expected number
      from the configuration.
    """
    # Check if the task object is created successfully
    assert task.name == task_cfg["name"]
    assert task.description == task_cfg["description"]
    assert task.domain == task_cfg["domain"]
    assert len(task._data) == task_cfg["test_args"]["size"]
    assert len(task._repr_samples) == task_cfg["data_args"]["num_repr_samples"]


def test_task_to_json_str():
    """
    Test the serialization of a task object to a JSON string.

    This test verifies that the `to_json_str` method of the task object correctly
    serializes the task into a JSON string. It checks the following:
    - The result of `to_json_str` is a string.
    - The JSON string contains the keys "name", "description", "domain", and "samples".
    - The "samples" key contains the expected number of samples as specified
    in the task configuration.
    - Each sample is correctly indexed in the JSON representation.

    Raises
    ------
        AssertionError: If any of the checks fail.
    """
    # Check if the task object can be serialized to JSON string
    task_repr_json_str = task.to_json_str()
    assert isinstance(task_repr_json_str, str)

    task_repr_json = json.loads(task_repr_json_str)
    assert "name" in task_repr_json
    assert "description" in task_repr_json
    assert "domain" in task_repr_json
    assert "samples" in task_repr_json
    assert len(task_repr_json["samples"]) == task_cfg["data_args"]["num_repr_samples"]
    for idx in range(task_cfg["data_args"]["num_repr_samples"]):
        assert str(idx + 1) in task_repr_json["samples"]
