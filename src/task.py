import json  # noqa: D100

from datasets import Dataset

from utils import load_data


class Task:
    """
    A class to represent a task with a dataset and its configuration.

    Attributes
    ----------
    name : str
        The name of the task.
    description : str
        The description of the task.
    domain : str
        The domain of the task.
    _cfg : dict
        The configuration dictionary for the task.
    _data : Dataset
        The dataset associated with the task.
    _repr_samples : Dataset
        Representative samples from the dataset.
    """

    def __init__(self, cfg: dict):
        self.name = cfg.name
        self.description = cfg.description
        self.domain = cfg.domain

        self._cfg = cfg

        self._load_dataset()

    def _load_dataset(self):
        self._data = load_data(
            dataset_name=self._cfg.data_args.source, **self._cfg.data_args
        )
        self._create_repr_samples(
            self._data, num_samples=self._cfg.data_args.num_repr_samples
        )

    def _create_repr_samples(self, data: Dataset, num_samples: int = 5, seed: int = 42):
        # create representative samples by randomly sampling from the data
        self._repr_samples = data.shuffle(seed=seed).take(num_samples)

    def _to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "samples": self._repr_samples,
        }

    def to_json_str(self):
        """
        Convert the task to a JSON string.

        Returns
        -------
        str
            A JSON string representation of the task.
        """
        task_dict = self._to_dict()
        task_dict["samples"] = {
            (idx + 1): sample for idx, sample in enumerate(task_dict["samples"])
        }
        return json.dumps(task_dict)

    def __str__(self):
        """
        Return a JSON string representation of the task.

        Returns
        -------
        str
            A JSON string representation of the task.
        """
        return self.to_json_str()

    # TODO: Get feedback on the following methods
    def to_metr_format(self):  # noqa: D102
        # convert the task to METR format
        raise NotImplementedError

    def evaluate_using_inspect(self, model):  # noqa: D102
        # evaluate the task using inspect-evals
        raise NotImplementedError
