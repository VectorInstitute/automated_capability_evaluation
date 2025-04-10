import importlib  # noqa: D100
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.model import Model
from src.utils.capability_utils import (
    parse_python_class_str,
    read_score_inspect_json,
    run_inspect_evals,
)
from src.utils.constants import (
    BASE_ARTIFACTS_DIR,
    BASE_INSPECT_EVALS_DIR,
    GCP_BASE_ARTIFACTS_DIR,
    NO_ANSWER_STR,
    NON_SEED_CAPABILITIES_SCORE_DIR,
    SEED_CAPABILITIES_SCORE_DIR,
    TAB_W_SPACES,
)
from src.utils.data_utils import (
    list_dir,
    load_data,
    path_exists,
    transfer_inspect_log_to_gcp,
)
from src.utils.inspect_eval_utils import (
    parse_submission,
)
from src.utils.prompts import TASK_SOLVER_SYSTEM_PROMPT
from src.utils.templates import (
    INSPECT_EVALS_INIT_FILE_TEMPLATE,
    INSPECT_EVALS_README_FILE_TEMPLATE,
    INSPECT_EVALS_SCRIPT_FILE_TEMPLATE,
)


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
        base_dir (str): The base directory where the capability
            directory will be created

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
        initial_tasks = [
            {"id": k, "problem": v["problem"], "answer": v["answer"]}
            for k, v in c_obj.repr_tasks().items()
        ]
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
        self._data: List[Dict[str, Any]] = _cfg["capability_data"]
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
        for model in list_dir(scores_dir):
            scores_file = os.path.join(
                scores_dir, model, self.domain, f"{self.name}.json"
            )
            if path_exists(scores_file):
                scores_dict[model] = read_score_inspect_json(scores_file)
        return scores_dict

    def get_repr_tasks(self) -> List[Dict[str, Any]]:
        """
        Get the representative tasks for the capability.

        Returns
        -------
        List[Dict[Any]]: A list of dictionaries containing the representative tasks.
            Each task dict consists of id, problem, and answer keys.
        """
        repr_tasks = []
        for task_id, task_data in self.capability_repr_class.repr_tasks().items():
            repr_tasks.append(
                {
                    "id": task_id,
                    "problem": task_data["problem"],
                    "answer": task_data["answer"],
                }
            )
        return repr_tasks

    def add_and_update_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """
        Add and/or update tasks for the capability.

        Args
        ----
            tasks (List[Dict[str, Any]]): A list of dictionaries containing the tasks
            to be added. Each task dict consists of id, problem, and answer keys.
        """
        if not all(
            "id" in task and "problem" in task and "answer" in task for task in tasks
        ):
            raise ValueError(
                "Each task must contain 'id', 'problem', and 'answer' keys."
            )

        existing_tasks = self.get_tasks()
        existing_task_ids = [task["id"] for task in existing_tasks]
        new_task_ids = [task["id"] for task in tasks]
        # Keep new task for overlapping tasks
        # TODO: Add `overwrite` flag to update existing tasks
        tasks_to_keep = [
            task
            for task in existing_tasks
            if task["id"]
            not in list(set.intersection(set(existing_task_ids), set(new_task_ids)))
        ] + tasks
        # Sort by task id
        tasks_to_keep.sort(key=lambda x: x["id"])

        # Check if the new task list consists of representative tasks
        # If yes, update the capability class python file
        repr_tasks = [
            task
            for task in tasks
            if task["id"] in self.capability_repr_class.repr_tasks()
        ]
        if repr_tasks:
            partial_repr_task_ids = [task["id"] for task in repr_tasks]
            missing_repr_tasks = {
                k: v
                for k, v in self.capability_repr_class.repr_tasks().items()
                if k not in partial_repr_task_ids
            }
            for task_id, task_data in missing_repr_tasks.items():
                repr_tasks.append({"id": task_id, **task_data})
            repr_tasks.sort(key=lambda x: x["id"])
            # Update the capability class python file
            # Extract str which contains the repr_tasks dictionary
            # TODO: Since these are hardcoded, update when the format changes
            prefix_str = f"def repr_tasks() -> dict[str, dict]:\n{TAB_W_SPACES}{TAB_W_SPACES}return "
            suffix_str = f"\n\n{TAB_W_SPACES}@staticmethod\n{TAB_W_SPACES}def get_instructions(t: dict) -> str:"
            prev_repr_tasks_str = self.capability_repr_class_str.split(prefix_str)[
                1
            ].split(suffix_str)[0]
            # Restructure to match the original format
            repr_tasks_dict = {}
            for elm in repr_tasks:
                repr_tasks_dict[elm["id"]] = {k: v for k, v in elm.items() if k != "id"}
            # Replace the repr_tasks dictionary in the capability class string
            # with the updated one
            updated_repr_tasks_str = json.dumps(repr_tasks_dict, indent=4)
            newline = "\n"
            capability_repr_class_str = self.capability_repr_class_str.lstrip(
                f"```python{newline}"
            ).rstrip(f"{newline}```")
            capability_repr_class_str = capability_repr_class_str.replace(
                prev_repr_tasks_str,
                updated_repr_tasks_str,
            )
            with open(os.path.join(self.source_dir, "capability.py"), "w") as f:
                f.write(capability_repr_class_str)

        # Update the capability data in the capability json file
        c_dict = {
            "capability_name": self.name,
            "capability_description": self.description,
            "capability_domain": self.domain,
            "capability_instructions": self.instructions,
            "capability_data": tasks_to_keep,
        }
        with open(os.path.join(self.source_dir, "capability.json"), "w") as f:
            json.dump(c_dict, f, indent=4)

        # Reload the capability class to reflect these changes
        self._load_capability_json()
        self._load_capability_repr_class()

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

    def __repr__(self) -> str:
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

    def _solve_task(
        self, task: Dict[str, Any], llm: Model, gen_cfg: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Solve the task using the given LLM.

        Args
        ----
            task (Dict[str, Any]): The task dictionary containing the ID
            and the problem to solve.
            llm (Model): The LLM to use for solving the task.
            gen_cfg (Dict[str, Any]): The generation configuration for the LLM.

        Returns
        -------
            Tuple[str, Dict[str, Any]]: A tuple containing the answer as a string
            and metadata as a dictionary, which includes raw response and
            input/output tokens.
        """
        # Generate answer using the LLM
        # TODO:
        #  1. Enable tool use
        #  2. How to link this function with the Inspect Solver
        #   to be used in _evaluate_using_inspect()?
        print(f"Solving task {task['id']} ...")
        sys_prompt = TASK_SOLVER_SYSTEM_PROMPT.format(
            capability_name=self.name, capability_domain=self.domain
        )
        user_prompt = self.capability_repr_class.get_instructions(task)
        response, metadata = llm.generate(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            generation_config=gen_cfg,
        )
        # Parse for "ANSWER" keyword only if the score function
        # uses the `parse_submission` function.
        # TODO:
        # 1. How to gracefully handle cases where tokens are insufficient
        #   and the answer is incomplete?
        if "parse_submission(submission)" in self.capability_repr_class_str:
            answer = parse_submission(response)
            # Handle case where no answer is found
            if answer == "":
                answer = NO_ANSWER_STR
        else:
            answer = response
        metadata = {
            "raw_response": response,
            "api_metadata": metadata,
        }
        return (answer, metadata)

    def solve_tasks(
        self, tasks: List[Dict[str, Any]], llm: Model, gen_cfg: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Solve the tasks using the given LLM.

        Args
        ----
            tasks (List[Dict[str, Any]]): The list of tasks to solve.
            llm (Model): The LLM to use for solving the tasks.
            gen_cfg (Dict[str, Any]): The generation configuration for the LLM.

        Returns
        -------
            Tuple[List[Dict[str, Any]], Dict[str, Any]]: A tuple containing a list of
            dictionaries with the solved tasks and a dictionary with metadata
            for each task.
        """
        solved_tasks = []
        metadata = {}
        for task in tasks:
            answer, _metadata = self._solve_task(
                task=task,
                llm=llm,
                gen_cfg=gen_cfg,
            )
            solved_tasks.append(
                {
                    "id": task["id"],
                    "problem": task["problem"],
                    "answer": answer,
                    "reasoning": _metadata["raw_response"],
                }
            )
            metadata[task["id"]] = _metadata["api_metadata"]
        return (solved_tasks, metadata)

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get the existing tasks for the capability.

        Returns
        -------
            List[Dict[str, Any]]: A list of dictionaries containing the tasks.
        """
        return self._data

    def _create_inspect_file(
        self,
        path: str,
        judge_llm: Model | None,
        judge_llm_gen_args: Dict[str, Any] | None,
    ) -> None:
        """
        Implement pipeline to evaluate the capability using the inspect framework.

        This involves converting the METR format to inspect solvers and scorers.
        """
        # Create JSONL dataset and store it under the inspect path
        dataset = self.get_tasks()
        dataset_metadata_keys = [
            k for k in list(dataset[0].keys()) if k not in ["id", "problem", "answer"]
        ]
        # Write data to a dataset JSONL file
        with open(os.path.join(path, "dataset.jsonl"), "w") as f:
            for elm in dataset:
                f.write(json.dumps(elm) + "\n")

        # Create __init__.py and README files
        # TODO: Add more details to the README file
        init_file_content = INSPECT_EVALS_INIT_FILE_TEMPLATE.format(
            capability_name=self.name,
        ).strip("\n")
        with open(os.path.join(path, "__init__.py"), "w") as f:
            f.write(init_file_content)
        readme_file_content = INSPECT_EVALS_README_FILE_TEMPLATE.format(
            capability_name=self.name,
            capability_description=self.description,
        ).strip("\n")
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(readme_file_content)

        # Create inspect evals script file
        # 1. Copy the helper functions in `utils/inspect_eval_utils.py`
        # to local `utils.py file`
        with open(
            os.path.join(os.path.dirname(__file__), "utils", "inspect_eval_utils.py"),
            "r",
        ) as f:
            utils_file_contents = f.read()
        # Update judge LLM if provided
        if judge_llm is not None:
            utils_file_contents = utils_file_contents.replace(
                'INSPECT_JUDGE_LLM = "openai/gpt-4o-mini"',
                f"INSPECT_JUDGE_LLM = {judge_llm.get_model_name(with_provider=True)}",
            )
        if judge_llm_gen_args is not None:
            utils_file_contents = utils_file_contents.replace(
                "INSPECT_JUDGE_LLM_GEN_CONFIG: Dict[str, Any] = {}",
                f"INSPECT_JUDGE_LLM_GEN_CONFIG: Dict[str, Any] = {json.dumps(judge_llm_gen_args, indent=0)}",
            )
        # Write the modified content to the local utils.py file
        with open(os.path.join(path, "utils.py"), "w") as f:
            f.write(utils_file_contents)

        # 2. Construct inspect evals script file
        # TODO: Do we need system prompt?
        instruction_template = self.capability_repr_class.get_instructions(
            {"problem": "{prompt}"}
        )
        score_func_prefix = f"@staticmethod\n{TAB_W_SPACES}def score"
        score_func_prefix_new = (
            f"async {score_func_prefix.split(TAB_W_SPACES)[1]}".replace(
                "score", "_score"
            )
        )
        score_func_str = f"{score_func_prefix_new}{self.capability_repr_class_str.split(score_func_prefix)[1].replace((TAB_W_SPACES + TAB_W_SPACES), TAB_W_SPACES)}".strip(
            "`"
        ).strip("\n")
        script_file_content = INSPECT_EVALS_SCRIPT_FILE_TEMPLATE.format(
            capability_name=self.name,
            dataset_metadata_keys=json.dumps(dataset_metadata_keys),
            prompt_template=instruction_template,
            score_func_t_dict_str='{"answer": target.text}',
            score_func_str=score_func_str,
        )
        script_file_path = os.path.join(path, f"{self.name}.py")
        with open(script_file_path, "w") as f:
            f.write(script_file_content)
        # TODO: Validate formatting of script file
        _ = _import_from_path(
            module_name=f"{self.name}_inspect_eval_script", file_path=script_file_path
        )

    def _evaluate_using_inspect(self, subject_llm: Model, **kwargs: Any) -> None:
        """
        Evaluate the subject LLM on the capability using the Inspect framework.

        This method uses the Inspect evaluation framework to assess the performance of
        the provided language model (LLM) on a specific capability. It ensures that the
        required evaluation files exist, temporarily stores logs locally, and transfers
        them to a GCP bucket after the evaluation is complete.

        Args:
            subject_llm (Model): The LLM model to evaluate.
            **kwargs (Any): Additional args for running the evals.

        Raises
        ------
            FileNotFoundError: If the required Inspect evaluation path does not exist.
        """
        inspect_path = os.path.join(BASE_INSPECT_EVALS_DIR, self.name)
        if not os.path.exists(inspect_path):
            raise FileNotFoundError(
                f"Inspect evaluation path does not exist: {inspect_path}. "
                "Please ensure the inspect files are created before evaluation."
            )
        # Temporarily store the logs locally and then transfer them to the GCP bucket,
        # since Inspect does not support GCP bucket paths for storing logs
        log_dir = os.path.join(
            self.score_dir.replace(GCP_BASE_ARTIFACTS_DIR, BASE_ARTIFACTS_DIR),
            subject_llm.get_model_name(),
            self.domain,
            self.name,
        )
        os.makedirs(log_dir, exist_ok=True)

        run_inspect_evals(
            path=self.name,
            model=subject_llm,
            log_dir=log_dir,
            **kwargs,
        )

        # Transfer the logs to the GCP bucket
        transfer_inspect_log_to_gcp(
            src_dir=log_dir,
            gcp_dir=log_dir.replace(BASE_ARTIFACTS_DIR, GCP_BASE_ARTIFACTS_DIR),
        )
        # Remove the local logs
        shutil.rmtree(log_dir)

    def evaluate(
        self,
        subject_llms: List[Model],
        gen_args: List[Dict[Any, Any]],
        judge_llm: Model | None = None,
        judge_llm_gen_args: Dict[str, Any] | None = None,
    ) -> None:
        """
        Evaluate the provided subject LLMs on the capability.

        Args
        ----
        subject_llms : List[Model]
            The list of LLMs to use for evaluation.
        gen_args : List[Dict[Any, Any]]
            The list of generation configurations corresponding to each LLM.
        """
        assert len(subject_llms) == len(gen_args), (
            "Each subject LLM must have a corresponding generation config."
        )
        # Create inspect script if evaluating for the first time
        inspect_path = os.path.join(BASE_INSPECT_EVALS_DIR, self.name)
        if not os.path.exists(inspect_path):
            os.makedirs(inspect_path)
            self._create_inspect_file(
                path=inspect_path,
                judge_llm=judge_llm,
                judge_llm_gen_args=judge_llm_gen_args,
            )

        # Change dir to where inspect eval scrips are stored
        # because inspect evals does not support non-relative paths
        cwd = os.getcwd()
        os.chdir(BASE_INSPECT_EVALS_DIR)
        # TODO: Run asynchronosly
        for model_idx, model in enumerate(subject_llms):
            self._evaluate_using_inspect(
                subject_llm=model,
                **gen_args[model_idx],
            )
        # Revert to original working dir after evaluation
        os.chdir(cwd)


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
