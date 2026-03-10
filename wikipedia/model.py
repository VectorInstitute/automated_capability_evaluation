"""Model class for LLMs with rate limiting and generation configuration."""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import SecretStr
from ratelimit import limits, sleep_and_retry

from src.utils import constants


RATE_LIMIT = {
    "calls": int(os.environ.get("RATE_LIMIT_CALLS", "5")),
    "period": int(os.environ.get("RATE_LIMIT_PERIOD", "60")),
}


logger = logging.getLogger(__name__)


class Model:
    """A class to represent a LLM with rate limiting and generation configuration."""  # noqa: W505

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the LLM with a name and additional keyword arguments.

        Args
        ----
            model_name (str): The name of the LLM.
            kwargs (Any): Additional keyword arguments.
        """
        self.model_name: str = model_name
        self.model_provider: str = kwargs.pop("model_provider", "openai")
        self.llm: ChatOpenAI | ChatGoogleGenerativeAI | ChatAnthropic = self._set_llm(
            **kwargs
        )

    def _set_llm(
        self, **kwargs: Any
    ) -> ChatOpenAI | ChatGoogleGenerativeAI | ChatAnthropic:
        if self.model_provider == "local":
            self.model_url = get_local_model_url(self.model_name, **kwargs)
            return ChatOpenAI(
                model=self.model_name,
                base_url=self.model_url,
                api_key=SecretStr(os.environ["LOCAL_API_KEY"]),
            )
        if self.model_provider == "anthropic":
            return ChatAnthropic(model=self.model_name)  # type: ignore
        if self.model_provider == "google":
            return ChatGoogleGenerativeAI(model=self.model_name)
        if self.model_provider == "openai":
            return ChatOpenAI(model=self.model_name)
        raise ValueError(f"Unsupported model provider: {self.model_provider}")

    @sleep_and_retry  # type: ignore
    @limits(**RATE_LIMIT)  # type: ignore
    @traceable
    def generate(
        self, sys_prompt: str, user_prompt: str, generation_config: Dict[str, Any]
    ) -> Tuple[str | None, Dict[str, int | Any]]:
        """
        Generate text based on the given system and user prompts using the LLM.

        Args
        ----
            sys_prompt (str): The system prompt for the model.
            user_prompt (str): The user prompt for the model.
            generation_config (Dict[str, Any]): A dictionary containing generation
                configuration parameters.

        Returns
        -------
            Tuple[str | None, Dict[str, int | Any]]: A tuple containing the
                generated text and metadata.
            - str | None: The generated text.
            - Dict[str, int | Any]: Metadata including input and output token counts.
        """
        messages = self._get_input_messages(
            sys_prompt=sys_prompt, user_prompt=user_prompt
        )
        generation_config = dict(generation_config)
        try:
            if "o1" in self.model_name:
                # Set temperature to 1 for o1
                generation_config.update({"temperature": 1})
            if any(model in self.model_name for model in ["o3-mini", "o3", "o4-mini"]):
                # Remove temperature for o3-mini, o3, o4-mini
                _ = generation_config.pop("temperature", None)
            chatopenai_response = self.llm.invoke(messages, **generation_config)
            generated_text = str(chatopenai_response.content)
            input_tokens = chatopenai_response.response_metadata["token_usage"][
                "prompt_tokens"
            ]
            output_tokens = chatopenai_response.response_metadata["token_usage"][
                "completion_tokens"
            ]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise e

        metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return generated_text, metadata

    @sleep_and_retry  # type: ignore
    @limits(**RATE_LIMIT)  # type: ignore
    @traceable
    async def async_generate(
        self, sys_prompt: str, user_prompt: str, generation_config: Dict[str, Any]
    ) -> Tuple[str | None, Dict[str, int | Any]]:
        """
        Generate text based on the given system and user prompts using the LLM (Async).

        Args
        ----
            sys_prompt (str): The system prompt for the model.
            user_prompt (str): The user prompt for the model.
            generation_config (Dict[str, Any]): A dictionary containing generation
                configuration parameters.

        Returns
        -------
            Tuple[str | None, Dict[str, int | Any]]: A tuple containing the
                generated text and metadata.
            - str | None: The generated text.
            - Dict[str, int | Any]: Metadata including input and output token counts.
        """
        messages = self._get_input_messages(
            sys_prompt=sys_prompt, user_prompt=user_prompt
        )
        generation_config = dict(generation_config)
        try:
            if "o1" in self.model_name:
                # Set temperature to 1 for o1
                generation_config.update({"temperature": 1})
            if any(model in self.model_name for model in ["o3-mini", "o3", "o4-mini"]):
                # Remove temperature for o3-mini, o3, o4-mini
                _ = generation_config.pop("temperature", None)
            chatopenai_response = await self.llm.ainvoke(messages, **generation_config)
            generated_text = str(chatopenai_response.content)
            input_tokens = chatopenai_response.response_metadata["token_usage"][
                "prompt_tokens"
            ]
            output_tokens = chatopenai_response.response_metadata["token_usage"][
                "completion_tokens"
            ]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise e

        metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return generated_text, metadata

    def get_model_name(self, with_provider: bool = False) -> str:
        """
        Get the name of the model.

        Args
        ----
            with_provider (bool): If True, include the model provider in the name.

        Returns
        -------
            str: The name of the model.
        """
        return (
            self.model_name
            if not with_provider
            else f"{self.model_provider}/{self.model_name}"
        )

    def _get_input_messages(
        self, sys_prompt: str, user_prompt: str
    ) -> List[Tuple[str, str]]:
        """
        Configure the input messages for the LLM based on the model name.

        Args
        ----
            sys_prompt (str): The system prompt for the model.
            user_prompt (str): The user prompt for the model.

        Returns
        -------
            List[Tuple[str, str]]: A list of tuples containing the input messages.
        """
        if "o1" in self.model_name:
            # o1 does not support system messages
            return [
                ("user", f"{sys_prompt}\n\n{user_prompt}"),
            ]
        return [
            ("system", sys_prompt),
            ("user", user_prompt),
        ]


def get_local_model_url(model_name: str, **kwargs: Any) -> str:
    """
    Get the base URL of a locally launched model.

    This function launches a local model using vector inference, waits for the model
    to be ready, and retrieves its base URL.

    Args
    ----
        model_name (str): The name of the model to launch.
        kwargs (Any): Additional keyword arguments for model launch.

    Returns
    -------
        str: The base URL of the launched local model.

    Raises
    ------
        RuntimeError: If the model launch fails or is cancelled.
    """
    # Check if the model is supported
    list_command = ["vec-inf", "list", "--json-mode"]
    list_out = _run_command(list_command)
    if model_name not in list_out:
        raise ValueError(
            f"Model {model_name} is not supported locally. Supported local models: {list_out}"
        )

    # Launch the local model using vector inference
    launch_command = ["vec-inf", "launch", model_name, "--json-mode"]
    # Add any additional arguments from kwargs
    for key, value in kwargs.items():
        c_arg = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                launch_command.append(f"--{c_arg}")
        else:
            launch_command.append(f"--{c_arg}={value}")
    logger.info(f"Launching local model with command: {' '.join(launch_command)}")
    launch_out = _run_command(launch_command)
    slurm_job_id = launch_out["slurm_job_id"]

    # Wait for the model to be ready
    vec_inf_status = constants.VecInfStatus
    status_command = ["vec-inf", "status", slurm_job_id, "--json-mode"]
    status = vec_inf_status.PENDING.value
    while status in [vec_inf_status.PENDING.value, vec_inf_status.LAUNCHING.value]:
        status_out = _run_command(status_command, model_status=True)
        status = status_out["model_status"]
        status = (
            vec_inf_status.PENDING.value if ("LOG FILE NOT FOUND" in status) else status
        )
        logger.info(f"Model status: {status}")
        time.sleep(5)  # Wait for 5 seconds before checking again
    if status == vec_inf_status.FAILED.value:
        raise RuntimeError(f"Model launch failed: {status_out['failed_reason']}")
    if status == vec_inf_status.SHUTDOWN.value:
        raise RuntimeError("Model launch cancelled")
    if status == vec_inf_status.UNAVAILABLE.value:
        raise RuntimeError("Model launch has either failed or is shutdown")

    # Check if the model is ready and get the base URL
    assert status == vec_inf_status.READY.value, f"Unknown model status: {status}"

    return str(status_out["base_url"])


def _sanitize_json(json_str: str) -> str:
    """
    Sanitize JSON string by replacing single quotes with double quotes.

    Args
    ----
        json_str (str): The JSON string to sanitize.

    Returns
    -------
        str: The sanitized JSON string.
    """
    return json_str.strip().replace("'", '"')


def _run_command(
    command: List[str], model_status: bool = False
) -> Dict[str, Any] | Any:
    """
    Run a command and return the parsed JSON output.

    Args
    ----
        command (List[str]): The command to run.
        model_status (bool): If True, parse the model status from the output.

    Returns
    -------
        Dict[str, Any]: The parsed JSON output.
    """
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"Failed to launch local model server: {stderr.strip()}")
    if model_status and constants.VEC_INF_LOG_DIR not in stdout:
        # Extract model status string and replace with
        # appropriate string which can be parsed
        str_to_replace = (
            stdout.split("'model_status':")[-1]
            .split("'base_url':")[0]
            .strip()
            .strip(",")
        )
        str_to_replace_by = str_to_replace.split(":")[-1].split(">")[0].strip()
        stdout = stdout.replace(str_to_replace, str_to_replace_by)
    try:
        logger.debug(f"Command output: {stdout.strip()}")
        stdout_dict = json.loads(_sanitize_json(stdout))
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON output: {stdout.strip()}") from None
    return stdout_dict
