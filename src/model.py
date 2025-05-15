"""Model class for LLMs with rate limiting and generation configuration."""

import logging
import os
from typing import Any, Dict, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import traceable
from ratelimit import limits, sleep_and_retry


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
            raise NotImplementedError(
                "Code to support local model is removed for annonymity purposes."
                "Please use your own local model."
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

        Returns
        -------
        str: The name of the model.
        with_provider (bool): If True, include the model provider in the name.
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
