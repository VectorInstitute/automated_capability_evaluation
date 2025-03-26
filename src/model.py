import os  # noqa: D100
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI
from ratelimit import limits, sleep_and_retry


RATE_LIMIT = {
    "calls": int(os.environ.get("RATE_LIMIT_CALLS", 5)),
    "period": int(os.environ.get("RATE_LIMIT_PERIOD", 60)),
}


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
        self.llm: ChatOpenAI = self._set_llm()

    def _set_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.model_name)

    @sleep_and_retry  # type: ignore
    @limits(**RATE_LIMIT)  # type: ignore
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
            if "o3-mini" in self.model_name:
                # Remove temperature for o3-mini
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
            print(f"Error generating text: {e}")
            raise e

        metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return generated_text, metadata

    def get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns
        -------
        str: The name of the model.
        """
        return self.model_name

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
