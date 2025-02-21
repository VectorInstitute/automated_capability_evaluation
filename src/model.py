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
        Initialize the LLM with a name, rate limit, generation configuration, and additional keyword arguments.

        :param model_name: The name of the LLM.
        :param rate_limit: A tuple containing the number of calls and the period for rate limiting in secs.
        :param generation_config: A dictionary containing generation configuration parameters.
        :param kwargs: Additional keyword arguments.
        """  # noqa: W505
        self.model_name: str = model_name
        self.llm: ChatOpenAI = self._set_llm()

        self._sys_msg: str = str(kwargs.get("sys_msg", ""))

    def _set_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.model_name)

    @sleep_and_retry  # type: ignore
    @limits(**RATE_LIMIT)  # type: ignore
    def generate(
        self, prompt: str, generation_config: Dict[str, Any]
    ) -> Tuple[str | None, Dict[str, int | Any]]:
        """
        Generate text based on the given prompt using the LLM.

        :param prompt: The input prompt for the model.
        :return: A tuple containing the generated text and metadata.
        """
        messages = self._get_input_messages(prompt)
        generation_config = dict(generation_config)
        try:
            if "o1" in self.model_name:
                # Set temperature to 1 for o1
                generation_config.update({"temperature": 1})
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

        :return: The name of the model.
        """
        return self.model_name

    def _get_input_messages(
        self, prompt: str
    ) -> List[Dict[str, str] | Tuple[str, str]]:
        if "o1" in self.model_name:
            # o1 does not support system messages
            return [
                ("user", f"{self._sys_msg}\n\n{prompt}"),
            ]
        return [
            ("system", self._sys_msg),
            ("user", prompt),
        ]
