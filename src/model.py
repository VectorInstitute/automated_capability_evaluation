import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ratelimit import limits, sleep_and_retry


RATE_LIMIT = {
    "calls": int(os.environ.get("RATE_LIMIT_CALLS", 5)),
    "period": int(os.environ.get("RATE_LIMIT_PERIOD", 60)),
}


class Model:
    """A class to represent a LLM with rate limiting and generation configuration."""  # noqa: W505

    def __init__(self, model_name: str, rate_limit: tuple = (5, 60), **kwargs):
        """
        Initialize the LLM with a name, rate limit, generation configuration, and additional keyword arguments.

        :param model_name: The name of the LLM.
        :param rate_limit: A tuple containing the number of calls and the period for rate limiting in secs.
        :param generation_config: A dictionary containing generation configuration parameters.
        :param kwargs: Additional keyword arguments.
        """  # noqa: W505
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name)

        self._sys_msg = kwargs.get("sys_msg", "")

        self.rate_limit_calls, self.rate_limit_period = rate_limit

    @sleep_and_retry
    @limits(**RATE_LIMIT)
    def generate(self, prompt: str, generation_config: dict = None):
        """
        Generate text based on the given prompt using the LLM.

        :param prompt: The input prompt for the model.
        :return: A tuple containing the generated text and metadata.
        """
        messages = [
            SystemMessage(self._sys_msg),
            HumanMessage(prompt),
        ]
        response = self.llm.invoke(messages, **generation_config)
        generated_text = response.content
        input_tokens = response.response_metadata["token_usage"]["prompt_tokens"]
        output_tokens = response.response_metadata["token_usage"]["completion_tokens"]
        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        return generated_text, metadata

    def get_model_name(self):
        """
        Get the name of the model.

        :return: The name of the model.
        """
        return self.model_name
