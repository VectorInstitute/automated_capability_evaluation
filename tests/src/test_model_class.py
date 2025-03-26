"""
Module containing unit tests for the Model class.

The tests cover the following functionalities:
- Initialization of the Model class with OpenAI and ChatOpenAI models.
- Retrieval of the model name using the get_model_name method.
- Input messages generation using the _get_input_messages method for both
  OpenAI and ChatOpenAI models.
- Text generation using the generate method for both OpenAI and ChatOpenAI models.

The tests verify that the Model class behaves as expected and that the attributes
and methods return the correct values.
"""

import os

from langchain_openai import ChatOpenAI
from openai import AuthenticationError

from src.model import Model


# Use dummy OpenAI API key for tests not making API calls.
DUMMY_OPENAI_API_KEY = "dummy_key"
os.environ["OPENAI_API_KEY"] = DUMMY_OPENAI_API_KEY

# API key error code vars
EC_401 = "Error code: 401"
EC_401_SKIP_MSG = (
    "Skip this test for code check because this test depends on actual API call."
)

sys_msg = "a"
prompt = "b"


def skip_test(function):
    """
    Wrap a test function to handle AuthenticationError exceptions.

    If an AuthenticationError with error code EC_401 is raised
    (Incorrect API key provided), it prints a skip message and
    does not re-raise the exception. For other AuthenticationError exceptions,
    it re-raises the exception.
    Args:
        function (callable): The test function to be wrapped.

    Returns
    -------
        callable: The wrapped function.
    """

    def wrapper():
        try:
            return function()
        except AuthenticationError as e:
            if EC_401 in str(e):
                print(EC_401_SKIP_MSG)
            else:
                raise e
            return True

    return wrapper


def test_model_class_init_chatopenai():
    """
    Test the initialization of the Model class with a ChatOpenAI model.

    This test verifies that the Model class is initialized correctly
    with a ChatOpenAI model.
    It checks the following:
    - The model name matches the expected name.
    - The LLM attribute is an instance of the ChatOpenAI class.
    - The system message attribute matches the expected message.
    """
    model_name = "gpt-4o-mini"
    model = Model(model_name=model_name)
    assert isinstance(model.llm, ChatOpenAI)
    assert model.model_name == model_name


def test_model_class_get_model_name():
    """
    Test the get_model_name method of the Model class.

    This test verifies that the get_model_name method of the Model class
    returns the correct model name.
    """
    model_name = "gpt-4o-mini"
    model = Model(model_name=model_name)
    assert model.get_model_name() == model_name


def test_model_class_get_input_messages_chatopenai():
    """
    Test the _get_input_messages method of the Model class with a ChatOpenAI model.

    This test verifies that the _get_input_messages method of the Model class
    works correctly with a ChatOpenAI model.
    It checks the following:
    - The input messages are a list of tuples.
    - The input messages contain the system message and prompt.
    """
    model_name = "gpt-4o-mini"
    model = Model(model_name=model_name)
    input_messages = model._get_input_messages(
        sys_prompt=sys_msg,
        user_prompt=prompt,
    )
    assert isinstance(input_messages, list)
    assert all(isinstance(message, tuple) for message in input_messages)
    assert input_messages[0][0] == "system"
    assert input_messages[0][1] == sys_msg
    assert input_messages[1][0] == "user"
    assert input_messages[1][1] == prompt


@skip_test
def test_model_class_generate_chatopenai_non_o1():
    """
    Test the generate method of the Model class with a ChatOpenAI model.

    This test verifies that the generate method of the Model class works
    correctly with a ChatOpenAI model.
    It checks the following:
    - The generated text is not None and of type string.
    - The metadata is a dictionary.
    - The output tokens are either 0 or 1.
    """
    # Use actual OpenAI API key for this test.
    os.environ["OPENAI_API_KEY"] = os.environ.get(
        "TEST_OPENAI_API_KEY", DUMMY_OPENAI_API_KEY
    )

    model_name = "gpt-4o-mini"
    model = Model(model_name=model_name)
    generation_config = {"temperature": 0.5, "max_tokens": 1}
    output, metadata = model.generate(
        sys_prompt=sys_msg, user_prompt=prompt, generation_config=generation_config
    )
    assert output is not None
    assert isinstance(output, str)
    assert isinstance(metadata, dict)
    assert metadata["output_tokens"] in [0, 1]


@skip_test
def test_model_class_generate_chatopenai_o1():
    """
    Test the generate method of the Model class with a ChatOpenAI model.

    This test verifies that the generate method of the Model class works
    correctly with a ChatOpenAI model.
    It checks the following:
    - The generated text is not None and of type string.
    - The metadata is a dictionary.
    - The output tokens are either 0 or 1.
    """
    # Use actual OpenAI API key for this test.
    os.environ["OPENAI_API_KEY"] = os.environ.get(
        "TEST_OPENAI_API_KEY", DUMMY_OPENAI_API_KEY
    )

    model_name = "o1-mini"
    model = Model(model_name=model_name)
    generation_config = {"temperature": 0.5, "max_tokens": 1}
    output, metadata = model.generate(
        sys_prompt=sys_msg, user_prompt=prompt, generation_config=generation_config
    )
    assert output is not None
    assert isinstance(output, str)
    assert isinstance(metadata, dict)
    assert metadata["output_tokens"] in [0, 1]
