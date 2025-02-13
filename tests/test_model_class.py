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
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.model import Model


sys_msg = "a"
prompt = "b"


def test_model_class_init_openai():
    """
    Test the initialization of the Model class with an OpenAI model.

    This test verifies that the Model class is initialized correctly
    with an OpenAI model.
    It checks the following:
    - The model name matches the expected name.
    - The LLM attribute is an instance of the OpenAI class.
    - The system message attribute matches the expected message.
    """
    model_name = "o1-mini"
    model = Model(model_name=model_name, sys_msg=sys_msg)
    assert isinstance(model.llm, OpenAI)
    assert model.model_name == model_name
    assert model._sys_msg == sys_msg


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
    model = Model(model_name=model_name, sys_msg=sys_msg)
    assert isinstance(model.llm, ChatOpenAI)
    assert model.model_name == model_name
    assert model._sys_msg == sys_msg


def test_model_class_get_model_name():
    """
    Test the get_model_name method of the Model class.

    This test verifies that the get_model_name method of the Model class
    returns the correct model name.
    """
    model_name = "gpt-4o-mini"
    model = Model(model_name=model_name)
    assert model.get_model_name() == model_name


def test_model_class_get_input_messages_openai():
    """
    Test the _get_input_messages method of the Model class with an OpenAI model.

    This test verifies that the _get_input_messages method of the Model class
    works correctly with an OpenAI model.
    It checks the following:
    - The input messages are a list of dictionaries.
    - The input messages contain the system message and prompt.
    """
    model_name = "o1-mini"
    model = Model(model_name=model_name, sys_msg=sys_msg)
    input_messages = model._get_input_messages(prompt=prompt)
    assert isinstance(input_messages, list)
    assert all(isinstance(message, dict) for message in input_messages)
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["content"] == f"{sys_msg}\n\n{prompt}"


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
    model = Model(model_name=model_name, sys_msg=sys_msg)
    input_messages = model._get_input_messages(prompt=prompt)
    assert isinstance(input_messages, list)
    assert all(isinstance(message, tuple) for message in input_messages)
    assert input_messages[0][0] == "system"
    assert input_messages[0][1] == sys_msg
    assert input_messages[1][0] == "user"
    assert input_messages[1][1] == prompt


def test_model_class_generate_openai():
    """
    Test the generate method of the Model class with an OpenAI model.

    This test verifies that the generate method of the Model class works
    correctly with an OpenAI model.
    It checks the following:
    - The generated text is not None and of type string.
    - The metadata is a dictionary.
    - The output tokens are either 0 or 1.
    """
    model_name = "o1-mini"
    model = Model(model_name=model_name, sys_msg=sys_msg)
    model.model_name = "gpt-4o-mini"  # Update to got-4o-mini to save costs
    generation_config = {"temperature": 0.5, "max_tokens": 1}
    output, metadata = model.generate(
        prompt=prompt, generation_config=generation_config
    )
    assert output is not None
    assert isinstance(output, str)
    assert isinstance(metadata, dict)
    assert metadata["output_tokens"] in [0, 1]


def test_model_class_generate_chatopenai():
    """
    Test the generate method of the Model class with a ChatOpenAI model.

    This test verifies that the generate method of the Model class works
    correctly with a ChatOpenAI model.
    It checks the following:
    - The generated text is not None and of type string.
    - The metadata is a dictionary.
    - The output tokens are either 0 or 1.
    """
    model_name = "gpt-4o-mini"
    model = Model(model_name=model_name, sys_msg=sys_msg)
    generation_config = {"temperature": 0.5, "max_tokens": 1}
    output, metadata = model.generate(
        prompt=prompt, generation_config=generation_config
    )
    assert output is not None
    assert isinstance(output, str)
    assert isinstance(metadata, dict)
    assert metadata["output_tokens"] in [0, 1]
