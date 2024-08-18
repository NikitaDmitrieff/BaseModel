from credentials import (
    OPENAI_API_KEY,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_API_KEY,
    LANGCHAIN_PROJECT,
)
import os


def setup_openai_api_key():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def setup_api_keys_and_langsmith(
    langsmith_tracking: bool = True, project_name: str = None
):
    setup_openai_api_key()

    if langsmith_tracking:
        os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        if project_name:
            os.environ["LANGCHAIN_PROJECT"] = project_name
        else:
            os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
