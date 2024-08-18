import os

from langchain.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable

from base_model_credentials import OPENAI_API_KEY
from base_model_prompts import HYDE_PROMPT_TEMPLATE


@traceable()
def load_model(model_type=None):

    if not model_type:
        model_type = "gpt-4o-mini"

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    model = ChatOpenAI(model=model_type)

    return model


@traceable()
def load_embedding(embedding_type=None):

    if not embedding_type:
        embedding_type = "gpt-4o-mini"

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    model = OpenAIEmbeddings(model=embedding_type)

    return model


def load_pdfs_as_documents(
    pdf_directory="/Users/nikita.dmitrieff/Desktop/Personal/Comet/data",
):

    if not pdf_directory:
        pdf_directory = "/Users/nikita.dmitrieff/Desktop/Personal/Comet/data"

    # Load all PDF files from the directory
    pdf_loaders = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_loaders.append(PyPDFLoader(os.path.join(pdf_directory, filename)))

    # Load and split documents
    docs = []
    for loader in pdf_loaders:
        docs.extend(loader.load())

    return docs


@traceable()
def basic_inquiry(
    system_prompt: str = "Translate the following from English into Italian",
    user_prompt: str = "hi!",
    model_type: str = None,
    model=None,
):

    if not model:
        model = load_model(model_type=model_type)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    result = model.invoke(messages).content

    return result


def reformat_query(user_question: str = "Hi", model_type: str = "gpt-3.5-turbo-0125"):

    reformatted_query = basic_inquiry(
        system_prompt="",
        user_prompt=HYDE_PROMPT_TEMPLATE.format(QUESTION=user_question),
        model_type=model_type,
    )

    return reformatted_query


def convert_documents_to_text(documents):

    documents.sort(key=lambda doc: doc.metadata.get("order"))
    text = "\n\n".join([doc.page_content for doc in documents])

    return text


def clean_text(text: str) -> str:

    cleaned_text = text.replace("The Comet Project   2022-2023", " ").replace(
        "The Comet Project  2022-2023", " "
    )

    return cleaned_text
