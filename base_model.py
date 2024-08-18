import os
from functools import reduce

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from credentials import OPENAI_API_KEY
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


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


DOC_ORDER_KEY = "chunk_order"


class BaseModel:
    def __init__(
        self,
        pdf_directory="/Users/nikita.dmitrieff/Desktop/Personal/Comet/data",
        system_template: str = SYSTEM_PROMPT_TEMPLATE,
        user_template: str = USER_PROMPT_TEMPLATE,
        vector_store_chunk_size_in_tokens: int = 250,
        vector_store_chunk_overlap_in_tokens: int = 25,
        vector_store_separators: Optional[List[str]] = None,
        number_of_documents_to_retrieve: int = 2,
        verbose: bool = False,
    ):

        self.verbose = verbose

        self.vector_store = None
        self.embeddings = load_embedding()
        self.vector_store_chunk_size_in_tokens = vector_store_chunk_size_in_tokens
        self.vector_store_chunk_overlap_in_tokens = vector_store_chunk_overlap_in_tokens

        if vector_store_separators is None:
            vector_store_separators = ["\n\n", "\n", " ", ""]
        self.vector_store_separators = vector_store_separators

        self.number_of_documents_to_retrieve = number_of_documents_to_retrieve
        self.pdf_directory = pdf_directory

        self.system_prompt_template = system_template
        self.user_prompt_template = user_template

        self._db_size = None
        self._db = None

        return

    def ingest_pdfs_to_vector_store(self) -> None:
        """
        Split text into chunks.
        Embed chunks and ingest those to a FAISS vector store
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.vector_store_chunk_size_in_tokens,
            chunk_overlap=self.vector_store_chunk_overlap_in_tokens,
            separators=self.vector_store_separators,
        )

        documents = load_pdf_documents(pdf_directory=self.pdf_directory)

        if documents == []:
            raise ValueError("Empty documents")

        chunks = text_splitter.split_documents(documents)
        self._db_size = len(chunks)

        # Chunk order is stored in metadata to be able to sort retrieve chunks by order of appearance in the document
        metadatas = [{DOC_ORDER_KEY: i} for i in range(len(documents))]

        # Assign metadata to each document
        for i, doc in enumerate(documents):
            doc.metadata.update(metadatas[i])

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        if self.verbose:
            for document in documents:
                print(document.page_content.lower().count("neoma"))
                print(document.page_content.lower()[40:55])

    def retrieve_documents(
        self,
        query: str = None,
        force_db_delete=False,
    ):

        if not self.vector_store or force_db_delete:
            try:
                self.vector_store.delete_collection()
            except AttributeError:
                pass

            self.ingest_pdfs_to_vector_store()

        if not query:
            query = "Combien d'écoles en post-bac"

        results = self.vector_store.similarity_search(
            query, k=self.number_of_documents_to_retrieve
        )

        return results

    def generate_answer(
        self,
        user_question: str = "Hi",
        model_type: str = None,
    ) -> str:

        reformatted_query = reformat_query(user_question=user_question)

        context_as_documents = self.retrieve_documents(query=reformatted_query)
        context_as_text = convert_documents_to_text(documents=context_as_documents)
        context_as_text_cleaned = clean_text(text=context_as_text)

        system_prompt, user_prompt = prompt_format(
            user_question=user_question,
            context=context_as_text_cleaned,
            system_template=self.system_prompt_template,
            user_template=self.user_prompt_template,
        )

        answer = basic_inquiry(
            system_prompt=system_prompt, user_prompt=user_prompt, model_type=model_type
        )

        answer = (
            answer.replace("< lang=" rf">", "").replace("html", "").replace("```", "")
        )

        print(answer)

        return answer, system_prompt, user_prompt


def reformat_query(user_question: str = "Hi", model_type: str = "gpt-3.5-turbo-0125"):

    reformatted_query = basic_inquiry(
        system_prompt="",
        user_prompt=HYDE_PROMPT_TEMPLATE.format(QUESTION=user_question),
        model_type=model_type,
    )

    return reformatted_query


def convert_documents_to_text(documents):

    documents.sort(key=lambda doc: doc.metadata.get(DOC_ORDER_KEY))
    text = "\n\n".join([doc.page_content for doc in documents])

    return text


def clean_text(text: str) -> str:

    cleaned_text = text.replace("The Comet Project   2022-2023", " ").replace(
        "The Comet Project  2022-2023", " "
    )

    return cleaned_text


if __name__ == "__main__":
    guidance_counselor = GuidanceCounselor(
        pdf_directory="/Users/nikita.dmitrieff/Desktop/Personal/Comet/data",
    )

    print(guidance_counselor.generate_answer(user_question="C'est quoi Néoma ?"))

    pass
