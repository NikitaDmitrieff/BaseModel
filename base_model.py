from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS

from base_model_config import setup_api_keys_and_langsmith
from base_model_prompts import (
    RAG_USER_PROMPT_TEMPLATE,
    rag_prompt_format,
    prompt_format,
)
from base_model_utils import (
    basic_inquiry,
    load_pdfs_as_documents,
    load_embedding,
    reformat_query,
    convert_documents_to_text,
    clean_text,
)


class BaseModel:
    def __init__(
        self,
        pdf_directory="/Users/nikita.dmitrieff/Desktop/Personal/Comet/data",
        system_template: str = "",
        user_template: str = "",
        rag_system_template: str = "",
        rag_user_template: str = RAG_USER_PROMPT_TEMPLATE,
        hyde_augmentation: bool = True,
        vector_store_chunk_size_in_tokens: int = 250,
        vector_store_chunk_overlap_in_tokens: int = 25,
        vector_store_separators: Optional[List[str]] = None,
        number_of_documents_to_retrieve: int = 4,
        verbose: bool = False,
        langsmith_tracking: bool = False,
        project_name: str = "to-delete",
    ):

        setup_api_keys_and_langsmith(
            langsmith_tracking=langsmith_tracking, project_name=project_name
        )

        # Basic generator
        self.verbose = verbose
        self.system_prompt_template = system_template
        self.user_prompt_template = user_template

        # RAG pipeline - important parameters
        self.vector_store = None
        self.similarity_search_by_vector = False
        self.hyde_augmentation = hyde_augmentation
        self.pdf_directory = pdf_directory
        self.rag_system_prompt_template = rag_system_template
        self.rag_user_prompt_template = rag_user_template

        # RAG pipeline - other tunable parameters
        self.embedding_model = load_embedding()
        self.vector_store_chunk_size_in_tokens = vector_store_chunk_size_in_tokens
        self.vector_store_chunk_overlap_in_tokens = vector_store_chunk_overlap_in_tokens

        if vector_store_separators is None:
            vector_store_separators = ["\n\n", "\n", " ", ""]
        self.vector_store_separators = vector_store_separators

        self.number_of_documents_to_retrieve = number_of_documents_to_retrieve
        self._db_size = None

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

        documents = load_pdfs_as_documents(pdf_directory=self.pdf_directory)

        if documents == []:
            raise ValueError("Empty documents")

        chunks = text_splitter.split_documents(documents)
        self._db_size = len(chunks)

        metadatas = [{"order": i} for i in range(len(documents))]

        for i, doc in enumerate(documents):
            doc.metadata.update(metadatas[i])

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        if self.verbose:
            for document in documents:
                print(document.page_content.lower()[40:60])

    def retrieve_documents(
        self,
        query: str = None,
        force_db_delete=False,
    ) -> List[Document]:
        """
        Main function for RAG:
        """

        # create vector store if necessary
        if not self.vector_store or force_db_delete:
            try:
                self.vector_store.delete_collection()
            except AttributeError:
                pass

            self.ingest_pdfs_to_vector_store()

        # retrieve documents from query
        if self.similarity_search_by_vector:
            embedding_vector = self.embeddings.embed_query(query)
            results = self.vector_store.similarity_search_by_vector(embedding_vector)
        else:
            results = self.vector_store.similarity_search(
                query, k=self.number_of_documents_to_retrieve
            )

        return results

    def generate_answer_using_rag(
        self,
        user_question: str = "Hi",
        model_type: str = None,
    ) -> tuple[str, str, str]:

        # apply HYDE
        if self.hyde_augmentation:
            reformatted_query = reformat_query(user_question=user_question)
        else:
            reformatted_query = user_question

        # retrieve and clean context as text
        context_as_documents = self.retrieve_documents(query=reformatted_query)
        context_as_text = convert_documents_to_text(documents=context_as_documents)
        context_as_text_cleaned = clean_text(text=context_as_text)

        # create prompt
        system_prompt, user_prompt = rag_prompt_format(
            user_question=user_question,
            context=context_as_text_cleaned,
            system_template=self.rag_system_prompt_template,
            user_template=self.rag_user_prompt_template,
        )

        # ask gpt and clean answer
        answer = basic_inquiry(
            system_prompt=system_prompt, user_prompt=user_prompt, model_type=model_type
        )

        answer = (
            answer.replace("< lang=" rf">", "").replace("html", "").replace("```", "")
        )

        return answer, system_prompt, user_prompt

    def generate_answer(
        self,
        user_question: str = "Hi",
        model_type: str = None,
    ) -> tuple[str, str, str]:

        system_prompt, user_prompt = prompt_format(
            user_question=user_question,
            system_template=self.system_prompt_template,
            user_template=self.user_prompt_template,
        )

        answer = basic_inquiry(
            system_prompt=system_prompt, user_prompt=user_prompt, model_type=model_type
        )

        answer = (
            answer.replace("< lang=" rf">", "").replace("html", "").replace("```", "")
        )

        return answer, system_prompt, user_prompt


if __name__ == "__main__":
    chat = BaseModel()

    print(chat.generate_answer(user_question="C'est quoi Néoma ?"))

    pass
