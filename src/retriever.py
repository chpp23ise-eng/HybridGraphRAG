"""
retriever.py

Hybrid GraphRAG Retriever

Responsibilities
----------------
1. Load ChromaDB
2. Retrieve relevant chunks
3. Support MMR retrieval
4. Return similarity scores
5. Return document metadata
"""

from langchain_chroma import Chroma

from src.embeddings import load_embedding_model


class Retriever:

    def __init__(
        self,
        persist_directory="data/chroma_db"
    ):

        self.embedding_model = load_embedding_model()

        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )

        # MMR Retriever
        self.retriever = self.vector_store.as_retriever(

            search_type="mmr",

            search_kwargs={

                "k": 5,

                "fetch_k": 20,

                "lambda_mult": 0.7
            }
        )

    # ---------------------------------------------------
    # Retrieve Documents
    # ---------------------------------------------------

    def retrieve(
        self,
        query
    ):

        return self.retriever.invoke(query)

    # ---------------------------------------------------
    # Retrieve Documents With Similarity Score
    # ---------------------------------------------------

    def retrieve_with_score(
        self,
        query,
        k=5
    ):

        return self.vector_store.similarity_search_with_score(

            query=query,

            k=k

        )

    # ---------------------------------------------------
    # Retrieve Only Clean Text
    # ---------------------------------------------------

    def retrieve_text(
        self,
        query
    ):

        documents = self.retrieve(query)

        chunks = []

        for doc in documents:

            chunks.append(doc.page_content)

        return chunks

    # ---------------------------------------------------
    # Retrieve Complete Information
    # ---------------------------------------------------

    def retrieve_context(
        self,
        query,
        k=5
    ):

        results = self.retrieve_with_score(
            query,
            k
        )

        retrieved = []

        for document, score in results:

            retrieved.append(

                {

                    "text": document.page_content,

                    "metadata": document.metadata,

                    "score": score

                }

            )

        return retrieved