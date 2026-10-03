"""
vector_store.py

Creates and loads the Chroma vector database.
"""

from langchain_chroma import Chroma


def create_vector_store(documents, embeddings):
    """
    Create a ChromaDB vector store from documents.
    """

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="data/chroma_db"
    )

    return vector_db


def load_vector_store(embeddings):
    """
    Load an existing ChromaDB database.
    """

    vector_db = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings
    )

    return vector_db