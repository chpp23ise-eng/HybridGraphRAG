"""
embeddings.py

Loads the Hugging Face embedding model
used for vector generation.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model():
    """
    Load the embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings