from src.document_loader import load_documents
from src.chunking import (
    filter_elements,
    build_sections,
    create_documents
)
from src.embeddings import load_embedding_model
from src.vector_store import create_vector_store

# -----------------------------------------
# Load Documents
# -----------------------------------------

elements = load_documents()

# -----------------------------------------
# Chunking Pipeline
# -----------------------------------------

filtered = filter_elements(elements)

sections = build_sections(filtered)

documents = create_documents(sections)

print(f"\nDocuments: {len(documents)}")

# -----------------------------------------
# Load Embedding Model
# -----------------------------------------

embeddings = load_embedding_model()

# -----------------------------------------
# Create Vector Store
# -----------------------------------------

vector_db = create_vector_store(
    documents,
    embeddings
)

print("\nVector Database Created Successfully!")

print(f"Documents Stored: {vector_db._collection.count()}")