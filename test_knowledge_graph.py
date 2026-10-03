from src.document_loader import load_documents
from src.chunking import (
    filter_elements,
    build_sections,
    create_documents
)
from src.knowledge_graph import KnowledgeGraphBuilder
from src.llm import load_llm

# -----------------------------------
# Load Documents
# -----------------------------------

elements = load_documents()

# -----------------------------------
# Chunking
# -----------------------------------

filtered = filter_elements(elements)

sections = build_sections(filtered)

documents = create_documents(sections)

print(f"\nDocuments: {len(documents)}")

# -----------------------------------
# Load LLM
# -----------------------------------

llm = load_llm()

# -----------------------------------
# Build Knowledge Graph
# -----------------------------------

builder = KnowledgeGraphBuilder(llm)

graph = builder.build_graph(documents)

builder.print_statistics()

builder.print_sample_relationships()

builder.save_graph()