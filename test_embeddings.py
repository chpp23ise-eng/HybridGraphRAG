from src.embeddings import load_embedding_model

embeddings = load_embedding_model()

print("Embedding model loaded successfully!")

vector = embeddings.embed_query(
    "Machine Learning is a subset of Artificial Intelligence."
)

print()

print("Vector Dimension:", len(vector))

print()

print("First 10 Values:")

print(vector[:10])