from collections import Counter
from src.document_loader import load_documents

elements = load_documents()

print(f"\nTotal Elements: {len(elements)}")

types = Counter(type(e).__name__ for e in elements)

print("\nElement Types\n")

for t, count in types.items():
    print(f"{t}: {count}")