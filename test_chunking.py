from src.document_loader import load_documents

from src.chunking import (

    filter_elements,

    build_sections,

    create_documents

)

# ----------------------------------------------------
# Load
# ----------------------------------------------------

elements = load_documents()

# ----------------------------------------------------
# Filter
# ----------------------------------------------------

filtered = filter_elements(elements)

# ----------------------------------------------------
# Build Sections
# ----------------------------------------------------

sections = build_sections(filtered)

# ----------------------------------------------------
# Documents
# ----------------------------------------------------

documents = create_documents(sections)

print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()

print(f"Total Documents : {len(documents)}")

# ----------------------------------------------------
# Show First Five
# ----------------------------------------------------

for i, doc in enumerate(documents[:5], start=1):

    print()

    print("=" * 80)

    print(f"DOCUMENT {i}")

    print("=" * 80)

    print()

    print("Metadata")

    print(doc.metadata)

    print()

    print("Content\n")

    print(doc.page_content[:1200])

    print()