from src.document_loader import load_documents
from src.ocr import extract_text_from_elements

elements = load_documents()

ocr_results = extract_text_from_elements(elements)

print()

for result in ocr_results:

    print("=" * 80)

    print("Image")

    print(result["image_path"])

    print()

    print(result["ocr_text"])

    print("=" * 80)