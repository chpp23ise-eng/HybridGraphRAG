"""
ocr.py

Extract text from images using Tesseract OCR.
"""

import pytesseract
from PIL import Image


def extract_text_from_image(image_path):
    """
    Extract text from a single image.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    str
    """

    try:

        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)

        return text.strip()

    except Exception as e:

        print(f"OCR Error: {e}")

        return ""


def extract_text_from_elements(elements):
    """
    Extract text from every Image element.

    Returns

    [
        {
            "image_path": "...",
            "ocr_text": "..."
        }
    ]
    """

    results = []

    for element in elements:

        if type(element).__name__ != "Image":
            continue

        image_path = getattr(
            element.metadata,
            "image_path",
            None
        )

        if not image_path:
            continue

        text = extract_text_from_image(image_path)

        if len(text.strip()) > 15:
            results.append({
                "image_path": image_path,
                "ocr_text": text
            })

    return results