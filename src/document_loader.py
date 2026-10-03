"""
document_loader.py

Loads supported documents from the docs folder.

Supported formats:
- PDF
- TXT
- DOCX

PDF files are loaded using Unstructured when available.
If the Unstructured PDF dependency chain is incompatible
with the installed NumPy/Numba versions, a lightweight
PyPDF fallback is used.
"""

import os
from pathlib import Path
from types import SimpleNamespace


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# ============================================================
# DOCS FOLDER
# ============================================================

DOCS_FOLDER = (
    PROJECT_ROOT
    / "docs"
)


# ============================================================
# PDF FALLBACK
# ============================================================

def _load_pdf_with_pypdf(
    file_path
):
    """
    Lightweight PDF fallback.

    This is used when Unstructured's PDF partitioner
    cannot be imported because of dependency
    incompatibility.

    Returns simple element-like objects containing:
    - text
    - category
    - metadata
    """

    try:

        from pypdf import PdfReader

    except ImportError as e:

        raise RuntimeError(
            "PDF loading failed because both "
            "Unstructured PDF loading and the PyPDF "
            "fallback are unavailable. "
            "Install pypdf with: pip install pypdf"
        ) from e

    elements = []

    reader = PdfReader(
        str(file_path)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

        except Exception:

            text = ""

        if not text:

            continue

        text = text.strip()

        if not text:

            continue

        element = SimpleNamespace(

            text=text,

            category="NarrativeText",

            metadata=SimpleNamespace(
                page_number=page_number,
                filename=str(file_path)
            )

        )

        elements.append(
            element
        )

    return elements


# ============================================================
# UNSTRUCTURED LOADER
# ============================================================

def _load_with_unstructured(
    file_path
):
    """
    Load a document using Unstructured.

    The import is intentionally performed inside the
    function so that a PDF dependency problem does not
    crash the entire document loader before we can use
    the PDF fallback.
    """

    from unstructured.partition.auto import partition

    extension = (
        file_path.suffix.lower()
    )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        return partition(

            filename=str(
                file_path
            ),

            strategy="hi_res",

            infer_table_structure=True,

            extract_image_block_types=[
                "Image"
            ],

            extract_image_block_output_dir=(
                "data/extracted_images"
            )

        )

    # --------------------------------------------------------
    # TXT / DOCX
    # --------------------------------------------------------

    return partition(

        filename=str(
            file_path
        )

    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents(
    folder_path=DOCS_FOLDER
):
    """
    Loads all supported documents from the docs folder
    and returns a list of document elements.

    Supported:
    - PDF
    - TXT
    - DOCX

    PDF loading first attempts the existing Unstructured
    pipeline. If the Unstructured PDF dependency chain
    fails, PyPDF is used as a fallback.
    """

    elements = []

    print(
        f"\nLoading documents from: "
        f"{folder_path}"
    )

    # --------------------------------------------------------
    # Validate Folder
    # --------------------------------------------------------

    if not folder_path.exists():

        raise FileNotFoundError(
            f"Documents folder not found: "
            f"{folder_path}"
        )

    if not folder_path.is_dir():

        raise NotADirectoryError(
            f"Not a directory: "
            f"{folder_path}"
        )

    # --------------------------------------------------------
    # Supported Extensions
    # --------------------------------------------------------

    supported_extensions = {

        ".pdf",

        ".txt",

        ".docx"

    }

    # --------------------------------------------------------
    # Process Files
    # --------------------------------------------------------

    for file_path in sorted(
        folder_path.iterdir()
    ):

        # ----------------------------------------------------
        # Ignore directories
        # ----------------------------------------------------

        if not file_path.is_file():

            continue

        extension = (
            file_path.suffix.lower()
        )

        # ----------------------------------------------------
        # Ignore unsupported files
        # ----------------------------------------------------

        if extension not in supported_extensions:

            print(
                f"Skipping: "
                f"{file_path.name}"
            )

            continue

        print(
            f"Processing: "
            f"{file_path.name}"
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if extension == ".pdf":

            try:

                document_elements = (
                    _load_with_unstructured(
                        file_path
                    )
                )

                print(
                    f"  PDF loaded using "
                    f"Unstructured: "
                    f"{len(document_elements)} elements"
                )

            except Exception as unstructured_error:

                print(
                    "  Unstructured PDF loading "
                    "failed."
                )

                print(
                    f"  Reason: "
                    f"{unstructured_error}"
                )

                print(
                    "  Using PyPDF fallback..."
                )

                try:

                    document_elements = (
                        _load_pdf_with_pypdf(
                            file_path
                        )
                    )

                    print(
                        f"  PDF loaded using "
                        f"PyPDF: "
                        f"{len(document_elements)} elements"
                    )

                except Exception as fallback_error:

                    print(
                        f"  Failed to load PDF: "
                        f"{fallback_error}"
                    )

                    continue

        # ----------------------------------------------------
        # TXT / DOCX
        # ----------------------------------------------------

        else:

            try:

                document_elements = (
                    _load_with_unstructured(
                        file_path
                    )
                )

                print(
                    f"  Loaded using "
                    f"Unstructured: "
                    f"{len(document_elements)} elements"
                )

            except Exception as e:

                print(
                    f"  Failed to load "
                    f"{file_path.name}: "
                    f"{e}"
                )

                continue

        # ----------------------------------------------------
        # Add Elements
        # ----------------------------------------------------

        elements.extend(
            document_elements
        )

    # --------------------------------------------------------
    # Final Statistics
    # --------------------------------------------------------

    print(
        f"\nTotal elements extracted: "
        f"{len(elements)}"
    )

    return elements