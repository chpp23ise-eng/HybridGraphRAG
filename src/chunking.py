"""
chunking.py

Creates clean LangChain Documents
from Unstructured Elements or PyPDF fallback elements.

The pipeline:

1. Filter useful textual elements
2. Keep documents separated by source file
3. Build logical sections
4. Clean OCR/noisy text
5. Remove clearly unwanted sections
6. Create LangChain Documents

The goal is to preserve useful academic content while
avoiding overly aggressive filtering.
"""

from langchain_core.documents import Document
import re


# ============================================================
# ALLOWED ELEMENT TYPES
# ============================================================

ALLOWED_TYPES = {
    "Title",
    "NarrativeText",
    "Text",
    "ListItem"
}


# ============================================================
# SECTIONS TO SKIP
# ============================================================

SKIP_SECTIONS = {
    "table of contents",
    "contents",
    "acknowledgement",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "preface",
    "foreword"
}


# ============================================================
# GET ELEMENT TYPE
# ============================================================

def _get_element_type(element):
    """
    Returns a normalized element type.

    Supports both:
    1. Normal Unstructured elements
    2. PyPDF fallback elements created with a category field
    """

    element_type = type(element).__name__

    if element_type in ALLOWED_TYPES:
        return element_type

    category = getattr(
        element,
        "category",
        None
    )

    if category in ALLOWED_TYPES:
        return category

    return element_type


# ============================================================
# GET ELEMENT TEXT
# ============================================================

def _get_element_text(element):
    """
    Returns the actual textual content.

    Unstructured elements generally provide text through
    their string representation.

    PyPDF fallback elements provide text through `.text`.
    """

    text = getattr(
        element,
        "text",
        None
    )

    if text is not None:

        return str(text).strip()

    return str(element).strip()


# ============================================================
# GET METADATA VALUE
# ============================================================

def _get_metadata_value(metadata, key, default=None):
    """
    Supports both:
    - Unstructured metadata objects
    - dictionaries
    """

    if metadata is None:
        return default

    if isinstance(metadata, dict):

        return metadata.get(
            key,
            default
        )

    return getattr(
        metadata,
        key,
        default
    )


# ============================================================
# FILTER ELEMENTS
# ============================================================

def filter_elements(elements):
    """
    Keep useful textual elements.

    Supports both Unstructured elements and the PyPDF
    fallback elements used when Unstructured PDF loading
    fails because of environment/dependency issues.
    """

    filtered = []

    for element in elements:

        element_type = _get_element_type(
            element
        )

        if element_type in ALLOWED_TYPES:

            filtered.append(
                element
            )

    print(
        f"After filtering elements: "
        f"{len(filtered)}"
    )

    return filtered


# ============================================================
# BUILD SECTIONS
# ============================================================

def build_sections(elements):
    """
    Builds logical sections while keeping different source
    documents separate.

    A new section is created when:

    - a Title is encountered
    - the source file changes
    """

    sections = []

    current_title = "Untitled"

    current_text = []

    current_page = 1

    current_source = ""

    # --------------------------------------------------------
    # Helper to save current section
    # --------------------------------------------------------

    def save_current_section():

        if not current_text:
            return

        text = "\n".join(
            current_text
        ).strip()

        if not text:
            return

        sections.append({

            "title":
                current_title,

            "text":
                text,

            "page":
                current_page,

            "source":
                current_source

        })

    # --------------------------------------------------------
    # Process Elements
    # --------------------------------------------------------

    for element in elements:

        element_type = _get_element_type(
            element
        )

        metadata = getattr(
            element,
            "metadata",
            None
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        page = _get_metadata_value(
            metadata,
            "page_number",
            1
        )

        if page is None:
            page = 1

        source = _get_metadata_value(
            metadata,
            "filename",
            ""
        )

        if source is None:
            source = ""

        source = str(
            source
        )

        # ----------------------------------------------------
        # Actual Text
        # ----------------------------------------------------

        text = _get_element_text(
            element
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Detect Source Change
        # ----------------------------------------------------

        if (
            current_source
            and source
            and source != current_source
        ):

            save_current_section()

            current_title = "Untitled"

            current_text = []

            current_page = page

            current_source = source

        elif not current_source:

            current_source = source

            current_page = page

        # ----------------------------------------------------
        # New Title
        # ----------------------------------------------------

        if element_type == "Title":

            if current_text:

                save_current_section()

            current_title = text

            current_text = []

            current_page = page

            current_source = source

        else:

            current_text.append(
                text
            )

    # --------------------------------------------------------
    # Save Last Section
    # --------------------------------------------------------

    save_current_section()

    print(
        f"Sections: "
        f"{len(sections)}"
    )

    return sections


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(text):
    """
    Cleans OCR/text extraction noise while preserving
    meaningful academic content.
    """

    cleaned = []

    SKIP_PREFIXES = (

        "Figure",

        "Fig.",

        "Table",

        "Source",

        "Source:",

        "Note",

        "Note:",

        "Chart",

        "Graph"

    )

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Remove Figure/Table captions
        # ----------------------------------------------------

        if line.startswith(
            SKIP_PREFIXES
        ):

            continue

        # ----------------------------------------------------
        # Remove standalone page numbers
        # ----------------------------------------------------

        if re.fullmatch(
            r"\d+",
            line
        ):

            continue

        # ----------------------------------------------------
        # Remove very short OCR fragments
        # ----------------------------------------------------

        words = line.split()

        if len(words) <= 2:

            if not any(
                c.isalpha()
                for c in line
            ):

                continue

        # ----------------------------------------------------
        # Symbol ratio
        # ----------------------------------------------------

        symbol_ratio = (

            sum(

                not c.isalnum()
                and not c.isspace()

                for c in line

            )

            /

            max(
                len(line),
                1
            )

        )

        if symbol_ratio > 0.35:
            continue

        # ----------------------------------------------------
        # Alphabetic character ratio
        # ----------------------------------------------------

        alpha = sum(
            c.isalpha()
            for c in line
        )

        ratio = (
            alpha
            /
            max(
                len(line),
                1
            )
        )

        if ratio < 0.45:
            continue

        cleaned.append(
            line
        )

    return "\n".join(
        cleaned
    )


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents(sections):
    """
    Converts logical sections into LangChain Documents.

    Page 1 is not blindly removed. A section is treated as a
    cover page only when it contains strong cover-page
    indicators and is relatively short.
    """

    documents = []

    for section in sections:

        title = str(
            section.get(
                "title",
                "Untitled"
            )
        ).strip()

        text = clean_text(
            section.get(
                "text",
                ""
            )
        )

        page = section.get(
            "page",
            1
        )

        source = section.get(
            "source",
            ""
        )

        # ----------------------------------------------------
        # Skip Empty Sections
        # ----------------------------------------------------

        if not text:
            continue

        # ----------------------------------------------------
        # Skip Unwanted Sections
        # ----------------------------------------------------

        normalized_title = re.sub(
            r"\s+",
            " ",
            title.lower()
        ).strip()

        if normalized_title in SKIP_SECTIONS:
            continue

        # ----------------------------------------------------
        # Minimum Content Check
        # ----------------------------------------------------

        word_count = len(
            text.split()
        )

        if word_count < 20:
            continue

        # ----------------------------------------------------
        # OCR Quality Filter
        # ----------------------------------------------------

        alpha = sum(
            c.isalpha()
            for c in text
        )

        ratio = (
            alpha
            /
            max(
                len(text),
                1
            )
        )

        if ratio < 0.45:
            continue

        # ----------------------------------------------------
        # Cover Page Detection
        # ----------------------------------------------------

        if page == 1:

            lower_text = text.lower()

            cover_indicators = (

                "submitted by",

                "submitted to",

                "department of",

                "certificate",

                "certification",

                "academic year",

                "college of",

                "university",

                "thesis submitted",

                "project report"

            )

            has_cover_indicator = any(

                indicator in lower_text

                for indicator in cover_indicators

            )

            if (
                has_cover_indicator
                and word_count < 150
            ):

                continue

        # ----------------------------------------------------
        # Create LangChain Document
        # ----------------------------------------------------

        document = Document(

            page_content=text,

            metadata={

                "section":
                    title,

                "page":
                    page,

                "source":
                    source

            }

        )

        documents.append(
            document
        )

    print(
        f"Documents: "
        f"{len(documents)}"
    )

    return documents