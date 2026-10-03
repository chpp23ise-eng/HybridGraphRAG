"""
schemas.py

Pydantic models for API requests
and responses.
"""

from typing import List, Optional

from pydantic import BaseModel


# ============================================================
# QUESTION PAPER SECTION
# ============================================================

class PaperSection(BaseModel):

    name: str

    type: str

    count: int

    marks_each: int

    blooms: List[str]


# ============================================================
# QUESTION PAPER PATTERN
# ============================================================

class PaperPattern(BaseModel):

    total_marks: int

    duration: str

    sections: List[PaperSection]


# ============================================================
# GENERATE REQUEST
# ============================================================

class GenerateRequest(BaseModel):

    topic: str

    blooms: Optional[List[str]] = [

        "Remember",

        "Understand",

        "Apply",

        "Analyze",

        "Evaluate",

        "Create"

    ]

    paper_pattern: Optional[PaperPattern] = None


# ============================================================
# GENERATE RESPONSE
# ============================================================

class GenerateResponse(BaseModel):

    question_paper: str