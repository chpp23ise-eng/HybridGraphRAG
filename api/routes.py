"""
routes.py

API Endpoints
"""

import time

from fastapi import APIRouter

from api.schemas import (
    GenerateRequest,
    GenerateResponse
)

from api.service import service


router = APIRouter()


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get("/health")
def health():

    return {
        "status": "Running"
    }


# ============================================================
# GENERATE QUESTION PAPER
# ============================================================

@router.post(
    "/generate-paper",
    response_model=GenerateResponse
)
def generate_question_paper(
    request: GenerateRequest
):

    start = time.time()

    # --------------------------------------------------------
    # Generate Question Paper
    # --------------------------------------------------------

    paper = service.generate_question_paper(

        topic=request.topic,

        blooms=request.blooms,

        paper_pattern=request.paper_pattern

    )

    # --------------------------------------------------------
    # Generation Time
    # --------------------------------------------------------

    end = time.time()

    print()

    print("=" * 60)

    print(
        f"Generation Time : "
        f"{round(end - start, 2)} sec"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return GenerateResponse(

        question_paper=paper

    )