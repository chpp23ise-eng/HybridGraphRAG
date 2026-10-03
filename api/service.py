"""
service.py

Business logic for the FastAPI backend.

Responsibilities:

1. Load the LLM.
2. Initialize the Retriever.
3. Initialize the Knowledge Graph traversal.
4. Initialize the QuestionGenerator.
5. Validate the user-defined question paper pattern.
6. Convert Pydantic models into dictionaries.
7. Pass the validated pattern to QuestionGenerator.
"""

from src.llm import load_llm
from src.retriever import Retriever
from src.question_generator import QuestionGenerator
from src.graph_traversal import GraphTraversal


class QuestionPaperService:

    # ==========================================================
    # SUPPORTED QUESTION TYPES
    # ==========================================================

    SUPPORTED_QUESTION_TYPES = {
        "MCQ",
        "Fill in the Blanks",
        "Match the Following",
        "Short Answer",
        "Long Answer"
    }

    # ==========================================================
    # SUPPORTED BLOOM LEVELS
    # ==========================================================

    SUPPORTED_BLOOMS = {
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create"
    }

    # ==========================================================
    # INITIALIZE SERVICE
    # ==========================================================

    def __init__(self):

        print(
            "\nLoading Mistral...\n"
        )

        # ------------------------------------------------------
        # Load Mistral
        # ------------------------------------------------------

        self.llm = load_llm()

        # ------------------------------------------------------
        # Initialize Retriever
        # ------------------------------------------------------

        self.retriever = Retriever()

        # ------------------------------------------------------
        # Initialize Knowledge Graph Traversal
        # ------------------------------------------------------

        self.graph_traversal = GraphTraversal()

        # ------------------------------------------------------
        # Initialize Question Generator
        # ------------------------------------------------------

        self.generator = QuestionGenerator(
            self.llm,
            self.retriever,
            self.graph_traversal
        )

    # ==========================================================
    # CONVERT PAPER PATTERN TO DICTIONARY
    # ==========================================================

    @staticmethod
    def _convert_pattern_to_dict(paper_pattern):

        if paper_pattern is None:
            return None

        # Pydantic v2
        if hasattr(
            paper_pattern,
            "model_dump"
        ):

            return paper_pattern.model_dump()

        # Pydantic v1
        if hasattr(
            paper_pattern,
            "dict"
        ):

            return paper_pattern.dict()

        # Already a dictionary
        if isinstance(
            paper_pattern,
            dict
        ):

            return paper_pattern

        raise ValueError(
            "Invalid paper pattern format."
        )

    # ==========================================================
    # NORMALIZE BLOOM LEVEL
    # ==========================================================

    @staticmethod
    def _normalize_bloom(bloom):

        bloom = str(
            bloom
        ).strip()

        # Example:
        # "L1 - Remember"
        if "-" in bloom:

            bloom = (
                bloom
                .split("-")[0]
                .strip()
            )

        # Convert full Bloom name to code
        bloom_map = {
            "remember": "L1",
            "understand": "L2",
            "apply": "L3",
            "analyze": "L4",
            "evaluate": "L5",
            "create": "L6"
        }

        lower_bloom = bloom.lower()

        if lower_bloom in bloom_map:

            return bloom_map[
                lower_bloom
            ]

        return bloom.upper()

    # ==========================================================
    # VALIDATE BLOOM LEVELS
    # ==========================================================

    @classmethod
    def _validate_blooms(
        cls,
        blooms,
        location
    ):

        if blooms is None:
            return

        if not isinstance(
            blooms,
            list
        ):

            raise ValueError(
                f"{location}.blooms must be a list."
            )

        if len(blooms) == 0:

            raise ValueError(
                f"{location}.blooms cannot be empty."
            )

        for bloom in blooms:

            normalized = cls._normalize_bloom(
                bloom
            )

            if normalized not in {
                "L1",
                "L2",
                "L3",
                "L4",
                "L5",
                "L6"
            }:

                raise ValueError(
                    f"Invalid Bloom level "
                    f"'{bloom}' in {location}. "
                    f"Use L1-L6 or valid Bloom names."
                )

    # ==========================================================
    # VALIDATE PAPER PATTERN
    # ==========================================================

    @classmethod
    def _validate_paper_pattern(
        cls,
        paper_pattern
    ):

        # ------------------------------------------------------
        # Pattern must be a dictionary
        # ------------------------------------------------------

        if not isinstance(
            paper_pattern,
            dict
        ):

            raise ValueError(
                "Paper pattern must be a dictionary."
            )

        # ------------------------------------------------------
        # Required fields
        # ------------------------------------------------------

        required_fields = [
            "total_marks",
            "duration",
            "sections"
        ]

        for field in required_fields:

            if field not in paper_pattern:

                raise ValueError(
                    f"Paper pattern is missing "
                    f"required field: '{field}'."
                )

        # ------------------------------------------------------
        # Validate total marks
        # ------------------------------------------------------

        total_marks = paper_pattern[
            "total_marks"
        ]

        try:

            total_marks = int(
                total_marks
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "total_marks must be an integer."
            )

        if total_marks <= 0:

            raise ValueError(
                "total_marks must be greater than 0."
            )

        # ------------------------------------------------------
        # Validate duration
        # ------------------------------------------------------

        duration = paper_pattern[
            "duration"
        ]

        if not isinstance(
            duration,
            str
        ) or not duration.strip():

            raise ValueError(
                "duration must be a non-empty string."
            )

        # ------------------------------------------------------
        # Validate sections
        # ------------------------------------------------------

        sections = paper_pattern[
            "sections"
        ]

        if not isinstance(
            sections,
            list
        ):

            raise ValueError(
                "sections must be a list."
            )

        if len(sections) == 0:

            raise ValueError(
                "At least one section is required."
            )

        # ------------------------------------------------------
        # Track section names
        # ------------------------------------------------------

        section_names = set()

        # ------------------------------------------------------
        # Calculate actual marks
        # ------------------------------------------------------

        calculated_total_marks = 0

        # ------------------------------------------------------
        # Validate each section
        # ------------------------------------------------------

        for index, section in enumerate(
            sections,
            start=1
        ):

            location = (
                f"section {index}"
            )

            if not isinstance(
                section,
                dict
            ):

                raise ValueError(
                    f"{location} must be an object."
                )

            # --------------------------------------------------
            # Section name
            # --------------------------------------------------

            section_name = section.get(
                "name"
            )

            if not section_name:

                raise ValueError(
                    f"{location} is missing 'name'."
                )

            section_name = str(
                section_name
            ).strip()

            if section_name in section_names:

                raise ValueError(
                    f"Duplicate section name: "
                    f"'{section_name}'."
                )

            section_names.add(
                section_name
            )

            # --------------------------------------------------
            # Question type
            # --------------------------------------------------

            question_type = section.get(
                "type"
            )

            if question_type not in (
                cls.SUPPORTED_QUESTION_TYPES
            ):

                supported = ", ".join(
                    sorted(
                        cls.SUPPORTED_QUESTION_TYPES
                    )
                )

                raise ValueError(
                    f"Invalid question type "
                    f"'{question_type}' in "
                    f"{location}. "
                    f"Supported types: {supported}."
                )

            # --------------------------------------------------
            # Question count
            # --------------------------------------------------

            count = section.get(
                "count"
            )

            try:

                count = int(
                    count
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"{location}.count must be "
                    f"an integer."
                )

            if count <= 0:

                raise ValueError(
                    f"{location}.count must be "
                    f"greater than 0."
                )

            # --------------------------------------------------
            # Marks per question
            # --------------------------------------------------

            marks_each = section.get(
                "marks_each"
            )

            try:

                marks_each = int(
                    marks_each
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"{location}.marks_each must "
                    f"be an integer."
                )

            if marks_each <= 0:

                raise ValueError(
                    f"{location}.marks_each must "
                    f"be greater than 0."
                )

            # --------------------------------------------------
            # Section marks
            # --------------------------------------------------

            section_marks = (
                count * marks_each
            )

            calculated_total_marks += (
                section_marks
            )

            # --------------------------------------------------
            # Bloom levels
            # --------------------------------------------------

            section_blooms = section.get(
                "blooms"
            )

            cls._validate_blooms(
                section_blooms,
                location
            )

        # ======================================================
        # TOTAL MARKS MUST MATCH SECTION MARKS
        # ======================================================

        if (
            calculated_total_marks
            != total_marks
        ):

            raise ValueError(
                "Total marks mismatch. "
                f"Paper pattern specifies "
                f"{total_marks} total marks, "
                f"but the sections add up to "
                f"{calculated_total_marks} marks."
            )

        # ------------------------------------------------------
        # Return validated pattern
        # ------------------------------------------------------

        return paper_pattern

    # ==========================================================
    # GENERATE QUESTION PAPER
    # ==========================================================

    def generate_question_paper(
        self,
        topic,
        blooms=None,
        paper_pattern=None
    ):

        # ======================================================
        # VALIDATE TOPIC
        # ======================================================

        if topic is None:

            raise ValueError(
                "Topic cannot be empty."
            )

        topic = str(
            topic
        ).strip()

        if not topic:

            raise ValueError(
                "Topic cannot be empty."
            )

        # ======================================================
        # CONVERT Pydantic PATTERN → DICTIONARY
        # ======================================================

        paper_pattern = (
            self._convert_pattern_to_dict(
                paper_pattern
            )
        )

        # ======================================================
        # VALIDATE PAPER PATTERN
        # ======================================================

        if paper_pattern is not None:

            paper_pattern = (
                self._validate_paper_pattern(
                    paper_pattern
                )
            )

        # ======================================================
        # VALIDATE TOP-LEVEL BLOOMS
        # ======================================================

        self._validate_blooms(
            blooms,
            "request"
        )

        # ======================================================
        # GENERATION LOG
        # ======================================================

        print()
        print("=" * 70)
        print("QUESTION PAPER REQUEST")
        print("=" * 70)
        print()
        print(
            f"Topic        : {topic}"
        )

        if paper_pattern is not None:

            print(
                f"Total Marks  : "
                f"{paper_pattern['total_marks']}"
            )

            print(
                f"Duration     : "
                f"{paper_pattern['duration']}"
            )

            print(
                f"Sections     : "
                f"{len(paper_pattern['sections'])}"
            )

            print()
            print(
                "Paper Pattern:"
            )

            for section in (
                paper_pattern["sections"]
            ):

                print(
                    f"  {section['name']} → "
                    f"{section['type']} → "
                    f"{section['count']} × "
                    f"{section['marks_each']} marks"
                )

        print()
        print("=" * 70)
        print()

        # ======================================================
        # GENERATE USING QUESTION GENERATOR
        # ======================================================

        paper = self.generator.generate(
            topic=topic,
            blooms=blooms,
            paper_pattern=paper_pattern
        )

        # ======================================================
        # BASIC RESPONSE VALIDATION
        # ======================================================

        if paper is None:

            raise ValueError(
                "Question generator returned None."
            )

        paper = str(
            paper
        ).strip()

        if not paper:

            raise ValueError(
                "Question generator returned an empty paper."
            )

        # ======================================================
        # RETURN PAPER
        # ======================================================

        return paper


# ============================================================
# SINGLETON SERVICE
# ============================================================

service = QuestionPaperService()