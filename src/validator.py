"""
validator.py

Final validation layer for Hybrid GraphRAG Question Paper Generator.

This validator is compatible with the current QuestionGenerator
interface:

    validator.validate(
        paper=paper,
        topic=topic,
        expected_questions=20
    )

It also supports the older interface:

    validator.validate(
        result,
        paper_pattern,
        source_context
    )
"""

import re
from difflib import SequenceMatcher


class QuestionPaperValidator:

    # ============================================================
    # CONFIGURATION
    # ============================================================

    GROUNDING_THRESHOLD = 0.15

    RAW_GRAPH_TERMS = {
        "FOLLOWED_BY",
        "ASSOCIATED_WITH",
        "RELATED_TO",
        "HAS_PART",
        "HAS_SUBTOPIC",
        "PREREQUISITE",
        "DEPENDS_ON",
    }

    BLOOM_LEVELS = {
        "L1": "Remember",
        "L2": "Understand",
        "L3": "Apply",
        "L4": "Analyze",
        "L5": "Evaluate",
        "L6": "Create",
    }

    STOP_WORDS = {
        "the", "a", "an", "is", "was", "were", "are",
        "of", "to", "in", "on", "for", "and", "or",
        "with", "how", "what", "which", "why", "who",
        "when", "where", "did", "has", "have", "had",
        "this", "that", "these", "those", "during",
        "from", "by", "as", "it", "its", "their",
        "there", "than", "into", "over", "through",
        "been", "being", "can", "could", "would",
        "should", "may", "might", "do", "does", "be",
        "such", "also", "very", "more", "most", "some",
        "any", "each", "one", "two", "three", "four",
        "five",
    }

    GENERIC_BLANK_ANSWERS = {
        "concept",
        "thing",
        "factor",
        "aspect",
        "item",
        "term",
        "trend",
        "phase",
        "period",
    }

    QUESTION_TYPES = {
        "MCQ",
        "Fill in the Blanks",
        "Match the Following",
        "Short Answer",
        "Long Answer",
    }

    # ============================================================
    # MAIN VALIDATION
    # ============================================================

    def validate(
        self,
        paper=None,
        topic=None,
        expected_questions=None,
        paper_pattern=None,
        source_context=None,
        result=None,
    ):
        """
        Validate a generated question paper.

        Current interface:
            validate(
                paper=...,
                topic=...,
                expected_questions=...
            )

        Older interface is also supported:
            validate(
                result,
                paper_pattern,
                source_context
            )

        Returns a dictionary:

            {
                "valid": True/False,
                "errors": [...]
            }
        """

        # --------------------------------------------------------
        # Backward compatibility
        # --------------------------------------------------------

        if paper is None and result is not None:
            paper = result

        # Older positional call:
        #
        # validate(result, paper_pattern, source_context)
        #
        # In that case Python places:
        #   paper = result
        #   topic = paper_pattern
        #   expected_questions = source_context
        #
        # Detect that situation.
        if (
            paper_pattern is None
            and isinstance(topic, dict)
            and "sections" in topic
        ):
            paper_pattern = topic
            topic = None

        if (
            expected_questions is not None
            and not isinstance(expected_questions, int)
            and source_context is None
            and (
                isinstance(expected_questions, list)
                or isinstance(expected_questions, str)
            )
        ):
            source_context = expected_questions
            expected_questions = None

        errors = []

        # --------------------------------------------------------
        # Empty paper
        # --------------------------------------------------------

        if paper is None:

            return {
                "valid": False,
                "errors": [
                    "Generated question paper is empty."
                ]
            }

        paper = str(paper).strip()

        if not paper:

            return {
                "valid": False,
                "errors": [
                    "Generated question paper is empty."
                ]
            }

        # --------------------------------------------------------
        # Parse questions
        # --------------------------------------------------------

        questions = self._extract_questions(paper)

        if not questions:

            return {
                "valid": False,
                "errors": [
                    "No questions could be extracted from "
                    "the generated paper."
                ]
            }

        # --------------------------------------------------------
        # Expected question count
        # --------------------------------------------------------

        if expected_questions is None:

            if paper_pattern is not None:

                expected_questions = 0

                for section in paper_pattern.get(
                    "sections",
                    []
                ):

                    expected_questions += int(
                        section.get(
                            "count",
                            0
                        )
                    )

            else:

                expected_questions = len(
                    questions
                )

        # --------------------------------------------------------
        # Validate count
        # --------------------------------------------------------

        if len(questions) != expected_questions:

            errors.append(
                f"Expected {expected_questions} "
                f"questions, found {len(questions)}."
            )

        # --------------------------------------------------------
        # Validate numbering
        # --------------------------------------------------------

        found_numbers = [
            q["number"]
            for q in questions
        ]

        expected_numbers = list(
            range(
                1,
                expected_questions + 1
            )
        )

        if found_numbers != expected_numbers:

            errors.append(
                "Incorrect question numbering. "
                f"Expected {expected_numbers}, "
                f"found {found_numbers}."
            )

        # --------------------------------------------------------
        # Build expected metadata if pattern exists
        # --------------------------------------------------------

        expected_metadata = {}

        if paper_pattern is not None:

            expected_metadata = (
                self._build_expected_metadata(
                    paper_pattern
                )
            )

        # --------------------------------------------------------
        # Validate every question
        # --------------------------------------------------------

        for question in questions:

            qno = question["number"]

            text = question.get(
                "text",
                ""
            )

            answer = question.get(
                "answer",
                ""
            )

            # ----------------------------------------------------
            # Question text
            # ----------------------------------------------------

            if not text.strip():

                errors.append(
                    f"Q{qno}: Question text is missing."
                )

            # ----------------------------------------------------
            # Bloom
            # ----------------------------------------------------

            actual_bloom = self._extract_bloom(
                text
            )

            if actual_bloom is None:

                errors.append(
                    f"Q{qno}: Bloom level is missing "
                    f"or invalid."
                )

            # If paper pattern is available, check expected Bloom.
            if qno in expected_metadata:

                expected_bloom = (
                    expected_metadata[qno]["bloom"]
                )

                if actual_bloom != expected_bloom:

                    errors.append(
                        f"Q{qno}: Expected Bloom "
                        f"{expected_bloom}, "
                        f"found "
                        f"{actual_bloom or 'missing'}."
                    )

            # ----------------------------------------------------
            # Answer
            # ----------------------------------------------------

            if not answer.strip():

                errors.append(
                    f"Q{qno}: Answer missing."
                )

            # ----------------------------------------------------
            # Raw KG terms
            # ----------------------------------------------------

            raw_terms = (
                self._find_raw_graph_terms(
                    text + " " + answer
                )
            )

            for term in raw_terms:

                errors.append(
                    f"Q{qno}: Raw KG term "
                    f"'{term}' detected."
                )

            # ----------------------------------------------------
            # Determine question type
            # ----------------------------------------------------

            qtype = None

            if qno in expected_metadata:

                qtype = expected_metadata[qno]["type"]

            else:

                qtype = self._detect_question_type(
                    question
                )

            # ----------------------------------------------------
            # Type-specific validation
            # ----------------------------------------------------

            if qtype == "MCQ":

                errors.extend(
                    self._validate_mcq(
                        question,
                        qno
                    )
                )

            elif qtype == "Fill in the Blanks":

                errors.extend(
                    self._validate_fill_blank(
                        question,
                        qno
                    )
                )

            elif qtype == "Match the Following":

                errors.extend(
                    self._validate_match(
                        question,
                        qno
                    )
                )

            elif qtype == "Short Answer":

                errors.extend(
                    self._validate_short(
                        question,
                        qno
                    )
                )

            elif qtype == "Long Answer":

                errors.extend(
                    self._validate_long(
                        question,
                        qno
                    )
                )

        # --------------------------------------------------------
        # Section validation
        #
        # Current QuestionGenerator joins questions directly
        # without adding SECTION headers.
        #
        # Therefore section headers are validated ONLY when
        # they actually exist in the generated paper.
        # --------------------------------------------------------

        if paper_pattern is not None:

            errors.extend(
                self._validate_sections_if_present(
                    paper,
                    paper_pattern
                )
            )

        # --------------------------------------------------------
        # Source grounding
        # --------------------------------------------------------

        if source_context:

            source_text = self._prepare_source(
                source_context
            )

            errors.extend(
                self._validate_grounding(
                    questions,
                    source_text
                )
            )

        # --------------------------------------------------------
        # Return current interface
        # --------------------------------------------------------

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # ============================================================
    # BUILD EXPECTED METADATA
    # ============================================================

    def _build_expected_metadata(
        self,
        paper_pattern
    ):

        metadata = {}

        number = 1

        for section in paper_pattern.get(
            "sections",
            []
        ):

            qtype = str(
                section.get(
                    "type",
                    ""
                )
            ).strip()

            count = int(
                section.get(
                    "count",
                    0
                )
            )

            blooms = section.get(
                "blooms",
                []
            )

            if not blooms:

                blooms = ["L1"]

            for position in range(count):

                metadata[number] = {
                    "type": qtype,
                    "bloom": self._normalize_bloom(
                        blooms[
                            position % len(blooms)
                        ]
                    )
                }

                number += 1

        return metadata

    # ============================================================
    # QUESTION PARSER
    # ============================================================

    def _extract_questions(
        self,
        text
    ):

        questions = []

        pattern = re.compile(
            r"(?ms)"
            r"^\s*Q\s*(\d+)"
            r"\s*[\.\):\-]\s*"
            r".*?"
            r"(?=^\s*Q\s*\d+\s*[\.\):\-]|\Z)"
        )

        for match in pattern.finditer(
            text
        ):

            number = int(
                match.group(1)
            )

            raw = match.group(0).strip()

            question = {
                "number": number,
                "raw": raw,
                "text": "",
                "lines": [],
                "options": {},
                "answer": "",
            }

            lines = raw.splitlines()

            for line in lines:

                stripped = line.strip()

                if not stripped:
                    continue

                # ------------------------------------------------
                # Question line
                # ------------------------------------------------

                qmatch = re.match(
                    r"^Q\s*\d+\s*"
                    r"[\.\):\-]\s*(.*)$",
                    stripped,
                    re.IGNORECASE
                )

                if qmatch:

                    question["text"] = (
                        qmatch.group(1).strip()
                    )

                    question["lines"].append(
                        stripped
                    )

                    continue

                # ------------------------------------------------
                # Answer line
                # ------------------------------------------------

                amatch = re.match(
                    r"^Answer\s*:\s*(.*)$",
                    stripped,
                    re.IGNORECASE
                )

                if amatch:

                    question["answer"] = (
                        amatch.group(1).strip()
                    )

                    continue

                # ------------------------------------------------
                # MCQ option
                # ------------------------------------------------

                option_match = re.match(
                    r"^\s*"
                    r"(?:\(([A-G])\)|([A-G]))"
                    r"\s*(?:[\.\):\-]\s*|\s+)"
                    r"(.+?)\s*$",
                    stripped,
                    re.IGNORECASE
                )

                if option_match:

                    letter = (
                        option_match.group(1)
                        or option_match.group(2)
                    ).upper()

                    value = (
                        option_match.group(3)
                        .strip()
                    )

                    question["options"][
                        letter
                    ] = value

                    question["lines"].append(
                        stripped
                    )

                    continue

                # ------------------------------------------------
                # Normal line
                # ------------------------------------------------

                question["lines"].append(
                    stripped
                )

            questions.append(
                question
            )

        return questions

    # ============================================================
    # QUESTION TYPE DETECTION
    # ============================================================

    def _detect_question_type(
        self,
        question
    ):

        raw = question.get(
            "raw",
            ""
        )

        options = question.get(
            "options",
            {}
        )

        if set(options.keys()) == {
            "A",
            "B",
            "C",
            "D"
        }:

            return "MCQ"

        if re.search(
            r"Column\s+A\s*:",
            raw,
            re.IGNORECASE
        ) and re.search(
            r"Column\s+B\s*:",
            raw,
            re.IGNORECASE
        ):

            return "Match the Following"

        if re.search(
            r"_{3,}",
            raw
        ):

            return "Fill in the Blanks"

        # There is no reliable way to distinguish a short answer
        # from a long answer from formatting alone.
        #
        # When no paper pattern is supplied, treat it as a
        # general text-answer question.
        return "Short Answer"

    # ============================================================
    # BLOOM NORMALIZATION
    # ============================================================

    def _normalize_bloom(
        self,
        bloom
    ):

        value = str(
            bloom
        ).strip()

        match = re.match(
            r"^(L[1-6])",
            value,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            ).upper()

        for code, name in (
            self.BLOOM_LEVELS.items()
        ):

            if value.lower() == name.lower():

                return code

        return value.upper()

    # ============================================================
    # BLOOM EXTRACTION
    # ============================================================

    def _extract_bloom(
        self,
        text
    ):

        # Supports:
        #
        # [L1]
        # [L1 - Remember]
        # [L1: Remember]
        # [Remember]
        #

        match = re.search(
            r"\[\s*"
            r"(L[1-6])"
            r"(?:\s*[-:]\s*"
            r"(?:Remember|Understand|Apply|Analyze|"
            r"Evaluate|Create))?"
            r"\s*\]",
            text,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            ).upper()

        name_match = re.search(
            r"\[\s*"
            r"(Remember|Understand|Apply|Analyze|"
            r"Evaluate|Create)"
            r"\s*\]",
            text,
            re.IGNORECASE
        )

        if name_match:

            return self._normalize_bloom(
                name_match.group(1)
            )

        return None

    # ============================================================
    # MCQ VALIDATION
    # ============================================================

    def _validate_mcq(
        self,
        question,
        qno
    ):

        errors = []

        options = question.get(
            "options",
            {}
        )

        required = {
            "A",
            "B",
            "C",
            "D"
        }

        # --------------------------------------------------------
        # Options
        # --------------------------------------------------------

        if set(options.keys()) != required:

            missing = (
                required
                - set(options.keys())
            )

            extra = (
                set(options.keys())
                - required
            )

            if missing:

                errors.append(
                    f"Q{qno}: MCQ is missing "
                    f"option(s): "
                    f"{', '.join(sorted(missing))}."
                )

            if extra:

                errors.append(
                    f"Q{qno}: MCQ contains "
                    f"invalid option(s): "
                    f"{', '.join(sorted(extra))}."
                )

            errors.append(
                f"Q{qno}: MCQ must contain "
                f"exactly A, B, C and D."
            )

            return errors

        # --------------------------------------------------------
        # Empty options
        # --------------------------------------------------------

        for letter in "ABCD":

            if not options[letter].strip():

                errors.append(
                    f"Q{qno}: Option {letter} "
                    f"is empty."
                )

        # --------------------------------------------------------
        # Duplicate options
        # --------------------------------------------------------

        normalized_options = [
            self._normalize_text(
                options[letter]
            )
            for letter in "ABCD"
        ]

        if len(
            set(normalized_options)
        ) != 4:

            errors.append(
                f"Q{qno}: MCQ contains "
                f"duplicate options."
            )

        # --------------------------------------------------------
        # Answer
        # --------------------------------------------------------

        answer = question.get(
            "answer",
            ""
        ).strip()

        answer_letter = (
            self._extract_answer_letter(
                answer,
                options
            )
        )

        if answer_letter not in required:

            errors.append(
                f"Q{qno}: MCQ answer must "
                f"identify exactly one of "
                f"A, B, C, or D."
            )

        return errors

    # ============================================================
    # EXTRACT MCQ ANSWER
    # ============================================================

    def _extract_answer_letter(
        self,
        answer,
        options
    ):

        if not answer:

            return None

        value = answer.strip()

        # A
        # A.
        # A)
        # Option A
        # Option A.
        match = re.match(
            r"^\s*"
            r"(?:Option\s*)?"
            r"\(?([A-D])\)?"
            r"(?:\s*[\.\):\-]\s*.*)?"
            r"\s*$",
            value,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            ).upper()

        # --------------------------------------------------------
        # Full option text
        # --------------------------------------------------------

        normalized_answer = (
            self._normalize_text(
                value
            )
        )

        for letter in "ABCD":

            option = options.get(
                letter,
                ""
            )

            if not option:
                continue

            if (
                normalized_answer
                ==
                self._normalize_text(
                    option
                )
            ):

                return letter

        return None

    # ============================================================
    # FILL IN THE BLANKS
    # ============================================================

    def _validate_fill_blank(
        self,
        question,
        qno
    ):

        errors = []

        raw = question.get(
            "raw",
            ""
        )

        blanks = re.findall(
            r"_{3,}",
            raw
        )

        if len(blanks) != 1:

            errors.append(
                f"Q{qno}: Fill in the Blanks "
                f"must contain exactly one blank."
            )

        if question.get(
            "options",
            {}
        ):

            errors.append(
                f"Q{qno}: Fill in the Blanks "
                f"must not contain MCQ options."
            )

        answer = question.get(
            "answer",
            ""
        ).strip()

        if not answer:

            errors.append(
                f"Q{qno}: Fill in the Blanks "
                f"answer is missing."
            )

        if answer.lower() in (
            self.GENERIC_BLANK_ANSWERS
        ):

            errors.append(
                f"Q{qno}: Fill in the Blanks "
                f"uses an overly generic answer "
                f"'{answer}'."
            )

        return errors

    # ============================================================
    # MATCH THE FOLLOWING
    # ============================================================

    def _validate_match(
        self,
        question,
        qno
    ):

        errors = []

        raw = question.get(
            "raw",
            ""
        )

        # --------------------------------------------------------
        # Column headers
        # --------------------------------------------------------

        has_a = re.search(
            r"Column\s+A\s*:",
            raw,
            re.IGNORECASE
        )

        has_b = re.search(
            r"Column\s+B\s*:",
            raw,
            re.IGNORECASE
        )

        if not has_a:

            errors.append(
                f"Q{qno}: Column A missing."
            )

        if not has_b:

            errors.append(
                f"Q{qno}: Column B missing."
            )

        if not has_a or not has_b:

            return errors

        # --------------------------------------------------------
        # Extract Column A
        #
        # Supports:
        #
        # 1. Item
        # 2. Item
        #
        # --------------------------------------------------------

        a_match = re.search(
            r"Column\s+A\s*:"
            r"(.*?)(?=Column\s+B\s*:)",
            raw,
            re.IGNORECASE | re.DOTALL
        )

        # --------------------------------------------------------
        # Extract Column B
        # --------------------------------------------------------

        b_match = re.search(
            r"Column\s+B\s*:"
            r"(.*?)(?=Answer\s*:)",
            raw,
            re.IGNORECASE | re.DOTALL
        )

        # --------------------------------------------------------
        # Column A can contain 1-5
        # --------------------------------------------------------

        if a_match:

            a_items = re.findall(
                r"(?m)^\s*[1-5]"
                r"\s*[\.\):\-]\s*.+$",
                a_match.group(1)
            )

            if len(a_items) != 5:

                errors.append(
                    f"Q{qno}: Column A must "
                    f"contain exactly five items."
                )

        # --------------------------------------------------------
        # Column B can contain A-E
        # --------------------------------------------------------

        if b_match:

            b_items = re.findall(
                r"(?m)^\s*[A-E]"
                r"\s*[\.\):\-]\s*.+$",
                b_match.group(1)
            )

            if len(b_items) != 5:

                errors.append(
                    f"Q{qno}: Column B must "
                    f"contain exactly five items."
                )

        # --------------------------------------------------------
        # Answer
        #
        # Expected:
        #
        # 1-A, 2-B, 3-C, 4-D, 5-E
        # --------------------------------------------------------

        answer_match = re.search(
            r"Answer\s*:\s*(.*?)$",
            raw,
            re.IGNORECASE | re.DOTALL
        )

        if not answer_match:

            errors.append(
                f"Q{qno}: Match answer missing."
            )

            return errors

        answer_text = (
            answer_match.group(1)
            .strip()
        )

        mappings = re.findall(
            r"([1-5])\s*[-:]\s*([A-E])",
            answer_text.upper()
        )

        if len(mappings) != 5:

            errors.append(
                f"Q{qno}: Match answer must "
                f"contain exactly five mappings."
            )

        else:

            left = [
                item[0]
                for item in mappings
            ]

            right = [
                item[1]
                for item in mappings
            ]

            if len(set(left)) != 5:

                errors.append(
                    f"Q{qno}: Match answer must "
                    f"use each Column A item exactly once."
                )

            if len(set(right)) != 5:

                errors.append(
                    f"Q{qno}: Match answer must "
                    f"use each Column B item exactly once."
                )

        return errors

    # ============================================================
    # SHORT ANSWER
    # ============================================================

    def _validate_short(
        self,
        question,
        qno
    ):

        errors = []

        if question.get(
            "options",
            {}
        ):

            errors.append(
                f"Q{qno}: Short Answer must "
                f"not contain MCQ options."
            )

        if not question.get(
            "text",
            ""
        ).strip():

            errors.append(
                f"Q{qno}: Short Answer "
                f"question is empty."
            )

        if not question.get(
            "answer",
            ""
        ).strip():

            errors.append(
                f"Q{qno}: Short Answer "
                f"answer is missing."
            )

        return errors

    # ============================================================
    # LONG ANSWER
    # ============================================================

    def _validate_long(
        self,
        question,
        qno
    ):

        errors = []

        if question.get(
            "options",
            {}
        ):

            errors.append(
                f"Q{qno}: Long Answer must "
                f"not contain MCQ options."
            )

        if not question.get(
            "text",
            ""
        ).strip():

            errors.append(
                f"Q{qno}: Long Answer "
                f"question is empty."
            )

        if not question.get(
            "answer",
            ""
        ).strip():

            errors.append(
                f"Q{qno}: Long Answer "
                f"answer is missing."
            )

        return errors

    # ============================================================
    # SECTION VALIDATION
    # ============================================================

    def _validate_sections_if_present(
        self,
        result,
        paper_pattern
    ):

        errors = []

        expected = [
            str(
                section.get(
                    "name",
                    ""
                )
            ).strip()
            for section
            in paper_pattern.get(
                "sections",
                []
            )
        ]

        found = re.findall(
            r"^\s*SECTION\s+(.+?)\s*$",
            result,
            re.IGNORECASE | re.MULTILINE
        )

        found = [
            x.strip()
            for x in found
        ]

        # --------------------------------------------------------
        # The current QuestionGenerator does not add SECTION
        # headings to the final paper.
        #
        # Therefore, only validate them if they are actually
        # present.
        # --------------------------------------------------------

        if found:

            if found != expected:

                errors.append(
                    f"Incorrect sections. "
                    f"Expected {expected}, "
                    f"found {found}."
                )

        return errors

    # ============================================================
    # GROUNDING
    # ============================================================

    def _validate_grounding(
        self,
        questions,
        source_text
    ):

        errors = []

        if not source_text.strip():

            return errors

        for question in questions:

            qno = question[
                "number"
            ]

            qtext = self._clean_for_grounding(
                question.get(
                    "text",
                    ""
                )
            )

            answer = self._clean_for_grounding(
                question.get(
                    "answer",
                    ""
                )
            )

            # ----------------------------------------------------
            # For MCQ, ground the selected option rather than
            # merely checking "A".
            # ----------------------------------------------------

            options = question.get(
                "options",
                {}
            )

            if options:

                letter = (
                    self._extract_answer_letter(
                        answer,
                        options
                    )
                )

                answer_text = (
                    options.get(
                        letter,
                        answer
                    )
                )

            else:

                answer_text = answer

            q_score = self._grounding_score(
                qtext,
                source_text
            )

            a_score = self._grounding_score(
                answer_text,
                source_text
            )

            score = max(
                q_score,
                a_score
            )

            if score < self.GROUNDING_THRESHOLD:

                errors.append(
                    f"Q{qno}: Low source grounding "
                    f"score ({score:.2f}). "
                    f"Expected at least "
                    f"{self.GROUNDING_THRESHOLD:.2f}."
                )

        return errors

    # ============================================================
    # GROUNDING SCORE
    # ============================================================

    def _grounding_score(
        self,
        text,
        source
    ):

        if not text or not source:

            return 0.0

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        normalized_source = (
            self._normalize_text(
                source
            )
        )

        if not normalized_text:

            return 0.0

        # Exact phrase
        if normalized_text in normalized_source:

            return 1.0

        text_words = (
            self._meaningful_words(
                text
            )
        )

        source_words = (
            self._meaningful_words(
                source
            )
        )

        if not text_words or not source_words:

            return 0.0

        source_set = set(
            source_words
        )

        overlap = sum(
            word in source_set
            for word in text_words
        )

        overlap_score = (
            overlap /
            len(text_words)
        )

        # --------------------------------------------------------
        # Local sentence score
        # --------------------------------------------------------

        local_score = 0.0

        sentences = re.split(
            r"(?<=[.!?])\s+",
            source
        )

        for sentence in sentences:

            sentence_words = set(
                self._meaningful_words(
                    sentence
                )
            )

            if not sentence_words:

                continue

            local_overlap = sum(
                word in sentence_words
                for word in text_words
            )

            local_score = max(
                local_score,
                local_overlap /
                len(text_words)
            )

        # --------------------------------------------------------
        # Sequence similarity
        # --------------------------------------------------------

        sequence_score = SequenceMatcher(
            None,
            " ".join(text_words),
            " ".join(source_words)
        ).ratio()

        score = max(
            overlap_score,
            local_score,
            sequence_score * 0.7
        )

        return round(
            min(
                max(
                    score,
                    0.0
                ),
                1.0
            ),
            2
        )

    # ============================================================
    # SOURCE PREPARATION
    # ============================================================

    def _prepare_source(
        self,
        source_context
    ):

        if isinstance(
            source_context,
            list
        ):

            parts = []

            for item in source_context:

                if isinstance(
                    item,
                    dict
                ):

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:

                        parts.append(
                            str(text)
                        )

                else:

                    parts.append(
                        str(item)
                    )

            return "\n\n".join(
                parts
            )

        return str(
            source_context
        )

    # ============================================================
    # TEXT NORMALIZATION
    # ============================================================

    def _normalize_text(
        self,
        text
    ):

        if not text:

            return ""

        text = text.lower()

        # Remove Bloom label
        text = re.sub(
            r"\[\s*l[1-6]"
            r"(?:\s*[-:]\s*"
            r"(?:remember|understand|apply|analyze|"
            r"evaluate|create))?"
            r"\s*\]",
            " ",
            text,
            flags=re.IGNORECASE
        )

        # Remove Answer prefix
        text = re.sub(
            r"^\s*answer\s*:\s*",
            " ",
            text,
            flags=re.IGNORECASE
        )

        # Remove question number
        text = re.sub(
            r"^\s*q\s*\d+\s*[\.\):\-]\s*",
            " ",
            text,
            flags=re.IGNORECASE
        )

        # Remove punctuation
        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        # Normalize whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ============================================================
    # MEANINGFUL WORDS
    # ============================================================

    def _meaningful_words(
        self,
        text
    ):

        words = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )

        return [
            word
            for word in words
            if word not in self.STOP_WORDS
        ]

    # ============================================================
    # RAW KG TERMS
    # ============================================================

    def _find_raw_graph_terms(
        self,
        text
    ):

        found = []

        upper = text.upper()

        for term in self.RAW_GRAPH_TERMS:

            if re.search(
                rf"\b{re.escape(term)}\b",
                upper
            ):

                found.append(
                    term
                )

        return found

    # ============================================================
    # CLEAN GROUNDING TEXT
    # ============================================================

    def _clean_for_grounding(
        self,
        text
    ):

        if not text:

            return ""

        text = re.sub(
            r"^\s*"
            r"\[\s*L[1-6]"
            r"(?:\s*[-:]\s*"
            r"(?:Remember|Understand|Apply|Analyze|"
            r"Evaluate|Create))?"
            r"\s*\]\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^\s*Answer\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        return text.strip()