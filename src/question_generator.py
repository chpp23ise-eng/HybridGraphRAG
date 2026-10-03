import re

from src.prompt_builder import PromptBuilder
from src.validator import QuestionPaperValidator


class QuestionGenerator:

    def __init__(self, llm, retriever, graph_traversal=None):
        self.llm = llm
        self.retriever = retriever
        self.graph_traversal = graph_traversal

    # ============================================================
    # PUBLIC GENERATION
    # ============================================================

    def generate(
        self,
        topic,
        blooms,
        paper_pattern=None
    ):

        graph_context = self._get_graph_context(topic)

        sections = self._get_sections(
            paper_pattern,
            blooms
        )

        questions = []

        previous_questions = []

        question_number = 1

        for section in sections:

            section_name = self._get(
                section,
                "name",
                "A"
            )

            question_type = self._get(
                section,
                "type",
                "MCQ"
            )

            count = int(
                self._get(
                    section,
                    "count",
                    0
                )
            )

            section_blooms = self._get(
                section,
                "blooms",
                blooms
            )

            print()
            print("=" * 80)
            print(
                f"SECTION {section_name} "
                f"({question_type.upper()})"
            )
            print("=" * 80)

            for position in range(count):

                qnum = question_number

                print()
                print(f"Generating Q{qnum}...")

                question = None
                response = ""

                # ------------------------------------------------
                # RETRIEVE CONTEXT FOR THIS QUESTION
                # ------------------------------------------------

                retrieval_query = self._build_retrieval_query(
                    topic=topic,
                    question_type=question_type,
                    bloom=section_blooms[
                        position % len(section_blooms)
                    ],
                    previous_questions=previous_questions
                )

                retrieved_chunks = self._retrieve(
                    retrieval_query
                )

                for attempt in range(1, 4):

                    print(
                        f"  Attempt {attempt}/3"
                    )

                    # ------------------------------------------------
                    # CHANGE RETRIEVAL CONTEXT ON RETRY
                    # ------------------------------------------------

                    if attempt > 1:

                        retry_query = (
                            self._build_retry_retrieval_query(
                                topic=topic,
                                question_type=question_type,
                                bloom=section_blooms[
                                    position % len(section_blooms)
                                ],
                                previous_questions=previous_questions,
                                attempt=attempt
                            )
                        )

                        retry_chunks = self._retrieve(
                            retry_query
                        )

                        if retry_chunks:
                            retrieved_chunks = retry_chunks

                    prompt = PromptBuilder.build_prompt(
                        topic=topic,
                        blooms=section_blooms,
                        configuration=None,
                        graph_context=graph_context,
                        retrieved_context="",
                        retrieved_chunks=retrieved_chunks,
                        paper_pattern=paper_pattern,
                        question_start=qnum,
                        question_number=qnum,
                        previous_questions=previous_questions
                    )

                    # ------------------------------------------------
                    # RETRY / FORMAT CORRECTION
                    # ------------------------------------------------

                    if attempt > 1:

                        prompt += f"""

IMPORTANT CORRECTION:

Your previous response was not accepted.

You MUST generate a NEW {question_type} question.

REQUIRED QUESTION TYPE:
{question_type}

DO NOT generate a different question type.

"""

                        if question_type.lower() == "mcq":

                            prompt += """
For MCQ:
- provide exactly four options
- use A, B, C and D
- provide exactly one Answer line
- ensure exactly one option is correct
"""

                        elif question_type.lower() in [
                            "fill in the blanks",
                            "fill in the blank",
                            "fill blanks"
                        ]:

                            prompt += """
For Fill in the Blanks:
- provide exactly one sentence/question
- include exactly ONE blank
- represent the blank using underscores such as _____
- provide exactly one Answer line
- DO NOT provide MCQ options
- DO NOT create more than one blank
- DO NOT ask about a fact already tested
- DO NOT reuse a statistic, percentage, quantity,
  crop, person, institution, event, cause, or effect
  already used in the previous questions
- select a DIFFERENT factual detail from the source material
"""

                        elif question_type.lower() in [
                            "short answer",
                            "short-answer"
                        ]:

                            prompt += """
For Short Answer:
- ask an explanatory question
- DO NOT provide MCQ options
- DO NOT use A/B/C/D options
- provide an appropriate explanatory Answer
"""

                        elif question_type.lower() in [
                            "long answer",
                            "long-answer"
                        ]:

                            prompt += """
For Long Answer:
- ask an explanatory/descriptive question
- the question should require a detailed answer
- DO NOT provide MCQ options
- DO NOT use A/B/C/D options
- provide an appropriate detailed Answer
"""

                        prompt += """

Also:
- do not repeat a previous question
- do not paraphrase a previous question
- do not ask about the same specific fact
- do not ask about the same statistic
- do not reuse the same question structure
- select a different fact, concept, relationship, event,
  cause, effect, comparison, or implication from the
  SOURCE MATERIAL

Follow the required output format exactly.
"""

                    response = self._invoke_llm(
                        prompt
                    )

                    normalized = self._normalize_question(
                        response=response,
                        question_type=question_type,
                        question_number=qnum,
                        expected_bloom=section_blooms[
                            position % len(section_blooms)
                        ],
                        source_context=self._source_text(
                            retrieved_chunks
                        )
                    )

                    if not normalized:

                        print(
                            "  Could not normalize response."
                        )

                        continue

                    # ------------------------------------------------
                    # QUESTION TYPE VALIDATION
                    # ------------------------------------------------

                    if not self._question_type_is_valid(
                        normalized,
                        question_type
                    ):

                        print(
                            "  Wrong question type generated."
                        )

                        print(
                            f"  Expected: {question_type}"
                        )

                        continue

                    # ------------------------------------------------
                    # EXTRA MCQ QUALITY CHECK
                    # ------------------------------------------------

                    if question_type.lower() == "mcq":

                        if not self._mcq_is_valid(
                            normalized,
                            self._source_text(
                                retrieved_chunks
                            )
                        ):

                            print(
                                "  Invalid/ambiguous MCQ."
                            )

                            continue

                    # ------------------------------------------------
                    # DUPLICATE QUESTION CHECK
                    # ------------------------------------------------

                    if self._is_duplicate_question(
                        normalized,
                        previous_questions
                    ):

                        print(
                            "  Duplicate/similar question detected."
                        )

                        continue

                    question = normalized

                    print(
                        f"Q{qnum} generated successfully."
                    )

                    break

                # ----------------------------------------------------
                # GENERATION FAILED AFTER THREE ATTEMPTS
                # ----------------------------------------------------

                if question is None:

                    print()
                    print(
                        "Last response for Q"
                        + str(qnum)
                        + ":"
                    )

                    print(
                        response if response else ""
                    )

                    print()

                    raise RuntimeError(
                        f"Unable to generate Q{qnum}."
                    )

                questions.append(
                    question
                )

                previous_questions.append(
                    question
                )

                question_number += 1

        # ========================================================
        # FINAL VALIDATION
        # ========================================================

        paper = "\n\n".join(
            questions
        )

        validator = QuestionPaperValidator()

        validation = validator.validate(
            paper=paper,
            topic=topic,
            expected_questions=question_number - 1
        )

        if not validation["valid"]:

            print()
            print("=" * 80)
            print("FINAL VALIDATION FAILED")
            print("=" * 80)

            for error in validation.get(
                "errors",
                []
            ):
                print(error)

            raise RuntimeError(
                "Generated question paper failed final validation."
            )

        return paper

    # ============================================================
    # RETRIEVAL QUERY
    # ============================================================

    def _build_retrieval_query(
        self,
        topic,
        question_type,
        bloom,
        previous_questions
    ):

        query_parts = [

            str(topic),

            str(question_type),

            str(bloom)

        ]

        if previous_questions:

            previous_topics = []

            for question in previous_questions:

                core = self._question_core(
                    question
                )

                if core:

                    previous_topics.append(
                        core
                    )

            if previous_topics:

                query_parts.append(
                    "different from: "
                    + " | ".join(
                        previous_topics[-5:]
                    )
                )

        return " ".join(
            query_parts
        )

    # ============================================================
    # RETRY RETRIEVAL QUERY
    # ============================================================

    def _build_retry_retrieval_query(
        self,
        topic,
        question_type,
        bloom,
        previous_questions,
        attempt
    ):

        focus_terms = [

            "facts details",

            "causes effects",

            "events people institutions",

            "changes trends statistics",

            "challenges impacts",

            "relationships examples"

        ]

        focus = focus_terms[
            (attempt - 2)
            % len(focus_terms)
        ]

        query = (

            f"{topic} "
            f"{question_type} "
            f"{bloom} "
            f"{focus}"

        )

        if previous_questions:

            previous_topics = []

            for question in previous_questions:

                core = self._question_core(
                    question
                )

                if core:

                    previous_topics.append(
                        core
                    )

            if previous_topics:

                query += (
                    " NOT previously tested: "
                    + " | ".join(
                        previous_topics[-5:]
                    )
                )

        return query

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def _retrieve(
        self,
        topic
    ):

        try:

            results = self.retriever.retrieve_context(
                topic,
                k=8
            )

            if results:
                return results

        except Exception:
            pass

        try:

            docs = self.retriever.retrieve(
                topic
            )

            return [
                {
                    "text": doc.page_content,
                    "metadata": getattr(
                        doc,
                        "metadata",
                        {}
                    )
                }
                for doc in docs
            ]

        except Exception:

            return []

    # ============================================================
    # GRAPH
    # ============================================================

    def _get_graph_context(
        self,
        topic
    ):

        if self.graph_traversal is None:
            return ""

        try:

            result = self.graph_traversal.retrieve(
                topic
            )

            if isinstance(
                result,
                str
            ):

                return result

            return str(
                result
            )

        except Exception:

            return ""

    # ============================================================
    # LLM
    # ============================================================

    def _invoke_llm(
        self,
        prompt
    ):

        result = self.llm.invoke(
            prompt
        )

        if hasattr(
            result,
            "content"
        ):

            return result.content

        return str(
            result
        )

    # ============================================================
    # QUESTION TYPE VALIDATION
    # ============================================================

    def _question_type_is_valid(
        self,
        question,
        question_type
    ):
        """
        Ensure the generated question actually matches
        the requested section type.

        This check happens BEFORE duplicate detection.
        """

        if not question:
            return False

        qtype = str(
            question_type
        ).lower().strip()

        text = str(
            question
        )

        # --------------------------------------------------------
        # MCQ
        # --------------------------------------------------------

        if qtype == "mcq":

            options = re.findall(
                r"^[ABCD]\.\s+.+$",
                text,
                flags=re.MULTILINE
            )

            answer = re.search(
                r"^Answer:\s*[ABCD]\b",
                text,
                flags=re.MULTILINE
                | re.IGNORECASE
            )

            return (
                len(options) == 4
                and answer is not None
            )

        # --------------------------------------------------------
        # Fill in the Blanks
        # --------------------------------------------------------

        if qtype in [
            "fill in the blanks",
            "fill in the blank",
            "fill blanks"
        ]:

            if re.search(
                r"^[ABCD]\.\s+",
                text,
                flags=re.MULTILINE
            ):

                return False

            blank_count = len(
                re.findall(
                    r"_{3,}",
                    text
                )
            )

            return (
                blank_count == 1
                and re.search(
                    r"^Answer:\s*.+",
                    text,
                    flags=re.MULTILINE
                    | re.IGNORECASE
                )
                is not None
            )

        # --------------------------------------------------------
        # Match
        # --------------------------------------------------------

        if qtype in [
            "match",
            "match the following"
        ]:

            return (
                "Column A:" in text
                and "Column B:" in text
                and re.search(
                    r"^Answer:",
                    text,
                    flags=re.MULTILINE
                    | re.IGNORECASE
                )
                is not None
            )

        # --------------------------------------------------------
        # Short / Long Answer
        # --------------------------------------------------------

        if qtype in [
            "short answer",
            "short-answer",
            "long answer",
            "long-answer"
        ]:

            if re.search(
                r"^[ABCD]\.\s+",
                text,
                flags=re.MULTILINE
            ):

                return False

            if re.search(
                r"^Answer:\s*.+",
                text,
                flags=re.MULTILINE
                | re.IGNORECASE
            ) is None:

                return False

            return True

        return True

    # ============================================================
    # DUPLICATE QUESTION CHECK
    # ============================================================

    def _is_duplicate_question(
        self,
        question,
        previous_questions
    ):

        if not previous_questions:
            return False

        current = self._question_core(
            question
        )

        if not current:
            return False

        for previous in previous_questions:

            previous_core = self._question_core(
                previous
            )

            if not previous_core:
                continue

            if current == previous_core:

                return True

            current_tokens = set(
                current.split()
            )

            previous_tokens = set(
                previous_core.split()
            )

            if (
                not current_tokens
                or not previous_tokens
            ):

                continue

            intersection = (
                current_tokens
                & previous_tokens
            )

            union = (
                current_tokens
                | previous_tokens
            )

            similarity = (
                len(intersection)
                / len(union)
            )

            if similarity >= 0.97:

                return True

        return False

    # ============================================================
    # EXTRACT QUESTION CORE
    # ============================================================

    def _question_core(
        self,
        question
    ):

        if not question:
            return ""

        text = str(
            question
        ).strip().lower()

        text = re.sub(
            r"^q\s*\d+\s*[\.:)]?\s*",
            "",
            text
        )

        text = re.sub(
            r"^\[[^\]]+\]\s*",
            "",
            text
        )

        text = re.split(
            r"\n\s*a\.\s*",
            text,
            maxsplit=1
        )[0]

        text = re.split(
            r"\n\s*answer\s*:",
            text,
            maxsplit=1
        )[0]

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_question(
        self,
        response,
        question_type,
        question_number,
        expected_bloom,
        source_context=""
    ):

        if not response:
            return None

        text = response.strip()

        text = re.sub(
            r"```.*?\n",
            "",
            text
        )

        text = text.replace(
            "```",
            ""
        ).strip()

        text = re.sub(
            rf"Q\s*{question_number}\s*\[\s*Bloom\s*\]",
            f"Q{question_number}. [{expected_bloom}]",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            rf"^Q\s*{question_number}\s*[:.)-]?",
            f"Q{question_number}.",
            text,
            flags=re.IGNORECASE
        )

        qtype = question_type.lower().strip()

        if qtype == "mcq":

            return self._normalize_mcq(
                text,
                question_number,
                expected_bloom
            )

        if qtype in [
            "fill in the blanks",
            "fill in the blank",
            "fill blanks"
        ]:

            return self._normalize_fill(
                text,
                question_number,
                expected_bloom
            )

        if qtype in [
            "match the following",
            "match"
        ]:

            return self._normalize_match(
                text,
                question_number,
                expected_bloom
            )

        if qtype in [
            "short answer",
            "short-answer"
        ]:

            return self._normalize_text(
                text,
                question_number,
                expected_bloom
            )

        if qtype in [
            "long answer",
            "long-answer"
        ]:

            return self._normalize_text(
                text,
                question_number,
                expected_bloom
            )

        return self._normalize_text(
            text,
            question_number,
            expected_bloom
        )

    # ============================================================
    # MCQ NORMALIZER
    # ============================================================

    def _normalize_mcq(
        self,
        text,
        question_number,
        bloom
    ):

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        question_lines = []
        options = {}
        answer = None

        for line in lines:

            if re.match(
                rf"^Q\s*{question_number}\s*[\.:)]?",
                line,
                re.IGNORECASE
            ):

                cleaned = re.sub(
                    rf"^Q\s*{question_number}\s*[\.:)]?\s*",
                    "",
                    line,
                    flags=re.IGNORECASE
                )

                cleaned = re.sub(
                    r"^\[[^\]]+\]\s*",
                    "",
                    cleaned
                )

                if cleaned:

                    question_lines.append(
                        cleaned
                    )

                continue

            option_match = re.match(
                r"^([ABCD])[\.\):\-]\s*(.+)$",
                line,
                re.IGNORECASE
            )

            if option_match:

                letter = option_match.group(
                    1
                ).upper()

                value = option_match.group(
                    2
                ).strip()

                options[letter] = value

                continue

            answer_match = re.match(
                r"^(?:Answer|Correct\s*Answer)\s*[:\-]\s*([ABCD])\b",
                line,
                re.IGNORECASE
            )

            if answer_match:

                answer = answer_match.group(
                    1
                ).upper()

                continue

            if line.lower() not in [
                "mcq",
                "multiple choice question"
            ]:

                question_lines.append(
                    line
                )

        if set(options.keys()) != {
            "A",
            "B",
            "C",
            "D"
        }:

            return None

        question = " ".join(
            question_lines
        ).strip()

        if not question:
            return None

        if answer not in {
            "A",
            "B",
            "C",
            "D"
        }:

            return None

        return (
            f"Q{question_number}. [{bloom}] "
            f"{question}\n\n"
            f"A. {options['A']}\n"
            f"B. {options['B']}\n"
            f"C. {options['C']}\n"
            f"D. {options['D']}\n\n"
            f"Answer: {answer}"
        )

    # ============================================================
    # MCQ QUALITY CHECK
    # ============================================================

    def _mcq_is_valid(
        self,
        text,
        source
    ):

        answer_match = re.search(
            r"Answer:\s*([ABCD])",
            text,
            re.IGNORECASE
        )

        if not answer_match:
            return False

        correct = answer_match.group(
            1
        ).upper()

        options = {}

        for letter, value in re.findall(
            r"^([ABCD])\.\s*(.+)$",
            text,
            flags=re.MULTILINE
        ):

            options[letter] = value.strip()

        if set(options) != {
            "A",
            "B",
            "C",
            "D"
        }:

            return False

        if not options.get(correct):
            return False

        normalized = [

            re.sub(
                r"\W+",
                " ",
                value.lower()
            ).strip()

            for value in options.values()

        ]

        if len(
            set(normalized)
        ) != 4:

            return False

        return True

    # ============================================================
    # FILL BLANK
    # ============================================================

    def _normalize_fill(
        self,
        text,
        question_number,
        bloom
    ):

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        question = ""
        answer = None

        for line in lines:

            if re.match(
                rf"^Q\s*{question_number}",
                line,
                re.IGNORECASE
            ):

                question = re.sub(
                    rf"^Q\s*{question_number}\s*[\.:)]?\s*",
                    "",
                    line,
                    flags=re.IGNORECASE
                )

                question = re.sub(
                    r"^\[[^\]]+\]\s*",
                    "",
                    question
                )

            elif re.match(
                r"^(?:Answer|Correct\s*Answer)\s*:",
                line,
                re.IGNORECASE
            ):

                answer = re.sub(
                    r"^(?:Answer|Correct\s*Answer)\s*:\s*",
                    "",
                    line,
                    flags=re.IGNORECASE
                ).strip()

        if not question:
            return None

        # --------------------------------------------------------
        # NORMALIZE BLANK FORMAT
        #
        # Accept:
        # ___
        # ____
        # _____
        # ______
        #
        # and convert all of them to:
        # _____
        # --------------------------------------------------------

        question = re.sub(
            r"_{3,}",
            "_____",
            question
        )

        # --------------------------------------------------------
        # EXACTLY ONE BLANK
        # --------------------------------------------------------

        blank_count = question.count(
            "_____"
        )

        if blank_count != 1:
            return None

        # --------------------------------------------------------
        # ANSWER IS REQUIRED
        # --------------------------------------------------------

        if not answer:
            return None

        # --------------------------------------------------------
        # DO NOT ALLOW MCQ OPTIONS
        # --------------------------------------------------------

        if re.search(
            r"^[ABCD]\.\s+",
            text,
            flags=re.MULTILINE
        ):

            return None

        return (
            f"Q{question_number}. [{bloom}] "
            f"{question}\n\n"
            f"Answer: {answer}"
        )

    # ============================================================
    # MATCH
    # ============================================================

    def _normalize_match(
        self,
        text,
        question_number,
        bloom
    ):

        column_a = []
        column_b = []
        answer = []

        mode = None

        for raw in text.splitlines():

            line = raw.strip()

            if not line:
                continue

            if re.match(
                r"^Q\s*" + str(question_number),
                line,
                re.I
            ):

                continue

            if re.match(
                r"^Column\s*A\s*:",
                line,
                re.I
            ):

                mode = "A"
                continue

            if re.match(
                r"^Column\s*B\s*:",
                line,
                re.I
            ):

                mode = "B"
                continue

            if re.match(
                r"^Answer\s*:",
                line,
                re.I
            ):

                mode = "ANSWER"

                value = re.sub(
                    r"^Answer\s*:\s*",
                    "",
                    line,
                    flags=re.I
                )

                answer.extend(
                    re.findall(
                        r"([1-5])\s*[-:]\s*([A-E])",
                        value.upper()
                    )
                )

                continue

            if mode == "A":

                match = re.match(
                    r"^([1-5])[\.\)]\s*(.+)$",
                    line
                )

                if match:

                    column_a.append(
                        (
                            match.group(1),
                            match.group(2).strip()
                        )
                    )

            elif mode == "B":

                match = re.match(
                    r"^([A-E])[\.\)]\s*(.+)$",
                    line
                )

                if match:

                    column_b.append(
                        (
                            match.group(1),
                            match.group(2).strip()
                        )
                    )

        if len(column_a) != 5:
            return None

        if len(column_b) != 5:
            return None

        if len(answer) != 5:
            return None

        answer_text = ", ".join(
            f"{a}-{b}"
            for a, b in answer
        )

        return (
            f"Q{question_number}. [{bloom}] "
            f"Match the following:\n\n"
            f"Column A:\n"
            f"1. {column_a[0][1]}\n"
            f"2. {column_a[1][1]}\n"
            f"3. {column_a[2][1]}\n"
            f"4. {column_a[3][1]}\n"
            f"5. {column_a[4][1]}\n\n"
            f"Column B:\n"
            f"A. {column_b[0][1]}\n"
            f"B. {column_b[1][1]}\n"
            f"C. {column_b[2][1]}\n"
            f"D. {column_b[3][1]}\n"
            f"E. {column_b[4][1]}\n\n"
            f"Answer: {answer_text}"
        )

    # ============================================================
    # SHORT/LONG ANSWER
    # ============================================================

    def _normalize_text(
        self,
        text,
        question_number,
        bloom
    ):

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        question = ""
        answer = None

        for line in lines:

            if re.match(
                rf"^Q\s*{question_number}",
                line,
                re.IGNORECASE
            ):

                question = re.sub(
                    rf"^Q\s*{question_number}\s*[\.:)]?\s*",
                    "",
                    line,
                    flags=re.IGNORECASE
                )

                question = re.sub(
                    r"^\[[^\]]+\]\s*",
                    "",
                    question
                )

            elif re.match(
                r"^Answer\s*:",
                line,
                re.IGNORECASE
            ):

                answer = re.sub(
                    r"^Answer\s*:\s*",
                    "",
                    line,
                    flags=re.IGNORECASE
                ).strip()

        if not question:
            return None

        if not answer:
            return None

        return (
            f"Q{question_number}. [{bloom}] "
            f"{question}\n\n"
            f"Answer: {answer}"
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _get(
        obj,
        key,
        default=None
    ):

        if hasattr(
            obj,
            key
        ):

            return getattr(
                obj,
                key
            )

        if isinstance(
            obj,
            dict
        ):

            return obj.get(
                key,
                default
            )

        return default

    def _get_sections(
        self,
        paper_pattern,
        blooms
    ):

        if paper_pattern is None:

            return [
                {
                    "name": "A",
                    "type": "MCQ",
                    "count": 5,
                    "blooms": blooms
                },
                {
                    "name": "B",
                    "type": "Fill in the Blanks",
                    "count": 5,
                    "blooms": blooms
                },
                {
                    "name": "C",
                    "type": "Match the Following",
                    "count": 5,
                    "blooms": blooms
                },
                {
                    "name": "D",
                    "type": "Short Answer",
                    "count": 5,
                    "blooms": blooms
                },
                {
                    "name": "E",
                    "type": "Long Answer",
                    "count": 5,
                    "blooms": blooms
                }
            ]

        sections = self._get(
            paper_pattern,
            "sections",
            []
        )

        return sections

    @staticmethod
    def _source_text(
        chunks
    ):

        parts = []

        for item in chunks:

            if isinstance(
                item,
                dict
            ):

                text = item.get(
                    "text",
                    ""
                )

            else:

                text = str(
                    item
                )

            if text:

                parts.append(
                    text
                )

        return "\n\n".join(
            parts
        )