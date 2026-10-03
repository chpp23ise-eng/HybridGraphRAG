class PromptBuilder:

    @staticmethod
    def build_prompt(
        topic,
        blooms,
        configuration=None,
        graph_context="",
        retrieved_context="",
        retrieved_chunks=None,
        paper_pattern=None,
        question_start=1,
        question_number=None,
        previous_questions=None,
        **kwargs
    ):
        if retrieved_chunks is None:
            retrieved_chunks = []

        if previous_questions is None:
            previous_questions = []

        # ---------------------------------------------------------
        # CONVERT RETRIEVED CHUNKS TO PLAIN TEXT
        # ---------------------------------------------------------
        source_parts = []

        if retrieved_chunks:
            for item in retrieved_chunks:

                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)

                if text.strip():
                    source_parts.append(text.strip())

        if retrieved_context:
            source_parts.append(
                str(retrieved_context).strip()
            )

        source = "\n\n".join(source_parts)

        if not source:
            source = "No source text available."

        # ---------------------------------------------------------
        # KNOWLEDGE GRAPH CONTEXT
        # ---------------------------------------------------------
        if graph_context:
            graph_text = str(graph_context).strip()
        else:
            graph_text = "No knowledge graph context available."

        # ---------------------------------------------------------
        # PREVIOUS QUESTIONS
        #
        # These are used to prevent the LLM from repeatedly asking
        # the same question or asking the same fact in different
        # wording.
        # ---------------------------------------------------------
        if previous_questions:

            previous_text = "\n\n".join(
                f"{index + 1}. {question}"
                for index, question in enumerate(previous_questions)
            )

        else:
            previous_text = "No previous questions have been generated."

        # ---------------------------------------------------------
        # NORMALIZE PAPER PATTERN
        # ---------------------------------------------------------
        if paper_pattern is None:

            paper_pattern = {
                "sections": [
                    {
                        "name": "A",
                        "type": "MCQ",
                        "count": 5,
                        "marks_each": 1,
                        "blooms": blooms
                    },
                    {
                        "name": "B",
                        "type": "Fill in the Blanks",
                        "count": 5,
                        "marks_each": 1,
                        "blooms": blooms
                    },
                    {
                        "name": "C",
                        "type": "Match the Following",
                        "count": 5,
                        "marks_each": 1,
                        "blooms": blooms
                    },
                    {
                        "name": "D",
                        "type": "Short Answer",
                        "count": 5,
                        "marks_each": 1,
                        "blooms": blooms
                    },
                    {
                        "name": "E",
                        "type": "Long Answer",
                        "count": 5,
                        "marks_each": 1,
                        "blooms": blooms
                    }
                ]
            }

        if hasattr(paper_pattern, "sections"):

            sections = paper_pattern.sections

        elif isinstance(paper_pattern, dict):

            sections = paper_pattern.get(
                "sections",
                []
            )

        else:

            sections = []

        # ---------------------------------------------------------
        # FIND SECTION AND POSITION OF QUESTION
        # ---------------------------------------------------------
        qnum = (
            question_number
            if question_number is not None
            else question_start
        )

        current_number = 1
        selected_section = None
        position = 0

        for section in sections:

            if hasattr(section, "count"):
                count = section.count
            else:
                count = section.get("count", 0)

            if (
                current_number
                <= qnum
                <= current_number + count - 1
            ):

                selected_section = section
                position = qnum - current_number
                break

            current_number += count

        if selected_section is None:

            selected_section = {
                "name": "A",
                "type": "MCQ",
                "count": 1,
                "blooms": blooms
            }

            position = 0

        if hasattr(selected_section, "type"):

            question_type = selected_section.type
            section_name = selected_section.name
            section_blooms = selected_section.blooms

        else:

            question_type = selected_section.get(
                "type",
                "MCQ"
            )

            section_name = selected_section.get(
                "name",
                "A"
            )

            section_blooms = selected_section.get(
                "blooms",
                blooms
            )

        if not section_blooms:
            section_blooms = blooms

        bloom = section_blooms[
            position % len(section_blooms)
        ]

        # ---------------------------------------------------------
        # COMMON RULES
        # ---------------------------------------------------------
        common_rules = f"""
You are generating ONLY question Q{qnum} for an educational question paper.

Topic:
{topic}

Section:
{section_name}

Question type:
{question_type}

Bloom level:
{bloom}

SOURCE MATERIAL:
----------------
{source}
----------------

KNOWLEDGE GRAPH CONTEXT:
----------------
{graph_text}
----------------

PREVIOUSLY GENERATED QUESTIONS:
----------------
{previous_text}
----------------

STRICT RULES:

1. Use ONLY information explicitly supported by the SOURCE MATERIAL
   and KNOWLEDGE GRAPH CONTEXT.

2. Do NOT introduce outside facts.

3. Do NOT invent dates, names, numbers, causes, effects, examples,
   relationships, or terminology.

4. The question must be answerable from the provided source material.

5. The Bloom level must genuinely match the question.

6. Generate EXACTLY ONE question.

7. Do not generate another question.

8. Do not add explanations.

9. Do not add introductory text.

10. Follow the output format EXACTLY.

11. The question MUST NOT duplicate any previously generated question.

12. Do NOT simply rephrase a previously generated question.

13. Do NOT ask about the same specific fact, statistic, relationship,
    event, definition, or statement used by a previous question
    unless the question requires a genuinely different concept or
    reasoning task.

14. Prefer a different fact, concept, relationship, event, cause,
    effect, comparison, application, or implication from the
    available source material.

15. Use the full range of relevant information available in the
    SOURCE MATERIAL rather than repeatedly selecting the most
    obvious or frequently mentioned fact.

16. If multiple source-supported facts are available, choose one
    that has NOT already been tested by the previous questions.

17. The answer must be directly supported by the source material.

18. Never sacrifice factual correctness merely to make the question
    different.
"""

        # ---------------------------------------------------------
        # MCQ
        # ---------------------------------------------------------
        if question_type.lower() == "mcq":

            return common_rules + f"""

MCQ REQUIREMENTS:

- Generate exactly FOUR options.
- Options MUST be labelled A, B, C and D.
- Exactly ONE option must be correct.
- The other three options must be plausible but clearly incorrect
  according to the SOURCE MATERIAL.
- NEVER create a question where two or more options could be correct.
- NEVER use a broad category as one option if another option is
  a subcategory of the same correct fact.
- The correct answer MUST be explicitly supported by the source.
- ALWAYS provide the answer.
- Avoid repeating the subject or fact tested by previous questions.

OUTPUT EXACTLY:

Q{qnum}. [{bloom}] <question>

A. <option>
B. <option>
C. <option>
D. <option>

Answer: <A/B/C/D>
"""

        # ---------------------------------------------------------
        # FILL IN THE BLANK
        # ---------------------------------------------------------
        if question_type.lower() in [
            "fill in the blanks",
            "fill in the blank",
            "fill blanks"
        ]:

            return common_rules + f"""

FILL-IN-THE-BLANK REQUIREMENTS:

- Generate exactly ONE sentence.
- Replace exactly ONE important source-supported term with _____.
- The answer must be explicitly present in the source.
- Do not create multiple blanks.
- Do not repeat a fact already tested by a previous question.

OUTPUT EXACTLY:

Q{qnum}. [{bloom}] <sentence containing _____>

Answer: <answer>
"""

        # ---------------------------------------------------------
        # MATCH THE FOLLOWING
        # ---------------------------------------------------------
        if question_type.lower() in [
            "match the following",
            "match"
        ]:

            return common_rules + f"""

MATCH-THE-FOLLOWING REQUIREMENTS:

Create exactly ONE matching question containing FIVE pairs.

Column A must contain:
1.
2.
3.
4.
5.

Column B must contain:
A.
B.
C.
D.
E.

Each item in Column A must have exactly one correct match
in Column B.

All matches must be directly supported by the SOURCE MATERIAL.

Prefer relationships, entities, events, concepts, or facts that
have not already been tested by previous questions.

OUTPUT EXACTLY:

Q{qnum}. [{bloom}] Match the following:

Column A:
1. <item>
2. <item>
3. <item>
4. <item>
5. <item>

Column B:
A. <item>
B. <item>
C. <item>
D. <item>
E. <item>

Answer:
1-A, 2-B, 3-C, 4-D, 5-E
"""

        # ---------------------------------------------------------
        # SHORT ANSWER
        # ---------------------------------------------------------
        if question_type.lower() in [
            "short answer",
            "short-answer"
        ]:

            return common_rules + f"""

SHORT-ANSWER REQUIREMENTS:

- Ask exactly ONE focused question.
- The answer should be reasonably short.
- The answer must be directly derivable from the source.
- Do not require outside knowledge.
- Prefer a fact, relationship, explanation, comparison, or concept
  that has not already been tested.

OUTPUT EXACTLY:

Q{qnum}. [{bloom}] <question>

Answer: <answer>
"""

        # ---------------------------------------------------------
        # LONG ANSWER
        # ---------------------------------------------------------
        if question_type.lower() in [
            "long answer",
            "long-answer"
        ]:

            return common_rules + f"""

LONG-ANSWER REQUIREMENTS:

- Ask exactly ONE comprehensive question.
- The answer must be constructible entirely from the source.
- Do not require information outside the source.
- Prefer synthesis of source-supported information that has not
  already been directly tested.

OUTPUT EXACTLY:

Q{qnum}. [{bloom}] <question>

Answer: <answer>
"""

        # ---------------------------------------------------------
        # DEFAULT
        # ---------------------------------------------------------
        return common_rules + f"""

OUTPUT EXACTLY:

Q{qnum}. [{bloom}] <question>

Answer: <answer>
"""