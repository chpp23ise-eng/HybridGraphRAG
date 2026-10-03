from src.llm import load_llm
from src.question_generator import QuestionGenerator


print("=" * 80)
print("QUESTION GENERATOR END-TO-END TEST")
print("=" * 80)
print()


# ==========================================================
# 1. LOAD LLM
# ==========================================================

print("Loading Mistral...")
llm = load_llm()

print("Mistral loaded.")
print()


# ==========================================================
# 2. CREATE QUESTION GENERATOR
# ==========================================================

generator = QuestionGenerator(llm)


# ==========================================================
# 3. PAPER PATTERN
# ==========================================================

paper_pattern = {

    "total_marks": 10,

    "duration": "2 Hours",

    "sections": [

        {
            "name": "A",
            "type": "MCQ",
            "count": 2,
            "marks_each": 1,
            "blooms": [
                "L1",
                "L2"
            ]
        },

        {
            "name": "B",
            "type": "Fill in the Blanks",
            "count": 2,
            "marks_each": 1,
            "blooms": [
                "L1",
                "L2"
            ]
        },

        {
            "name": "C",
            "type": "Match the Following",
            "count": 1,
            "marks_each": 2,
            "blooms": [
                "L2"
            ]
        },

        {
            "name": "D",
            "type": "Short Answer",
            "count": 1,
            "marks_each": 2,
            "blooms": [
                "L3"
            ]
        },

        {
            "name": "E",
            "type": "Long Answer",
            "count": 1,
            "marks_each": 2,
            "blooms": [
                "L4"
            ]
        }

    ]
}


# ==========================================================
# 4. GENERATE PAPER
# ==========================================================

print("=" * 80)
print("STARTING QUESTION PAPER GENERATION")
print("=" * 80)
print()

paper = generator.generate(

    topic="Green Revolution",

    blooms=[
        "L1",
        "L2",
        "L3",
        "L4"
    ],

    paper_pattern=paper_pattern
)


# ==========================================================
# 5. DISPLAY RESULT
# ==========================================================

print()
print("=" * 80)
print("FINAL GENERATED PAPER")
print("=" * 80)
print()

print(paper)

print()
print("=" * 80)
print("END OF TEST")
print("=" * 80)