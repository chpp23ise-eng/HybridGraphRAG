from src.validator import QuestionPaperValidator


validator = QuestionPaperValidator()

source = """
The adoption of Green Revolution technology led to an
improvement in the situation, and per capita food production
reached the level of 365 kg in the early 1970s.

Growth of the agriculture sector has led to the unsustainable
use of natural resources like land, water and biodiversity,
spread of insects and pests, indiscriminate use of agro
chemicals and adverse impact on ecology and environment.
"""

source_context = """
The Green Revolution introduced high-yielding varieties of crops,
especially wheat and rice. It increased agricultural production
through improved seeds, irrigation, fertilizers, and pesticides.
The Green Revolution had a significant impact on agriculture
and food production in developing countries, including India.
Per capita food production increased during this period.
"""


# ==========================================================
# TEST 1: SUPPORTED
# ==========================================================

supported = """
Q1. [L1 - Remember] What happened to per capita food
production after the adoption of Green Revolution technology?

Answer: It reached 365 kg in the early 1970s.
"""

score1 = validator._grounding_score(
    supported,
    source
)

print("TEST 1 - Supported")
print("Score:", round(score1, 2))
print()


# ==========================================================
# TEST 2: UNRELATED
# ==========================================================

unrelated = """
Q1. [L1 - Remember] What is the capital of France?

Answer: Paris
"""

score2 = validator._grounding_score(
    unrelated,
    source
)

print("TEST 2 - Unrelated")
print("Score:", round(score2, 2))
print()


# ==========================================================
# TEST 3: PARTIALLY RELATED
# ==========================================================

partially_supported = """
Q1. [L1 - Remember] What natural resources were affected
by agricultural growth?

Answer: Land and water.
"""

score3 = validator._grounding_score(
    partially_supported,
    source
)

print("TEST 3 - Partially supported")
print("Score:", round(score3, 2))
print()


# ==========================================================
# TEST 4: MCQ
# ==========================================================

mcq = """
Q1. [L1 - Remember] What was the level of per capita food
production in the early 1970s?

A. 296 kg
B. 365 kg
C. 683 kg
D. 500 kg

Answer: B
"""

grounding_text = validator._get_grounding_text(
    mcq,
    "MCQ"
)

score4 = validator._grounding_score(
    grounding_text,
    source
)

print("TEST 4 - MCQ")
print("Grounding text:")
print(grounding_text)
print("Score:", round(score4, 2))
print()

# ==========================================================
# TEST 5: UNSUPPORTED BUT TOPIC-RELATED
# ==========================================================

print("\nTEST 5 - Topic-related but unsupported")

question = """
Q1. [L1 - Remember] Who was the scientist who invented the Green Revolution?

Answer: Norman Borlaug
"""

question_score = validator._question_grounding_score(
    question,
    source_context
)

answer_text = validator._get_answer_grounding_text(
    question,
    "Short Answer"
)

answer_score = validator._answer_grounding_score(
    answer_text,
    source_context
)

print("Question score:", round(question_score, 2))
print("Answer text:", answer_text)
print("Answer score:", round(answer_score, 2))