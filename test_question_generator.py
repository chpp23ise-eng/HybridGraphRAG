"""
test_question_generator.py

Tests the complete Hybrid GraphRAG
Question Generator.
"""

from src.llm import load_llm
from src.question_generator import QuestionGenerator


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

print("\nLoading Mistral...\n")

llm = load_llm()


# --------------------------------------------------
# Initialize Generator
# --------------------------------------------------

generator = QuestionGenerator(llm)


# --------------------------------------------------
# User Input
# --------------------------------------------------

topic = input("Enter Topic: ").strip()


# --------------------------------------------------
# Bloom's Levels
# --------------------------------------------------

blooms = [

    "Remember",

    "Understand",

    "Apply",

    "Analyze",

    "Evaluate",

    "Create"

]


# --------------------------------------------------
# Generate Question Paper
# --------------------------------------------------

paper = generator.generate(

    topic=topic,

    blooms=blooms

)


# --------------------------------------------------
# Display Result
# --------------------------------------------------

print()

print("=" * 80)

print("QUESTION PAPER")

print("=" * 80)

print()

print(paper)