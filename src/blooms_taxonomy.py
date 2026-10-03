"""
blooms_taxonomy.py

Contains Bloom's Taxonomy descriptions
used while generating the prompt.
"""

BLOOMS = {

    "Remember": {
        "code": "L1",
        "description": """
Generate questions that test the student's ability
to recall facts, definitions, terminology,
people, places and basic concepts.

Use verbs such as:
Define, List, Name, Recall, Identify, State.
"""
    },

    "Understand": {
        "code": "L2",
        "description": """
Generate questions that require explaining ideas,
interpreting concepts or summarizing information.

Use verbs such as:
Explain, Describe, Summarize, Discuss,
Interpret, Classify.
"""
    },

    "Apply": {
        "code": "L3",
        "description": """
Generate questions that require applying
the learned concept to a new situation
or solving a problem.

Use verbs such as:
Apply, Solve, Demonstrate, Illustrate,
Use, Implement.
"""
    },

    "Analyze": {
        "code": "L4",
        "description": """
Generate questions that require breaking
information into parts and identifying
relationships or patterns.

Use verbs such as:
Analyze, Compare, Differentiate,
Categorize, Examine, Investigate.
"""
    },

    "Evaluate": {
        "code": "L5",
        "description": """
Generate questions that require making
judgements based on evidence.

Use verbs such as:
Evaluate, Justify, Critique,
Assess, Defend, Recommend.
"""
    },

    "Create": {
        "code": "L6",
        "description": """
Generate questions that require producing
new ideas, solutions or designs.

Use verbs such as:
Design, Create, Develop,
Construct, Propose, Formulate.
"""
    }

}


def build_bloom_prompt(levels):
    """
    Builds Bloom's Taxonomy instructions
    for the Prompt Builder.
    """

    prompt = ""

    for level in levels:

        if level not in BLOOMS:
            continue

        info = BLOOMS[level]

        prompt += (
            f"{info['code']} - {level}\n"
        )

        prompt += info["description"]

        prompt += "\n"

    return prompt