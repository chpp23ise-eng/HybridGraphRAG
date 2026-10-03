"""
blooms_distribution.py

Defines the question paper pattern
using Bloom's Taxonomy.
"""

QUESTION_PATTERN = {

    "legend": [

        ("L1", "Remember"),
        ("L2", "Understand"),
        ("L3", "Apply"),
        ("L4", "Analyze"),
        ("L5", "Evaluate"),
        ("L6", "Create")

    ],

    "section_a": [

        ("Q1", "L1", "Remember"),
        ("Q2", "L2", "Understand"),
        ("Q3", "L3", "Apply"),
        ("Q4", "L4", "Analyze"),
        ("Q5", "L5", "Evaluate")

    ],

    "section_b": [

        ("Q6", "L2", "Understand"),
        ("Q7", "L4", "Analyze"),
        ("Q8", "L5", "Evaluate")

    ],

    "section_c": [

        ("Q9", "L5", "Evaluate"),
        ("Q10", "L6", "Create")

    ]

}