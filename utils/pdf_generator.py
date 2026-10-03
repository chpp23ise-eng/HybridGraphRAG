"""
pdf_generator.py

Creates PDF from generated
question paper.
"""

import os

from reportlab.lib.pagesizes import A4

from reportlab.platypus import SimpleDocTemplate

from reportlab.platypus import Paragraph

from reportlab.lib.styles import getSampleStyleSheet


class PDFGenerator:

    @staticmethod
    def generate(question_paper):

        os.makedirs("exports", exist_ok=True)

        pdf_path = "exports/question_paper.pdf"

        doc = SimpleDocTemplate(

            pdf_path,

            pagesize=A4

        )

        styles = getSampleStyleSheet()

        story = []

        for line in question_paper.split("\n"):

            story.append(

                Paragraph(

                    line,

                    styles["BodyText"]

                )

            )

        doc.build(story)

        return pdf_path