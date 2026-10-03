"""
app.py

Main entry point for the Hybrid GraphRAG
Question Paper Generator.

This file launches the Streamlit UI.
"""

import os
import subprocess
import sys


def main():

    # --------------------------------------------------
    # Locate UI application
    # --------------------------------------------------

    project_root = os.path.dirname(
        os.path.abspath(__file__)
    )

    ui_app = os.path.join(
        project_root,
        "ui",
        "app.py"
    )

    # --------------------------------------------------
    # Check UI file
    # --------------------------------------------------

    if not os.path.exists(ui_app):

        print(
            "Error: ui/app.py not found."
        )

        sys.exit(1)

    # --------------------------------------------------
    # Start Streamlit
    # --------------------------------------------------

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            ui_app
        ],
        cwd=project_root
    )


if __name__ == "__main__":

    main()