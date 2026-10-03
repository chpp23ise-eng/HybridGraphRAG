"""
llm.py

Loads the local mistral  model using Ollama.
"""

from langchain_ollama import ChatOllama


def load_llm(
        model_name="mistral",
        temperature=0.0
):
    """
    Load Phi-3 from Ollama.

    Parameters
    ----------
    model_name : str
        Ollama model name.

    temperature : float
        LLM temperature.

    Returns
    -------
    ChatOllama
    """

    llm = ChatOllama(
        model=model_name,
        temperature=temperature
    )

    return llm