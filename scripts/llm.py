"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    llm.py

Author:
    Emir Yalçınkaya

Description:
    Sends the generated prompt to a local Llama model
    running with Ollama and returns the generated response.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

import ollama


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_NAME = "llama3"


# ----------------------------------------------------------
# Generate Answer
# ----------------------------------------------------------

def generate_answer(prompt):
    """
    Sends the prompt to the local Llama model.

    Parameters
    ----------
    prompt : str

    Returns
    -------
    str
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]