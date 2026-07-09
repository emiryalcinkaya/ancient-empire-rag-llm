"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    main.py

Author:
    Emir Yalçınkaya

Description:
    Main entry point of the application.

    Receives a user's question, retrieves the most
    relevant document context using RAG, sends the
    generated prompt to Gemini, and displays the
    final answer.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

from scripts.rag import retrieve
from scripts.llm import generate_answer


def main():
    """
    Runs the complete RAG question answering system.
    """

    print("=" * 60)
    print("Mesopotamia RAG Assistant")
    print("=" * 60)

    while True:

        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        print("\nSearching relevant information...")

        prompt = retrieve(question)

        print("Generating answer...\n")

        answer = generate_answer(prompt)

        print(answer)


if __name__ == "__main__":
    main()