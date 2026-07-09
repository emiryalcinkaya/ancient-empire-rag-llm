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
    generated prompt to the LLM, and displays the
    final answer.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

from scripts.llm import generate_answer
from scripts.rag import (
    retrieve,
    load_faiss_index,
    load_metadata,
    load_embedding_model
)


def main():
    """
    Runs the complete RAG question answering system.
    """

    print("=" * 60)
    print("Mesopotamia RAG Assistant")
    print("=" * 60)

    print("\nLoading resources...")

    index = load_faiss_index()
    metadata = load_metadata()
    model = load_embedding_model()

    print("System ready!")

    while True:

        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        print("\nSearching relevant information...")

        prompt = retrieve(
            question=question,
            model=model,
            index=index,
            metadata=metadata
        )

        print("Generating answer...\n")

        answer = generate_answer(prompt)

        print(answer)


if __name__ == "__main__":
    main()