"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    rag.py

Author:
    Emir Yalçınkaya

Description:
    Loads the FAISS index and metadata, retrieves the most
    relevant text chunks for a user question, and builds
    the context that will later be sent to an LLM.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DATA_DIRECTORY = "data"

FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = os.path.join(DATA_DIRECTORY, FAISS_INDEX_FILE)
METADATA_PATH = os.path.join(DATA_DIRECTORY, METADATA_FILE)


# ----------------------------------------------------------
# Load resources
# ----------------------------------------------------------

def load_faiss_index():
    """
    Loads the saved FAISS index.

    Returns
    -------
    faiss.Index
    """

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found:\n{INDEX_PATH}"
        )

    print("Loading FAISS index...")

    return faiss.read_index(INDEX_PATH)


def load_metadata():
    """
    Loads metadata corresponding to every chunk.

    Returns
    -------
    list
    """

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    print("Loading metadata...")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def load_embedding_model():
    """
    Loads the embedding model.

    Returns
    -------
    SentenceTransformer
    """

    print("Loading embedding model...")

    return SentenceTransformer(MODEL_NAME)


# ----------------------------------------------------------
# Retrieval
# ----------------------------------------------------------

def search_similar_chunks(
    question,
    model,
    index,
    metadata,
    top_k=3
):
    """
    Retrieves the most similar chunks.

    Parameters
    ----------
    question : str

    model : SentenceTransformer

    index : faiss.Index

    metadata : list

    top_k : int

    Returns
    -------
    list
    """

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = index.search(
        question_embedding.astype(np.float32),
        top_k
    )

    results = []

    for i in indices[0]:

        if i >= 0:

            results.append(metadata[i])

    return results


# ----------------------------------------------------------
# Prompt Builder
# ----------------------------------------------------------

def build_context(retrieved_chunks):
    """
    Combines retrieved chunks into one context.

    Parameters
    ----------
    retrieved_chunks : list

    Returns
    -------
    str
    """

    context = ""

    for chunk in retrieved_chunks:

        context += (
            f"\nSection: {chunk['section_title']}\n"
            f"Page: {chunk['source_page']}\n"
            f"{chunk['text']}\n"
        )

    return context.strip()


def build_prompt(question, context):
    """
    Creates the final prompt for the LLM.

    Parameters
    ----------
    question : str

    context : str

    Returns
    -------
    str
    """

    prompt = f"""
You are an AI assistant answering questions ONLY using the provided context.

If the answer is not found in the context,
respond with:

"I could not find that information in the provided document."

Context
-------
{context}

Question
--------
{question}

Answer:
"""

    return prompt.strip()


# ----------------------------------------------------------
# Main Retrieval Pipeline
# ----------------------------------------------------------

def retrieve(question, top_k=3):
    """
    Complete retrieval pipeline.

    Parameters
    ----------
    question : str

    top_k : int

    Returns
    -------
    str
    """

    index = load_faiss_index()

    metadata = load_metadata()

    model = load_embedding_model()

    retrieved_chunks = search_similar_chunks(
        question=question,
        model=model,
        index=index,
        metadata=metadata,
        top_k=top_k
    )

    context = build_context(retrieved_chunks)

    prompt = build_prompt(question, context)

    return prompt