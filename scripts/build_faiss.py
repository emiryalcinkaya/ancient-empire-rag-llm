"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    build_faiss.py

Author:
    Emir Yalçınkaya

Description:
    Loads the generated embedding vectors and creates
    a FAISS vector index for efficient semantic search.

    The generated FAISS index is stored on disk and
    will later be used by the RAG pipeline to retrieve
    the most relevant text chunks.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

# Used for checking whether files exist.
import os

# NumPy is used to load embedding vectors.
import numpy as np

# FAISS is used for similarity search.
import faiss


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DATA_DIRECTORY = "data"

EMBEDDINGS_FILE_NAME = "embeddings.npy"
FAISS_INDEX_FILE_NAME = "faiss_index.bin"

EMBEDDINGS_PATH = os.path.join(DATA_DIRECTORY, EMBEDDINGS_FILE_NAME)
FAISS_INDEX_PATH = os.path.join(DATA_DIRECTORY, FAISS_INDEX_FILE_NAME)


def load_embeddings():
    """
    Loads the generated embedding vectors.

    Returns
    -------
    numpy.ndarray
        Embedding vectors.
    """

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Could not find the embeddings file:\n{EMBEDDINGS_PATH}"
        )

    embeddings = np.load(EMBEDDINGS_PATH)

    return embeddings


def build_faiss_index(embeddings):
    """
    Creates a FAISS index from the embedding vectors.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Embedding vectors.

    Returns
    -------
    faiss.Index
        FAISS vector index.
    """

    print("Building FAISS index...")

    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings.astype("float32"))

    return index


def save_index(index):
    """
    Saves the FAISS index to disk.

    Parameters
    ----------
    index : faiss.Index
        Generated FAISS index.
    """

    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"FAISS index saved to:\n{FAISS_INDEX_PATH}")


def main():
    """
    Executes the complete FAISS index creation process.
    """

    print("Starting FAISS index creation...\n")

    embeddings = load_embeddings()

    print(f"Loaded {len(embeddings)} embeddings.\n")

    index = build_faiss_index(embeddings)

    save_index(index)

    print("\nFAISS index created successfully.")

    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Indexed vectors: {index.ntotal}")


# ----------------------------------------------------------
# Program Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":
    main()