"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    create_embeddings.py

Author:
    Emir Yalçınkaya

Description:
    Loads the generated text chunks and converts each chunk
    into a dense vector embedding using a Sentence Transformer
    model.

    The generated embeddings are saved as a NumPy array.
    Metadata is also saved separately for later retrieval
    using FAISS.

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

# Used for file paths and existence checks.
import os

# Used to read and write JSON files.
import json

# NumPy is used to store embeddings efficiently.
import numpy as np

# Sentence Transformer model used to generate embeddings.
from sentence_transformers import SentenceTransformer


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DATA_DIRECTORY = "data"

CHUNKS_FILE_NAME = "chunks.json"
EMBEDDINGS_FILE_NAME = "embeddings.npy"
METADATA_FILE_NAME = "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNKS_PATH = os.path.join(DATA_DIRECTORY, CHUNKS_FILE_NAME)
EMBEDDINGS_PATH = os.path.join(DATA_DIRECTORY, EMBEDDINGS_FILE_NAME)
METADATA_PATH = os.path.join(DATA_DIRECTORY, METADATA_FILE_NAME)


def load_chunks():
    """
    Loads all generated text chunks.

    Returns
    -------
    list
        List containing all chunk dictionaries.
    """

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Could not find the chunks file:\n{CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def load_embedding_model():
    """
    Loads the Sentence Transformer embedding model.

    Returns
    -------
    SentenceTransformer
        Loaded embedding model.
    """

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    return model


def generate_embeddings(model, chunks):
    """
    Generates vector embeddings for every text chunk.

    Parameters
    ----------
    model : SentenceTransformer
        Loaded embedding model.

    chunks : list
        List of chunk dictionaries.

    Returns
    -------
    numpy.ndarray
        Embedding vectors.
    """

    print("Generating embeddings...")

    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


def save_embeddings(embeddings):
    """
    Saves all embedding vectors.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Generated embedding vectors.
    """

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Embeddings saved to:\n{EMBEDDINGS_PATH}")


def save_metadata(chunks):
    """
    Saves chunk metadata separately.

    Only useful information required during retrieval
    is stored.

    Parameters
    ----------
    chunks : list
        Original chunk list.
    """

    metadata = []

    for chunk in chunks:

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "source_page": chunk["source_page"],
            "section_title": chunk["section_title"],
            "text": chunk["text"],
            "word_count": len(chunk["text"].split())
        })

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Metadata saved to:\n{METADATA_PATH}")


def main():
    """
    Executes the complete embedding generation process.
    """

    print("Starting embedding generation...\n")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} text chunks.\n")

    model = load_embedding_model()

    embeddings = generate_embeddings(model, chunks)

    save_embeddings(embeddings)

    save_metadata(chunks)

    print("\nEmbedding generation completed successfully.")

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


# ----------------------------------------------------------
# Program Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":
    main()