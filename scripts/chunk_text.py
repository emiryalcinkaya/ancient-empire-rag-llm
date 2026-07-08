"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    chunk_text.py

Author:
    Deniz Gözcü

Description:
    Reads the text extracted from the Mesopotamia PDF
    detects page and section information, cleans the text
    and splits it into overlapping chunks

    Each chunk keeps useful metadata such as the source page
    and section title so the RAG system can show where an
    answer came from

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

# os is used to create file paths
import os

# re is used for regular expressions.
# We use it to detect page markers, section titles, and extra spaces
import re

# json is used to save the final chunks in JSON format
import json


# ----------------------------------------------------------
# File and directory constants
# ----------------------------------------------------------

DATA_DIRECTORY = "data"

INPUT_FILE_NAME = "mesopotamia_text.txt"
CLEAN_TEXT_FILE_NAME = "mesopotamia_clean_text.txt"
CHUNKS_FILE_NAME = "chunks.json"


# ----------------------------------------------------------
# file paths
# ----------------------------------------------------------

INPUT_PATH = os.path.join(DATA_DIRECTORY, INPUT_FILE_NAME)
CLEAN_TEXT_PATH = os.path.join(DATA_DIRECTORY, CLEAN_TEXT_FILE_NAME)
CHUNKS_OUTPUT_PATH = os.path.join(DATA_DIRECTORY, CHUNKS_FILE_NAME)


# ----------------------------------------------------------
# Chunking settings
# ----------------------------------------------------------

# Number of words in each chunk.
# 450 is large enough to keep useful historical context,
# but small enough for embedding and retrieval.
CHUNK_SIZE = 350

# Number of words repeated between consecutive chunks.
# Overlap prevents important information from being lost
# when an idea is split between two chunks.
CHUNK_OVERLAP = 75


def parse_pages(raw_text):
    """
    Splits the extracted PDF text into separate pages

    extract_text.py adds page markers such as
    ========== PAGE 2 ==========

    We use those markers to keep track of which PDF page
    each chunk came from
    """

    # Split the text by page markers while also keeping the page number.
    pages = re.split(r"========== PAGE (\d+) ==========", raw_text)

    parsed_pages = []

    # The split result alternates between
    # text before marker, page number, page text, page number, page text...
    # The split result alternates between page numbers and page text:
    #
    # Index 1 -> Page number
    # Index 2 -> Page text
    #
    # we iterate by 2 so that each loop processes
    # one complete page (page number + page text)
    for index in range(1, len(pages), 2):
        page_number = int(pages[index])
        page_text = pages[index + 1].strip()

        parsed_pages.append({
            "page": page_number,
            "text": page_text
        })

    return parsed_pages


def detect_section_title(page_text):
    """
    Detects the section title from one PDF page

    Example:
        1. Introduction to Mesopotamia

    If a title is not found, the page is probably the cover page
    """

    # Detect a title that starts with a number and a dot.
    # Example: "4. The Akkadian Empire"
    match = re.search(r"\d+\.\s+[A-Z][^\n]+", page_text)

    if match:
        return match.group(0).strip()

    return "Cover Page / General"


def clean_text(text):
    """
    Cleans extracted PDF text

    PDF extraction can create unnecessary line breaks and
    repeated spaces. Cleaning makes the text easier for
    embedding models to process
    """

    # Replace multiple spaces and line breaks with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces.
    text = text.strip()

    return text


def create_chunks_from_pages(pages):
    """
    Splits extracted PDF pages into overlapping chunks

    Each chunk contains only useful metadata:
    - chunk_id
    - source_page
    - section_title
    - text
    """

    chunks = []
    chunk_id = 1

    # Process each page separately so the source page is preserved
    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        # Detect section title before cleaning because line breaks
        # make title detection more reliable
        section_title = detect_section_title(page_text)

        # Clean the page text before splitting it into words
        cleaned_page_text = clean_text(page_text)

        words = cleaned_page_text.split()
        start_index = 0

        while start_index < len(words):
            end_index = start_index + CHUNK_SIZE
            chunk_words = words[start_index:end_index]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "source_page": page_number,
                "section_title": section_title,
                "text": chunk_text
            })

            chunk_id += 1

            # Move forward while keeping overlap from the previous chunk.
            start_index += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def save_clean_text(pages):
    """
    Saves the cleaned full text into a separate text file
    """

    cleaned_pages = []

    for page in pages:
        cleaned_pages.append(clean_text(page["text"]))

    cleaned_text = "\n\n".join(cleaned_pages)

    with open(CLEAN_TEXT_PATH, "w", encoding="utf-8") as file:
        file.write(cleaned_text)


def save_chunks(chunks):
    """
    Saves the generated chunks into a JSON file
    """

    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)


def main():
    """
    Runs the full parsing, cleaning, and chunking process
    """

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print("Reading extracted text...")

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        raw_text = file.read()

    print("Parsing PDF pages...")
    pages = parse_pages(raw_text)

    print("Saving cleaned text...")
    save_clean_text(pages)

    print("Splitting text into chunks with metadata...")
    chunks = create_chunks_from_pages(pages)

    save_chunks(chunks)

    print("\nChunking completed successfully.")
    print(f"Clean text saved to: {CLEAN_TEXT_PATH}")
    print(f"Chunks saved to: {CHUNKS_OUTPUT_PATH}")
    print(f"Total chunks created: {len(chunks)}")


if __name__ == "__main__":
    main()