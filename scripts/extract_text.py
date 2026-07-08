"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    extract_text.py

Author:
    Deniz Gözcü

Description:
    Reads the generated Mesopotamia PDF and extracts
    all textual content into a plain text file

    The extracted text will later be cleaned and split
    into chunks for the RAG pipeline

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

# Used for checking whether files and folders exist.
import os

# PyPDF is used to read PDF documents page by page.
from pypdf import PdfReader


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DATA_DIRECTORY = "data"

PDF_FILE_NAME = "mesopotamia.pdf"
TEXT_FILE_NAME = "mesopotamia_text.txt"

PDF_PATH = os.path.join(DATA_DIRECTORY, PDF_FILE_NAME)
TEXT_OUTPUT_PATH = os.path.join(DATA_DIRECTORY, TEXT_FILE_NAME)


def extract_text_from_pdf():
    """
    Extracts all text from the generated PDF document

    Every page is read individually and appended to one
    large text string. Page separators are included so
    that the original document structure is preserved

    Returns
    -------
    string
        Complete text extracted from the PDF
    """

    # Verify that the PDF exists before attempting to read it
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Could not find the PDF file:\n{PDF_PATH}"
        )

    # Open the PDF document
    reader = PdfReader(PDF_PATH)

    extracted_text = []

    # Read each page one by one
    for page_number, page in enumerate(reader.pages, start=1):

        print(f"Reading page {page_number}...")

        page_text = page.extract_text()

        # Skip empty pages
        if not page_text:
            continue

        # Add a page separator so we know where each page begins
        extracted_text.append(f"\n========== PAGE {page_number} ==========\n")

        extracted_text.append(page_text)

    return "\n".join(extracted_text)


def save_text(text):
    """
    Saves the extracted PDF text into a text file

    Parameters
    ----------
    text : string
        Text extracted from the PDF
    """

    with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"\nText successfully saved to:\n{TEXT_OUTPUT_PATH}")


def main():
    """
    Executes the complete text extraction process
    """

    print("Starting PDF text extraction...\n")

    text = extract_text_from_pdf()

    save_text(text)

    print("\nExtraction completed successfully.")




# ----------------------------------------------------------
# Program Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":
    main()