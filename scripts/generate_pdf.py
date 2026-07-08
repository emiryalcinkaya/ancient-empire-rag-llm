"""
------------------------------------------------------------
Project:
    RAG-Based Question Answering System for the
    Mesopotamian Civilization

File:
    generate_pdf.py

Author:
    Deniz Gözcü

Description:
    Generates a structured PDF containing historical
    information about Mesopotamian civilization

    The generated PDF will be used as the knowledge source
    for the Retrieval-Augmented Generation system

Course:
    Machine Learning and Smart Systems
------------------------------------------------------------
"""

# Used for creating folders and file paths.
import os

# ReportLab page size constant for standard A4 pages.
from reportlab.lib.pagesizes import A4

# ReportLab document and layout components.
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

# Provides default paragraph, title, and heading styles.
from reportlab.lib.styles import getSampleStyleSheet

# Import the historical content from the separate data file.
from mesopotamia_content import sections


# Folder where generated files will be stored
DATA_DIRECTORY = "data"

# Name of the generated historical document
OUTPUT_FILE_NAME = "mesopotamia.pdf"

# Complete output path
OUTPUT_PATH = os.path.join(DATA_DIRECTORY, OUTPUT_FILE_NAME)


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DATA_DIRECTORY = "data"
OUTPUT_FILE_NAME = "mesopotamia.pdf"
OUTPUT_PATH = os.path.join(DATA_DIRECTORY, OUTPUT_FILE_NAME)

PAGE_MARGIN = 50
TITLE_SPACING = 20
SECTION_SPACING = 12


def create_pdf_document():
    """
    Creates the PDF document object with page size and margins.

    Returns
    -------
    SimpleDocTemplate
        ReportLab document object used to build the PDF.
    """

    return SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=PAGE_MARGIN,
        leftMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN
    )



def build_pdf_story(styles):
    """
    Builds the list of PDF elements such as titles, paragraphs,
    spaces, and page breaks

    Parameters
    ----------
    styles : StyleSheet1
        ReportLab style sheet containing text styles

    Returns
    -------
    list
        A list of PDF elements that ReportLab will convert into a PDF
    """

    story = []

    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    body_style = styles["BodyText"]

    # Increase line spacing so the PDF is easier to read.
    body_style.leading = 16

    # Add the cover title.
    story.append(Paragraph("History of Mesopotamian Civilization", title_style))
    story.append(Spacer(1, TITLE_SPACING))

    # Add a short explanation of why this document exists.
    story.append(Paragraph(
        "Generated historical document for a RAG machine learning project.",
        body_style
    ))

    # Start the historical content on a new page.
    story.append(PageBreak())

    # Add each historical section to the PDF.
    for section in sections:
        story.append(Paragraph(section["title"], heading_style))
        story.append(Spacer(1, SECTION_SPACING))

        # Split each section into paragraphs using double line breaks.
        paragraphs = section["text"].strip().split("\n\n")

        for paragraph in paragraphs:
            story.append(Paragraph(paragraph.strip(), body_style))
            story.append(Spacer(1, SECTION_SPACING))

        # Each section starts on a new page for clean structure.
        story.append(PageBreak())

    return story



def generate_pdf():
    """
    Generates the Mesopotamia PDF file.

    This function creates the output folder, prepares the PDF
    structure, builds the content, and saves the final PDF.
    """

    # Create the data folder if it does not already exist.
    os.makedirs(DATA_DIRECTORY, exist_ok=True)

    # Create the PDF document object.
    document = create_pdf_document()

    # Load default ReportLab text styles.
    styles = getSampleStyleSheet()

    # Build all PDF content elements.
    story = build_pdf_story(styles)

    # Convert the story elements into a final PDF file.
    document.build(story)

    print(f"PDF created successfully: {OUTPUT_PATH}")




# ----------------------------------------------------------
# Program Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":
    generate_pdf()