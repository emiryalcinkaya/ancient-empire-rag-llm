# RAG-Based Question Answering System for Mesopotamian Civilization

A Retrieval-Augmented Generation (RAG) system that answers questions about Mesopotamian civilization using a locally generated historical document.

The project builds a searchable knowledge base from a PDF document, retrieves the most relevant information using semantic search with FAISS, and generates grounded answers using a local Llama 3 model through Ollama.

---

## Features

- Generate a historical PDF about Mesopotamian civilization
- Extract and preprocess text from the PDF
- Split the document into overlapping text chunks
- Generate semantic embeddings using Sentence Transformers
- Store embeddings in a FAISS vector database
- Retrieve the most relevant document sections for each question
- Generate grounded answers using a local Llama 3 model
- Interactive command-line interface

---

## Project Structure

```text
ancient-empire-rag-llm/
│
├── data/
│   ├── mesopotamia.pdf
│   ├── mesopotamia_text.txt
│   ├── mesopotamia_clean_text.txt
│   ├── chunks.json
│   ├── metadata.json
│   ├── embeddings.npy
│   └── faiss_index.bin
│
├── scripts/
│   ├── mesopotamia_content.py
│   ├── generate_pdf.py
│   ├── extract_text.py
│   ├── chunk_text.py
│   ├── create_embeddings.py
│   ├── build_faiss.py
│   ├── rag.py
│   └── llm.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Technologies

- Python 3.11
- Ollama
- Llama 3
- FAISS
- Sentence Transformers
- NumPy
- PyPDF
- ReportLab

---

## Workflow

1. Generate the Mesopotamian history PDF.
2. Extract and clean the document text.
3. Split the text into overlapping chunks.
4. Generate semantic embeddings.
5. Build the FAISS vector database.
6. Ask a question.
7. Retrieve the most relevant chunks.
8. Send the retrieved context to Llama 3 through Ollama.
9. Display the generated answer.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/EmirYalcinkaya/ancient-empire-rag-llm.git
cd ancient-empire-rag-llm
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install Ollama and download Llama 3:

```bash
ollama pull llama3
```

---

## Usage

Run the application:

```bash
python main.py
```

Example:

```text
============================================================
Mesopotamia RAG Assistant
============================================================

Ask a question:
Who founded the Akkadian Empire?

Answer:
Sargon of Akkad founded the Akkadian Empire around 2334 BCE.
```

---

## How It Works

The system converts the generated historical document into semantic embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model.

When a user asks a question, the same embedding model converts the question into a vector. FAISS retrieves the three most relevant document chunks based on semantic similarity.

These retrieved chunks are combined into a prompt and passed to a local Llama 3 model through Ollama, which generates an answer using only the retrieved document context.

---

## Project Information

**Course:** Machine Learning and Smart Systems

**Project Type:** Retrieval-Augmented Generation (RAG) Question Answering System